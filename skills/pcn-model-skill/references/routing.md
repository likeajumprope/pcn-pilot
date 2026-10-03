# Routing and the recipe

## Contents

1. Traits measured by `choose_recipe.py`
2. Decision order
3. Recipe schema
4. BLR options
5. HBR options
6. Changing a recipe by hand

## 1. Traits

`choose_recipe.py` reads the training split and reports:

| Trait | How it is measured | Used for |
|---|---|---|
| `n_train`, `n_response_vars` | counts | BLR versus HBR (compute cost) |
| `covariate_span` | range of the first covariate | linear versus spline mean |
| `n_batch_levels`, `min_level_n` | counts per batch effect | fixed effects versus hierarchical pooling |
| `frac_nonlinear` | share of variables where a cubic fits better than a line (R2 gain above 0.01) | basis |
| `frac_site_shift` | share where batch dummies add more than 0.02 R2 | batch effect in the mean |
| `frac_site_variance` | share where site residual SDs differ by more than 1.5x | batch effect in the variance |
| `frac_skewed`, `frac_heavy_tailed` | share with residual skewness above 0.5 or excess kurtosis above 1 | warp or SHASH |
| `frac_heteroskedastic` | share where residual spread correlates with the covariate | variance model |
| `all_in_unit_interval` | every value strictly between 0 and 1 | Beta likelihood |
| `repeated_measures` | duplicate subject IDs | caveat only |

Residuals come from a quick least-squares fit (cubic in the covariate plus batch dummies). They
steer the proposal; they are not the model. With more than 400 variables a random sample of 400
is examined.

## 2. Decision order

1. A reference model was given: transfer or extend (see `transfer_extend.md`). The algorithm and
   basis come from the reference model.
2. `--prefer` was given: that algorithm.
3. Every value in (0, 1): HBR with a Beta likelihood and min-max scaling.
4. More than 50 variables or more than 20,000 training rows: BLR. It fits each variable in seconds;
   HBR runs MCMC per variable.
5. At least 5 batch levels and the smallest has fewer than 30 rows: HBR, because hierarchical priors
   share strength across small sites.
6. Otherwise BLR.

Then, for BLR: fixed batch effects when batch effects exist; a variance batch effect when more than
20% of variables show site variance differences; heteroskedastic noise when more than 20% show it
or a spline basis is used; a sinh-arcsinh warp when more than 20% are skewed or heavy-tailed.
For HBR: random site intercept in the mean; SHASH likelihood under the same non-Gaussianity rule.

One recipe applies to every variable of a run. When a minority of variables need something else,
that surfaces in QC, and the fix is a second run for those variables only.

## 3. Recipe schema

```json
{
  "algorithm": "blr",
  "mode": "fit_predict",
  "reference_model": null,
  "basis": {"type": "bspline", "nknots": 5, "degree": 3, "basis_column": 0},
  "basis_var": null,
  "inscaler": "standardize",
  "outscaler": "standardize",
  "y_transform": null,
  "saveplots": false,
  "blr": {
    "n_iter": 1000, "tol": 1e-8, "optimizer": "l-bfgs-b",
    "l_bfgs_b_epsilon": 0.1, "l_bfgs_b_l": 0.1, "l_bfgs_b_norm": "l2",
    "fixed_effect": true, "fixed_effect_slope": false,
    "heteroskedastic": true, "fixed_effect_var": false, "fixed_effect_var_slope": false,
    "warp_name": "warpsinharcsinh", "warp_reparam": true
  },
  "hbr": {
    "likelihood": "Normal",
    "random_intercept_mu": true, "random_slope_mu": false,
    "linear_sigma": true, "random_intercept_sigma": false, "random_slope_sigma": false,
    "draws": 1500, "tune": 500, "chains": 4, "cores": 4, "nuts_sampler": "nutpie"
  },
  "transfer_kwargs": {},
  "n_synth_samples": null,
  "rationale": ["..."], "alternatives": ["..."], "traits": {}
}
```

- `mode`: `fit_predict`, `transfer_predict` or `extend_predict`. The last two need `reference_model`.
- `basis.type`: `linear`, `polynomial` (`degree`) or `bspline` (`nknots`, `degree`). `basis_column`
  is the index of the covariate the basis expands (0 = the first covariate in the data spec).
- `basis_var`: basis for the BLR variance; defaults to `basis` when `heteroskedastic` is true.
- `blr` keys are passed to `pcntoolkit.BLR` as written. `hbr` sampler keys (`draws`, `tune`,
  `chains`, `cores`, `nuts_sampler`, `init`) are passed to `pcntoolkit.HBR`; the other `hbr` keys
  build the priors and likelihood (`scripts/pcn_bridge.py`).
- `y_transform`: `log1p` or `log`, only in PCNtoolkit versions that support it.
- `rationale`, `alternatives`, `traits` are documentation and do not affect the fit.

Unknown keys are refused at plan time with the list of keys the installed version accepts.

## 4. BLR options in plain terms

| Option | Effect | Turn on when |
|---|---|---|
| `fixed_effect` | a separate offset per batch level in the mean | sites or sexes differ in level (almost always with multi-site data) |
| `fixed_effect_slope` | the covariate slope differs per batch level | QC still shows site shifts after offsets |
| `heteroskedastic` | variance changes along the covariate | spread grows or shrinks with age |
| `fixed_effect_var` | a variance offset per batch level | sites differ in noise level |
| `warp_name: warpsinharcsinh` | models skewness and tail weight | skewed or heavy-tailed measures (ventricles, lesion load) |
| `warp_reparam` | a better-conditioned warp parameterisation | whenever the warp is used |
| `optimizer: powell` | gradient-free optimiser | L-BFGS-B fails or returns non-finite likelihoods |

Other warps accepted by PCNtoolkit: `warpboxcox`, `warpaffine`, `warplog`, and `warpcompose(...)`.

## 5. HBR options in plain terms

| Option | Effect |
|---|---|
| `likelihood` | `Normal`; `SHASHb` adds skewness and tail parameters; `Beta` for (0, 1) data |
| `random_intercept_mu` | site-specific mean offsets drawn from a shared distribution (partial pooling) |
| `random_slope_mu` | site-specific covariate slopes; harder to sample |
| `linear_sigma` | variance depends on the covariate through the same basis |
| `random_intercept_sigma` | site-specific variance |
| `draws`, `tune`, `chains` | MCMC length. Raise `tune` first when R-hat is high |
| `cores` | parallel chains; match the cores requested per job |

Priors follow the PCNtoolkit tutorials (Normal priors on slopes and intercepts, softplus mapping
for scale parameters). To use other priors, edit `_hbr_likelihood` in `common/pcn_bridge.py`,
run `tools/sync_common.py`, and record the change in the recipe's `rationale`.

Cost: HBR samples once per response variable. Budget minutes per variable for a Normal likelihood
and more for SHASH. For hundreds of variables use a cluster backend, or BLR.

## 6. Changing a recipe by hand

Edit the JSON, keep `rationale` truthful (add a line saying who asked for what), and plan under a
new run name. To compare two recipes on the same data, fit both as separate runs and let
`pcn-qc-skill` grade both; `qc_review.py compare` puts them side by side.
