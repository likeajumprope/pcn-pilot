# The five checks: codes, thresholds, repairs

## Contents

1. Inputs
2. Statistical guards
3. Flag codes by step
4. Repair ladder
5. Reading the diagnostic plot
6. Metrics in `metrics.csv`

## 1. Inputs

`qc_detect.py` reads, and never writes to, the run folder:

- `results/Z_test.csv`: z-scores of held-out reference subjects (the basis of S2, S4, S5)
- `results/Z_train.csv`: for the train/test comparison
- `results/statistics_test.csv`: PCNtoolkit's own metrics (S3)
- `results/centiles_test.csv`: centile values per subject (centile ordering, finite values)
- `model/<variable>/regression_model.json`, `idata.nc`: existence, BLR likelihood, MCMC diagnostics
- `<project>/data/test.csv`, `train.csv`: covariates and batch labels of the same rows

Z-score rows are matched to the test table by row order, and the match is verified against the
subject IDs. If that fails it falls back to joining on unique IDs; if that fails too, S4 and S5
are SKIPPED and say why.

## 2. Statistical guards

A threshold alone would flag noise in small samples and miss small but certain effects in large
ones, so most rules need both an effect size and significance:

- mean z: beyond the limit AND more than 3 standard errors from 0
- SD of z: outside the band AND more than 3 x 1/sqrt(2n) from 1
- skewness / kurtosis: beyond the limit AND more than 3 standard errors (sqrt(6/n), sqrt(24/n))
- batch level: at least 10 subjects; mean beyond the limit AND more than 3.5 standard errors
- fewer than 50 held-out subjects: FAIL is downgraded to WARN for S2, S4, S5 and the card says so
- values with |z| above 10 are reported once as `EXTREME_Z` and left out of the other statistics,
  so one corrupt value produces one clear flag

## 3. Flag codes

Thresholds are in `assets/qc_thresholds.json`; the numbers below are its defaults (warn / fail).

### S1 Completion and convergence

| Code | Rule | Meaning |
|---|---|---|
| `NOT_COMPLETED` | no saved model or no test z-scores: FAIL | the fit failed or never ran; the error text from the model stage is attached |
| `NONFINITE_Z` | any NaN or infinite z, or a non-finite BLR likelihood: FAIL | numerical breakdown |
| `HBR_RHAT` | max R-hat above 1.05 / 1.10 | chains disagree: the posterior is not reliably sampled |
| `HBR_ESS` | min effective sample size below 400 / 100 | too few independent draws |
| `HBR_DIVERGENT` | any divergent transition / above 1% | the sampler could not follow the posterior geometry |

MCMC diagnostics come from `idata.nc` through arviz. If it is missing or arviz is not installed
the check is SKIPPED for HBR models: say so.

PCNtoolkit 1.3.0 writes only the posterior group to `idata.nc`. R-hat and effective sample size
are recomputed from those draws (constants and empty variables, such as the offset of a batch
effect with a single level, are left out). The sampler statistics are not in the file, so:

- local runs: `run_model.py` reads the divergence count from the model in memory right after each
  fit and stores it in `status/<variable>.json`; `HBR_DIVERGENT` is evaluated from that record
- cluster runs, and models fitted elsewhere: there is no record. S1 lists "divergent transitions"
  under SKIPPED for those variables. Say so; if it matters, refit the variable locally

### S2 Calibration (held-out reference subjects)

| Code | Rule | Meaning |
|---|---|---|
| `EXTREME_Z` | any |z| above 10: FAIL | almost always a data error (units, failed segmentation); lists the subjects |
| `Z_MEAN` | abs mean above 0.15 / 0.30 | the norm is shifted: everyone looks slightly abnormal in one direction |
| `Z_SD` | outside 0.85 to 1.15 / 0.70 to 1.30 | predicted variance too small (deviations overstated) or too large (understated) |
| `Z_SKEW` | abs skewness above 0.5 / 1.0 | asymmetric response the model treats as symmetric |
| `Z_KURT` | abs excess kurtosis above 1 / 3 | tails heavier or lighter than modelled |
| `Z_TAILS` | share with abs z above 1.96 outside 2% to 9% (binomial p below 0.001): WARN | too many or too few "abnormal" reference subjects |
| `MACE` | mean absolute centile error above 0.05 / 0.10 | centile curves do not contain the stated share of subjects |

### S3 Fit relative to the cohort

| Code | Rule | Meaning |
|---|---|---|
| `WORSE_THAN_BASELINE` | MSLL above 0: WARN | no better than a constant mean and variance |
| `NEG_EXPV` | explained variance below 0: FAIL | predictions are worse than the mean |
| `COHORT_OUTLIER` | modified Z of MSLL, EXPV or MACE beyond 3.5 on the bad side: WARN | this variable fits much worse than its peers in the same run |

Modified Z is 0.6745 x (x - median) / MAD (Iglewicz and Hoaglin). It needs at least 10 variables
in the run; smaller runs (for example fix runs) use the absolute floors only.

### S4 Batch effects

| Code | Rule | Meaning |
|---|---|---|
| `SITE_MEAN` | a level's mean z beyond 0.3 / 0.6 | a site or sex offset was not removed: deviations there are biased |
| `SITE_SD` | a level's SD outside 0.7 to 1.3, with 30+ subjects: WARN | noise level differs by site and the model assumes it does not |

### S5 Covariate structure and centiles

| Code | Rule | Meaning |
|---|---|---|
| `COV_TREND` | Spearman of z with the covariate beyond 0.15, p below 0.001: WARN | the mean curve is too rigid |
| `COV_SPREAD` | Spearman of abs z with the covariate beyond 0.15: WARN | variance changes along the covariate and the model's does not |
| `COV_BINS` | mean z in a covariate quintile beyond 0.4 and significant: WARN | local misfit (often at the ends of the age range) |
| `CENTILE_CROSS` | centiles out of order for any subject / more than 1%: WARN / FAIL | invalid centile curves |
| `CENTILE_NONFINITE` | any saved centile value is infinite or missing: FAIL | the centile file is unusable. With PCNtoolkit up to 1.3.0 this is what `y_transform` produces (the model stage refuses that option for this reason) |
| `OVERFIT` | SD of z on test exceeds train by more than 0.15: WARN | the model fits training data better than new data |

Test subjects outside the training covariate range are reported once as a dataset note, since
that affects every variable alike.

## 4. Repair ladder

Candidates are ranked: cheapest and most targeted first. Each is a change to the run's recipe.

| Finding | BLR candidates | HBR candidates |
|---|---|---|
| skew, kurtosis, tails, MACE | sinh-arcsinh warp | SHASH likelihood |
| SD off, spread changes with covariate | heteroskedastic variance | covariate-dependent sigma |
| site mean shift | fixed batch effect; then batch-specific slope; then HBR | random site intercept |
| site variance differs | batch effect on the variance; then HBR | random site effect on sigma |
| trend or local misfit along the covariate | B-spline basis; then more knots | same |
| overfitting, crossing centiles | fewer knots | fewer knots |
| not completed, non-finite | Powell optimiser; drop the warp (more iterations only when the recipe uses `optimizer: cg`, the one optimiser that reads `n_iter` and `tol`) | longer warm-up; simpler hierarchy; BLR |
| R-hat, ESS, divergences | | longer warm-up and more draws; simpler hierarchy; BLR |
| extreme z | review the input data first | review the input data first |

For transferred or extended models with a shifted mean or variance: use extend instead of
transfer, or fit from scratch if the site has a few hundred controls.

"Review this variable's input data" is not a refit. Go to `pcn-data-skill`: look at the listed
subjects in the source, then correct the source or dismiss the subject, rebuild the dataset and
start a new run.

Things that are not repairs: changing thresholds, removing test subjects until the statistics
pass, or clipping z-scores.

## 5. Reading the diagnostic plot

One figure per flagged variable (`qc/<run>/plots/`), four panels:

- **z-scores along the covariate**: points should scatter evenly around 0 between the dashed lines
  at plus and minus 1.96. The black line (binned mean) should stay flat near 0: a curve means
  `COV_TREND` / `COV_BINS`; a funnel shape means `COV_SPREAD`
- **QQ plot**: points on the diagonal mean normal z-scores. An S-shape is heavy or light tails; a
  bow is skewness
- **Batch levels**: mean z with a 95% interval for the levels furthest from 0. Intervals that
  exclude 0 by a wide margin are `SITE_MEAN`
- **Raw response values**: train in grey, test in blue. Use it to spot unit errors, floor or
  ceiling effects and gaps in coverage

## 6. Metrics in `metrics.csv`

`n_test`, `z_mean`, `z_sd`, `z_skew`, `z_kurt`, `shapiro_w`, `frac_abs_z_gt_1_96`,
`frac_abs_z_gt_3`, `mace_z`, `n_extreme_z`; PCNtoolkit's `EXPV`, `MSLL`, `SMSE`, `Rho`, `RMSE`,
`R2`, `MACE`; `rho_z_<cov>`, `rho_absz_<cov>`, `max_bin_mean_z_<cov>`; `centile_cross_frac`,
`centile_nonfinite_frac`;
`z_sd_train`, `z_sd_gap`; `rhat_max`, `ess_min`, `divergent_frac` (empty with PCNtoolkit 1.3.0, see
S1); then the grade, the status of each step and the flag codes.

PCNtoolkit 1.3.0 writes these rows to `statistics_<name>.csv`: `EXPV`, `Kurtosis`, `MACE`, `MAPE`,
`MLL`, `MSLL`, `R2`, `RMSE`, `Rho`, `Rho_p`, `SMSE`, `ShapiroW`, `Skewness`.
