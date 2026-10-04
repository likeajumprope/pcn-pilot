# Transfer and extend

Both adapt an existing (reference or pretrained) model to data from a site it has not seen,
without access to the reference data. Only model parameters move between sites.

## Contents

1. Which one
2. What PCNtoolkit 1.3.0 can and cannot do
3. Data requirements
4. Compatibility checks at plan time (R1 to R7)
5. Pretrained models
6. After adaptation
7. Options

## 1. Which one

| | Transfer | Extend |
|---|---|---|
| What it does | keeps the reference model and re-estimates it for the new site: BLR adds a mean offset and a variance factor per batch level; HBR resamples with the reference posterior as prior | synthesises data from the reference model, pools it with the local data, refits a full model |
| Result covers | the new site only, within the covariate range of the new data | the reference sites and range plus the new site |
| Repeatable | no: a model should be transferred once, from the original reference | yes: site after site across a consortium |
| Cost | low for BLR; one MCMC run per variable for HBR | a full refit on (reference size + local) rows |
| Works from | HBR; BLR only if the reference was fitted with a warp | every BLR and HBR reference |
| Typical use | "score my patients against a published model" | "add our site to the consortium model and pass it on" |

Ask the user which outcome they need. `choose_recipe.py --reference <dir> [--goal transfer|extend]`
writes the recipe. Without `--goal` it proposes transfer, or extend when transfer is not available
for that reference, and says which and why.

## 2. What PCNtoolkit 1.3.0 can and cannot do

These were found by running the calls against an installed PCNtoolkit 1.3.0. The plan checks for
them (R7) so that nothing fails in the middle of a run. `scripts/pcn_bridge.py` holds the details.

- **A BLR model without a warp cannot be transferred.** `BLR.transfer` computes its target only
  for warped models and stops with `UnboundLocalError` otherwise (every release from 1.0 to
  1.3.0). The plan refuses with R7. Use extend, or a reference fitted with a warp. If you fit a
  reference that others will transfer, fit it with `warp_name` set.
- **`NormativeModel.extend` fails for a new site** (`KeyError` on a reference site label, 1.2.0 to
  1.3.0), because the pooled dataset keeps the batch-effect registry of the local data only.
  PCN-Pilot therefore performs extend's three steps itself with public calls: synthesise from the
  reference (`NormativeModel.synthesize`, covariate ranges per batch level), pool with the local
  rows, and fit a new model built from the reference's template, scalers and settings. The result
  is what `extend` is documented to produce. The pooled table is scored under the name
  `extend_fit` (`results/Z_extend_fit.csv`, synthetic rows have IDs `synth_...`); `Z_train.csv`
  holds the real local rows only.
- **`y_transform` breaks centiles**, and an adapted model inherits it from its reference. R7 fails
  for such a reference.
- **A transfer always writes PCNtoolkit's plots** (`<run>/plots/`), whatever `saveplots` says, and
  the adapted model is saved with plotting switched on. PCN-Pilot switches it off again whenever it
  reloads the model, because plotting fails when a reloaded model predicts a subset of its
  variables. Load an adapted model yourself and you must do the same (`model.saveplots = False`).
- **Options are swallowed.** `transfer(**kwargs)` ignores names it does not know. The plan refuses
  any `transfer_kwargs` entry the reference's algorithm does not read (section 7).

## 3. Data requirements

- **Adaptation set** (the project's `train.csv`): healthy controls from the new site. PCNtoolkit's
  guidance is 20 to 100 per site. The plan warns below 20.
- **Test set**: any size, down to one subject. Calibration checks in QC need a few dozen held-out
  controls to mean anything, so a 50/50 split of the controls is a reasonable default when
  controls are scarce (`"split": {"test_fraction": 0.5}` in the data spec).
- **Clinical subjects**: scored after adaptation, never used in it.
- Same measure, same atlas, same units and the same covariates as the reference model.
- Extend can only help with site differences the reference's model structure can express. A BLR
  reference fitted without batch effects (`fixed_effect` false) has no parameter for a site
  offset: extending it pools the sites but cannot centre the new one. QC will show `Z_MEAN` or
  `SITE_MEAN`; the repair is a fresh fit with batch effects, not another extend.

## 4. Compatibility checks at plan time

`run_model.py plan` loads the reference model and compares it with the standardized data.

| ID | Check | FAIL means | Fix |
|---|---|---|---|
| R1 | covariate names and order equal the model's | the model cannot be applied | rename or reorder in the data spec |
| R2 | batch-effect dimensions equal the model's | same | add or rename the batch-effect columns |
| R3 | labels of each batch effect | a small dimension (such as sex) shares no label with the model, for example `0/1` against `F/M` | add `recode` to the data spec. WARN: a new level has fewer than 20 adaptation rows |
| R4 | response variables | none is shared with the model. WARN lists data variables without a fitted counterpart: they are left out of the run | match names exactly (atlas and measure) |
| R5 | covariate range | WARN: data lies outside the model's range | consider restricting the data or using extend |
| R6 | lineage | transfer requested from a model that is itself a transfer | transfer from the original reference, or use extend |
| R7 | the installed PCNtoolkit can run this route from this reference | transfer from a BLR model without a warp; a reference fitted with `y_transform`; extend from a model saved without the batch-effect counts and ranges that synthesis needs | change the route (transfer to extend), or use another reference |

A new site label under R3 is expected and passes. R4 is the drop-check of this stage: tell the
user which variables are left out and why. R7 names the reference's algorithm and warp on PASS, so
the user can see what is being adapted.

## 5. Pretrained models

PCNtoolkit publishes pretrained lifespan models (cortical thickness and others; see the
"Transfer a pretrained model to your own data" tutorial in the PCNtoolkit documentation for the
current list and download location). Download and unzip one, and pass the unzipped folder (the one containing `model/normative_model.json`) as
`--reference`.

Before building the dataset, load the model's expectations so the spec matches them:

```python
from pcntoolkit import NormativeModel
m = NormativeModel.load("<folder>")
print(m.covariates)                       # e.g. ['age']
print(list(m.unique_batch_effects))       # e.g. ['sex', 'site']
print(m.unique_batch_effects['sex'])      # the labels the model knows
print(m.response_vars[:10])               # exact variable names
print(type(m.template_regression_model).__name__,                 # BLR or HBR
      getattr(m.template_regression_model, "warp_name", None))    # None: a BLR model that needs extend
```

Pretrained models were trained on healthy subjects: adapt on healthy controls only. A reference
model is never written to: results go to the run folder.

## 6. After adaptation

Run `pcn-qc-skill` exactly as for a fitted model. The calibration checks (mean and SD of held-out
control z-scores) are the direct test of whether the adaptation worked. If they fail, QC's
candidates are: extend instead of transfer, or fit from scratch when the site has a few hundred
controls.

A transferred model's centiles are valid only inside the new data's covariate range, and it knows
only the new site's batch labels. Say so when reporting scores for subjects near the edges.

## 7. Options

`transfer_kwargs` in the recipe is passed to the reference's transfer:

| Reference | Options read by PCNtoolkit 1.3.0 |
|---|---|
| HBR | `freedom` (multiplies the width of the priors built from the reference posterior: above 1 lets the new site move further from the reference, below 1 holds it closer; default 1), and sampler overrides `draws`, `tune`, `chains`, `cores`, `nuts_sampler` |
| BLR | none: any entry is refused |

Leave it empty unless the user asks.

For extend: `n_synth_samples` sets the number of synthetic rows (default: as many as the reference
was fitted on, which keeps the reference's weight in the pooled fit), and `seed` makes them
reproducible. In a local run each variable gets its own synthetic rows, derived from the seed and
the variable's name; in a cluster run one synthetic table is drawn for the submitted variables,
derived from the seed and their names (recorded in `submissions.jsonl`).
