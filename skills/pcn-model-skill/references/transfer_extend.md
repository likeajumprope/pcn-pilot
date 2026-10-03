# Transfer and extend

Both adapt an existing (reference or pretrained) model to data from a site it has not seen,
without access to the reference data. Only model parameters move between sites.

## Which one

| | Transfer | Extend |
|---|---|---|
| What it does | re-estimates the model for the new site using the reference model as prior information | synthesises data from the reference model, pools it with the local data, refits a full model |
| Result covers | the covariate range of the new data only | the reference sites and range plus the new site |
| Repeatable | no: a model should be transferred once, from the original reference | yes: site after site across a consortium |
| Cost | low | a full refit |
| Typical use | "score my patients against a published model" | "add our site to the consortium model and pass it on" |

Ask the user which outcome they need. `choose_recipe.py --reference <dir> --goal transfer|extend`
writes the recipe; the default is transfer.

## Data requirements

- **Adaptation set** (the project's `train.csv`): healthy controls from the new site. PCNtoolkit's
  guidance is 20 to 100 per site. The plan warns below 20.
- **Test set**: any size, down to one subject. Calibration checks in QC need a few dozen held-out
  controls to mean anything, so a 50/50 split of the controls is a reasonable default when
  controls are scarce (`"split": {"test_fraction": 0.5}` in the data spec).
- **Clinical subjects**: scored after adaptation, never used in it.
- Same measure, same atlas, same units and the same covariates as the reference model.

## Compatibility checks at plan time

`run_model.py plan` loads the reference model and compares it with the standardized data.

| ID | Check | FAIL means | Fix |
|---|---|---|---|
| R1 | covariate names and order equal the model's | the model cannot be applied | rename or reorder in the data spec |
| R2 | batch-effect dimensions equal the model's | same | add or rename the batch-effect columns |
| R3 | labels of each batch effect | a small dimension (such as sex) shares no label with the model, for example `0/1` against `F/M` | add `recode` to the data spec. WARN: a new level has fewer than 20 adaptation rows |
| R4 | response variables | none is shared with the model. WARN lists data variables without a fitted counterpart: they are left out of the run | match names exactly (atlas and measure) |
| R5 | covariate range | WARN: data lies outside the model's range | consider restricting the data or using extend |
| R6 | lineage | transfer requested from a model that is itself a transfer | transfer from the original reference, or use extend |

A new site label under R3 is expected and passes. R4 is the drop-check of this stage: tell the
user which variables are left out and why.

## Pretrained models

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
```

Pretrained models were trained on healthy subjects: adapt on healthy controls only.

## After adaptation

Run `pcn-qc-skill` exactly as for a fitted model. The calibration checks (mean and SD of held-out
control z-scores) are the direct test of whether the adaptation worked. If they fail, QC's
candidates are: extend instead of transfer, or fit from scratch when the site has a few hundred
controls.

A transferred model's centiles are valid only inside the new data's covariate range. Say so when
reporting scores for subjects near its edges.

## HBR transfer options

`transfer_kwargs` in the recipe is passed to PCNtoolkit's transfer for HBR models (sampler
overrides such as `draws`, `tune`, `chains`, `cores`, and `freedom`; see the PCNtoolkit documentation for their
exact meaning in your version). Leave it empty unless the user asks.
`n_synth_samples` sets the number of synthetic observations for extend.
