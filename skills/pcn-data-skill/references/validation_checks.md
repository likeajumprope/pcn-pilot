# Validation checks V01 to V14

`validate_dataset.py` writes the result of each check to `validation.json` and `data_report.md`.
PASS needs nothing. FAIL blocks the model stage. WARN needs a decision from the user: accept it
and say why, or change the spec and rebuild.

| ID | Check | FAIL when | WARN when |
|---|---|---|---|
| V01 | columns present and complete | a required column is absent or has missing cells | |
| V02 | response variables vary | non-finite values or zero variance | fewer than 10 distinct values |
| V03 | no leakage | a subject is in both train and test | repeated measures exist |
| V04 | batch levels in train | a level in test or clinical is absent from train | a level has fewer than `min_n_per_batch_level` training rows |
| V05 | covariate coverage | | test or clinical rows outside the training range; a batch level covering under 20% of the range |
| V06 | batch labels | labels that differ only by case or whitespace | a site that contains a single sex (or the reverse) |
| V07 | extreme values | | any value beyond the modified-Z threshold |
| V08 | drop-check | a row or a requested variable is unaccounted for | variables were dropped with a recorded reason |
| V09 | PHI | an identifying column is in the standardized tables | |
| V10 | units | | age above 130 or negative; negative values in response variables |
| V11 | sample size | fewer than 30 training rows or no test rows | fewer than 200 training or 50 test rows |
| V12 | reference purity | non-reference rows in train or test | no reference group declared |
| V13 | split balance | | a covariate differs by more than 0.3 SD between train and test |
| V14 | storable names | a response-variable name ends in `_old`, equals `observations`, `subject_ids`, `centile` or `statistic`, contains a path separator, or has surrounding whitespace | |

## What each one means for a normative model, and what to do

**V03 repeated measures.** The split is by subject, so there is no leakage. The models still treat
rows as independent. Options: keep (state the caveat), or keep one visit per subject.

**V04 small or missing batch levels.** A level absent from train cannot be predicted: the model
has no parameter for it, and PCNtoolkit refuses the whole table ("Data is not compatible with the
model"), with BLR and HBR alike. Either dismiss the level, or move toward transfer/extend, which is
designed for new sites. A level with few training rows is estimated poorly by BLR fixed effects;
HBR pools small sites, and `choose_recipe.py` takes this into account. Tell the user which sites
are small.

**V05 coverage.** Rows outside the training covariate range get extrapolated z-scores. Report
how many. A site that covers a narrow age band is confounded with age: its site effect and the
age effect cannot be separated there, which matters most when that site is a large share of one
age range. The user decides whether to keep it.

**V06 confounded batch effects.** If a site scanned only one sex, site and sex effects cannot be
separated for that site. Usually acceptable; the user should know.

**V07 extreme values.** The threshold is on the raw distribution, ignoring age, so it is
deliberately loose: it catches unit errors and failed segmentations, not ordinary tails. Skewed
measures (lesion volumes) will show many flags that are not errors. Look at the listed values. For
a true error: fix the source, or dismiss the subject. Removal by rule (`remove_rows`) is the
user's choice and is recorded row by row in `dropped.csv`.

**V08 drop-check.** This is the backstop against silent loss. A WARN here lists variables dropped
for sparsity or dismissal: confirm with the user that losing them is intended. A FAIL means the
bookkeeping does not add up and the build must not be used.

**V10 units.** Mixed age units across sites (years at one, months at another) produce a model that
looks fine and is wrong. If the range looks odd, check per site.

**V11 sample size.** Centile tails (5th, 95th) need many observations. With a small reference
sample, prefer transferring a pretrained model over fitting from scratch.

**V12 no reference group.** If the table holds only healthy subjects this is fine; say so in the
summary. If it holds a mix and no label exists, the model is not a norm. Ask.

**V14 storable names.** PCNtoolkit keeps one result file per split and merges each newly fitted
variable into it. During that merge the previous copy of a shared column gets the suffix `_old`
and every column ending in `_old` is then dropped, so a variable called `hippocampus_old` loses
its z-scores as soon as the next variable is written (verified with PCNtoolkit 1.3.0). The other
refused names are the files' own index columns, and names that cannot be a folder name (each
model is saved in a folder named after its variable). Fix: rename the column in a copy of the
source table and point the spec at the copy; sources themselves stay untouched.

## Batch-level dismissal

Batch levels are dismissed as `column=value`:

```
dismiss.py --project <project> --kind batch_levels --id site=Tiny --reason "6 subjects, cannot estimate" --by "A. Reviewer (u123)"
```

All rows of that level are then dropped at the next build and listed in `dropped.csv`.
