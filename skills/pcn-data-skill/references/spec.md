# `spec.json` reference

The spec is the single description of how source tables become the standardized dataset.
It is written once the user has confirmed the decisions in step 3 of `SKILL.md`, and it is the
only thing `build_dataset.py` reads besides the sources and the dismissal list.

## Full example

```json
{
  "name": "mycohort",
  "sources": [
    {"path": "/data/study/participants.tsv", "id_column": "participant_id"},
    {"path": "/data/study/derivatives/aparc_thickness_lh.tsv", "id_column": "lh.aparc.thickness"},
    {"path": "/data/study/clinical.xlsx", "id_column": "ID", "sheet": "baseline"}
  ],
  "id_normalization": "strip_prefix:sub-",
  "visit_column": null,
  "covariates": ["age"],
  "batch_effects": ["sex", "site"],
  "response_vars": {"regex": "^(lh|rh)_.*_thickness$"},
  "recode": {"sex": {"0": "F", "1": "M"}},
  "reference": {"column": "diagnosis", "values": ["HC", "CN"]},
  "missing": {"max_feature_missing_fraction": 0.2},
  "outliers": {"policy": "flag", "threshold": 5.0},
  "split": {"test_fraction": 0.2, "seed": 42, "min_train_per_stratum": 5},
  "min_n_per_batch_level": 10
}
```

## Fields

| Field | Required | Meaning |
|---|---|---|
| `name` | no | dataset name used in reports |
| `sources[].path` | yes | csv, tsv, txt, xlsx or parquet. Delimiter is detected |
| `sources[].id_column` | yes | subject ID column in that table |
| `sources[].sheet` | no | Excel sheet name or index |
| `id_normalization` | no | `strip` (default), `none`, `alnum`, `alnum_lower`, `strip_prefix:<text>` |
| `visit_column` | no | set for longitudinal data; tables that both have it are joined on ID + visit |
| `covariates` | yes | numeric columns the model conditions on. The first one is the main axis (age) |
| `batch_effects` | no | categorical columns: site, scanner, sex. Stored as strings |
| `response_vars` | yes | a list, `{"regex": "..."}`, or `{"all_numeric": true}` |
| `recode` | no | per column, source value to label. Every value present must be covered |
| `reference` | no | column and values that define the reference group. Absent = everyone |
| `missing.max_feature_missing_fraction` | no | variables above it are dropped and reported (default 0.2) |
| `outliers.policy` | no | `flag` (default) or `remove_rows` |
| `outliers.threshold` | no | modified Z (Iglewicz and Hoaglin) above which a value is flagged (default 5) |
| `split.test_fraction`, `split.seed` | no | defaults 0.2 and 42 |
| `split.min_train_per_stratum` | no | small sex x site cells keep at least this many subjects in train (default 5) |
| `min_n_per_batch_level` | no | levels with fewer training observations get a WARN (default 10) |

## How the build works

1. Each source is read and its ID column normalised. Sources are joined on subject ID (and visit).
   Subjects missing from any source go to `dropped.csv`.
2. Recoding is applied; covariates and response variables are coerced to numbers.
3. Only the ID, visit, covariates, batch effects, the reference column and the response variables
   are kept. Nothing else is copied.
4. Dismissed subjects, variables and batch levels are removed (`dismissed.json`).
5. Variables that are too sparse are dropped; then rows with any missing value are dropped
   (complete cases). Both are recorded.
6. Outliers are flagged, and removed only under `remove_rows`.
7. The reference group is split into train and test: stratified on the batch-effect combination and
   grouped by subject, so a subject is never on both sides. Non-reference subjects go to
   `clinical.csv`.

The standardized tables always use `subject_id` for the ID and add `pcn_group`
(`reference` or `clinical`).

## Choosing the ID normalisation

`profile_table.py` reports matches between tables under each mode. Choose the weakest mode that
matches everything you expect to match:

- `strip`: whitespace only
- `strip_prefix:sub-`: BIDS IDs against bare IDs
- `alnum`: removes separators, so `001_S_1000` equals `001S1000`
- `alnum_lower`: also ignores case

If a stronger mode makes two different subjects collide, the profile reports it under
`id_normalisation_collisions`. Do not use that mode.

## Common layouts

**FreeSurfer `aparcstats2table` / `asegstats2table`.** The first column is named after the
measure (`lh.aparc.thickness`, `Measure:volume`) and holds the subject ID: use it as `id_column`.
Left and right hemisphere tables are separate sources. Column names already differ by hemisphere.

**BIDS `participants.tsv`.** `participant_id` values start with `sub-`. Use
`"id_normalization": "strip_prefix:sub-"` when the measure tables use bare IDs.

**Longitudinal data.** Set `visit_column`. The split is still by subject. BLR and HBR treat
observations as independent, so tell the user that uncertainty will be somewhat too narrow, and
ask whether they want one visit per subject instead (filter the source, or dismiss rows).

**Several feature sets with different missingness.** Complete-case filtering over a wide table can
discard most subjects. Create one project per feature set (for example thickness, volumes,
diffusion) so each keeps its own subjects.

**Transfer to a pretrained model.** Covariate names, batch-effect names and labels, and
response-variable names must equal the model's exactly. Recode here (for example sex to `F`/`M`),
and rename columns in a copy of the source if needed. `pcn-model-skill` checks this before fitting.

## Why the reference group matters

A normative model describes a reference population. Fitting on patients moves the norm toward the
patients and shrinks their deviations. That is why non-reference subjects are held out of both
train and test: train defines the norm, test checks its calibration on unseen reference subjects,
and clinical subjects are only scored.
