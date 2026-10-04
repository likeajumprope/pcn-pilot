---
name: pcn-data-skill
description: Turn raw tabular data (demographics, FreeSurfer or other imaging-derived measures, clinical labels) into a validated, standardized dataset ready for normative modelling with PCNtoolkit. Use this skill whenever the user wants to prepare, clean, merge, split or check data for normative models, z-scores, centiles or brain charts, mentions PCNtoolkit input data, NormData, covariates, batch effects or site effects, or hands over CSV/TSV/Excel tables of regional brain measures, even if they only say "get my data ready" or "can I fit a normative model on this".
---

# pcn-data-skill: data standardization for PCNtoolkit

Stage 1 of three. It produces the dataset that `pcn-model-skill` fits and `pcn-qc-skill` reviews.
You handle the mechanics; the user makes every decision that changes which subjects or variables
enter the model. Never guess those.

`$SKILL` below is the folder that holds this file. `$PCN_PYTHON` is the interpreter named in
`pipeline.env` (any Python with pandas works for this stage; only step 6 needs PCNtoolkit).

## What comes out

A project folder (the user chooses where). Everything later stages need is under `<project>/data/`:

| File | Content |
|---|---|
| `spec.json` | the confirmed column roles and policies (the "conversion config") |
| `clean.csv`, `train.csv`, `test.csv`, `clinical.csv` | standardized tables; `clinical.csv` only if non-reference subjects exist |
| `dropped.csv` | every observation that was left out, with the reason |
| `build_manifest.json` | counts, source checksums, split seed |
| `validation.json`, `data_report.md` | the 14 checks and the human-readable report |

Sources are read-only. Only columns named in the spec are copied, so identifying columns never
reach the project.

## Workflow

Run the steps in order. Do not skip the checkpoints: they are what lets you report verified
state instead of an assumption.

**1. Pre-run check.** Confirms inputs exist and the project folder is writable.

```
$PCN_PYTHON $SKILL/scripts/check_status.py pre data --project <project> --input <table> [<table> ...]
```

**2. Inspect and profile.** Proposes a role for every column and measures ID overlap between tables.

```
$PCN_PYTHON $SKILL/scripts/profile_table.py --project <project> --input <table> [...] --out <project>/data/profile.json
```

Read the profile. Look at example values yourself when a role is uncertain (`*_candidate`, low
confidence). Roles: `id`, `covariate`, `batch_effect`, `reference_group`, `response_var`, `phi`, `ignore`.

**3. Collect every open decision in one round-trip.** Present the proposed mapping and ask all of
these together, not one prompt at a time:

- project folder and dataset name
- which column identifies subjects in each table, and the ID normalisation (the profile shows how
  many IDs match under `strip`, `alnum`, `alnum_lower`)
- covariates (age and its unit; any others) and batch effects (site/scanner, sex)
- how categorical codes map to labels, for example sex `0/1` to `F/M`
- which response variables to model (explicit list, a regex, or all numeric). Names ending in
  `_old`, and the names `observations`, `subject_ids`, `centile` and `statistic`, cannot be used:
  PCNtoolkit's result files would lose them (check V14). Rename such a column in a copy of the source
- the reference group: which column and values mark the population the norm should describe
  (usually healthy controls). Everyone else is scored but never used to fit
- repeated measures: is there a visit column, and is a longitudinal design intended
- missing data (default: drop variables with more than 20% missing, then complete cases)
- outliers (default: flag only; removal is opt-in)
- test fraction and seed (default 0.2 and 42)
- columns flagged `phi`: confirm they are excluded

**4. Write `<project>/data/spec.json`.** Field-by-field reference: `references/spec.md`.

**5. Dry run.** `build_dataset.py --project <project> --dry-run` prints the counts without writing.
If the numbers surprise you or the user (many unmatched IDs, most rows dropped), stop and resolve
it before building.

**6. Build, then self-test against PCNtoolkit.**

```
$PCN_PYTHON $SKILL/scripts/build_dataset.py --project <project>
$PCN_PYTHON $SKILL/scripts/normdata_selftest.py --project <project> --fit-one
```

The self-test builds `NormData` from a few variables and fits a minimal BLR on one, so a format
problem shows up now and not in job 37 of 200.

**7. Validate and report.**

```
$PCN_PYTHON $SKILL/scripts/validate_dataset.py --project <project>
$PCN_PYTHON $SKILL/scripts/check_status.py post data --project <project>
```

Exit code 1 means a FAIL check: fix the cause (usually in the spec) and rebuild. The meaning of
each check and the usual remedy: `references/validation_checks.md`.

**8. Report to the user.** Summarise from `data_report.md`: subjects kept and dropped with reasons,
variables dropped, and every WARN. Each WARN needs an explicit decision from the user (accept, or
change the spec and rebuild). Record what they decided in your summary.

## Decisions that are the user's, not yours

- including or excluding any subject, site or variable
- the reference group definition
- removing outliers
- accepting a WARN

To exclude something after the fact, use the dismissal list. It deletes nothing and every later
step honours it:

```
$PCN_PYTHON $SKILL/scripts/dismiss.py --project <project> --kind subjects|response_vars|batch_levels \
    --id <id> [<id> ...] --reason "<why>" --by "<name (ID)>"
```

Then rebuild and re-validate. Models fitted on an earlier build are never mixed with a new one:
the model stage will ask for a new run name.

## Rules

- Never edit, move or delete a source table. Never write inside the sources' folders.
- Never copy a column that is not in the spec. Never print identifying values in your messages.
- A build without `data_report.md` is incomplete. Do not hand over to the model stage until
  `check_status.py post data` returns OK.
- If two sources disagree (duplicate columns, IDs that match only partly), show the user the
  numbers and ask. Do not pick silently.
- Every script appends to `<project>/audit/commands.jsonl`. Do not edit that file.

## Reference files

- `references/spec.md`: every field of `spec.json`, with examples for common layouts
  (FreeSurfer stats tables, BIDS `participants.tsv`, longitudinal data)
- `references/validation_checks.md`: checks V01 to V14, what each means for a normative model,
  and what to do
