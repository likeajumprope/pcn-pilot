---
name: pcn-qc-skill
description: Quality-control fitted normative models (PCNtoolkit) with a human in the loop. Grades every response variable in five steps (convergence, calibration of z-scores, fit relative to the cohort, residual site effects, covariate structure and centiles), proposes ranked repairs, serves a review dashboard, records every decision under the reviewer's name, and gates the final export behind a signed, tamper-evident ledger. Use this skill whenever the user wants to check, validate, review, QC, debug or sign off a normative model or its z-scores, asks whether a model is well calibrated or why deviations look wrong, wants to fix a badly fitting model, or wants to export deviation scores for analysis.
---

# pcn-qc-skill: detect, propose, approve, fix, verify, sign

Stage 3 of three. Input: a run from `pcn-model-skill` whose post-check found at least one
completed variable. Output: a signed export holding only the variables a named person accepted,
plus a report of everything that was dismissed.

`$SKILL` is the folder that holds this file. Use `$PCN_PYTHON` from `pipeline.env`; detection and
review need only pandas, numpy, scipy and matplotlib (arviz for HBR convergence).

## Two rules that hold throughout

**Rule 1. Detection and fixing stay on separate tools.** This skill computes metrics, localises
problems and prepares repairs. A repair is always a new recipe that `pcn-model-skill` fits under
a new run name. Never edit a z-score, a result file or a model file. Never refit from here.

**Rule 2. Grading is cohort-relative, with absolute floors.** There is no universal threshold for
"a good normative model", so a variable is compared with the other variables of its run (modified
Z) and conservative absolute limits act as a safety net. Thresholds are fixed globally in
`assets/qc_thresholds.json` and are not tuned per cohort.

## What the agent owns and what the human owns

| Agent | Human |
|---|---|
| compute metrics, localise defects, draw the plots | pass, accept with caveat, fix, dismiss, escalate |
| propose ranked repair candidates and explain them | which candidate to apply |
| prepare the fix recipe, hand it to the model skill, verify the result | whether the result is good enough |
| build the dashboard and the report | the signed export |

Your grade is a recommendation. Record a decision only when a person has made it, under that
person's name and institutional ID. Do not invent either.

## Ask first: where results go and who signs

Before any check, settle with the user:

1. the export name (a new folder under `<project>/export/`; exports are never overwritten)
2. who reviews (name and institutional ID) and who the supervisor is for escalations
3. whether they will review in the dashboard or have you record decisions they dictate

## Workflow

**1. Pre-check and detect.**

```
$PCN_PYTHON $SKILL/scripts/check_status.py pre qc --project <project> --run <run>
$PCN_PYTHON $SKILL/scripts/qc_detect.py --project <project> --run <run> [--plots flagged|all|none]
```

Five steps per response variable, each PASS, WARN, FAIL or SKIPPED:

| Step | Question | Main evidence |
|---|---|---|
| S1 Completion and convergence | did it fit; are z-scores finite; did MCMC converge | model and result files, R-hat, ESS, divergences (local runs) |
| S2 Calibration | do held-out reference z-scores look like N(0, 1) | mean, SD, skewness, kurtosis, tail share, centile error, extreme values |
| S3 Fit relative to the cohort | is this variable an outlier among its peers | MSLL, explained variance, MACE as modified Z; floors at MSLL 0 and EXPV 0 |
| S4 Batch effects | is any site or sex level left shifted or rescaled | mean and SD of z per level |
| S5 Covariate structure and centiles | any trend or spread left along the covariate; do centiles cross; train versus test | Spearman of z and of abs z, binned means, centile order, SD gap |

SKIPPED means the check could not be computed, with the reason. It is not a pass: tell the user
what was not checked. One SKIPPED is expected for HBR models fitted by cluster jobs: PCNtoolkit
saves only the posterior draws, so divergent transitions are known only for local runs, where the
model stage records them at fit time. Say so.
Codes, formulas and thresholds: `references/checks.md`.

**2. Summarise for the user.** Counts per grade, the dataset notes, the most common flags, and
anything SKIPPED. Lead with what needs their attention.

**3. Review.** Build and serve the dashboard:

```
$PCN_PYTHON $SKILL/scripts/qc_review.py serve --project <project> --run <run> [--port 8765]
```

It listens on 127.0.0.1 only and prints the SSH tunnel command for remote machines. Each card
shows the grade, the five steps with their findings, the diagnostic plot, the metrics and the
ranked repair candidates. The reviewer enters name and ID once; every decision is saved when made.
Keyboard: `j`/`k` move, `p` pass, `a` accept with caveat, `f` fix, `d` dismiss, `e` escalate.

Without a server, `qc_review.py dashboard` writes a standalone page; decisions are then downloaded
as a file and loaded with `qc_review.py import`. Or record a dictated decision:

```
$PCN_PYTHON $SKILL/scripts/qc_review.py decide --project <project> --run <run> --response-var <v> \
    --decision pass|accept_warn|fix|dismiss|escalate [--fix-id A] [--note "<reason>"] --by "<name>" --id "<ID>" [--role supervisor]
```

The decision rules are enforced by the store, in every path: name and ID are mandatory; passing a
variable you graded FAIL is an override and needs a written reason; accept with caveat, dismiss
and escalate need a reason; an escalated variable can only be closed by a supervisor who is not
the person who escalated it. Details: `references/review.md`.

**4. Fix (delegate).**

```
$PCN_PYTHON $SKILL/scripts/qc_review.py fix --project <project> --run <run>
```

Groups the approved fixes, writes one recipe and variable list per fix under
`qc/<run>/fixes/<run>_fixN/`, and prints the `pcn-model-skill` commands. Follow that skill's
workflow for each (plan, approval, run, post-check). Candidates labelled "Review this variable's
input data" go back to `pcn-data-skill` instead: inspect the subject, dismiss or correct, rebuild,
and start a new run.

**5. Verify.** Grade the fix run and compare:

```
$PCN_PYTHON $SKILL/scripts/qc_detect.py --project <project> --run <run>_fixN
$PCN_PYTHON $SKILL/scripts/qc_review.py compare --project <project> --before <run> --after <run>_fixN
```

Show the before/after table. A fix is never accepted automatically, improved or not: the reviewer
decides in the fix run's dashboard, or chooses the next candidate (which creates
`<run>_fixN_fix1`, and so on). After two failed repairs of the same variable, recommend escalation
or dismissal instead of a third attempt.

**6. Sign-off and export.** When every variable is accepted or dismissed:

```
$PCN_PYTHON $SKILL/scripts/qc_review.py export --project <project> --runs <run>,<run>_fix1[,...] \
    --name <export name> --by "<name>" --id "<ID>" --confirm
$PCN_PYTHON $SKILL/scripts/check_status.py post qc --project <project> --run <run>
```

List the original run first and fix runs after it; a variable is taken from the last run in which
it was accepted. Pass `--confirm` only after the signer has confirmed in the conversation, having
seen the counts. The export refuses to run while any variable is undecided, awaiting a fix or
escalated (`--allow-open` exports the accepted ones and lists the rest as excluded, if the signer
asks for that). It also refuses when the project has clinical subjects and an accepted variable
has no clinical z-scores (cluster runs score train and test only); it prints the
`run_model.py predict --data clinical` command to run first.

It writes `z_test.csv`, `z_clinical.csv`, `z_train.csv` for the accepted variables,
`accepted_models.csv`, and `qc_report.md`, and appends a row to the ledger
(`<project>/audit/ledger.jsonl`: who, ID, step, when, summary, hash-chained).

**7. Close.** Give the user the report's headline numbers and the list of dismissed variables with
reasons. Nothing was deleted: dismissed models stay in their run folders.

## Rules

- Never change `assets/qc_thresholds.json` to make a run pass. A threshold change is a
  supervisor's decision, applies to every run, and must be stated in the report.
- Never record a decision the user did not make. Never sign for them.
- Never edit `decisions.jsonl`, `ledger.jsonl` or `commands.jsonl`. If `qc_review.py ledger`
  reports tampering, stop and tell the user and the supervisor.
- Z-scores of clinical subjects are expected to deviate. Calibration is judged on held-out
  reference subjects only. Do not "fix" a model because patients look abnormal.
- Low explained variance alone is not a defect: a measure that barely changes with age can have a
  perfectly calibrated norm. Calibration (S2), batch effects (S4) and structure (S5) decide.

## Reference files

- `references/checks.md`: every flag code, its formula, threshold and meaning, and the repair ladder
- `references/review.md`: dashboard, decision rules, roles, escalation, ledger, export, report contents
