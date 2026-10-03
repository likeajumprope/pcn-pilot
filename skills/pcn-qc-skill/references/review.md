# Review, decisions, ledger and export

## Decisions

| Decision | Meaning | Needs |
|---|---|---|
| `pass` | accepted as is | a written reason if the agent graded FAIL (recorded as an override) |
| `accept_warn` | accepted with a documented caveat | a written reason |
| `fix` | apply one repair candidate and review again | the candidate's letter |
| `dismiss` | exclude the variable from the export and from later builds | a written reason |
| `escalate` | hand to the supervisor | a written reason |

Every decision carries the reviewer's name, institutional ID, role and time. The latest decision
per variable counts; earlier ones stay in `decisions.jsonl` as history.

## Roles and escalation

- **Reviewer**: grades the cohort, decides the clear cases, escalates the unclear ones.
- **Supervisor**: closes escalations and signs exports. A supervisor closing an escalation must be
  a different person (different ID) from the one who escalated.

Path of a hard case: the agent flags and proposes; the reviewer accepts, overrides or escalates;
the supervisor decides. Each step is in the record.

## The dashboard

`qc_review.py serve` starts a small server bound to 127.0.0.1.

- On a workstation: open the printed URL.
- On a cluster or remote server: run it on the login node, then from your laptop
  `ssh -N -L 8765:127.0.0.1:8765 <user>@<host>` and open `http://127.0.0.1:8765/`.

Nothing is exposed to the network; access is whoever can reach that port on that machine.
Plots are loaded only for the open card, so cohorts of thousands of variables stay responsive.
Filters: grade, undecided, awaiting fix or escalated; plus search by name.

Offline alternative: `qc_review.py dashboard` writes `qc/<run>/dashboard.html`. Copy the `qc/<run>`
folder (page and `plots/`) to the reviewer. Their decisions are held in the page and saved with
"Download decisions"; load the file with `qc_review.py import --file <file>`. The import applies
the same rules and reports any decision it refuses.

## Where things are stored

| File | Content | Written by |
|---|---|---|
| `qc/<run>/grades.json`, `metrics.csv`, `plots/` | the agent's grading | `qc_detect.py` |
| `qc/<run>/decisions.jsonl` | every decision, append-only | dashboard, `decide`, `import` |
| `qc/<run>/fixes/index.json`, `fixes/<new run>/` | prepared repairs | `qc_review.py fix` |
| `qc/<run>/compare_*.csv` | before/after | `qc_review.py compare` |
| `<project>/dismissed.json` | the shared dismissal list | `dismiss` decisions, `pcn-data-skill` |
| `<project>/audit/ledger.jsonl` | sign-offs, hash-chained | `qc_review.py export` |
| `<project>/audit/commands.jsonl` | every script call | all scripts |
| `<project>/export/<name>/` | the signed deliverable | `qc_review.py export` |

## Dismissal

A dismissed variable is added to `dismissed.json` at once. From then on it is left out of dataset
rebuilds, model plans and exports. Its model and results stay on disk and it is listed, with the
reason and the reviewer, in every later report. To bring one back, a person edits `dismissed.json`
deliberately; no script does it.

## The ledger

Each export appends one row: `by`, `id`, `step`, `at`, `summary`, a payload (runs, counts, file
checksums), the hash of the previous row and its own hash. Changing, removing or reordering any
row breaks the chain. `qc_review.py ledger --project <project>` lists the rows and verifies them;
the export and both QC checkpoints verify it too. A failed verification stops the export.

## Export

```
qc_review.py export --project <project> --runs <run>,<fix runs...> --name <name> --by "<name>" --id "<ID>" --confirm
```

- refuses without `--confirm`, without a signer, with a broken ledger, when the runs were fitted
  on different dataset builds, when the name already exists, and while variables are open
- a variable's z-scores come from the last listed run in which it was accepted
- `accepted_models.csv` maps each accepted variable to its run and model folder, with the
  reviewer and any caveat

`qc_report.md` contains: signer and ledger hash; counts; accepted with caveat and overrides (with
reasons); every dismissed variable (with reasons); variables excluded as open; the project
dismissal list; fixes applied; the method and the exact thresholds; checksums of the exported
files; the full ledger.

## What to tell the user at the end

- how many variables were accepted, accepted with caveat, overridden, fixed, dismissed
- the dismissed ones by name, with the reason
- any check that was SKIPPED for the whole run (for example no MCMC diagnostics)
- where the export is, and that z-scores for clinical subjects are in `z_clinical.csv`
- the caveats that travel with the data: extrapolated subjects, repeated measures, a transferred
  model's limited covariate range
