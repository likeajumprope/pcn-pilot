# Execution: `pipeline.env`, backends, resume

## `pipeline.env`

The one file that changes between machines. The scripts look for it in this order:
`--env <path>`, `$PCN_PIPELINE_ENV`, `<project>/pipeline.env`, `~/.config/pcnpilot/pipeline.env`.
Real environment variables override the file. It is parsed as plain `KEY=VALUE` lines; nothing in
it is executed.

| Key | Meaning | Default |
|---|---|---|
| `PCN_PYTHON` | interpreter with PCNtoolkit 1.3.0 (Python 3.11 or 3.12) | the running interpreter |
| `PCN_BACKEND` | `local`, `slurm` or `torque` | `local` |
| `PCN_CONDA_ENV` | root folder of the environment the jobs use; it must contain `bin/python` | empty |
| `PCN_PREAMBLE` | shell line run at the top of each job, e.g. `module load anaconda3` | `module load anaconda3` |
| `PCN_TIME_LIMIT` | per job, `HH:MM:SS` | `02:00:00` |
| `PCN_MEMORY` | per job | `8GB` |
| `PCN_N_CORES` | cores per job | `1` |
| `PCN_N_BATCHES` | number of jobs the variables are spread over | `10` |
| `PCN_MAX_RETRIES` | in-job retries (PCNtoolkit's own) | `3` |
| `PCN_SLURM_PARTITION`, `PCN_SLURM_ACCOUNT`, `PCN_SLURM_QOS` | exported as `SBATCH_PARTITION` etc., which `sbatch` reads | empty |
| `PCN_QC_PORT` | port of the QC dashboard | `8765` |

How to find the environment path: activate it and run
`python -c "import sys; print(sys.prefix)"`.

Each job script PCNtoolkit writes runs the preamble, then `source activate <PCN_CONDA_ENV>`, then
`python`. With conda that is all that is needed. With a virtualenv, activate it in the preamble
(`PCN_PREAMBLE="source /path/to/venv/bin/activate"`); the `source activate` line that follows then
re-runs the same script and is harmless.

Always run the scripts with `$PCN_PYTHON`. The data and QC-detection scripts only need pandas,
numpy, scipy and matplotlib, so they also run where PCNtoolkit is not installed.

## Local backend

`run_model.py run` fits one response variable at a time in the current process. Each variable
gets fresh data objects, its own try/except and a status file, so a failure never stops the
cohort and never contaminates the next variable. All variables write into the same `model/` and
`results/` folders: PCNtoolkit merges each result file on write, and `NormativeModel.load` picks
up every variable folder it finds. The train split is scored as part of the fit (QC's overfitting
check uses it), and the clinical table afterwards if there is one.

Good for BLR on up to a few thousand variables, and for HBR on a handful.

## SLURM and Torque

Submission goes through PCNtoolkit's own `Runner`:

- variables are spread over `PCN_N_BATCHES` jobs (`NormData.chunk`), all writing into the same
  run folder. PCNtoolkit's result files are merged on write under a file lock, so this is safe
- each job: one node, `PCN_N_CORES` cores, `PCN_MEMORY`, `PCN_TIME_LIMIT`; it runs the preamble,
  activates `PCN_CONDA_ENV`, and executes the pickled fit function
- the call returns after submission. Runner state is saved under `<run>/tmp/<task>/runner_state.json`;
  each submission is recorded in `<run>/submissions.jsonl` (jobs, variables, seed)
- job output: `<run>/logs/<task>/<job>.out` and `.err`; a `.success` file marks a job that ended well
- the jobs score the train and test splits. The clinical table is not scored by the jobs: run
  `run_model.py predict --data clinical` when they have finished. The post-check reminds you
- extend on a cluster: the synthetic data are drawn in the submitting process, and the jobs then run
  an ordinary fit on the pooled table. Afterwards also run `predict --data train`
- Torque: PCNtoolkit 1.1 to 1.3.0 cannot write a Torque job script on its own (its writer reads a
  Runner attribute the constructor no longer sets). `pcn_bridge.py` sets that attribute, so the
  backend works; the value only reaches an argument the job ignores

Run it from a node where `sbatch` (or `qsub`) works. For HBR set `PCN_N_CORES` equal to
`hbr.cores` in the recipe (one core per chain).

Sizing: BLR needs little (minutes, a few GB). HBR: start from 4 cores, 8 GB, and about
10 minutes per variable per job for a Normal likelihood, more for SHASH; then adjust from the
first jobs' logs. Tell the user the total: jobs x time x cores.

## Status

`run_model.py status` prints two views:

- **on disk**: variables with a saved model and test z-scores. This is the truth
- **scheduler**: running, finished and failed jobs of the latest submission, as PCNtoolkit's Runner
  sorts them (a job is finished when the scheduler no longer lists it and its `.success` file
  exists; failed when neither)

If the two disagree, trust the disk. If the scheduler view is unavailable (state file gone,
scheduler command missing) the script says so and still reports the disk view.

`check_status.py post model` writes `status_summary.json` with the lists `done`, `failed`
(with the error text) and `not_started`.

## Resume and retry

`run` is idempotent. It recomputes the pending list from disk every time:

- complete variables are skipped
- local: failed variables are tried again
- cluster: refuses to submit while jobs of the previous submission still run (override with
  `--force`). A retry (attempt 2 and later) uses one job per remaining variable. PCNtoolkit fits a
  batch's variables in sequence without isolating them, so one failing variable ends its whole
  job; isolating on retry finds the culprit and lets the others finish

Typical causes of failure and what to do:

| Symptom | Likely cause | Action |
|---|---|---|
| job killed, `TIME LIMIT` or `OUT OF MEMORY` in the `.err` log | resources too small | raise `PCN_TIME_LIMIT` / `PCN_MEMORY`, re-plan (same run), run again |
| `LinAlgError`, non-finite likelihood (BLR) | optimiser diverged, often with a warp | leave it; QC proposes Powell, more iterations, or no warp |
| sampler errors, zero-probability start (HBR) | priors or scaling do not suit the variable | leave it; QC proposes longer warm-up, a simpler hierarchy, or BLR |
| every variable fails the same way | environment or data problem | read one traceback in `status/<variable>.json`; run `normdata_selftest.py --fit-one` |
| `sbatch: command not found` | not on a login node | run from a login node or use `--backend local` |
| `Invalid environment` when submitting | `PCN_CONDA_ENV` is not an environment root (no `bin/python` inside) | set it to the output of `python -c "import sys; print(sys.prefix)"` |
| `UnboundLocalError ... 'y'` in a transfer | a BLR reference without a warp; only possible with a PCNtoolkit newer than the plan's R7 check knows | use extend |
| `Data is not compatible with the model` | the table has a batch-effect label the model was not fitted on | adapt the model (transfer or extend) or recode the label |

Re-planning the same run with different resources is allowed (the recipe is unchanged). A
different recipe needs a new run name.

## Audit trail

Every script call appends a row to `<project>/audit/commands.jsonl`: time, script, arguments,
user, host, Python and PCNtoolkit versions, exit code, duration. Approvals for each `run` are in
`<run>/approvals.jsonl`. Do not edit either.
