---
name: pcn-model-skill
description: Choose, fit, transfer, extend and apply normative models with PCNtoolkit v1.x (BLR and HBR), locally or on a SLURM/Torque cluster, with resumable per-variable execution. Use this skill whenever the user wants to fit or run a normative model, compute z-scores, deviation scores or centiles, pick between BLR and HBR, warping or SHASH likelihoods, adapt a pretrained or reference model to a new site (transfer, extend), score patients against a norm, or resume or check a PCNtoolkit run, even if they do not name PCNtoolkit.
---

# pcn-model-skill: routing and running normative models

Stage 2 of three. Input: a dataset standardized by `pcn-data-skill` (its post-check must be OK).
Output: one run folder per model configuration, reviewed afterwards by `pcn-qc-skill`.

`$SKILL` is the folder that holds this file. `$PCN_PYTHON` is the interpreter that has
PCNtoolkit 1.x (set in `pipeline.env`).

## First use on a machine

1. Copy `assets/pipeline.env.template` to `<project>/pipeline.env` (or `~/.config/pcnpilot/pipeline.env`)
   and fill it in with the user. It is the only site-specific file: interpreter, backend, conda
   environment path, job resources. Details: `references/execution.md`.
2. Run the package self-test once with the real toolkit (`selftest/run_selftest.py` in the PCN-Pilot
   folder). It confirms the installed PCNtoolkit accepts the calls made here and writes the files
   read here. If it fails, stop and show the user the failing checks.

## What comes out

`<project>/models/<run>/`

| Path | Content |
|---|---|
| `recipe.json`, `plan.json`, `lineage.json` | the frozen configuration, what was planned, where the model came from |
| `approvals.jsonl` | who approved each execution |
| `model/`, `results/` | written by PCNtoolkit: one folder per response variable, and `Z_*.csv`, `centiles_*.csv`, `statistics_*.csv` |
| `status/<variable>.json`, `status_summary.json` | per-variable outcome |
| `logs/`, `tmp/`, `submissions.jsonl` | cluster runs only |

A run name is bound to one recipe and one build of the dataset. Anything different is a new run.
Nothing is ever overwritten.

## Workflow

**1. Route.** Measure the data and propose a recipe.

```
$PCN_PYTHON $SKILL/scripts/choose_recipe.py --project <project> --out <project>/recipe_<name>.json
        [--reference <saved model folder> --goal transfer|extend] [--prefer blr|hbr]
```

| Situation | Route |
|---|---|
| Own reference data, many variables or large N | BLR, B-spline mean, fixed batch effects; warp if residuals are non-Gaussian |
| Own reference data, several small sites, few variables | HBR with a random site intercept; SHASH likelihood if non-Gaussian |
| Responses bounded in (0, 1) | HBR with a Beta likelihood |
| New site, a reference or pretrained model exists, 20 to 100+ controls | transfer |
| Reference model exists and its sites and covariate range must be kept, or more sites will follow | extend |

The script prints the measured traits, the reasoning for each setting and ranked alternatives.
Read `references/routing.md` before changing a recipe by hand.

**2. Get the recipe approved.** Show the user the route, the reasons and the alternatives in plain
language. Choosing a modelling recipe is the user's decision. Edit the recipe file if they want
something else.

**3. Plan.** Freezes the recipe, checks a reference model for compatibility, and states what would
run. It computes nothing.

```
$PCN_PYTHON $SKILL/scripts/run_model.py plan --project <project> --run <run> --recipe <recipe.json>
        [--response-vars a,b,c | --response-vars-file <file>] [--backend local|slurm|torque] [--n-batches N]
$PCN_PYTHON $SKILL/scripts/check_status.py pre model --project <project> --run <run>
```

If the plan prints FAIL lines (R1 to R6) the reference model does not fit the data: the usual
cause is naming or coding in the data spec. See `references/transfer_extend.md`, fix it with
`pcn-data-skill`, and plan again.

**4. Get the plan approved.** Tell the user: how many variables, which backend, how many jobs with
which time and memory, and where output goes. Wait for an explicit yes. This applies to every
execution, and above all to cluster submissions.

**5. Run.**

```
$PCN_PYTHON $SKILL/scripts/run_model.py run --project <project> --run <run> --approved-by "<name of the person who approved>"
```

Local runs fit one variable at a time and continue past failures. Cluster runs submit jobs through
PCNtoolkit's `Runner` and return immediately.

**6. Check what actually happened.** Never report success from the absence of an error message.

```
$PCN_PYTHON $SKILL/scripts/run_model.py status --project <project> --run <run>      # cluster: is it still running?
$PCN_PYTHON $SKILL/scripts/check_status.py post model --project <project> --run <run>
```

The post-check counts, on disk, the variables that have both a saved model and test z-scores.
Exit 0 = all done, 3 = partial, 1 = nothing.

**7. Resume.** If partial, read the errors in `status_summary.json` (and the job logs for cluster
runs), tell the user what failed and why, and with their approval run step 5 again. Completed
variables are skipped. On a cluster, a retry puts each remaining variable in its own job, so one
bad variable cannot take a batch down twice. A variable that fails twice for the same reason is
not a scheduling problem: report it, and leave it for `pcn-qc-skill` (it will be graded FAIL at
step 1 with repair options).

**8. Score other data (optional).** Clinical subjects are scored automatically in local runs. For
cluster runs, or for new tables:

```
$PCN_PYTHON $SKILL/scripts/run_model.py predict --project <project> --run <run> --data clinical|train|<standardized csv> [--name <name>]
```

**9. Hand over.** Report per-variable outcomes from the post-check, then continue with
`pcn-qc-skill`. A fitted model is not a validated model.

## Needs the user's explicit approval

- the recipe, and any change to it
- every `run` (the `--approved-by` name must be a real person who said yes in this conversation)
- cluster submissions and their resources
- transfer versus extend
- re-planning under a new run name after the dataset was rebuilt

## Rules

- Do not call PCNtoolkit yourself in ad hoc Python for work these scripts cover. They add the
  checks, the per-variable isolation and the audit trail.
- A recipe option the installed PCNtoolkit does not accept stops the plan with a message naming
  it. Do not remove the option to get past the error without telling the user: the model they
  approved would silently become a different model.
- Never fit on `clinical.csv`. Never delete a run folder. Never change files under `model/` or
  `results/` by hand.
- Transfer a model at most once. To add sites repeatedly, use extend.
- If the dataset is rebuilt, existing runs stay valid for the old build only. Start a new run.

## Reference files

- `references/routing.md`: the routing logic, the full recipe schema, what each BLR and HBR option does
- `references/execution.md`: `pipeline.env`, local versus SLURM/Torque, resume and retry, where logs are
- `references/transfer_extend.md`: transfer versus extend, compatibility checks R1 to R6, pretrained models
