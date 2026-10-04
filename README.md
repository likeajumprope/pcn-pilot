# PCN-Pilot

Three agent skills that take tabular brain measures to quality-controlled, signed-off
normative-model deviation scores with [PCNtoolkit](https://pcntoolkit.readthedocs.io) 1.3.0.
The design follows NeuroPilot (Chen et al., arXiv:2608.07541): each skill is a description, reference
procedures read on demand, parameterized scripts, and checkpoints, and a human approves every
consequential step.

| Stage | Skill | Does |
|---|---|---|
| 1 | `pcn-data-skill` | profiles tables, assigns column roles, merges sources, separates the reference group from clinical subjects, splits by subject, validates (14 checks), writes a data report |
| 2 | `pcn-model-skill` | routes the data to a recipe (BLR / HBR / transfer / extend), plans, runs locally or on SLURM/Torque, resumes per response variable |
| 3 | `pcn-qc-skill` | grades each variable in five steps, proposes repairs, serves a review dashboard, records named decisions, exports behind a hash-chained sign-off ledger |

## Install

```
./install.sh                    # copies the three skills to ~/.claude/skills
./install.sh <project>/.claude/skills
```

Requirements: Python 3.11 or 3.12 with PCNtoolkit 1.3.0 for fitting (PCNtoolkit 1.x does not install
on Python 3.13); pandas, numpy, scipy and matplotlib for data preparation and QC; arviz for HBR
convergence checks (installed with PCNtoolkit).

```
python3.12 -m venv ~/envs/pcn && ~/envs/pcn/bin/pip install "pcntoolkit==1.3.0"
```

Then configure the one site-specific file and run the self-test with that interpreter:

```
mkdir -p ~/.config/pcnpilot
cp skills/pcn-model-skill/assets/pipeline.env.template ~/.config/pcnpilot/pipeline.env   # edit it
$PCN_PYTHON selftest/run_selftest.py            # about 4 minutes on 2 cores; --hbr adds the MCMC path (1 to 2 more)
```

## What was verified, and against what

Every PCNtoolkit call these skills make was run against an installed **PCNtoolkit 1.3.0** (Python
3.12, with the pandas 3.0, numpy 2.4, pymc 5.28 and arviz 0.23 that pip resolved for it), and every
file they read was written by it. `selftest/run_selftest.py --hbr` passes all of its checks there:
data preparation, BLR and HBR fits, the router's full recipe (B-spline, batch effects,
heteroskedastic noise, warp), prediction with reloaded models, transfer and extend for a new
site, SLURM and Torque submission through PCNtoolkit's own `Runner`, the five QC steps and the
signed export.

The cluster checks use stand-in `sbatch` / `qsub` commands that run each job script at once on the
local machine. That exercises everything PCNtoolkit does (pickled job, job script, state file,
status) but not a real scheduler's queueing: the first submission on your cluster is still the
first real one. Run a two-variable plan before a large one.

Run the self-test once on each machine. If you install a different PCNtoolkit version it is what
tells you whether the calls still hold; the only file that talks to PCNtoolkit is
`common/pcn_bridge.py`: adapt it there and run `tools/sync_common.py`.

`run_selftest.py --fake` replaces PCNtoolkit with a stand-in that mirrors 1.3.0, quirks included.
It needs only pandas, numpy, scipy and matplotlib, and additionally injects failures to test resume,
retry isolation, the decision rules and the ledger.

### Where PCNtoolkit 1.3.0 does not do what its documentation says

Found by running it; each one is handled in `common/pcn_bridge.py` and covered by the self-test.

| PCNtoolkit 1.3.0 | What PCN-Pilot does |
|---|---|
| `transfer` fails for a BLR model fitted without a warp (`UnboundLocalError`) | refused at plan time (check R7) with the reason; the router proposes extend for such a reference |
| `extend` fails as soon as the new data contains a site the reference lacks (`KeyError`) | performs extend's three steps itself with public calls: synthesise from the reference, pool, refit |
| `y_transform` gives correct z-scores but infinite centiles | refused at plan time; QC also fails any variable whose centile file holds non-finite values |
| a transfer always writes plots and saves the model with plotting on; a reloaded multi-variable model then fails when it predicts one variable | plotting is set from the recipe whenever a run's model is reloaded |
| a loaded model writes results to the absolute path stored when it was fitted | results go to the folder the model was loaded from, so a moved project keeps working |
| `Runner.check_jobs_status()` returns (running, finished, failed), and after `load_from_state()` it only sees running jobs | job counts are read from the attributes `load_from_state()` sets |
| the Torque job writer reads a `Runner` attribute that is never set (`AttributeError`) | the attribute is set before submitting |
| cluster jobs score train and test only | `predict --data clinical` afterwards; the post-check says so and the QC export refuses without it |
| basis functions and `transfer()` swallow unknown options; `cg` is silently replaced by L-BFGS-B for warped or heteroskedastic models; `n_iter` and `tol` are read by `cg` only | misspelt or ignored options are refused at plan time; the router and QC no longer propose `n_iter` |
| result files drop every column whose name ends in `_old` when a second variable is written | validation check V14 refuses such variable names |
| `idata.nc` holds the posterior only (no divergences); a transferred single-site model has empty and constant parameters | divergences are recorded at fit time for local runs and reported as SKIPPED otherwise; R-hat and ESS skip non-sampled parameters |
| a Beta likelihood with min-max output scaling gives infinite z-scores beyond the training range | refused; the router uses no output scaling for (0, 1) data |
| HBR sampling takes no seed | stated in the plan; extend's synthetic data are seeded (`seed` in the recipe) |

## Use

Ask the agent in plain language, for example:

- "Prepare `participants.tsv` and the FreeSurfer thickness tables for normative modelling."
- "Fit normative models for all regions and tell me which ones I can trust."
- "Adapt the pretrained lifespan model to our site and score our patients."
- "QC run `main` and open the review dashboard."

Each skill tells the agent which script to run and where it must stop for your decision.

## Project layout

```
<project>/
  pipeline.env                 site configuration (optional here; or ~/.config/pcnpilot/)
  dismissed.json               subjects / variables / batch levels excluded by a named person
  data/                        spec.json, clean/train/test/clinical.csv, dropped.csv, data_report.md
  models/<run>/                recipe.json, plan.json, lineage.json, approvals.jsonl,
                               model/ and results/ (PCNtoolkit's own layout), status/
  qc/<run>/                    grades.json, metrics.csv, plots/, dashboard.html, decisions.jsonl, fixes/
  export/<name>/               z_test.csv, z_clinical.csv, accepted_models.csv, qc_report.md
  audit/commands.jsonl         every script call: arguments, versions, exit code
  audit/ledger.jsonl           sign-offs, hash-chained
```

## Guarantees the scripts enforce

- sources are read-only; only declared columns are copied; every dropped row has a recorded reason
- clinical subjects are never used to fit or to judge calibration
- a subject is never in both train and test
- nothing is fitted without a plan and a named approval
- one failing response variable never stops the others; re-running only does what is missing
- a recipe option the installed PCNtoolkit does not declare, would swallow, or would silently
  replace is refused, not dropped
- a reference model's folder is never written to
- runs, exports and results are never overwritten; a dataset rebuild needs a new run name
- QC never edits results; a repair is a new run fitted by the model skill
- a check that cannot be computed is SKIPPED with a reason, never PASS
- decisions need a name and an institutional ID; overrides, dismissals and escalations need a reason
- exports are refused while anything is undecided, and when the ledger fails verification

## Limits

- Tabular response variables. Voxel- or vertex-wise images (`NormData.from_bids` / `from_fsl`) are
  not wired in.
- One complete-case table per project; feature sets with different missingness go in separate projects.
- BLR and HBR only. Cross-validation through the Runner is not used; a single held-out test split is.
- With PCNtoolkit 1.3.0: no transfer from a BLR model without a warp (use extend), and no
  `y_transform` (use a warp or a SHASH likelihood). See the table above.
- HBR fits are not exactly reproducible (the sampler takes no seed in 1.3.0), and divergences are
  known only for local runs.
- Extend is carried out by PCN-Pilot from PCNtoolkit's public building blocks, because
  `NormativeModel.extend` fails in 1.2.0 to 1.3.0. When a later release fixes it, compare the two
  before switching back.
- QC thresholds are sensible defaults, not calibrated on your cohorts. Review
  `skills/pcn-qc-skill/assets/qc_thresholds.json` with whoever supervises QC before relying on grades.
- The agent's grade is a screen. Subtle misfit still needs a person looking at the plots.

## Development

Shared code lives in `common/` and is vendored into each skill by `tools/sync_common.py` so every
skill works when installed alone. Edit `common/`, sync, then run `selftest/run_selftest.py --fake`
(170 checks, under two minutes) and the real-mode self-test (180 checks with `--hbr`).

When PCNtoolkit behaves differently from what `selftest/fake_pcntoolkit` assumes, change the
stand-in as well, so that fake mode keeps testing the guards that real mode needs.
