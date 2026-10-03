# PCN-Pilot

Three agent skills that take tabular brain measures to quality-controlled, signed-off
normative-model deviation scores with [PCNtoolkit](https://pcntoolkit.readthedocs.io) v1.x.
The design follows NeuroPilot (Chen et al., arXiv:2608.07541): each skill is a description, reference
procedures read on demand, parameterized scripts, and checkpoints, and a human approves every
consequential step.

| Stage | Skill | Does |
|---|---|---|
| 1 | `pcn-data-skill` | profiles tables, assigns column roles, merges sources, separates the reference group from clinical subjects, splits by subject, validates (13 checks), writes a data report |
| 2 | `pcn-model-skill` | routes the data to a recipe (BLR / HBR / transfer / extend), plans, runs locally or on SLURM/Torque, resumes per response variable |
| 3 | `pcn-qc-skill` | grades each variable in five steps, proposes repairs, serves a review dashboard, records named decisions, exports behind a hash-chained sign-off ledger |

## Install

```
./install.sh                    # copies the three skills to ~/.claude/skills
./install.sh <project>/.claude/skills
```

Requirements: Python 3.11 or 3.12 with `pcntoolkit>=1.0` for fitting; pandas, numpy, scipy and
matplotlib for data preparation and QC; arviz for HBR convergence checks (installed with PCNtoolkit).

Then configure the one site-specific file and run the self-test with the real toolkit:

```
mkdir -p ~/.config/pcnpilot
cp skills/pcn-model-skill/assets/pipeline.env.template ~/.config/pcnpilot/pipeline.env   # edit it
$PCN_PYTHON selftest/run_selftest.py            # add --hbr to also fit one small HBR model
```

**Run the self-test before first use.** These skills were written against the PCNtoolkit 1.3 API
from its documentation and source, and tested end to end against a stand-in for the toolkit
(`selftest/fake_pcntoolkit`, `run_selftest.py --fake`), not against an installed PCNtoolkit. The
real-mode self-test is what confirms that your installed version accepts the calls made here and
writes the files read here. If a check fails, the only file that talks to PCNtoolkit is
`common/pcn_bridge.py`: adapt it there and run `tools/sync_common.py`.

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
- a recipe option the installed PCNtoolkit does not declare is refused, not dropped
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
- QC thresholds are sensible defaults, not calibrated on your cohorts. Review
  `skills/pcn-qc-skill/assets/qc_thresholds.json` with whoever supervises QC before relying on grades.
- The agent's grade is a screen. Subtle misfit still needs a person looking at the plots.

## Development

Shared code lives in `common/` and is vendored into each skill by `tools/sync_common.py` so every
skill works when installed alone. Edit `common/`, sync, then run `selftest/run_selftest.py --fake`
(104 checks, about a minute) and the real-mode self-test.
