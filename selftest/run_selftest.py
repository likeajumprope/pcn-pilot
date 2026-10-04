#!/usr/bin/env python3
"""End-to-end self-test of the three PCN-Pilot skills on synthetic data.

    $PCN_PYTHON selftest/run_selftest.py                 # against the real, installed PCNtoolkit  <- run this once per site
    python      selftest/run_selftest.py --fake          # against the bundled stand-in (orchestration only)
    ...                                   --keep DIR     # keep the working directory for inspection
    ...                                   --hbr          # also fit, transfer and QC one small HBR model (real mode; ~2 minutes)
    ...                                   --no-cluster   # skip the SLURM/Torque submission checks

Real mode is the test that matters: it confirms that the installed PCNtoolkit accepts
the calls PCN-Pilot makes and writes the files PCN-Pilot reads. It was last run clean
against PCNtoolkit 1.3.0. Fake mode replaces PCNtoolkit with selftest/fake_pcntoolkit
(which mirrors 1.3.0, quirks included) and additionally injects failures to test resume,
retry isolation, the decision rules and the ledger.

The cluster checks never reach a real scheduler: stand-in `sbatch`, `squeue`, `qsub` and
`qstat` commands are put first on PATH for the test's own subprocesses and run each job
script at once, so PCNtoolkit's Runner is exercised end to end (pickled job, job script,
state file, status) on any machine with bash.

Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "skills" / "pcn-data-skill" / "scripts"
MODEL = ROOT / "skills" / "pcn-model-skill" / "scripts"
QC = ROOT / "skills" / "pcn-qc-skill" / "scripts"
results: list[tuple[bool, str]] = []


def check(cond: bool, label: str, detail: str = "") -> bool:
    results.append((bool(cond), label))
    print(f"  [{'ok' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail and not cond else ""), flush=True)
    return bool(cond)


def run(script: Path, *args, expect: int | tuple = 0, env: dict | None = None, label: str | None = None) -> str:
    cmd = [sys.executable, str(script), *map(str, args)]
    p = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **(env or {})})
    exp = (expect,) if isinstance(expect, int) else expect
    words = [str(x) for x in args if not str(x).startswith(("/", "-")) and len(str(x)) < 24][:3]
    name = label or f"{script.name} {' '.join(words)}"
    check(p.returncode in exp, f"{name} -> exit {p.returncode}",
          (p.stdout[-600:] + p.stderr[-900:]).replace("\n", " | "))
    return p.stdout + p.stderr


def spec(work: Path, name: str, demo: str, idp: str, rvs: list[str], frac: float) -> dict:
    return {"name": name,
            "sources": [{"path": str(work / "syn" / demo), "id_column": "participant_id"},
                        {"path": str(work / "syn" / idp), "id_column": "SubjID"}],
            "id_normalization": "alnum", "covariates": ["age"], "batch_effects": ["sex", "site"],
            "response_vars": rvs, "recode": {"sex": {"0": "F", "1": "M"}},
            "reference": {"column": "diagnosis", "values": ["HC"]},
            "missing": {"max_feature_missing_fraction": 0.2}, "outliers": {"policy": "flag", "threshold": 5.0},
            "split": {"test_fraction": frac, "seed": 42, "min_train_per_stratum": 5}, "min_n_per_batch_level": 10}


STUBS = {
    "sbatch": '#!/bin/bash\n# stand-in scheduler for the self-test: runs the job script at once\n'
              'n=$(( $(cat "$0.n" 2>/dev/null || echo 7000) + 1 )); echo $n > "$0.n"\n'
              'bash "$1" >/dev/null 2>&1\necho "Submitted batch job $n"\n',
    "qsub": '#!/bin/bash\nn=$(( $(cat "$0.n" 2>/dev/null || echo 7000) + 1 )); echo $n > "$0.n"\n'
            'bash "$1" >/dev/null 2>&1\necho "$n.selftest"\n',
    "squeue": '#!/bin/bash\necho "JOBID PARTITION NAME USER ST TIME NODES NODELIST(REASON)"\n',   # nothing is running
    "qstat": '#!/bin/bash\nexit 153\n',                                                          # unknown job id
    "activate": '# stand-in for `source activate <env>`: the job inherits the test\'s interpreter through PATH\n',
}


def cluster_env(work: Path, fake: bool) -> dict:
    """A stand-in scheduler on PATH, and an environment folder PCNtoolkit's Runner accepts."""
    bindir = work / "stub_bin"
    bindir.mkdir(exist_ok=True)
    for name, text in STUBS.items():
        (bindir / name).write_text(text)
        (bindir / name).chmod((bindir / name).stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    envdir = work / "env"                      # the Runner only checks that <env>/bin/python exists
    (envdir / "bin").mkdir(parents=True, exist_ok=True)
    if not (envdir / "bin" / "python").exists():
        (envdir / "bin" / "python").symlink_to(sys.executable)
    pydir = Path(sys.executable).parent        # not resolved: a virtualenv's python must stay inside the venv
    return {"PATH": f"{bindir}{os.pathsep}{pydir}{os.pathsep}{os.environ.get('PATH', '')}",
            "PCN_CONDA_ENV": str(envdir), "PCN_PREAMBLE": "true", "PCN_MAX_RETRIES": "0",
            **({"PCN_SKIP_SCHEDULER_CHECK": "1"} if fake else {})}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--keep")
    ap.add_argument("--hbr", action="store_true")
    ap.add_argument("--no-cluster", action="store_true")
    a = ap.parse_args()
    for k in [k for k in os.environ if k.startswith("PCN_")]:
        os.environ.pop(k)                      # the test must not inherit a site's pipeline settings
    if a.fake:
        os.environ["PYTHONPATH"] = str(ROOT / "selftest" / "fake_pcntoolkit") + os.pathsep + os.environ.get("PYTHONPATH", "")
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="pcnpilot_selftest_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    work = work.resolve()
    envfile = work / "pipeline.env"            # takes precedence over ~/.config/pcnpilot/pipeline.env
    envfile.write_text("PCN_BACKEND=local\n")
    os.environ["PCN_PIPELINE_ENV"] = str(envfile)
    P, P2, P3 = work / "proj", work / "proj_new", work / "proj_names"
    rvs = ["thick_1", "thick_2", "thick_3", "thick_4", "vol_skewed", "vol_hetero", "vol_site_var"]
    ref_rvs = ["thick_2", "thick_3", "thick_4", "vol_hetero"]
    print(f"mode: {'FAKE stand-in' if a.fake else 'REAL PCNtoolkit'}   python: {sys.executable}\nwork dir: {work}\n")

    def plan(project, name, recipe, *extra, expect=0, label=None, env=None):
        return run(MODEL / "run_model.py", "plan", "--project", project, "--run", name, "--recipe", recipe, *extra,
                   expect=expect, label=label, env=env)

    def fit(project, name, by="Self Test", expect=0, label=None, env=None):
        return run(MODEL / "run_model.py", "run", "--project", project, "--run", name, "--approved-by", by,
                   expect=expect, label=label, env=env)

    def post(project, name, expect=0, label=None):
        return run(MODEL / "check_status.py", "post", "model", "--project", project, "--run", name, expect=expect, label=label)

    def recipe_file(path: Path, obj: dict) -> Path:
        path.write_text(json.dumps(obj))
        return path

    def z_columns(project, name, split="test") -> list[str]:
        f = project / "models" / name / "results" / f"Z_{split}.csv"
        return f.read_text().splitlines()[0].split(",") if f.is_file() else []

    print("0. package integrity")
    run(ROOT / "tools" / "sync_common.py", "--check", label="shared modules in sync across skills")
    run(ROOT / "selftest" / "make_synthetic.py", work / "syn", label="synthetic cohort written")

    print("\n1. pcn-data-skill")
    srcs = [work / "syn" / "demographics.csv", work / "syn" / "idps.tsv"]
    run(DATA / "check_status.py", "pre", "data", "--project", P, "--input", *srcs)
    run(DATA / "profile_table.py", "--project", P, "--input", *srcs, "--out", P / "data" / "profile.json")
    prof = json.loads((P / "data" / "profile.json").read_text())
    roles = {c["column"]: c["role"] for s in prof["sources"] for c in s["columns"]}
    check(roles.get("last_name") == "phi" and roles.get("age") == "covariate" and roles.get("site") == "batch_effect"
          and roles.get("diagnosis") == "reference_group", "column roles proposed (PHI, covariate, batch, group)", str(roles))
    check(prof["id_overlap"][0]["alnum"]["matched"] > 1400 and prof["id_overlap"][0]["strip"]["matched"] == 0,
          "ID formats reconciled only after normalisation")
    (P / "data" / "spec.json").write_text(json.dumps(spec(work, "synthetic", "demographics.csv", "idps.tsv",
                                                          rvs + ["sparse_measure"], 0.2)))
    run(DATA / "build_dataset.py", "--project", P)
    out = run(DATA / "validate_dataset.py", "--project", P)
    man = json.loads((P / "data" / "build_manifest.json").read_text())
    val = json.loads((P / "data" / "validation.json").read_text())
    st = {c["id"]: c["status"] for c in val["checks"]}
    check(man["n_clinical"] > 0 and st["V12"] == "PASS", "patients held out of train/test")
    check(st["V03"] == "PASS", "no subject in both train and test")
    check([f["response_var"] for f in man["features_dropped"]] == ["sparse_measure"] and st["V08"] == "WARN",
          "sparse variable dropped with a recorded reason, drop-check reports it")
    check(st["V07"] == "WARN" and st["V09"] == "PASS", "outliers flagged, no PHI column carried over")
    check(st.get("V14") == "PASS", "variable names can be stored by PCNtoolkit")
    check("last_name" not in (P / "data" / "clean.csv").read_text().splitlines()[0], "PHI column absent from clean.csv")
    run(DATA / "check_status.py", "post", "data", "--project", P)
    run(DATA / "normdata_selftest.py", "--project", P, "--fit-one", label="PCNtoolkit accepts the tables (NormData + minimal BLR)")
    (P3 / "data").mkdir(parents=True)
    (P3 / "data" / "spec.json").write_text(json.dumps(spec(work, "names", "demographics.csv", "idps.tsv",
                                                           ["thick_2", "vol_old"], 0.2)))
    run(DATA / "build_dataset.py", "--project", P3, label="build with a variable named *_old")
    out = run(DATA / "validate_dataset.py", "--project", P3, expect=1, label="a name PCNtoolkit's result files would drop fails validation")
    check("V14" in out and "vol_old" in json.dumps(json.loads((P3 / "data" / "validation.json").read_text())["checks"][-1]),
          "the refusal names the variable (check V14)")

    print("\n2. pcn-model-skill")
    run(MODEL / "choose_recipe.py", "--project", P, "--out", P / "recipe_auto.json", "--prefer", "blr")
    plain = recipe_file(P / "recipe_plain.json", {"algorithm": "blr", "mode": "fit_predict", "basis": {"type": "linear"},
                                                  "blr": {}})
    warped = recipe_file(P / "recipe_warped.json", {"algorithm": "blr", "basis": {"type": "linear"},
                                                    "blr": {"fixed_effect": True, "warp_name": "warpsinharcsinh",
                                                            "warp_reparam": True}})
    out = plan(P, "strict", recipe_file(P / "recipe_bad.json", {"algorithm": "blr", "basis": {"type": "linear"},
                                                                "blr": {"no_such_option": 1}}),
               expect=2, label="unknown model option is refused, not dropped")
    check("does not declare" in out, "refusal names the unsupported option")
    out = plan(P, "strict", recipe_file(P / "recipe_bad2.json", {"algorithm": "blr", "basis": {"type": "bspline", "nknot": 7}}),
               expect=2, label="misspelt basis option is refused (the toolkit itself would swallow it)")
    check("nknot" in out, "refusal names the misspelt option")
    plan(P, "strict", recipe_file(P / "recipe_bad3.json", {"algorithm": "blr", "basiss": {"type": "linear"}}),
         expect=2, label="misspelt recipe key is refused")
    out = plan(P, "strict", recipe_file(P / "recipe_bad4.json", {"algorithm": "blr", "basis": {"type": "linear"},
                                                                 "y_transform": "log"}),
               expect=2, label="y_transform is refused on PCNtoolkit <= 1.3.0")
    check("centiles" in out, "refusal explains what y_transform does to the centiles")
    plan(P, "strict", recipe_file(P / "recipe_bad5.json", {"algorithm": "hbr", "outscaler": "minmax", "hbr": {"likelihood": "Beta"}}),
         expect=2, label="Beta likelihood with min-max output scaling is refused")
    plan(P, "strict", recipe_file(P / "recipe_bad6.json", {"algorithm": "blr", "basis": {"type": "linear"},
                                                           "blr": {"optimizer": "cg", "heteroskedastic": True}}),
         expect=2, label="an optimiser PCNtoolkit would silently replace is refused")
    check(not (P / "models" / "strict").exists(), "a refused plan leaves no run folder")
    run(MODEL / "check_status.py", "pre", "model", "--project", P, "--run", "base", expect=1, label="pre-check fails without a plan")
    plan(P, "base", plain)
    check(not (P / "models" / "base" / "model").exists(), "plan computed nothing")
    run(MODEL / "run_model.py", "run", "--project", P, "--run", "base", expect=2, label="run refused without --approved-by")
    if a.fake:
        fit(P, "base", expect=3, env={"PCN_FAKE_FAIL": "thick_3"}, label="one failing variable does not stop the others")
        post(P, "base", expect=3, label="post-check reports PARTIAL")
        s = json.loads((P / "models" / "base" / "status_summary.json").read_text())
        check(s["n_done"] == 6 and list(s["failed"]) == ["thick_3"], "exactly the failing variable is reported")
    out = fit(P, "base")
    if a.fake:
        check("6 complete, 1 pending" in out, "resume redid only the missing variable")
    post(P, "base")
    out = fit(P, "base")
    check("nothing to do" in out, "re-running a finished run is a no-op")
    res = P / "models" / "base" / "results"
    check(all((res / f).is_file() for f in ("Z_test.csv", "statistics_test.csv", "centiles_test.csv")),
          "PCNtoolkit result files found where expected",
          str(sorted(p.name for p in res.glob("*"))) if res.exists() else "no results dir")
    check(set(z_columns(P, "base")) == set(rvs) | {"observations", "subject_ids"},
          "per-variable fits were merged into one z-score file", str(z_columns(P, "base")))
    check((res / "Z_clinical.csv").is_file(), "clinical cohort scored")
    check((res / "Z_train.csv").is_file(), "train z-scores available for the overfitting check")
    plan(P, "base", P / "recipe_auto.json", expect=2, label="existing run cannot be re-planned with another recipe")
    run(MODEL / "run_model.py", "predict", "--project", P, "--run", "base", "--data", "clinical", "--force",
        label="predict with a reloaded model (clinical)")
    moved = work / "moved"                     # a project folder that was moved after fitting
    shutil.copytree(P, moved)
    run(MODEL / "run_model.py", "predict", "--project", moved, "--run", "base", "--data", "test", "--name", "moved",
        label="predict in a moved copy of the project")
    check((moved / "models" / "base" / "results" / "Z_moved.csv").is_file()
          and not (P / "models" / "base" / "results" / "Z_moved.csv").exists(),
          "results go to the folder the model was loaded from, not to the path stored at fit time")
    run(MODEL / "run_model.py", "predict", "--project", moved, "--run", "base", "--data", moved / "data" / "train.csv",
        "--name", "moved", expect=2, label="a result name cannot be reused for another table")
    shutil.rmtree(moved)
    if not a.fake:                             # the full recipe the router proposes: spline, batch effects, warp
        plan(P, "auto", P / "recipe_auto.json", "--response-vars", "thick_2,vol_skewed")
        fit(P, "auto", label="auto-routed recipe fits (B-spline, fixed effects, heteroskedastic, warp)")
        post(P, "auto")
    plan(P, "ref", warped, "--response-vars", ",".join(ref_rvs))        # references for section 4
    fit(P, "ref", label="warped BLR reference fitted (4 variables)")
    post(P, "ref")
    plan(P, "fe", recipe_file(P / "recipe_fe.json", {"algorithm": "blr", "basis": {"type": "linear"},
                                                     "blr": {"fixed_effect": True}}), "--response-vars", ",".join(ref_rvs))
    fit(P, "fe", label="unwarped BLR reference with batch effects fitted")
    post(P, "fe")

    if not a.no_cluster:
        print("\n2b. cluster submission through PCNtoolkit's Runner (stand-in scheduler, nothing leaves this machine)")
        cenv = cluster_env(work, a.fake)
        envfile.write_text("PCN_BACKEND=slurm\nPCN_N_BATCHES=3\n")
        plan(P, "clus", plain, env=cenv)
        run(MODEL / "check_status.py", "pre", "model", "--project", P, "--run", "clus", env=cenv,
            label="pre-check accepts the scheduler and the environment path")
        if a.fake:
            fit(P, "clus", env={**cenv, "PCN_FAKE_FAIL": "thick_1"}, label="cluster submission (one batch will fail)")
            out = run(MODEL / "run_model.py", "status", "--project", P, "--run", "clus", expect=3, env=cenv,
                      label="a failed batch leaves its variables pending")
            check("2 finished, 1 failed" in out, "scheduler view counts finished and failed jobs", out[-300:])
            out = fit(P, "clus", env=cenv)
            check("submitted 3 job(s) for 3 variables (attempt 2)" in out, "retry isolates each remaining variable in its own job")
        else:
            out = fit(P, "clus", env=cenv, label="SLURM submission through the real Runner")
            check("submitted 3 job(s) for 7 variables" in out, "three batches submitted", out[-300:])
        out = run(MODEL / "run_model.py", "status", "--project", P, "--run", "clus", env=cenv)
        check("0 running, 3 finished, 0 failed" in out, "scheduler view read back from the Runner's state file", out[-300:])
        post(P, "clus")
        check(not (P / "models" / "clus" / "results" / "Z_clinical.csv").exists(), "cluster jobs do not score the clinical table")
        run(MODEL / "run_model.py", "predict", "--project", P, "--run", "clus", "--data", "clinical", env=cenv)
        check(set(z_columns(P, "clus", "clinical")) == set(rvs) | {"observations", "subject_ids"}, "clinical table scored afterwards")
        envfile.write_text("PCN_BACKEND=torque\nPCN_N_BATCHES=2\n")
        plan(P, "tq", plain, "--response-vars", "thick_2,thick_4", env=cenv)
        fit(P, "tq", env=cenv, label="Torque submission (needs the Runner attribute 1.3.0 leaves unset)")
        post(P, "tq")
        envfile.write_text("PCN_BACKEND=local\n")

    print("\n3. pcn-qc-skill")
    run(QC / "check_status.py", "pre", "qc", "--project", P, "--run", "base")
    run(QC / "qc_detect.py", "--project", P, "--run", "base", "--plots", "flagged")
    g = json.loads((P / "qc" / "base" / "grades.json").read_text())
    feats = {f["response_var"]: f for f in g["features"]}
    check("verified" in g["alignment"] or "subject ID" in g["alignment"], "z-scores matched back to test subjects", g["alignment"])
    codes = {rv: {fl["code"] for s in f["steps"].values() for fl in s["flags"]} for rv, f in feats.items()}
    skipped = {rv: [t for s in f["steps"].values() for t in s["skipped"]] for rv, f in feats.items()}
    check(all(f["steps"]["S2"]["status"] != "SKIPPED" for f in feats.values()), "calibration computed for every variable")
    check(all(f["steps"]["S3"]["status"] != "SKIPPED" for f in feats.values()), "PCNtoolkit's statistics file parsed", str(skipped))
    check(all(f["steps"]["S4"]["status"] != "SKIPPED" and f["steps"]["S5"]["status"] != "SKIPPED" for f in feats.values()),
          "batch and covariate checks computed", str(skipped))
    check(not any("centile" in t for ts in skipped.values() for t in ts)
          and all(f["metrics"].get("centile_nonfinite_frac") == 0 for rv, f in feats.items() if rv != "thick_1"),
          "centile file parsed, centiles finite", str(skipped))
    check(all("z_sd_train" in f["metrics"] for rv, f in feats.items()), "train z-scores read for the overfitting check")
    check("EXTREME_Z" in codes["thick_1"] or json.loads((P / "data" / "build_manifest.json").read_text())["n_test"] == 0
          or "006S0005" not in (P / "data" / "test.csv").read_text(), "planted unit error caught as an extreme z-score")
    check("SITE_MEAN" in codes["thick_2"], "unmodelled site effect detected", str(codes["thick_2"]))
    check(codes["vol_skewed"] & {"Z_SKEW", "Z_KURT", "MACE"}, "skewed variable detected", str(codes["vol_skewed"]))
    labels = [c["label"] for f in feats.values() for c in f["fix_candidates"]]
    check(not any("iterations" in lb for lb in labels), "no repair is proposed that PCNtoolkit would ignore (n_iter with L-BFGS-B)")
    run(QC / "qc_review.py", "dashboard", "--project", P, "--run", "base")
    html = (P / "qc" / "base" / "dashboard.html").read_text()
    check("__PCN_DATA__" not in html and "thick_2" in html, "dashboard built with the run's data embedded")

    def decide(run_, rv, decision, *extra, expect=0, label=None, project=P):
        return run(QC / "qc_review.py", "decide", "--project", project, "--run", run_, "--response-var", rv,
                   "--decision", decision, *extra, expect=expect, label=label or f"decide {rv} {decision}")
    who = ("--by", "Ana Reviewer", "--id", "u123")
    decide("base", "thick_2", "pass", "--by", "Ana Reviewer", "--id", "", expect=1, label="decision refused without an ID")
    fail_rv = next((rv for rv, f in feats.items() if f["grade"] == "FAIL" and rv != "thick_1"), None)
    if fail_rv:
        decide("base", fail_rv, "pass", *who, expect=1, label="override of a FAIL refused without a reason")
    site_fix = next(c["id"] for c in feats["thick_2"]["fix_candidates"] if "fixed batch effects" in c["label"])
    decide("base", "thick_2", "fix", "--fix-id", site_fix, *who)
    decide("base", "thick_4", "escalate", "--note", "unsure", *who)
    decide("base", "thick_4", "accept_warn", "--note", "ok", *who, expect=1, label="escalation cannot be closed by a reviewer")
    decide("base", "thick_4", "accept_warn", "--note", "ok", *who, "--role", "supervisor", expect=1,
           label="nor by the same person as supervisor")
    decide("base", "thick_4", "accept_warn", "--note", "site shift documented", "--by", "Dr Lee", "--id", "s777",
           "--role", "supervisor")
    for rv in ("thick_1", "thick_3"):
        decide("base", rv, "dismiss", "--note", "selftest", *who)
    for rv in ("vol_skewed", "vol_hetero", "vol_site_var"):
        decide("base", rv, "accept_warn", "--note", "selftest", *who)
    out = run(QC / "qc_review.py", "fix", "--project", P, "--run", "base")
    fx = P / "qc" / "base" / "fixes" / "base_fix1"
    check((fx / "recipe.json").is_file() and "run_model.py plan" in out, "fix prepared as a new recipe and delegated")
    check(json.loads((fx / "recipe.json").read_text())["blr"].get("fixed_effect") is True, "fix recipe carries the change")
    run(QC / "qc_review.py", "export", "--project", P, "--runs", "base", "--name", "early", "--by", "Dr Lee", "--id", "s777",
        "--confirm", expect=1, label="export refused while a fix is open")
    plan(P, "base_fix1", fx / "recipe.json", "--response-vars-file", fx / "response_vars.txt")
    fit(P, "base_fix1", by="Ana Reviewer")
    post(P, "base_fix1")
    run(QC / "qc_detect.py", "--project", P, "--run", "base_fix1", "--plots", "none")
    out = run(QC / "qc_review.py", "compare", "--project", P, "--before", "base", "--after", "base_fix1")
    g2 = json.loads((P / "qc" / "base_fix1" / "grades.json").read_text())
    c2 = {fl["code"] for s in g2["features"][0]["steps"].values() for fl in s["flags"]}
    check("SITE_MEAN" not in c2, "the fix removed the site effect (verify step)", str(c2))
    f2 = g2["features"][0]
    decide("base_fix1", "thick_2", "pass" if f2["grade"] != "FAIL" else "accept_warn", "--note", "after fix", *who)
    run(QC / "qc_review.py", "export", "--project", P, "--runs", "base,base_fix1", "--name", "v1", "--by", "Dr Lee",
        "--id", "s777", label="export refused without --confirm", expect=2)
    run(QC / "qc_review.py", "export", "--project", P, "--runs", "base,base_fix1", "--name", "v1", "--by", "Dr Lee",
        "--id", "s777", "--confirm")
    exp = P / "export" / "v1"
    accepted = {"thick_2", "thick_4", "vol_skewed", "vol_hetero", "vol_site_var"}
    for split in ("test", "clinical"):
        head = (exp / f"z_{split}.csv").read_text().splitlines()[0].split(",") if (exp / f"z_{split}.csv").is_file() else []
        check(set(head) - {"observations", "subject_ids"} == accepted,
              f"export holds exactly the accepted variables ({split})", str(head))
    acc = (exp / "accepted_models.csv").read_text() if exp.exists() else ""
    check("thick_2,base_fix1" in acc, "the fixed variable is exported from the fix run")
    rep = (exp / "qc_report.md").read_text() if exp.exists() else ""
    check("thick_1" in rep and "thick_3" in rep and "Dr Lee" in rep, "report lists every dismissed variable and the signer")
    check((P / "models" / "base" / "model" / "thick_1").exists(), "dismissed models were not deleted")
    run(QC / "qc_review.py", "export", "--project", P, "--runs", "base,base_fix1", "--name", "v1", "--by", "Dr Lee",
        "--id", "s777", "--confirm", expect=2, label="an export is never overwritten")
    run(QC / "check_status.py", "post", "qc", "--project", P, "--run", "base")
    run(QC / "qc_review.py", "ledger", "--project", P, label="ledger verifies")
    led = P / "audit" / "ledger.jsonl"
    orig = led.read_text()
    led.write_text(orig.replace("Dr Lee", "Dr Evil"))
    run(QC / "qc_review.py", "ledger", "--project", P, expect=1, label="edited ledger is detected")
    led.write_text(orig)
    n_cmd = len((P / "audit" / "commands.jsonl").read_text().splitlines())
    check(n_cmd > 30, f"audit log recorded every command ({n_cmd} rows)")
    run(DATA / "build_dataset.py", "--project", P)
    man2 = json.loads((P / "data" / "build_manifest.json").read_text())
    check("thick_1" not in man2["response_vars"] and "thick_3" not in man2["response_vars"],
          "dismissed variables stay out of a rebuilt dataset")
    fit(P, "base", expect=1, label="old plan refused after the dataset was rebuilt")
    run(MODEL / "run_model.py", "predict", "--project", P, "--run", "base", "--data", "clinical", "--force", expect=1,
        label="old models are not applied to the rebuilt splits")

    print("\n4. a new site: transfer and extend")
    (P2 / "data").mkdir(parents=True)
    sp = spec(work, "newsite", "newsite_demographics.csv", "newsite_idps.tsv", rvs, 0.5)
    nocode = {k: v for k, v in sp.items() if k != "recode"}
    (P2 / "data" / "spec.json").write_text(json.dumps(nocode))
    run(DATA / "build_dataset.py", "--project", P2)
    run(DATA / "validate_dataset.py", "--project", P2)
    unwarped, ref = P / "models" / "fe", P / "models" / "ref"
    run(MODEL / "choose_recipe.py", "--project", P2, "--out", P2 / "recipe_ext.json", "--reference", unwarped)
    check(json.loads((P2 / "recipe_ext.json").read_text())["mode"] == "extend_predict",
          "router proposes extend for a BLR reference without a warp")
    run(MODEL / "choose_recipe.py", "--project", P2, "--out", P2 / "recipe_tr_bad.json", "--reference", unwarped,
        "--goal", "transfer")
    out = plan(P2, "tr0", P2 / "recipe_tr_bad.json", expect=1, label="transfer of an unwarped BLR model is refused at plan time")
    check("R7" in out and "warp" in out, "refusal explains why and points to extend (check R7)")
    run(MODEL / "choose_recipe.py", "--project", P2, "--out", P2 / "recipe.json", "--reference", ref, "--goal", "transfer")
    out = plan(P2, "tr", P2 / "recipe.json", expect=1, label="sex coded 0/1 against a model coded F/M is refused")
    check("recode" in out, "refusal explains the recode")
    (P2 / "data" / "spec.json").write_text(json.dumps(sp))
    run(DATA / "build_dataset.py", "--project", P2)
    run(DATA / "validate_dataset.py", "--project", P2)
    run(MODEL / "run_model.py", "predict", "--project", P, "--run", "ref", "--data", P2 / "data" / "test.csv", "--name", "newsite",
        expect=1, label="scoring a site the model was not fitted on is refused with a reason")
    r_tr = json.loads((P2 / "recipe.json").read_text())
    out = plan(P2, "trk", recipe_file(P2 / "recipe_kw.json", {**r_tr, "transfer_kwargs": {"freedom": 0.5}}), expect=2,
               label="a transfer option the reference's algorithm ignores is refused")
    check("freedom" in out, "refusal names the option")
    out = plan(P2, "tr", P2 / "recipe.json")
    check("R4" in out and "3 in the data have no fitted" in out, "variables the reference lacks are reported and left out", out[-600:])
    fit(P2, "tr", label="transfer runs (warped BLR reference)")
    post(P2, "tr")
    check(set(z_columns(P2, "tr")) == set(ref_rvs) | {"observations", "subject_ids"}, "transferred z-scores written", str(z_columns(P2, "tr")))
    run(MODEL / "run_model.py", "predict", "--project", P2, "--run", "tr", "--data", "clinical", "--force",
        label="reloaded transferred model predicts (PCNtoolkit saves it with plotting switched on)")
    run(QC / "qc_detect.py", "--project", P2, "--run", "tr", "--plots", "none")
    gt = json.loads((P2 / "qc" / "tr" / "grades.json").read_text())
    check(all(abs(f["metrics"].get("z_mean", 9)) < 0.5 for f in gt["features"]),
          "transferred model is centred on the new site's held-out controls",
          str({f["response_var"]: f["metrics"].get("z_mean") for f in gt["features"]}))
    plan(P2, "tr2", recipe_file(P2 / "recipe_again.json", {**r_tr, "reference_model": str(P2 / "models" / "tr")}), expect=1,
         label="transferring an already transferred model is refused")
    plan(P2, "ex", P2 / "recipe_ext.json")
    fit(P2, "ex", label="extend runs (synthesise from the reference, pool, refit)")
    post(P2, "ex")
    meta = json.loads((P2 / "models" / "ex" / "model" / "normative_model.json").read_text())
    sites = set(meta["unique_batch_effects"]["site"])
    check({"NewSite", "SiteA", "SiteB"} <= sites, "extended model covers the reference sites and the new one", str(sites))
    n_train = json.loads((P2 / "data" / "build_manifest.json").read_text())["n_train"]
    zt = (P2 / "models" / "ex" / "results" / "Z_train.csv").read_text().splitlines()
    zf = (P2 / "models" / "ex" / "results" / "Z_extend_fit.csv").read_text().splitlines()
    check(len(zt) - 1 == n_train and len(zf) - 1 > n_train and "synth_" in zf[-1],
          "train z-scores hold the real rows only; the pooled fit table is kept apart", f"{len(zt) - 1} / {len(zf) - 1} / {n_train}")
    run(QC / "qc_detect.py", "--project", P2, "--run", "ex", "--plots", "none")
    ge = json.loads((P2 / "qc" / "ex" / "grades.json").read_text())
    check(all(abs(f["metrics"].get("z_mean", 9)) < 0.5 for f in ge["features"]),
          "extended model is centred on the new site's held-out controls",
          str({f["response_var"]: f["metrics"].get("z_mean") for f in ge["features"]}))
    if not a.no_cluster:
        cenv = cluster_env(work, a.fake)
        envfile.write_text("PCN_BACKEND=slurm\nPCN_N_BATCHES=2\n")
        plan(P2, "exc", P2 / "recipe_ext.json", "--response-vars", "thick_2,thick_4", env=cenv)
        fit(P2, "exc", env=cenv, label="extend on the cluster backend (synthesis here, pooled fit in the jobs)")
        post(P2, "exc")
        plan(P2, "trc", P2 / "recipe.json", "--response-vars", "thick_2,thick_4", env=cenv)
        fit(P2, "trc", env=cenv, label="transfer on the cluster backend")
        post(P2, "trc")
        envfile.write_text("PCN_BACKEND=local\n")
        # a cluster run has no clinical scores until `predict` is run: the export must not paper over that
        run(QC / "qc_detect.py", "--project", P2, "--run", "exc", "--plots", "none")
        for rv in ("thick_2", "thick_4"):
            decide("exc", rv, "accept_warn", "--note", "selftest", *who, project=P2)
        export = (QC / "qc_review.py", "export", "--project", P2, "--runs", "exc", "--name", "site", "--by", "Dr Lee",
                  "--id", "s777", "--confirm")
        out = run(*export, expect=1, label="export refused while accepted variables lack clinical scores")
        check("predict" in out and "clinical" in out, "refusal gives the command that scores them")
        run(MODEL / "run_model.py", "predict", "--project", P2, "--run", "exc", "--data", "clinical")
        run(*export, label="export succeeds once the clinical table is scored")

    if a.hbr and not a.fake:
        print("\n5. HBR: fit, QC, transfer")
        hb = recipe_file(P / "recipe_hbr.json", {
            "algorithm": "hbr", "basis": {"type": "bspline", "nknots": 5, "degree": 3, "basis_column": 0},
            "hbr": {"likelihood": "Normal", "random_intercept_mu": True, "draws": 300, "tune": 300,
                    "chains": 2, "cores": 2, "nuts_sampler": "nutpie"}})
        plan(P, "hbr", hb, "--response-vars", "thick_2")
        fit(P, "hbr", label="HBR fit (Normal likelihood, random site intercept)")
        post(P, "hbr")
        run(QC / "qc_detect.py", "--project", P, "--run", "hbr", "--plots", "none")
        gh = json.loads((P / "qc" / "hbr" / "grades.json").read_text())["features"][0]
        check("rhat_max" in gh["metrics"], "R-hat and ESS recomputed from idata.nc", str(gh["steps"]["S1"]))
        check(gh["metrics"].get("divergent_frac") is not None and not gh["steps"]["S1"]["skipped"],
              "divergences recorded at fit time (idata.nc holds the posterior only)", str(gh["steps"]["S1"]))
        stf = P / "models" / "hbr" / "status" / "thick_2.json"     # what a cluster run looks like: no such record
        st0 = stf.read_text()
        stf.write_text(json.dumps({**json.loads(st0), "sampler": None}))
        run(QC / "qc_detect.py", "--project", P, "--run", "hbr", "--plots", "none", label="qc_detect.py without a divergence record")
        gh = json.loads((P / "qc" / "hbr" / "grades.json").read_text())["features"][0]
        check(any("divergent" in t for t in gh["steps"]["S1"]["skipped"]),
              "without that record the divergence check is SKIPPED, not PASS", str(gh["steps"]["S1"]))
        stf.write_text(st0)
        run(MODEL / "choose_recipe.py", "--project", P2, "--out", P2 / "recipe_hbr_tr.json", "--reference", P / "models" / "hbr",
            "--goal", "transfer")
        rh = json.loads((P2 / "recipe_hbr_tr.json").read_text())
        rh["transfer_kwargs"] = {"freedom": 0.5, "draws": 200, "tune": 200}
        plan(P2, "hbr_tr", recipe_file(P2 / "recipe_hbr_tr.json", rh))
        fit(P2, "hbr_tr", label="HBR transfer with freedom and sampler overrides")
        post(P2, "hbr_tr")
        run(QC / "qc_detect.py", "--project", P2, "--run", "hbr_tr", "--plots", "none")
        gx = json.loads((P2 / "qc" / "hbr_tr" / "grades.json").read_text())["features"][0]
        check("rhat_max" in gx["metrics"] and abs(gx["metrics"].get("z_mean", 9)) < 0.5,
              "transferred HBR model converged and is centred", str(gx["metrics"]))

    n_fail = sum(not ok for ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed" + (f"   work dir kept: {work}" if a.keep or n_fail else ""))
    if n_fail:
        print("FAILED checks:")
        for ok, label in results:
            if not ok:
                print(f"  - {label}")
        if not a.fake:
            print("\nA failure in real mode usually means the installed PCNtoolkit differs from the 1.3.0 this was "
                  "verified against.\nThe single place to adapt is scripts/pcn_bridge.py (edit common/pcn_bridge.py, "
                  "then run tools/sync_common.py).")
    elif not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
