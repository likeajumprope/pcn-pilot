#!/usr/bin/env python3
"""End-to-end self-test of the three PCN-Pilot skills on synthetic data.

    $PCN_PYTHON selftest/run_selftest.py                 # against the real, installed PCNtoolkit  <- run this once per site
    python      selftest/run_selftest.py --fake          # against the bundled stand-in (orchestration only)
    ...                                   --keep DIR     # keep the working directory for inspection
    ...                                   --hbr          # also fit one small HBR model (real mode; takes minutes)

Real mode is the test that matters: it confirms that the installed PCNtoolkit accepts
the calls PCN-Pilot makes and writes the files PCN-Pilot reads. Fake mode replaces
PCNtoolkit with selftest/fake_pcntoolkit and additionally injects failures to test
resume, retry isolation, the decision rules and the ledger.

Exit code 0 = every check passed.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--keep")
    ap.add_argument("--hbr", action="store_true")
    a = ap.parse_args()
    if a.fake:
        os.environ["PYTHONPATH"] = str(ROOT / "selftest" / "fake_pcntoolkit") + os.pathsep + os.environ.get("PYTHONPATH", "")
        os.environ["PCN_SKIP_SCHEDULER_CHECK"] = "1"
    os.environ.pop("PCN_PIPELINE_ENV", None)
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="pcnpilot_selftest_"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    P, P2 = work / "proj", work / "proj_new"
    rvs = ["thick_1", "thick_2", "thick_3", "thick_4", "vol_skewed", "vol_hetero", "vol_site_var"]
    print(f"mode: {'FAKE stand-in' if a.fake else 'REAL PCNtoolkit'}   python: {sys.executable}\nwork dir: {work}\n")

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
    check("last_name" not in (P / "data" / "clean.csv").read_text().splitlines()[0], "PHI column absent from clean.csv")
    run(DATA / "check_status.py", "post", "data", "--project", P)
    run(DATA / "normdata_selftest.py", "--project", P, "--fit-one", label="PCNtoolkit accepts the tables (NormData + minimal BLR)")

    print("\n2. pcn-model-skill")
    run(MODEL / "choose_recipe.py", "--project", P, "--out", P / "recipe_auto.json", "--prefer", "blr")
    plain = P / "recipe_plain.json"
    plain.write_text(json.dumps({"algorithm": "blr", "mode": "fit_predict", "basis": {"type": "linear"},
                                 "blr": {"n_iter": 200}}))
    bad = P / "recipe_bad.json"
    bad.write_text(json.dumps({"algorithm": "blr", "basis": {"type": "linear"}, "blr": {"no_such_option": 1}}))
    out = run(MODEL / "run_model.py", "plan", "--project", P, "--run", "strict", "--recipe", bad, expect=2,
              label="unknown model option is refused, not dropped")
    check("does not declare" in out, "refusal names the unsupported option")
    run(MODEL / "check_status.py", "pre", "model", "--project", P, "--run", "base", expect=1, label="pre-check fails without a plan")
    run(MODEL / "run_model.py", "plan", "--project", P, "--run", "base", "--recipe", plain)
    check(not (P / "models" / "base" / "model").exists(), "plan computed nothing")
    run(MODEL / "run_model.py", "run", "--project", P, "--run", "base", expect=2, label="run refused without --approved-by")
    if a.fake:
        run(MODEL / "run_model.py", "run", "--project", P, "--run", "base", "--approved-by", "Self Test",
            expect=3, env={"PCN_FAKE_FAIL": "thick_3"}, label="one failing variable does not stop the others")
        run(MODEL / "check_status.py", "post", "model", "--project", P, "--run", "base", expect=3, label="post-check reports PARTIAL")
        s = json.loads((P / "models" / "base" / "status_summary.json").read_text())
        check(s["n_done"] == 6 and list(s["failed"]) == ["thick_3"], "exactly the failing variable is reported")
    out = run(MODEL / "run_model.py", "run", "--project", P, "--run", "base", "--approved-by", "Self Test")
    if a.fake:
        check("6 complete, 1 pending" in out, "resume redid only the missing variable")
    run(MODEL / "check_status.py", "post", "model", "--project", P, "--run", "base")
    out = run(MODEL / "run_model.py", "run", "--project", P, "--run", "base", "--approved-by", "Self Test")
    check("nothing to do" in out, "re-running a finished run is a no-op")
    res = P / "models" / "base" / "results"
    check(all((res / f).is_file() for f in ("Z_test.csv", "statistics_test.csv")), "PCNtoolkit result files found where expected",
          str(sorted(p.name for p in res.glob("*"))) if res.exists() else "no results dir")
    check((res / "Z_clinical.csv").is_file(), "clinical cohort scored")
    check((res / "Z_train.csv").is_file(), "train z-scores available for the overfitting check")
    run(MODEL / "run_model.py", "plan", "--project", P, "--run", "base", "--recipe", P / "recipe_auto.json", expect=2,
        label="existing run cannot be re-planned with another recipe")
    if a.fake:
        (P / "pipeline.env").write_text("PCN_BACKEND=slurm\nPCN_CONDA_ENV=/opt/env\nPCN_N_BATCHES=3\n")
        run(MODEL / "run_model.py", "plan", "--project", P, "--run", "clus", "--recipe", plain)
        run(MODEL / "run_model.py", "run", "--project", P, "--run", "clus", "--approved-by", "Self Test",
            env={"PCN_FAKE_FAIL": "thick_1"}, label="cluster submission (stand-in scheduler)")
        run(MODEL / "run_model.py", "status", "--project", P, "--run", "clus", expect=3, label="a failed batch leaves its variables pending")
        out = run(MODEL / "run_model.py", "run", "--project", P, "--run", "clus", "--approved-by", "Self Test")
        check("submitted 3 job(s) for 3 variables (attempt 2)" in out, "retry isolates each remaining variable in its own job")
        run(MODEL / "run_model.py", "status", "--project", P, "--run", "clus")
        run(MODEL / "run_model.py", "predict", "--project", P, "--run", "clus", "--data", "clinical")
        (P / "pipeline.env").unlink()

    print("\n3. pcn-qc-skill")
    run(QC / "check_status.py", "pre", "qc", "--project", P, "--run", "base")
    run(QC / "qc_detect.py", "--project", P, "--run", "base", "--plots", "flagged")
    g = json.loads((P / "qc" / "base" / "grades.json").read_text())
    feats = {f["response_var"]: f for f in g["features"]}
    check("verified" in g["alignment"] or "subject ID" in g["alignment"], "z-scores matched back to test subjects", g["alignment"])
    codes = {rv: {fl["code"] for s in f["steps"].values() for fl in s["flags"]} for rv, f in feats.items()}
    skipped = {rv: [t for s in f["steps"].values() for t in s["skipped"]] for rv, f in feats.items()}
    check(all(f["steps"]["S2"]["status"] != "SKIPPED" for f in feats.values()), "calibration computed for every variable")
    check(all(f["steps"]["S4"]["status"] != "SKIPPED" and f["steps"]["S5"]["status"] != "SKIPPED" for f in feats.values()),
          "batch and covariate checks computed", str(skipped))
    check(not any("centile" in t for ts in skipped.values() for t in ts), "centile file parsed", str(skipped))
    check("EXTREME_Z" in codes["thick_1"] or json.loads((P / "data" / "build_manifest.json").read_text())["n_test"] == 0
          or "006S0005" not in (P / "data" / "test.csv").read_text(), "planted unit error caught as an extreme z-score")
    check("SITE_MEAN" in codes["thick_2"], "unmodelled site effect detected", str(codes["thick_2"]))
    check(codes["vol_skewed"] & {"Z_SKEW", "Z_KURT", "MACE"}, "skewed variable detected", str(codes["vol_skewed"]))
    run(QC / "qc_review.py", "dashboard", "--project", P, "--run", "base")
    html = (P / "qc" / "base" / "dashboard.html").read_text()
    check("__PCN_DATA__" not in html and "thick_2" in html, "dashboard built with the run's data embedded")

    def decide(run_, rv, decision, *extra, expect=0, label=None):
        return run(QC / "qc_review.py", "decide", "--project", P, "--run", run_, "--response-var", rv,
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
    run(MODEL / "run_model.py", "plan", "--project", P, "--run", "base_fix1", "--recipe", fx / "recipe.json",
        "--response-vars-file", fx / "response_vars.txt")
    run(MODEL / "run_model.py", "run", "--project", P, "--run", "base_fix1", "--approved-by", "Ana Reviewer")
    run(MODEL / "check_status.py", "post", "model", "--project", P, "--run", "base_fix1")
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
    head = (exp / "z_test.csv").read_text().splitlines()[0].split(",") if (exp / "z_test.csv").is_file() else []
    check(set(head) - {"observations", "subject_ids"} == {"thick_2", "thick_4", "vol_skewed", "vol_hetero", "vol_site_var"},
          "export holds exactly the accepted variables", str(head))
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
    run(MODEL / "run_model.py", "run", "--project", P, "--run", "base", "--approved-by", "Self Test", expect=1,
        label="old plan refused after the dataset was rebuilt")

    print("\n4. transfer to a new site")
    (P2 / "data").mkdir(parents=True)
    sp = spec(work, "newsite", "newsite_demographics.csv", "newsite_idps.tsv", rvs, 0.5)
    ref = P / "models" / ("clus" if a.fake else "base")
    nocode = {k: v for k, v in sp.items() if k != "recode"}
    (P2 / "data" / "spec.json").write_text(json.dumps(nocode))
    run(DATA / "build_dataset.py", "--project", P2)
    run(DATA / "validate_dataset.py", "--project", P2)
    run(MODEL / "choose_recipe.py", "--project", P2, "--out", P2 / "recipe.json", "--reference", ref, "--goal", "transfer")
    out = run(MODEL / "run_model.py", "plan", "--project", P2, "--run", "tr", "--recipe", P2 / "recipe.json", expect=1,
              label="sex coded 0/1 against a model coded F/M is refused")
    check("recode" in out, "refusal explains the recode")
    (P2 / "data" / "spec.json").write_text(json.dumps(sp))
    run(DATA / "build_dataset.py", "--project", P2)
    run(DATA / "validate_dataset.py", "--project", P2)
    run(MODEL / "run_model.py", "plan", "--project", P2, "--run", "tr", "--recipe", P2 / "recipe.json")
    run(MODEL / "run_model.py", "run", "--project", P2, "--run", "tr", "--approved-by", "Self Test")
    run(MODEL / "check_status.py", "post", "model", "--project", P2, "--run", "tr")
    run(QC / "qc_detect.py", "--project", P2, "--run", "tr", "--plots", "none")
    r2 = json.loads((P2 / "recipe.json").read_text())
    r2["reference_model"] = str(P2 / "models" / "tr")
    (P2 / "recipe_again.json").write_text(json.dumps(r2))
    run(MODEL / "run_model.py", "plan", "--project", P2, "--run", "tr2", "--recipe", P2 / "recipe_again.json", expect=1,
        label="transferring an already transferred model is refused")
    r2["mode"] = "extend_predict"
    (P2 / "recipe_extend.json").write_text(json.dumps(r2))
    run(MODEL / "run_model.py", "plan", "--project", P2, "--run", "ex", "--recipe", P2 / "recipe_extend.json")
    run(MODEL / "run_model.py", "run", "--project", P2, "--run", "ex", "--approved-by", "Self Test", label="extend runs")
    run(MODEL / "check_status.py", "post", "model", "--project", P2, "--run", "ex")

    if a.hbr and not a.fake:
        print("\n5. one HBR model")
        hb = P2 / "recipe_hbr.json"
        hb.write_text(json.dumps({"algorithm": "hbr", "basis": {"type": "bspline", "nknots": 5, "degree": 3, "basis_column": 0},
                                  "hbr": {"likelihood": "Normal", "random_intercept_mu": True, "draws": 300, "tune": 300,
                                          "chains": 2, "cores": 2, "nuts_sampler": "nutpie"}}))
        run(MODEL / "run_model.py", "plan", "--project", P2, "--run", "hbr", "--recipe", hb, "--response-vars", "thick_2")
        run(MODEL / "run_model.py", "run", "--project", P2, "--run", "hbr", "--approved-by", "Self Test")
        run(MODEL / "check_status.py", "post", "model", "--project", P2, "--run", "hbr")
        run(QC / "qc_detect.py", "--project", P2, "--run", "hbr", "--plots", "none")
        gh = json.loads((P2 / "qc" / "hbr" / "grades.json").read_text())["features"][0]
        check("rhat_max" in gh["metrics"], "MCMC convergence read from idata.nc", str(gh["steps"]["S1"]))

    n_fail = sum(not ok for ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed" + (f"   work dir kept: {work}" if a.keep or n_fail else ""))
    if n_fail:
        print("FAILED checks:")
        for ok, label in results:
            if not ok:
                print(f"  - {label}")
        if not a.fake:
            print("\nA failure in real mode usually means the installed PCNtoolkit differs from the 1.3 API this was "
                  "written against.\nThe single place to adapt is scripts/pcn_bridge.py (edit common/pcn_bridge.py, "
                  "then run tools/sync_common.py).")
    elif not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
