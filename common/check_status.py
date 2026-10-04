#!/usr/bin/env python3
"""Checkpoints that bracket every PCN-Pilot stage.

    python check_status.py pre  <data|model|qc> --project P [--run R] [--input f ...]
    python check_status.py post <data|model|qc> --project P [--run R]

pre   confirms the required inputs exist, and aborts before compute is wasted
post  scans the outputs on disk and reports what actually succeeded

Exit codes: 0 ok, 1 failed, 3 partial (some items done, some not).
The agent reads these results instead of assuming a stage worked.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (EXIT_FAIL, EXIT_OK, EXIT_PARTIAL, Project, audited, ledger_verify, load_env,  # noqa: E402
                     pcntoolkit_version, read_json, read_jsonl, say, write_json)


VERIFIED_PCNTOOLKIT = "1.3.0"


class Report:
    def __init__(self, title: str):
        self.title, self.lines, self.failed, self.partial = title, [], False, False

    def ok(self, msg: str) -> None:
        self.lines.append(("ok  ", msg))

    def fail(self, msg: str) -> None:
        self.failed = True
        self.lines.append(("FAIL", msg))

    def part(self, msg: str) -> None:
        self.partial = True
        self.lines.append(("part", msg))

    def note(self, msg: str) -> None:
        self.lines.append(("note", msg))

    def finish(self) -> int:
        say(f"== {self.title} ==")
        for tag, msg in self.lines:
            say(f"[{tag}] {msg}")
        code = EXIT_FAIL if self.failed else (EXIT_PARTIAL if self.partial else EXIT_OK)
        say({EXIT_OK: "RESULT: OK", EXIT_FAIL: "RESULT: FAILED (do not continue)",
             EXIT_PARTIAL: "RESULT: PARTIAL (resume or review before continuing)"}[code])
        return code


def need_file(r: Report, path: Path, what: str) -> bool:
    if path.is_file() and path.stat().st_size > 0:
        r.ok(f"{what}: {path}")
        return True
    r.fail(f"{what} missing or empty: {path}")
    return False


# ------------------------------------------------------------------ data
def pre_data(p: Project, a) -> Report:
    r = Report("pre-run check: data")
    if not a.input:
        r.fail("no --input tables given")
    for f in a.input or []:
        need_file(r, Path(f), "input table")
        if Path(f).is_file() and not os.access(f, os.R_OK):
            r.fail(f"not readable: {f}")
    try:
        p.ensure()
        probe = p.data / ".write_probe"
        probe.write_text("x")
        probe.unlink()
        r.ok(f"project directory is writable: {p.root}")
    except OSError as e:
        r.fail(f"cannot write to project directory {p.root}: {e}")
    for f in a.input or []:
        try:
            if Path(f).resolve().is_relative_to(p.data):
                r.fail(f"input {f} lives inside the project's data/ folder; sources must stay outside it (read-only)")
        except (OSError, ValueError):
            pass
    return r


def post_data(p: Project, a) -> Report:
    r = Report("post-run check: data")
    d = p.data
    files_ok = all([need_file(r, d / n, n) for n in
                    ("spec.json", "clean.csv", "train.csv", "test.csv", "dropped.csv", "build_manifest.json")])
    v = read_json(p.validation)
    if v is None:
        r.fail("validation.json missing: validate_dataset.py has not been run")
    elif v["overall"] == "FAIL":
        r.fail("validation has FAIL checks: " + ", ".join(c["id"] for c in v["checks"] if c["status"] == "FAIL"))
    else:
        warn = [c["id"] for c in v["checks"] if c["status"] == "WARN"]
        r.ok(f"validation overall {v['overall']}" + (f" (WARN: {', '.join(warn)}: each needs a human decision)" if warn else ""))
    need_file(r, d / "data_report.md", "data report (required closing artifact)")
    if files_ok and v:
        man = read_json(d / "build_manifest.json")
        newer = [n for n in ("clean.csv", "train.csv", "test.csv")
                 if (d / n).stat().st_mtime > p.validation.stat().st_mtime + 1]
        if newer:
            r.fail(f"{newer} were rebuilt after the last validation: re-run validate_dataset.py")
        r.note(f"{man['n_train']} train / {man['n_test']} test / {man['n_clinical']} clinical observations, "
               f"{len(man['response_vars'])} response variables")
    return r


# ------------------------------------------------------------------ model
def _run_features(p: Project, run: str) -> tuple[list[str], Path]:
    rd = p.run_dir(run)
    plan = read_json(rd / "plan.json") or {}
    return list(plan.get("response_vars", [])), rd


def pre_model(p: Project, a) -> Report:
    r = Report(f"pre-run check: model ({a.run})")
    env = load_env(p.root, a.env)
    r.note(f"pipeline.env: {env['_ENV_FILE'] or 'not found, using defaults'}")
    v = read_json(p.validation)
    if v is None or v["overall"] == "FAIL":
        r.fail("data stage is not validated (run pcn-data-skill to a non-FAIL validation first)")
    else:
        r.ok(f"data validated: {v['overall']}")
    ver = pcntoolkit_version()
    if ver is None:
        r.fail(f"pcntoolkit not importable in {sys.executable}; set PCN_PYTHON in pipeline.env")
    elif ver.split(".")[0].isdigit() and int(ver.split(".")[0]) < 1:
        r.fail(f"pcntoolkit {ver} is a legacy 0.x release; v1.x is required")
    else:
        r.ok(f"pcntoolkit {ver}")
        if not ver.startswith(VERIFIED_PCNTOOLKIT):
            r.note(f"PCN-Pilot was verified against PCNtoolkit {VERIFIED_PCNTOOLKIT}; with {ver}, run "
                   "selftest/run_selftest.py once with this interpreter before relying on the results")
    rd = p.run_dir(a.run)
    plan = read_json(rd / "plan.json")
    if plan is None:
        r.fail(f"no plan for run '{a.run}': run `run_model.py plan` and get it approved")
        return r
    r.ok(f"plan: {plan['mode']} / {plan['algorithm']} / backend {plan['backend']} / "
         f"{len(plan['response_vars'])} response variables / {plan['n_jobs']} job(s)")
    bad = [c for c in plan.get("compatibility", []) if c["status"] == "FAIL"]
    for c in bad:
        r.fail(f"reference-model compatibility {c['id']}: {c['detail']}")
    if plan["backend"] in ("slurm", "torque"):
        exe = "sbatch" if plan["backend"] == "slurm" else "qsub"
        if shutil.which(exe):
            r.ok(f"{exe} found")
        else:
            r.fail(f"backend {plan['backend']} selected but `{exe}` is not on PATH")
        envdir = env.get("PCN_CONDA_ENV")
        if not envdir:
            r.fail("PCN_CONDA_ENV is empty: cluster jobs need the environment path")
        elif not (Path(envdir) / "bin" / "python").exists():
            r.fail(f"PCN_CONDA_ENV={envdir} has no bin/python: PCNtoolkit's Runner refuses such a path. Give the "
                   "environment's root folder (the parent of bin/), not the interpreter")
    from pcn_bridge import unstorable_names
    weird = unstorable_names(plan["response_vars"])
    if weird:
        r.fail(f"response-variable names PCNtoolkit cannot store (see check V14 of the data stage): {list(weird)[:5]}")
    try:
        rd.mkdir(parents=True, exist_ok=True)
        (rd / ".write_probe").write_text("x")
        (rd / ".write_probe").unlink()
        r.ok(f"run directory is writable: {rd}")
    except OSError as e:
        r.fail(f"cannot write to {rd}: {e}")
    return r


def post_model(p: Project, a) -> Report:
    from pcn_bridge import feature_done, results_columns
    r = Report(f"post-run check: model ({a.run})")
    rvs, rd = _run_features(p, a.run)
    if not rvs:
        r.fail(f"run '{a.run}' has no plan.json")
        return r
    zc = results_columns(rd, "test")
    done = [v for v in rvs if feature_done(rd, v, zc)]
    todo = [v for v in rvs if v not in done]
    failed = {}
    for v in todo:
        st = read_json(rd / "status" / f"{v}.json") or {}
        if st.get("state") == "failed":
            failed[v] = st.get("error", "unknown error")
    summary = {"run": a.run, "n_total": len(rvs), "n_done": len(done), "n_failed": len(failed),
               "n_not_started": len(todo) - len(failed), "done": done, "failed": failed,
               "not_started": [v for v in todo if v not in failed]}
    write_json(rd / "status_summary.json", summary)
    r.note(f"{len(done)}/{len(rvs)} response variables have a saved model and test z-scores")
    if failed:
        for v, err in list(failed.items())[:10]:
            r.part(f"failed: {v}: {err[:160]}")
        if len(failed) > 10:
            r.part(f"... and {len(failed) - 10} more failures (see status_summary.json)")
    if summary["n_not_started"]:
        r.part(f"{summary['n_not_started']} response variables have no outputs yet (still running, or never ran)")
    if not done:
        r.fail("nothing completed")
    elif not todo:
        r.ok("all response variables completed")
    for extra in ("train", "clinical"):
        if extra == "clinical" and not (p.data / "clinical.csv").exists():
            continue
        cols = results_columns(rd, extra)
        miss = [v for v in done if v not in cols]
        if miss:
            r.note(f"{len(miss)} completed variables have no '{extra}' predictions yet "
                   f"(run `run_model.py predict --data {extra}`)")
    return r


# ------------------------------------------------------------------ qc
def pre_qc(p: Project, a) -> Report:
    r = Report(f"pre-run check: qc ({a.run})")
    rd = p.run_dir(a.run)
    s = read_json(rd / "status_summary.json")
    if s is None:
        r.fail("no status_summary.json: run `check_status.py post model` first")
    elif s["n_done"] == 0:
        r.fail("the run has no completed response variable")
    else:
        r.ok(f"{s['n_done']} completed response variables to review")
        if s["n_done"] < s["n_total"]:
            r.note(f"{s['n_total'] - s['n_done']} variables did not complete and will be graded FAIL at step 1")
    need_file(r, rd / "results" / "Z_test.csv", "test z-scores")
    need_file(r, p.data / "test.csv", "test split")
    ok, msg = ledger_verify(p)
    (r.ok if ok else r.fail)(f"sign-off ledger: {msg}")
    return r


def _settled_in_fix_run(p: Project, run: str, rv: str, depth: int = 0) -> bool:
    """A 'fix' decision is closed once the variable is accepted or dismissed in a fix run (followed recursively)."""
    if depth > 10:
        return False
    index = read_json(p.qc_dir(run) / "fixes" / "index.json", {}) or {}
    for new_run, entry in index.items():
        if rv not in entry.get("response_vars", []):
            continue
        last = None
        for row in read_jsonl(p.qc_dir(new_run) / "decisions.jsonl"):
            if row["response_var"] == rv:
                last = row
        if last is None:
            continue
        if last["decision"] in ("pass", "accept_warn", "dismiss"):
            return True
        if last["decision"] == "fix" and _settled_in_fix_run(p, new_run, rv, depth + 1):
            return True
    return False


def post_qc(p: Project, a) -> Report:
    r = Report(f"post-run check: qc ({a.run})")
    qd = p.qc_dir(a.run)
    grades = read_json(qd / "grades.json")
    if grades is None:
        r.fail("grades.json missing: qc_detect.py has not been run")
        return r
    latest = {}
    for row in read_jsonl(qd / "decisions.jsonl"):
        latest[row["response_var"]] = row
    undecided = [f["response_var"] for f in grades["features"] if f["response_var"] not in latest]
    counts = {}
    for row in latest.values():
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    r.note("decisions: " + (", ".join(f"{k} {n}" for k, n in sorted(counts.items())) or "none"))
    if undecided:
        r.part(f"{len(undecided)} response variables have no human decision yet: {undecided[:8]}")
    open_fix = [v for v, row in latest.items()
                if row["decision"] == "escalate" or (row["decision"] == "fix" and not _settled_in_fix_run(p, a.run, v))]
    if open_fix:
        r.part(f"{len(open_fix)} variables are waiting for a fix, its review, or a supervisor: {open_fix[:8]}")
    ok, msg = ledger_verify(p)
    (r.ok if ok else r.fail)(f"sign-off ledger: {msg}")
    if (qd / "qc_report.md").is_file():
        r.ok("qc_report.md present")
    else:
        r.part("qc_report.md missing: the QC stage is not closed until qc_export.py writes it")
    return r


STAGES = {("pre", "data"): pre_data, ("post", "data"): post_data, ("pre", "model"): pre_model,
          ("post", "model"): post_model, ("pre", "qc"): pre_qc, ("post", "qc"): post_qc}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("when", choices=["pre", "post"])
    ap.add_argument("stage", choices=["data", "model", "qc"])
    ap.add_argument("--project", required=True)
    ap.add_argument("--run")
    ap.add_argument("--input", nargs="*")
    ap.add_argument("--env", help="explicit pipeline.env path")
    a = ap.parse_args(argv)
    if a.stage in ("model", "qc") and not a.run:
        ap.error("--run is required for the model and qc stages")
    project = Project(a.project)
    with audited(project if project.root.exists() else None, "check_status.py", argv):
        raise SystemExit(STAGES[(a.when, a.stage)](project, a).finish())


if __name__ == "__main__":
    raise SystemExit(main())
