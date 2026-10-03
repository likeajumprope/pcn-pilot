#!/usr/bin/env python3
"""PROPOSE / APPROVE / FIX / VERIFY / SIGN-OFF half of pcn-qc-skill.

    python qc_review.py dashboard --project P --run R
    python qc_review.py serve     --project P --run R [--port 8765]
    python qc_review.py decide    --project P --run R --response-var V --decision pass|accept_warn|fix|dismiss|escalate
                                  --by NAME --id ID [--fix-id A] [--note TEXT] [--role reviewer|supervisor]
    python qc_review.py import    --project P --run R --file decisions_R.json
    python qc_review.py fix       --project P --run R
    python qc_review.py compare   --project P --before R --after R_fix1
    python qc_review.py export    --project P --runs R,R_fix1 --name NAME --by NAME --id ID --confirm
    python qc_review.py ledger    --project P [--verify]

dashboard  writes a self-contained review page (qc/<run>/dashboard.html)
serve      serves it on 127.0.0.1 and saves each decision the moment it is made
decide     records one decision from the command line (same rules as the dashboard)
import     loads decisions downloaded from an offline copy of the dashboard
fix        turns approved fixes into new recipes and prints the pcn-model-skill commands (it never fits)
compare    before/after table for refitted variables (the verify step)
export     the signed gate: writes accepted z-scores + the QC report and appends to the ledger
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (EXIT_PARTIAL, Project, audited, die, ledger_append, ledger_verify, load_dismissed,  # noqa: E402
                     load_env, now, read_json, read_jsonl, say, write_json)
from qc_detect import STEP_TITLES, deep_merge, read_z  # noqa: E402
from qc_store import ACCEPTING, DecisionError, latest_decisions, record_decision  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"


def need_grades(project: Project, run: str) -> dict:
    g = read_json(project.qc_dir(run) / "grades.json")
    if g is None:
        die(f"run '{run}' has not been graded: run qc_detect.py first", 2)
    return g


# ---------------------------------------------------------------- dashboard
def build_dashboard(project: Project, run: str) -> Path:
    g = need_grades(project, run)
    data = {"run": run, "counts": g["counts"], "step_titles": g["step_titles"], "features": g["features"],
            "decisions": latest_decisions(project, run), "graded_at": g["graded_at"],
            "dataset_notes": g.get("dataset_notes", [])}
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    data["features"] = sorted(data["features"], key=lambda f: (order.get(f["grade"], 3), f["response_var"]))
    html = (ASSETS / "dashboard_template.html").read_text(encoding="utf-8")
    blob = json.dumps(data, default=str).replace("</", "<\\/")
    out = project.qc_dir(run) / "dashboard.html"
    out.write_text(html.replace("__PCN_DATA__", blob), encoding="utf-8")
    return out


def cmd_dashboard(a, project: Project) -> int:
    out = build_dashboard(project, a.run)
    say(f"dashboard written: {out}")
    say(f"To record decisions directly: qc_review.py serve --project {project.root} --run {a.run}")
    return 0


# ---------------------------------------------------------------- server
def make_handler(project: Project, run: str):
    qd = project.qc_dir(run).resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "PCNPilotQC/1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html", "/dashboard.html"):
                return self._send(200, build_dashboard(project, run).read_bytes(), "text/html; charset=utf-8")
            if path == "/api/decisions":
                return self._json(200, latest_decisions(project, run))
            if path.startswith("/plots/") and path.endswith(".png"):
                f = (qd / path.lstrip("/")).resolve()
                if f.is_file() and qd in f.parents:              # no path traversal
                    return self._send(200, f.read_bytes(), "image/png")
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/api/decision":
                return self._json(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if n > 100_000:
                    return self._json(413, {"error": "payload too large"})
                row = json.loads(self.rfile.read(n) or b"{}")
                self._json(200, record_decision(project, run, row, source="dashboard"))
            except DecisionError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # noqa: BLE001
                self._json(500, {"error": f"{type(e).__name__}: {e}"})
    return Handler


def cmd_serve(a, project: Project) -> int:
    need_grades(project, a.run)
    port = int(a.port or load_env(project.root)["PCN_QC_PORT"])
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(project, a.run))
    say(f"QC dashboard for run '{a.run}': http://127.0.0.1:{port}/")
    say("It listens on this machine only. From a laptop, tunnel first:")
    say(f"  ssh -N -L {port}:127.0.0.1:{port} <user>@<this host>     then open http://127.0.0.1:{port}/")
    say("Decisions are saved as they are made. Stop with Ctrl-C.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        say("stopped")
    finally:
        httpd.server_close()
    return 0


# ---------------------------------------------------------------- decide / import
def cmd_decide(a, project: Project) -> int:
    try:
        row = record_decision(project, a.run, {
            "response_var": a.response_var, "decision": a.decision, "fix_id": a.fix_id, "note": a.note,
            "by": a.by, "id": a.id, "role": a.role}, source="cli")
    except DecisionError as e:
        die(str(e))
    say(f"recorded: {row['response_var']} -> {row['decision']}"
        + (f" ({row['fix_label']})" if row["fix_label"] else "") + f" by {row['by']} ({row['id']})")
    return 0


def cmd_import(a, project: Project) -> int:
    payload = read_json(a.file)
    if payload is None:
        die(f"file not found: {a.file}", 2)
    if payload.get("run") != a.run:
        die(f"the file holds decisions for run '{payload.get('run')}', not '{a.run}'")
    n_ok, problems = 0, []
    for row in payload.get("decisions", []):
        try:
            record_decision(project, a.run, row, source="import")
            n_ok += 1
        except DecisionError as e:
            problems.append(f"{row.get('response_var')}: {e}")
    say(f"imported {n_ok} decisions, refused {len(problems)}")
    for p in problems:
        say(f"  refused {p}")
    if problems:
        raise SystemExit(EXIT_PARTIAL)
    return 0


# ---------------------------------------------------------------- fix (delegation only)
def cmd_fix(a, project: Project) -> int:
    g = need_grades(project, a.run)
    feats = {f["response_var"]: f for f in g["features"]}
    base = read_json(project.run_dir(a.run) / "recipe.json")
    groups: dict[str, dict] = {}
    data_review = []
    for rv, d in latest_decisions(project, a.run).items():
        if d["decision"] != "fix":
            continue
        cand = next(c for c in feats[rv]["fix_candidates"] if c["id"] == d["fix_id"])
        if cand["change"] is None:
            data_review.append(rv)
            continue
        key = json.dumps(cand["change"], sort_keys=True)
        groups.setdefault(key, {"label": cand["label"], "change": cand["change"], "vars": [], "by": set()})
        groups[key]["vars"].append(rv)
        groups[key]["by"].add(f"{d['by']} ({d['id']})")
    if not groups and not data_review:
        say("no approved fixes to prepare")
        return 0
    fixes_dir = project.qc_dir(a.run) / "fixes"
    index = read_json(fixes_dir / "index.json", {}) or {}
    existing = set(project.runs()) | set(index)
    k = 1
    for key, grp in sorted(groups.items()):
        prior = next((name for name, e in index.items() if e["change_key"] == key
                      and sorted(e["response_vars"]) == sorted(grp["vars"])), None)
        if prior:
            new_run = prior
        else:
            while f"{a.run}_fix{k}" in existing:
                k += 1
            new_run = f"{a.run}_fix{k}"
            existing.add(new_run)
        recipe = deep_merge({kk: v for kk, v in base.items() if kk not in ("traits", "alternatives")}, grp["change"])
        recipe["rationale"] = [f"QC fix for run '{a.run}': {grp['label']}", f"approved by {', '.join(sorted(grp['by']))}"]
        d = fixes_dir / new_run
        d.mkdir(parents=True, exist_ok=True)
        write_json(d / "recipe.json", recipe)
        (d / "response_vars.txt").write_text("\n".join(sorted(grp["vars"])) + "\n", encoding="utf-8")
        index[new_run] = {"from_run": a.run, "label": grp["label"], "change": grp["change"], "change_key": key,
                          "response_vars": sorted(grp["vars"]), "prepared_at": now()}
        say(f"\nFix '{grp['label']}' for {len(grp['vars'])} variable(s) -> new run '{new_run}'")
        say("  Delegate to pcn-model-skill (QC never fits or edits model outputs itself):")
        say(f"    run_model.py plan --project {project.root} --run {new_run} --recipe {d / 'recipe.json'} "
            f"--response-vars-file {d / 'response_vars.txt'}")
        say(f"    run_model.py run  --project {project.root} --run {new_run} --approved-by \"<name>\"")
        say(f"  Then verify:  qc_detect.py --project {project.root} --run {new_run}")
        say(f"                qc_review.py compare --project {project.root} --before {a.run} --after {new_run}")
    write_json(fixes_dir / "index.json", index)
    if data_review:
        say(f"\n{len(data_review)} variable(s) need their input data reviewed with pcn-data-skill before any refit: "
            f"{data_review[:10]}")
    return 0


# ---------------------------------------------------------------- compare (verify)
COMPARE_KEYS = ("z_mean", "z_sd", "z_skew", "z_kurt", "mace_z", "MSLL", "EXPV", "centile_cross_frac")


def cmd_compare(a, project: Project) -> int:
    gb, ga = need_grades(project, a.before), need_grades(project, a.after)
    fb = {f["response_var"]: f for f in gb["features"]}
    rows = []
    rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
    for f in ga["features"]:
        rv = f["response_var"]
        if rv not in fb:
            continue
        b = fb[rv]
        row = {"response_var": rv, "grade_before": b["grade"], "grade_after": f["grade"],
               "flags_before": ";".join(fl["code"] for s in b["steps"].values() for fl in s["flags"]),
               "flags_after": ";".join(fl["code"] for s in f["steps"].values() for fl in s["flags"])}
        for k in COMPARE_KEYS:
            row[f"{k}_before"], row[f"{k}_after"] = b["metrics"].get(k), f["metrics"].get(k)
        row["verdict"] = ("improved" if rank[f["grade"]] < rank[b["grade"]] else
                          "worse" if rank[f["grade"]] > rank[b["grade"]] else "same grade")
        rows.append(row)
    if not rows:
        die("the two runs share no response variable")
    df = pd.DataFrame(rows)
    out = project.qc_dir(a.after) / f"compare_{a.before}_vs_{a.after}.csv"
    df.to_csv(out, index=False)
    say(f"{'variable':<34} {'before':<6} {'after':<6} verdict       flags after")
    for r in rows:
        say(f"{r['response_var'][:34]:<34} {r['grade_before']:<6} {r['grade_after']:<6} {r['verdict']:<13} "
            f"{r['flags_after'] or '-'}")
    counts = df["verdict"].value_counts().to_dict()
    say(f"\n{counts}.  A fix that did not improve the grade is not accepted automatically: the reviewer decides "
        f"in the '{a.after}' dashboard, or picks the next candidate.")
    say(f"table: {out}")
    return 0


# ---------------------------------------------------------------- export (signed gate)
def cmd_export(a, project: Project) -> int:
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    if not a.confirm:
        die("export is a signed step: pass --confirm together with --by and --id after the reviewer has confirmed", 2)
    ok, msg = ledger_verify(project)
    if not ok:
        die(f"sign-off ledger failed verification ({msg}). Do not export; escalate to the supervisor.")
    built = {read_json(project.run_dir(r) / "plan.json")["data_built_at"] for r in runs}
    if len(built) != 1:
        die("the runs were fitted on different builds of the dataset and cannot be exported together")

    final: dict[str, dict] = {}          # later runs (fixes) override earlier ones
    all_vars: dict[str, str] = {}
    for r in runs:
        g = need_grades(project, r)
        dec = latest_decisions(project, r)
        for f in g["features"]:
            rv = f["response_var"]
            all_vars.setdefault(rv, r)
            d = dec.get(rv)
            if d is None:
                final.setdefault(rv, {"state": "undecided", "run": r, "grade": f["grade"]})
            elif d["decision"] in ACCEPTING:
                final[rv] = {"state": "accepted", "run": r, "grade": f["grade"], **d}
            elif d["decision"] == "dismiss":
                final[rv] = {"state": "dismissed", "run": r, "grade": f["grade"], **d}
            else:                                        # fix / escalate: open unless a later run settles it
                final[rv] = {"state": "open", "run": r, "grade": f["grade"], **d}
    accepted = {rv: v for rv, v in final.items() if v["state"] == "accepted"}
    open_ = {rv: v for rv, v in final.items() if v["state"] in ("open", "undecided")}
    dismissed = {rv: v for rv, v in final.items() if v["state"] == "dismissed"}
    if open_ and not a.allow_open:
        say(f"{len(open_)} variable(s) are undecided, awaiting a fix, or escalated:")
        for rv, v in list(open_.items())[:20]:
            say(f"  {rv}: {v['state']} in run {v['run']}" + (f" ({v.get('decision')})" if v.get("decision") else ""))
        die("export refused. Settle them, or pass --allow-open to export only the accepted variables "
            "(the open ones are then listed as excluded in the report).")
    if not accepted:
        die("no accepted variable to export")

    out = project.export_dir(a.name)
    if out.exists() and any(out.iterdir()):
        die(f"{out} already exists. Exports are never overwritten: choose another --name.", 2)
    out.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for split in ("test", "clinical", "train"):
        frames = None
        for r in runs:
            cols = [rv for rv, v in accepted.items() if v["run"] == r]
            z = read_z(project.run_dir(r), split)
            if z is None or not cols:
                continue
            keys = [c for c in ("observations", "subject_ids") if c in z.columns]
            part = z[keys + [c for c in cols if c in z.columns]]
            frames = part if frames is None else frames.merge(part, on=keys, how="outer")
        if frames is not None and frames.shape[1] > 2:
            p = out / f"z_{split}.csv"
            frames.to_csv(p, index=False)
            checksums[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    pd.DataFrame([{"response_var": rv, "run": v["run"], "model_dir": str(project.run_dir(v["run"]) / "model" / rv),
                   "agent_grade": v["grade"], "decision": v["decision"], "override": v.get("override", False),
                   "reviewer": v["by"], "reviewer_id": v["id"], "reason": v.get("note", "")}
                  for rv, v in sorted(accepted.items())]).to_csv(out / "accepted_models.csv", index=False)

    row = ledger_append(project, a.by, a.id, "qc_export",
                        f"exported {len(accepted)} accepted variables from runs {runs}; "
                        f"{len(dismissed)} dismissed, {len(open_)} excluded as open",
                        {"export": a.name, "runs": runs, "n_accepted": len(accepted), "n_dismissed": len(dismissed),
                         "n_open": len(open_), "checksums": checksums})

    L = [f"# QC report: {a.name}", "",
         f"Signed off by **{a.by}** ({a.id}) at {row['at']}. Ledger hash `{row['hash'][:16]}`.", "",
         f"Runs: {', '.join(runs)}. {len(all_vars)} response variables reviewed: "
         f"**{len(accepted)} accepted**, **{len(dismissed)} dismissed**, **{len(open_)} excluded as open**.", ""]
    n_over = sum(bool(v.get("override")) for v in accepted.values())
    n_caveat = sum(v["decision"] == "accept_warn" for v in accepted.values())
    n_fixed = sum(v["run"] != runs[0] for v in accepted.values())
    L += ["## Accepted", "", f"- {len(accepted) - n_caveat - n_over} passed",
          f"- {n_caveat} accepted with a documented caveat", f"- {n_over} passed against the agent's FAIL grade (overrides)",
          f"- {n_fixed} accepted from a fix run rather than the original run", ""]
    for title, pick in (("Accepted with a caveat or as an override", lambda v: v["decision"] == "accept_warn" or v.get("override")),):
        rows = [(rv, v) for rv, v in sorted(accepted.items()) if pick(v)]
        if rows:
            L += [f"### {title}", "", "| Variable | Run | Agent grade | Reviewer | Reason |", "|---|---|---|---|---|"]
            L += [f"| {rv} | {v['run']} | {v['grade']} | {v['by']} ({v['id']}) | {v.get('note', '')} |" for rv, v in rows]
            L.append("")
    L += ["## Dismissed (nothing was deleted; models and results remain in the run folders)", ""]
    if dismissed:
        L += ["| Variable | Run | Agent grade | Reviewer | Reason |", "|---|---|---|---|---|"]
        L += [f"| {rv} | {v['run']} | {v['grade']} | {v['by']} ({v['id']}) | {v.get('note', '')} |"
              for rv, v in sorted(dismissed.items())]
    else:
        L.append("None.")
    L += ["", "## Excluded as open", ""]
    L += [f"- {rv}: {v['state']} in run {v['run']}" for rv, v in sorted(open_.items())] or ["None."]
    dis = load_dismissed(project)
    L += ["", "## Project dismissal list", ""]
    for kind, items in dis.items():
        L.append(f"- {kind}: {len(items)}")
        L += [f"  - {it['id']}: {it['reason']} ({it['by']}, {it['step']}, {it['at']})" for it in items]
    L += ["", "## Fixes applied", ""]
    any_fix = False
    for r in runs:
        idx = read_json(project.qc_dir(r) / "fixes" / "index.json", {}) or {}
        for new_run, e in idx.items():
            any_fix = True
            L.append(f"- `{new_run}` from `{r}`: {e['label']} for {len(e['response_vars'])} variable(s); "
                     f"change `{json.dumps(e['change'])}`")
    if not any_fix:
        L.append("None.")
    g0 = need_grades(project, runs[0])
    L += ["", "## Method", "",
          "Five checks per response variable: " + "; ".join(f"{k} {t}" for k, t in STEP_TITLES.items()) + ".",
          "Calibration is judged on held-out reference subjects only. Relative rules grade a variable against "
          "the other variables of its run (modified Z, Iglewicz and Hoaglin), with absolute floors as a safety net.",
          f"Thresholds file: `{g0['thresholds_file']}`.", "", "```json", json.dumps(g0["thresholds"], indent=1), "```",
          "", "## Files", ""]
    L += [f"- `{name}` sha256 `{h[:16]}...`" for name, h in checksums.items()]
    L += ["- `accepted_models.csv`: accepted variable -> run and model folder", "",
          "## Sign-off ledger", "", "| At | By | ID | Step | Summary |", "|---|---|---|---|---|"]
    L += [f"| {r_['at']} | {r_['by']} | {r_['id']} | {r_['step']} | {r_['summary']} |" for r_ in read_jsonl(project.ledger)]
    text = "\n".join(L) + "\n"
    (out / "qc_report.md").write_text(text, encoding="utf-8")
    for r in runs:
        (project.qc_dir(r) / "qc_report.md").write_text(text, encoding="utf-8")
    say(f"exported {len(accepted)} accepted variables to {out}")
    say(f"signed by {a.by} ({a.id}); ledger row {row['hash'][:16]}")
    say(f"report: {out / 'qc_report.md'}")
    return 0


def cmd_ledger(a, project: Project) -> int:
    ok, msg = ledger_verify(project)
    for r in read_jsonl(project.ledger):
        say(f"{r['at']}  {r['by']} ({r['id']})  {r['step']}: {r['summary']}")
    say(("VERIFIED: " if ok else "TAMPERING SUSPECTED: ") + msg)
    if not ok:
        raise SystemExit(1)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, run=True):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True)
        if run:
            p.add_argument("--run", required=True)
        return p
    add("dashboard")
    add("serve").add_argument("--port", type=int)
    p = add("decide")
    p.add_argument("--response-var", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--fix-id")
    p.add_argument("--note", default="")
    p.add_argument("--by", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--role", default="reviewer")
    add("import").add_argument("--file", required=True)
    add("fix")
    p = add("compare", run=False)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p = add("export", run=False)
    p.add_argument("--runs", required=True, help="comma-separated, original run first, fix runs after")
    p.add_argument("--name", required=True, help="export folder name under <project>/export/")
    p.add_argument("--by", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--confirm", action="store_true")
    p.add_argument("--allow-open", action="store_true")
    add("ledger", run=False).add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    project = Project(a.project)
    with audited(project, f"qc_review.py {a.cmd}", argv):
        return {"dashboard": cmd_dashboard, "serve": cmd_serve, "decide": cmd_decide, "import": cmd_import,
                "fix": cmd_fix, "compare": cmd_compare, "export": cmd_export, "ledger": cmd_ledger}[a.cmd](a, project)


if __name__ == "__main__":
    raise SystemExit(main())
