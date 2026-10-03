"""Decision store for pcn-qc-skill: one append-only file per run, shared by the dashboard and the CLI.

A decision is always a named person's. The rules enforced here (not in the UI, so they
cannot be bypassed by posting to the server directly):

  * reviewer name and institutional ID are mandatory
  * pass on a variable the agent graded FAIL is an override and needs a written reason
  * accept_warn, dismiss and escalate need a written reason
  * fix must name one of the candidates that qc_detect.py proposed
  * an escalated variable can only be closed by a supervisor who is not the person who escalated it
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Project, add_dismissal, append_jsonl, now, read_json, read_jsonl  # noqa: E402

DECISIONS = ("pass", "accept_warn", "fix", "dismiss", "escalate")
ACCEPTING = ("pass", "accept_warn")


class DecisionError(ValueError):
    pass


def decisions_path(project: Project, run: str) -> Path:
    return project.qc_dir(run) / "decisions.jsonl"


def latest_decisions(project: Project, run: str) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in read_jsonl(decisions_path(project, run)):
        latest[row["response_var"]] = row
    return latest


def record_decision(project: Project, run: str, row: dict, source: str = "cli") -> dict:
    grades = read_json(project.qc_dir(run) / "grades.json")
    if grades is None:
        raise DecisionError(f"run '{run}' has not been graded (no grades.json)")
    feats = {f["response_var"]: f for f in grades["features"]}
    rv = str(row.get("response_var", ""))
    decision = str(row.get("decision", "")).lower()
    by, rid = str(row.get("by", "")).strip(), str(row.get("id", "")).strip()
    note = str(row.get("note", "") or "").strip()
    role = str(row.get("role", "reviewer") or "reviewer").lower()
    fix_id = row.get("fix_id")
    if rv not in feats:
        raise DecisionError(f"unknown response variable '{rv}'")
    if decision not in DECISIONS:
        raise DecisionError(f"decision must be one of {DECISIONS}")
    if not by or not rid:
        raise DecisionError("reviewer name and institutional ID are both required")
    if role not in ("reviewer", "supervisor"):
        raise DecisionError("role must be reviewer or supervisor")
    feat = feats[rv]
    if decision == "pass" and feat["grade"] == "FAIL" and not note:
        raise DecisionError("passing a variable the agent graded FAIL is an override: a written reason is required")
    if decision in ("accept_warn", "dismiss", "escalate") and not note:
        raise DecisionError(f"'{decision}' needs a written reason")
    fix = None
    if decision == "fix":
        fix = next((c for c in feat["fix_candidates"] if c["id"] == fix_id), None)
        if fix is None:
            raise DecisionError(f"fix needs fix_id, one of {[c['id'] for c in feat['fix_candidates']]}")
    prev = latest_decisions(project, run).get(rv)
    if prev and prev["decision"] == "escalate" and decision != "escalate":
        if role != "supervisor":
            raise DecisionError("this variable was escalated: only a supervisor can close it")
        if rid == prev["id"]:
            raise DecisionError("the supervisor closing an escalation must not be the person who escalated it")
    out = {"run": run, "response_var": rv, "decision": decision, "fix_id": fix_id if fix else None,
           "fix_label": fix["label"] if fix else None, "note": note, "by": by, "id": rid, "role": role,
           "agent_grade": feat["grade"], "override": decision == "pass" and feat["grade"] == "FAIL",
           "at": now(), "source": source}
    append_jsonl(decisions_path(project, run), out)
    if decision == "dismiss":
        add_dismissal(project, "response_vars", rv, note, f"{by} ({rid})", f"qc:{run}")
    return out
