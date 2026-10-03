#!/usr/bin/env python3
"""Add a subject, response variable or batch level to the project's dismissal list.

    python dismiss.py --project P --kind subjects|response_vars|batch_levels --id ID --reason TEXT --by "Name (ID)"
    python dismiss.py --project P --list

A dismissal deletes nothing. It adds a row to <project>/dismissed.json, which
build_dataset.py and every later stage filter on, so a dismissed item never
returns downstream. For batch levels use the form  column=value  (e.g. site=Tiny).
Dismissing is a human decision: only run this after the user has said so.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DISMISS_KINDS, Project, add_dismissal, audited, die, load_dismissed, say  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--kind", choices=DISMISS_KINDS)
    ap.add_argument("--id", nargs="+")
    ap.add_argument("--reason")
    ap.add_argument("--by")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    project = Project(a.project).ensure()
    with audited(project, "dismiss.py", argv):
        if a.list:
            for kind, items in load_dismissed(project).items():
                say(f"{kind}: {len(items)}")
                for it in items:
                    say(f"  {it['id']}: {it['reason']} ({it['by']}, {it['step']}, {it['at']})")
            return 0
        if not (a.kind and a.id and (a.reason or "").strip() and (a.by or "").strip()):
            die("--kind, --id, --reason and --by are all required", 2)
        for ident in a.id:
            if a.kind == "batch_levels" and "=" not in ident:
                die("batch levels are written column=value, for example site=Tiny", 2)
            added = add_dismissal(project, a.kind, ident, a.reason.strip(), a.by.strip(), "data")
            say(f"{'dismissed' if added else 'already dismissed'}: {a.kind} {ident}")
        say("Rebuild the dataset (build_dataset.py, validate_dataset.py) for this to take effect, "
            "then re-plan any model run under a new run name.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
