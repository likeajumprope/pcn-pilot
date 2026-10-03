#!/usr/bin/env python3
"""Copy the shared modules from common/ into every skill's scripts/ folder.

Each skill must work when installed on its own, so the shared code is vendored
rather than imported across skills. Edit files in common/ only, then run this.
`--check` exits 1 if any copy has drifted (used by the self-test).
"""
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHARED = ["_common.py", "check_status.py", "pcn_bridge.py"]
SKILLS = ["pcn-data-skill", "pcn-model-skill", "pcn-qc-skill"]


def main() -> int:
    check = "--check" in sys.argv
    drift = []
    for skill in SKILLS:
        for name in SHARED:
            src, dst = ROOT / "common" / name, ROOT / "skills" / skill / "scripts" / name
            if check:
                if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
                    drift.append(str(dst.relative_to(ROOT)))
            else:
                shutil.copyfile(src, dst)
    if check and drift:
        print("out of sync:", *drift, sep="\n  ")
        return 1
    print("in sync" if check else f"synced {len(SHARED)} modules into {len(SKILLS)} skills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
