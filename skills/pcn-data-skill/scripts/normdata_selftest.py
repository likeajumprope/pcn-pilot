#!/usr/bin/env python3
"""Step 6 of pcn-data-skill: self-test the standardized tables against PCNtoolkit.

Dry-runs the conversion on a small sample before the whole cohort depends on it:
builds NormData from train/test for a few response variables, checks the two are
compatible, and (with --fit-one) fits a minimal BLR on one variable.

    $PCN_PYTHON normdata_selftest.py --project <project> [--n-vars 3] [--fit-one]

Needs the interpreter that has PCNtoolkit (PCN_PYTHON in pipeline.env).
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Project, audited, die, read_json, say  # noqa: E402
import pcn_bridge as B  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--n-vars", type=int, default=3)
    ap.add_argument("--fit-one", action="store_true", help="also fit a minimal BLR on the first variable")
    a = ap.parse_args(argv)
    project = Project(a.project)
    with audited(project, "normdata_selftest.py", argv):
        man = read_json(project.data / "build_manifest.json")
        if man is None:
            die("build_manifest.json missing: run build_dataset.py first", 2)
        B.import_pcn()
        B.quiet()
        rvs = man["response_vars"][: max(1, a.n_vars)]
        train = B.make_normdata("train", B.read_split(project.data / "train.csv", man), man, rvs)
        test = B.make_normdata("test", B.read_split(project.data / "test.csv", man), man, rvs)
        say(f"NormData built for {len(rvs)} response variables: {rvs}")
        if hasattr(train, "check_compatibility"):
            ok = train.check_compatibility(test)
            say(f"train/test compatibility: {ok}")
            if ok is False:
                die("PCNtoolkit reports train and test as incompatible (covariates or batch-effect dimensions differ)")
        if a.fit_one:
            recipe = {"algorithm": "blr", "basis": {"type": "linear"}, "blr": {}}
            with tempfile.TemporaryDirectory() as tmp:
                model = B.make_model(recipe, tmp)
                model.fit_predict(B.select(train, rvs[:1]), B.select(test, rvs[:1]))
                done = B.feature_done(tmp, rvs[0])
                say(f"minimal BLR on '{rvs[0]}': model + z-scores written = {done}")
                if not done:
                    die("PCNtoolkit ran but did not write the expected model/ and results/ files; "
                        "the installed version may use a different layout than PCN-Pilot expects")
        say("self-test OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
