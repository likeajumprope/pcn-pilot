#!/usr/bin/env python3
"""Step 7 of pcn-data-skill: validate the standardized dataset (checks V01 to V14) and write the report.

    python validate_dataset.py --project <project>

Writes <project>/data/validation.json and <project>/data/data_report.md.
Exit code 0 = no FAIL, 1 = at least one FAIL (the model stage refuses to start).

The report is a required closing artifact: a build without it is incomplete.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Project, audited, die, now, read_json, say, worst, write_json  # noqa: E402
from profile_table import RX  # noqa: E402
import pcn_bridge as B  # noqa: E402  (string rules only here; PCNtoolkit itself is not imported)

ID, GROUP = "subject_id", "pcn_group"


class Checks:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, cid: str, title: str, status: str, detail: str, items: list | None = None) -> None:
        self.rows.append({"id": cid, "title": title, "status": status, "detail": detail,
                          "items": (items or [])[:200]})


def run_checks(project: Project) -> dict:
    d = project.data
    spec = read_json(project.spec)
    man = read_json(d / "build_manifest.json")
    if spec is None or man is None:
        die("spec.json or build_manifest.json missing: run build_dataset.py first", 2)
    cov, batch, rvs = man["covariates"], man["batch_effects"], man["response_vars"]
    str_cols = {c: str for c in [ID] + batch}
    train = pd.read_csv(d / "train.csv", dtype=str_cols)
    test = pd.read_csv(d / "test.csv", dtype=str_cols)
    clean = pd.read_csv(d / "clean.csv", dtype=str_cols)
    clinical = pd.read_csv(d / "clinical.csv", dtype=str_cols) if (d / "clinical.csv").exists() else clean.iloc[0:0]
    min_n = int(spec.get("min_n_per_batch_level", 10))
    c = Checks()

    # V01 completeness -------------------------------------------------------
    need = [ID] + cov + batch + rvs
    absent = [x for x in need if x not in clean.columns]
    n_na = int(clean[[x for x in need if x in clean.columns]].isna().sum().sum())
    c.add("V01", "Required columns present and complete",
          "FAIL" if absent or n_na else "PASS",
          f"{len(absent)} columns absent, {n_na} missing cells in covariates / batch effects / response variables",
          absent)

    # V02 response variables numeric, finite, non-constant --------------------
    bad = []
    for r in rvs:
        v = pd.to_numeric(clean[r], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).all():
            bad.append(f"{r}: non-finite values")
        elif np.nanstd(v) == 0:
            bad.append(f"{r}: zero variance")
        elif len(np.unique(v)) < 10:
            bad.append(f"{r}: only {len(np.unique(v))} distinct values (not continuous)")
    hard = [b for b in bad if "distinct" not in b]
    c.add("V02", "Response variables are numeric, finite and vary",
          "FAIL" if hard else ("WARN" if bad else "PASS"), f"{len(bad)} of {len(rvs)} response variables flagged", bad)

    # V03 subject leakage and repeated measures -------------------------------
    leak = sorted(set(train[ID]) & set(test[ID]))
    rep = int(clean[ID].duplicated().sum())
    status = "FAIL" if leak else ("WARN" if rep else "PASS")
    c.add("V03", "No subject is in both train and test",
          status, f"{len(leak)} subjects leak across the split; {rep} repeated-measure rows"
          + (" (models assume independent observations: confirm the longitudinal design is intended)" if rep else ""),
          leak)

    # V04 batch-effect levels --------------------------------------------------
    items, status = [], "PASS"
    for b in batch:
        tr = train[b].value_counts()
        for lvl in sorted(set(test[b]) | set(clinical[b]) if len(clinical) else set(test[b])):
            if lvl not in tr.index:
                items.append(f"{b}={lvl}: in test/clinical but absent from train (cannot be predicted)")
                status = "FAIL"
        for lvl, n in tr.items():
            if n < min_n:
                items.append(f"{b}={lvl}: only {n} training observations (< {min_n})")
                status = worst([status, "WARN"])
    c.add("V04", "Every batch-effect level is represented in train", status,
          f"{len(items)} level issues across {batch}", items)

    # V05 covariate coverage ---------------------------------------------------
    items, status = [], "PASS"
    for x in cov:
        lo, hi = train[x].min(), train[x].max()
        for name, part in (("test", test), ("clinical", clinical)):
            if len(part):
                n_out = int(((part[x] < lo) | (part[x] > hi)).sum())
                if n_out:
                    items.append(f"{x}: {n_out} {name} observations outside the training range [{lo:g}, {hi:g}] "
                                 f"(extrapolation)")
                    status = "WARN"
        span = hi - lo
        for b in batch:
            for lvl, grp in train.groupby(b):
                if len(grp) >= min_n and span > 0 and (grp[x].max() - grp[x].min()) < 0.2 * span:
                    items.append(f"{x}: {b}={lvl} covers only [{grp[x].min():g}, {grp[x].max():g}] "
                                 f"({100 * (grp[x].max() - grp[x].min()) / span:.0f}% of the range): "
                                 f"its batch effect is confounded with {x}")
                    status = "WARN"
    c.add("V05", "Covariate ranges overlap across splits and batch levels", status,
          f"{len(items)} coverage issues", items)

    # V06 batch-effect confounding / label hygiene ----------------------------
    items, status = [], "PASS"
    for b in batch:
        labels = pd.Series(clean[b].unique()).astype(str)
        folded = labels.str.strip().str.lower()
        for key in folded[folded.duplicated()].unique():
            items.append(f"{b}: labels {sorted(labels[folded == key])} differ only by case/whitespace")
            status = "FAIL"
    if len(batch) >= 2:
        for i, b1 in enumerate(batch):
            for b2 in batch[i + 1:]:
                tab = pd.crosstab(train[b1], train[b2])
                # report per level of the higher-cardinality factor (usually site)
                if tab.shape[0] < tab.shape[1]:
                    tab, b1_, b2_ = tab.T, b2, b1
                else:
                    b1_, b2_ = b1, b2
                for lvl, row in tab.iterrows():
                    if (row > 0).sum() == 1 and tab.shape[1] > 1:
                        items.append(f"{b1_}={lvl} contains a single {b2_} level: the two effects are inseparable there")
                        status = worst([status, "WARN"])
    c.add("V06", "Batch-effect labels are clean and not fully confounded", status, f"{len(items)} issues", items)

    # V07 outliers ---------------------------------------------------------------
    out = man.get("outliers", {})
    n_flag = out.get("n_flagged_rows", 0)
    frac = n_flag / max(man["n_clean"], 1)
    c.add("V07", "Extreme values in the response variables",
          "WARN" if n_flag else "PASS",
          f"{out.get('n_flagged_values', 0)} values in {n_flag} rows exceed modified Z {out.get('threshold')} "
          f"({100 * frac:.1f}% of rows); policy = {out.get('policy')}. Flagged values are listed, never silently removed.",
          [f"{r[ID]} {r['response_var']} = {r['value']:g} (M = {r['modified_z']})" for r in out.get("top", [])])

    # V08 drop-check: nothing lost silently -----------------------------------
    dropped = pd.read_csv(d / "dropped.csv") if (d / "dropped.csv").exists() else pd.DataFrame()
    after_merge_drops = len(dropped) - man.get("n_unmatched_between_sources", 0)
    accounted = man["n_rows_after_merge"] == man["n_clean"] + after_merge_drops
    parts_ok = man["n_clean"] == len(train) + len(test) + len(clinical)
    declared = spec["response_vars"] if isinstance(spec["response_vars"], list) else None
    silently_lost = []
    if declared is not None:
        known = set(rvs) | {f["response_var"] for f in man["features_dropped"]}
        silently_lost = [r for r in declared if r not in known]
    cols_ok = all(r in train.columns and r in test.columns for r in rvs)
    status = "PASS" if (accounted and parts_ok and cols_ok and not silently_lost) else "FAIL"
    if status == "PASS" and man["features_dropped"]:
        status = "WARN"
    c.add("V08", "Drop-check: every row and every requested variable is accounted for", status,
          f"rows after merge {man['n_rows_after_merge']} = clean {man['n_clean']} + dropped {after_merge_drops}: "
          f"{'yes' if accounted else 'NO'}; clean = train + test + clinical: {'yes' if parts_ok else 'NO'}; "
          f"{len(man['features_dropped'])} response variables dropped with a recorded reason, "
          f"{len(silently_lost)} lost without one",
          [f"{f['response_var']}: {f['reason']}" for f in man["features_dropped"]] + silently_lost)

    # V09 PHI --------------------------------------------------------------------
    phi = [x for x in clean.columns if RX["phi"].search(x)]
    c.add("V09", "No identifying columns in the standardized tables", "FAIL" if phi else "PASS",
          f"{len(phi)} columns look like PHI", phi)

    # V10 units / plausibility -----------------------------------------------
    items = []
    for x in cov:
        if RX["age"].match(x):
            if clean[x].max() > 130:
                items.append(f"{x}: max {clean[x].max():g} > 130. Months or days? Normative models expect one consistent unit")
            if clean[x].min() < 0:
                items.append(f"{x}: negative values")
    neg = [r for r in rvs if (clean[r] < 0).any()]
    if neg:
        items.append(f"{len(neg)} response variables contain negative values (fine for z-scored or signed measures, "
                     f"wrong for volumes / thickness / areas): {neg[:8]}")
    c.add("V10", "Units and value ranges are plausible", "WARN" if items else "PASS", f"{len(items)} notes", items)

    # V11 sample size --------------------------------------------------------------
    n_tr, n_te = len(train), len(test)
    status = "FAIL" if n_tr < 30 or n_te < 1 else ("WARN" if n_tr < 200 or n_te < 50 else "PASS")
    c.add("V11", "Enough reference data to fit and to evaluate", status,
          f"train n = {n_tr}, test n = {n_te}. Below ~200 training observations centile tails are unstable; "
          f"below ~50 test observations calibration metrics are noisy.")

    # V12 reference group purity ---------------------------------------------
    impure = int((train[GROUP] != "reference").sum() + (test[GROUP] != "reference").sum())
    ref = spec.get("reference")
    c.add("V12", "Only the reference group is used to fit and evaluate",
          "FAIL" if impure else ("WARN" if not ref else "PASS"),
          f"{impure} non-reference rows in train/test; {len(clinical)} clinical rows held out for prediction only"
          + ("" if ref else ". No reference group was declared: every subject is treated as reference. "
                          "Confirm the table holds only the population the norm should describe."))

    # V13 split balance --------------------------------------------------------
    items = []
    for x in cov:
        a, b_ = train[x].to_numpy(float), test[x].to_numpy(float)
        if len(a) > 5 and len(b_) > 5:
            pooled = np.sqrt((a.var() + b_.var()) / 2) or 1.0
            dlt = abs(a.mean() - b_.mean()) / pooled
            if dlt > 0.3:
                items.append(f"{x}: train/test means differ by {dlt:.2f} SD")
    c.add("V13", "Train and test are comparable on the covariates", "WARN" if items else "PASS",
          f"{len(items)} covariates imbalanced", items)

    # V14 names PCNtoolkit can store ---------------------------------------------
    bad_names = B.unstorable_names(rvs)
    c.add("V14", "Response-variable names survive PCNtoolkit's result files", "FAIL" if bad_names else "PASS",
          f"{len(bad_names)} of {len(rvs)} names cannot be stored", [f"{n}: {why}" for n, why in bad_names.items()])

    overall = worst(r["status"] for r in c.rows)
    return {"validated_at": now(), "overall": overall, "checks": c.rows,
            "summary": {"n_train": n_tr, "n_test": n_te, "n_clinical": int(len(clinical)),
                        "n_response_vars": len(rvs), "covariates": cov, "batch_effects": batch,
                        "batch_levels": {b: train[b].value_counts().to_dict() for b in batch},
                        "covariate_ranges": {x: [float(train[x].min()), float(train[x].max())] for x in cov}}}


def write_report(project: Project, v: dict) -> Path:
    man = read_json(project.data / "build_manifest.json")
    s = v["summary"]
    L = [f"# Data report: {man['name']}", "",
         f"Validated {v['validated_at']}. Overall: **{v['overall']}**", "",
         "## Dataset", "",
         f"- Train {s['n_train']}, test {s['n_test']}, clinical (predict only) {s['n_clinical']} observations",
         f"- {s['n_response_vars']} response variables; covariates: {', '.join(s['covariates'])}; "
         f"batch effects: {', '.join(s['batch_effects']) or 'none'}",
         f"- Rows after merging sources: {man['n_rows_after_merge']}; kept {man['n_clean']}; dropped {man['n_dropped']}",
         f"- Split: seed {man['split'].get('seed', 42)}, test fraction {man['split'].get('test_fraction', 0.2)}, "
         f"grouped by subject, stratified by {', '.join(man['split']['stratified_by']) or 'nothing'}", ""]
    for x, (lo, hi) in s["covariate_ranges"].items():
        L.append(f"- Training range of `{x}`: {lo:g} to {hi:g}")
    L += ["", "## Sources", ""]
    for src in man["sources"]:
        L.append(f"- `{src['path']}`: {src['n_rows']} rows, {src['n_subjects']} subjects, sha256 `{src['sha256'][:16]}...`")
    L += ["", "## Dropped observations", ""]
    if man["drop_reasons"]:
        L += [f"- {n} x {reason}" for reason, n in man["drop_reasons"].items()]
        L.append("\nFull list: `dropped.csv`.")
    else:
        L.append("None.")
    L += ["", "## Dropped response variables", ""]
    L += [f"- `{f['response_var']}`: {f['reason']}" for f in man["features_dropped"]] or ["None."]
    L += ["", "## Checks", "", "| ID | Check | Status | Detail |", "|---|---|---|---|"]
    for r in v["checks"]:
        L.append(f"| {r['id']} | {r['title']} | {r['status']} | {r['detail'].replace('|', '/')} |")
    for r in v["checks"]:
        if r["items"] and r["status"] != "PASS":
            L += ["", f"### {r['id']} details", ""] + [f"- {it}" for it in r["items"][:40]]
            if len(r["items"]) > 40:
                L.append(f"- ... {len(r['items']) - 40} more in validation.json")
    L += ["", "## Batch-effect levels (training n)", ""]
    for b, counts in s["batch_levels"].items():
        L.append(f"**{b}**: " + ", ".join(f"{k} ({n})" for k, n in sorted(counts.items(), key=lambda kv: -kv[1])))
        L.append("")
    path = project.data / "data_report.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    a = ap.parse_args(argv)
    project = Project(a.project).ensure()
    with audited(project, "validate_dataset.py", argv):
        v = run_checks(project)
        write_json(project.validation, v)
        report = write_report(project, v)
        for r in v["checks"]:
            say(f"{r['status']:<5} {r['id']} {r['title']}: {r['detail']}")
        say(f"\noverall: {v['overall']}\nreport: {report}")
        if v["overall"] == "FAIL":
            raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
