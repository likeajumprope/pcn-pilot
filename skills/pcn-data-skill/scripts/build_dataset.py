#!/usr/bin/env python3
"""Steps 5 and 6 of pcn-data-skill: turn the confirmed spec into the standardized dataset.

    python build_dataset.py --project <project> [--spec spec.json] [--dry-run]

Writes into <project>/data/ :
    clean.csv       every kept observation, one row each
    train.csv       reference-group observations used to fit
    test.csv        held-out reference-group observations (evaluation)
    clinical.csv    non-reference observations (predict only; never used to fit)
    dropped.csv     every observation that did not make it, with the reason
    build_manifest.json   counts, feature list, source checksums, split seed

Guarantees
  * sources are read-only; only declared columns are copied (PHI never is)
  * every source row is either in clean.csv or in dropped.csv (accounted for)
  * a subject never appears in both train and test
  * re-running with the same spec and seed reproduces the same files
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (Project, audited, die, dismissed_ids, modified_z, now, read_json,  # noqa: E402
                     say, write_json)
from profile_table import normalise_ids, read_table  # noqa: E402

ID, GROUP = "subject_id", "pcn_group"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_sources(spec: dict) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    mode = spec.get("id_normalization", "strip")
    visit = spec.get("visit_column")
    frames, meta = [], []
    for src in spec["sources"]:
        df = read_table(src["path"], src.get("sheet"))
        if src["id_column"] not in df.columns:
            die(f"id column '{src['id_column']}' not in {src['path']}", 2)
        df = df.rename(columns={src["id_column"]: ID})
        df[ID] = normalise_ids(df[ID], mode)
        meta.append({"path": str(Path(src["path"]).resolve()), "sha256": sha256(src["path"]),
                     "n_rows": int(len(df)), "n_subjects": int(df[ID].nunique())})
        frames.append(df)
    merged = frames[0]
    dropped = []
    for i, other in enumerate(frames[1:], start=1):
        keys = [ID] + ([visit] if visit and visit in merged.columns and visit in other.columns else [])
        dup = [c for c in other.columns if c in merged.columns and c not in keys]
        if dup:
            say(f"NOTE: columns present in several sources, keeping the first occurrence: {dup[:10]}")
            other = other.drop(columns=dup)
        if other.duplicated(keys).any() and merged.duplicated(keys).any():
            die(f"source {i} and earlier sources both have repeated keys {keys}; set visit_column in the spec", 2)
        both = merged.merge(other, on=keys, how="outer", indicator=True)
        lost = both[both["_merge"] != "both"]
        for _, r in lost.iterrows():
            dropped.append({ID: r[ID], "reason": "not present in every source",
                            "detail": "missing from " + (spec["sources"][i]["path"] if r["_merge"] == "left_only"
                                                         else "an earlier source")})
        merged = both[both["_merge"] == "both"].drop(columns="_merge")
    return merged.reset_index(drop=True), meta, pd.DataFrame(dropped, columns=[ID, "reason", "detail"])


def resolve_response_vars(spec: dict, df: pd.DataFrame, reserved: set[str]) -> list[str]:
    rv = spec["response_vars"]
    if isinstance(rv, list):
        missing = [c for c in rv if c not in df.columns]
        if missing:
            die(f"response variables not found in the sources: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''}", 2)
        return list(rv)
    if isinstance(rv, dict) and "regex" in rv:
        rx = re.compile(rv["regex"])
        cols = [c for c in df.columns if rx.search(c) and c not in reserved]
    elif isinstance(rv, dict) and rv.get("all_numeric"):
        cols = [c for c in df.columns if c not in reserved
                and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5]
    else:
        die("spec.response_vars must be a list, {'regex': ...} or {'all_numeric': true}", 2)
    if not cols:
        die("response_vars selector matched no column", 2)
    return cols


def split_reference(ref: pd.DataFrame, batch: list[str], cfg: dict) -> pd.Series:
    """Subject-grouped split, stratified on the batch-effect combination."""
    frac = float(cfg.get("test_fraction", 0.2))
    seed = int(cfg.get("seed", 42))
    min_train = int(cfg.get("min_train_per_stratum", 5))
    rng = np.random.default_rng(seed)
    first = ref.drop_duplicates(ID).copy()
    first["_stratum"] = first[batch].astype(str).agg("|".join, axis=1) if batch else "all"
    test_ids: set[str] = set()
    for _, grp in sorted(first.groupby("_stratum"), key=lambda kv: kv[0]):
        ids = sorted(grp[ID].tolist())
        rng.shuffle(ids)
        n_test = int(round(frac * len(ids)))
        n_test = max(0, min(n_test, len(ids) - min_train))
        test_ids.update(ids[:n_test])
    return ref[ID].isin(test_ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--spec", help="defaults to <project>/data/spec.json")
    ap.add_argument("--dry-run", action="store_true", help="report what would be written, write nothing")
    a = ap.parse_args(argv)
    project = Project(a.project).ensure()
    with audited(project, "build_dataset.py", argv):
        spec_path = Path(a.spec) if a.spec else project.spec
        spec = read_json(spec_path)
        if spec is None:
            die(f"spec not found: {spec_path}", 2)
        for key in ("sources", "covariates", "response_vars"):
            if key not in spec:
                die(f"spec is missing '{key}'", 2)
        cov = list(spec["covariates"])
        batch = list(spec.get("batch_effects", []))
        visit = spec.get("visit_column")
        ref_cfg = spec.get("reference") or None

        df, src_meta, dropped = load_sources(spec)
        n_merged, n_unmatched = len(df), len(dropped)

        # recode categorical labels (e.g. sex 0/1 -> F/M) before anything else looks at them
        for col, mapping in (spec.get("recode") or {}).items():
            if col not in df.columns:
                die(f"recode column '{col}' not found", 2)
            m = {str(k): v for k, v in mapping.items()}
            as_str = df[col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
            unknown = sorted(set(as_str[df[col].notna()]) - set(m))
            if unknown:
                die(f"recode for '{col}' does not cover values {unknown[:10]}", 2)
            df[col] = as_str.map(m).where(df[col].notna())

        reserved = {ID, *cov, *batch, *([visit] if visit else []), *([ref_cfg["column"]] if ref_cfg else [])}
        missing_cols = [c for c in reserved if c not in df.columns]
        if missing_cols:
            die(f"columns named in the spec are not in the sources: {missing_cols}", 2)
        rvs = resolve_response_vars(spec, df, reserved)

        # --- dismissals recorded by earlier QC rounds are honoured on every rebuild
        dis_rv = dismissed_ids(project, "response_vars")
        dis_sub = dismissed_ids(project, "subjects")
        dis_lvl = dismissed_ids(project, "batch_levels")
        features_dropped = [{"response_var": r, "reason": "dismissed"} for r in rvs if r in dis_rv]
        rvs = [r for r in rvs if r not in dis_rv]

        for c in cov + rvs:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in batch:
            df[c] = df[c].where(df[c].isna(), df[c].astype(str).str.strip())

        # --- features too sparse to keep under a complete-case policy
        miss_cfg = spec.get("missing") or {}
        max_frac = float(miss_cfg.get("max_feature_missing_fraction", 0.2))
        sparse = [r for r in rvs if df[r].isna().mean() > max_frac]
        features_dropped += [{"response_var": r, "reason": f"missing fraction {df[r].isna().mean():.2f} > {max_frac}"}
                             for r in sparse]
        rvs = [r for r in rvs if r not in sparse]
        if not rvs:
            die("no response variable left after dismissals and the sparsity rule")

        keep_cols = [ID] + ([visit] if visit else []) + cov + batch + ([ref_cfg["column"]] if ref_cfg else []) + rvs
        df = df[keep_cols].copy()            # nothing else (PHI included) is carried over

        drops: list[pd.DataFrame] = [dropped]

        def drop(mask: pd.Series, reason: str, detail) -> None:
            nonlocal df
            if mask.any():
                d = df.loc[mask, [ID]].copy()
                d["reason"] = reason
                d["detail"] = detail[mask] if isinstance(detail, pd.Series) else detail
                drops.append(d)
                df = df.loc[~mask].copy()

        drop(df[ID].isin(dis_sub), "dismissed subject", "listed in dismissed.json")
        for lvl in sorted(dis_lvl):
            col, _, val = lvl.partition("=")
            if col in df.columns:
                drop(df[col].astype(str) == val, "dismissed batch level", lvl)

        need = cov + batch + rvs
        na = df[need].isna()
        detail = na.apply(lambda r: ",".join([c for c in need if r[c]][:5]), axis=1)
        drop(na.any(axis=1), "missing value", "missing: " + detail)

        if visit:
            drop(df.duplicated([ID, visit], keep="first"), "duplicate subject+visit row", "kept first")
        else:
            drop(df.duplicated(keep="first"), "exact duplicate row", "kept first")

        # --- outliers: always reported, removed only when the spec says so
        out_cfg = spec.get("outliers") or {}
        thr = float(out_cfg.get("threshold", 5.0))
        mz = pd.DataFrame({r: np.abs(modified_z(df[r].to_numpy())) for r in rvs}, index=df.index)
        flagged = mz.gt(thr)
        outlier_rows = [{ID: df.at[i, ID], "response_var": r, "value": float(df.at[i, r]),
                         "modified_z": round(float(mz.at[i, r]), 2)}
                        for r in rvs for i in flagged.index[flagged[r]]]
        if out_cfg.get("policy", "flag") == "remove_rows":
            rows = flagged.any(axis=1)
            detail = pd.Series("", index=df.index)
            if rows.any():
                detail[rows] = f"modified Z > {thr} in " + mz[rows].idxmax(axis=1).astype(str)
            drop(rows, "outlier", detail)

        # --- reference group
        if ref_cfg:
            vals = {str(v).strip() for v in ref_cfg["values"]}
            is_ref = df[ref_cfg["column"]].astype(str).str.strip().isin(vals)
            unknown_group = df[ref_cfg["column"]].isna()
            drop(unknown_group, "missing value", "missing: " + ref_cfg["column"])
            is_ref = is_ref.loc[df.index]
        else:
            is_ref = pd.Series(True, index=df.index)
        df[GROUP] = np.where(is_ref, "reference", "clinical")
        if not is_ref.any():
            die(f"no observation matches the reference group {ref_cfg}", 2)

        df = df.sort_values([ID] + ([visit] if visit else [])).reset_index(drop=True)
        ref = df[df[GROUP] == "reference"]
        in_test = split_reference(ref, batch, spec.get("split") or {})
        train = ref[~in_test].reset_index(drop=True)
        test = ref[in_test].reset_index(drop=True)
        clinical = df[df[GROUP] == "clinical"].reset_index(drop=True)
        dropped_all = pd.concat(drops, ignore_index=True) if drops else pd.DataFrame(columns=[ID, "reason", "detail"])

        manifest = {
            "built_at": now(), "spec": str(spec_path), "name": spec.get("name", project.root.name),
            "sources": src_meta, "id_column": ID, "visit_column": visit, "covariates": cov,
            "batch_effects": batch, "response_vars": rvs, "features_dropped": features_dropped,
            "n_rows_after_merge": int(n_merged), "n_unmatched_between_sources": int(n_unmatched),
            "n_clean": int(len(df)), "n_dropped": int(len(dropped_all)),
            "n_train": int(len(train)), "n_test": int(len(test)), "n_clinical": int(len(clinical)),
            "n_subjects": {"train": int(train[ID].nunique()), "test": int(test[ID].nunique()),
                           "clinical": int(clinical[ID].nunique())},
            "drop_reasons": dropped_all["reason"].value_counts().to_dict(),
            "outliers": {"threshold": thr, "policy": out_cfg.get("policy", "flag"),
                         "n_flagged_values": len(outlier_rows),
                         "n_flagged_rows": int(flagged.any(axis=1).sum()), "top": sorted(
                             outlier_rows, key=lambda r: -r["modified_z"])[:50]},
            "split": {**(spec.get("split") or {}), "grouped_by": ID, "stratified_by": batch},
        }
        say(f"merged rows: {n_merged}   clean: {len(df)}   dropped: {len(dropped_all)}")
        say(f"train: {len(train)}   test: {len(test)}   clinical (predict only): {len(clinical)}")
        say(f"response variables kept: {len(rvs)}   dropped: {len(features_dropped)}")
        for reason, n in manifest["drop_reasons"].items():
            say(f"  dropped {n:>6}  {reason}")
        if a.dry_run:
            say("dry run: nothing written")
            return 0
        d = project.data
        df.to_csv(d / "clean.csv", index=False)
        train.to_csv(d / "train.csv", index=False)
        test.to_csv(d / "test.csv", index=False)
        if len(clinical):
            clinical.to_csv(d / "clinical.csv", index=False)
        elif (d / "clinical.csv").exists():
            (d / "clinical.csv").rename(d / "clinical.stale.csv")
        dropped_all.to_csv(d / "dropped.csv", index=False)
        write_json(d / "build_manifest.json", manifest)
        if spec_path.resolve() != project.spec.resolve():
            write_json(project.spec, spec)
        say(f"written to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
