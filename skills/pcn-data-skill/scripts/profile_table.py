#!/usr/bin/env python3
"""Step 1-3 of pcn-data-skill: inspect tables and propose a role for every column.

Reads one or more tabular files (csv / tsv / txt / xlsx / parquet, FreeSurfer
*stats2table output included), and writes a JSON profile with a proposed role
per column:

    id | covariate | batch_effect | reference_group | response_var | phi | ignore

The proposal is a starting point for the agent and the human. Nothing is
converted here and the sources are only read.

    python profile_table.py --input demo.csv idps.tsv --out <project>/data/profile.json
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Project, audited, die, say, write_json  # noqa: E402

RX = {
    "id": re.compile(r"^(participant[_ ]?id|subject[_ ]?id|subj(ect)?|sub[_ ]?id|subjid|eid|ptid|rid|src_subject_id|"
                     r"subjectkey|id|bids[_ ]?id|measure:.*|[lr]h\.aparc.*|.*\.aseg.*)$", re.I),
    "visit": re.compile(r"^(visit|session|ses|timepoint|tp|wave|eventname|viscode\d*)$", re.I),
    "phi": re.compile(r"(^|_)(name|first[_ ]?name|last[_ ]?name|surname|dob|birth[_ ]?date|date[_ ]?of[_ ]?birth|"
                      r"address|street|phone|email|ssn|mrn|nhs|postcode|zip([_ ]?code)?|insurance)($|_)", re.I),
    "age": re.compile(r"^(age|age[_ ]?(years|yrs|y|at[_ ]?scan|at[_ ]?visit|scan|mri)|interview[_ ]?age|"
                      r"age[_ ]?in[_ ]?(years|months|days)|age[_ ]?(months|days))$", re.I),
    "sex": re.compile(r"^(sex|gender|sex[_ ]?at[_ ]?birth|ptgender|demo[_ ]?sex.*)$", re.I),
    "site": re.compile(r"^(site|site[_ ]?id|scanner|scanner[_ ]?id|center|centre|cohort|study|dataset|protocol|"
                       r"vendor|manufacturer|field[_ ]?strength|imaging[_ ]?site)$", re.I),
    "group": re.compile(r"^(dx|diagnosis|diag|group|status|clinical[_ ]?status|case[_ ]?control|patient|"
                        r"control|dx[_ ]?bl|research[_ ]?group|cohort[_ ]?group|hc)$", re.I),
    "cov_candidate": re.compile(r"^(etiv|icv|tiv|e?tiv[_ ]?vol.*|euler.*|education|edu.*|bmi|handedness|"
                                r"mean[_ ]?fd|fd[_ ]?mean|snr|cnr)$", re.I),
    "control_value": re.compile(r"^(hc|cn|ctl|ctrl|control|controls|healthy|healthy[_ ]?control|td|nc|normal|0)$", re.I),
}


def read_table(path: str | Path, sheet: str | int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        die(f"input not found: {p}", 2)
    suf = "".join(p.suffixes[-2:]).lower()
    if suf.endswith((".parquet", ".pq")):
        return pd.read_parquet(p)
    if suf.endswith((".xlsx", ".xls")):
        return pd.read_excel(p, sheet_name=sheet if sheet is not None else 0)
    # delimiter sniffing; engine="python" handles ragged whitespace-separated stats tables
    compression = "infer"
    try:
        df = pd.read_csv(p, sep=None, engine="python", compression=compression)
    except Exception as e:  # noqa: BLE001
        die(f"could not parse {p}: {e}")
    if df.shape[1] == 1:
        df = pd.read_csv(p, sep=r"\s+", engine="python", compression=compression)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalise_ids(s: pd.Series, mode: str) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)          # ints read as floats
    if mode in ("none", "strip"):
        return out
    if mode.startswith("strip_prefix:"):
        return out.str.replace("^" + re.escape(mode.split(":", 1)[1]), "", regex=True)
    if mode == "alnum":
        return out.str.replace(r"[^0-9A-Za-z]", "", regex=True)
    if mode == "alnum_lower":
        return out.str.replace(r"[^0-9A-Za-z]", "", regex=True).str.lower()
    die(f"unknown id_normalization '{mode}' (none|strip|alnum|alnum_lower|strip_prefix:<p>)", 2)


def looks_like_date(s: pd.Series) -> bool:
    if s.dtype.kind in "biuf":
        return False
    sample = s.dropna().astype(str).head(50)
    if sample.empty:
        return False
    hit = sample.str.match(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}").mean()
    return bool(hit > 0.8)


def profile_column(name: str, s: pd.Series, n: int) -> dict:
    nonnull = s.dropna()
    n_unique = int(nonnull.nunique())
    numeric = pd.to_numeric(nonnull, errors="coerce")
    frac_numeric = float(numeric.notna().mean()) if len(nonnull) else 0.0
    is_numeric = frac_numeric > 0.98 and len(nonnull) > 0
    info = {
        "column": name, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()),
        "missing_fraction": round(float(s.isna().mean()), 4), "n_unique": n_unique,
        "examples": [str(v) for v in nonnull.unique()[:5]],
    }
    if is_numeric:
        v = numeric.dropna().astype(float)
        info.update(min=float(v.min()), max=float(v.max()), median=float(v.median()))
        integer_like = bool(np.allclose(v, np.round(v)))
    else:
        integer_like = False

    role, conf, why = "ignore", 0.3, "no rule matched"
    if RX["phi"].search(name) or looks_like_date(s):
        role, conf, why = "phi", 0.9, "name/date pattern suggests identifying information"
    elif RX["id"].match(name) and n_unique >= 0.5 * max(len(nonnull), 1):
        role, conf, why = "id", 0.9, "name looks like a subject ID and values are mostly unique"
    elif RX["visit"].match(name):
        role, conf, why = "visit", 0.8, "visit / session label"
    elif RX["age"].match(name) and is_numeric:
        role, conf, why = "covariate", 0.95, "age"
        if info["max"] > 130:
            why += " (max > 130: probably months or days, convert to years)"
            info["unit_warning"] = True
    elif RX["sex"].match(name):
        role, conf, why = "batch_effect", 0.9, "sex"
        if is_numeric:
            why += " (numeric coding: recode to labels such as F/M)"
            info["needs_recode"] = True
    elif RX["site"].match(name):
        role, conf, why = "batch_effect", 0.85, "site / scanner"
    elif RX["group"].match(name) and n_unique <= 20:
        role, conf, why = "reference_group", 0.8, "diagnosis / group column"
        info["control_like_values"] = [str(v) for v in nonnull.unique() if RX["control_value"].match(str(v).strip())]
    elif RX["cov_candidate"].match(name) and is_numeric:
        role, conf, why = "covariate_candidate", 0.5, "plausible extra covariate; confirm with the user"
    elif n_unique <= 1:
        role, conf, why = "ignore", 0.95, "constant column"
    elif info["missing_fraction"] > 0.5:
        role, conf, why = "ignore", 0.7, "more than half missing"
    elif is_numeric and n_unique > 20 and not (integer_like and n_unique >= 0.98 * len(nonnull)):
        role, conf, why = "response_var", 0.7, "continuous numeric measure"
    elif is_numeric and integer_like and n_unique >= 0.98 * len(nonnull):
        role, conf, why = "id", 0.5, "unique integers: an ID or row counter"
    elif (not is_numeric) and n_unique >= 0.9 * max(len(nonnull), 1):
        role, conf, why = "id", 0.4, "nearly unique strings"
    elif n_unique <= 50:
        role, conf, why = "batch_effect_candidate", 0.4, "low-cardinality categorical; confirm with the user"
    info.update(role=role, confidence=conf, reason=why, numeric=bool(is_numeric))
    return info


def profile_file(path: str, sheet=None) -> dict:
    df = read_table(path, sheet)
    cols = [profile_column(c, df[c], len(df)) for c in df.columns]
    ids = [c for c in cols if c["role"] == "id"]
    ids.sort(key=lambda c: -c["confidence"])
    out = {"path": str(Path(path).resolve()), "n_rows": int(len(df)), "n_columns": int(df.shape[1]),
           "id_column_guess": ids[0]["column"] if ids else None, "columns": cols}
    if ids:
        raw = df[ids[0]["column"]]
        dup = int(raw.astype(str).duplicated().sum())
        out["duplicate_ids"] = dup
        out["id_normalisation_collisions"] = {
            mode: int(normalise_ids(raw, mode).nunique() < raw.astype(str).nunique())
            for mode in ("strip", "alnum", "alnum_lower")}
    return out


def overlap(profiles: list[dict]) -> list[dict]:
    """Pairwise ID overlap between sources under each normalisation mode."""
    res = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a, b = profiles[i], profiles[j]
            if not a["id_column_guess"] or not b["id_column_guess"]:
                continue
            da, db = read_table(a["path"]), read_table(b["path"])
            row = {"a": a["path"], "b": b["path"]}
            for mode in ("strip", "alnum", "alnum_lower"):
                sa = set(normalise_ids(da[a["id_column_guess"]], mode))
                sb = set(normalise_ids(db[b["id_column_guess"]], mode))
                row[mode] = {"matched": len(sa & sb), "only_a": len(sa - sb), "only_b": len(sb - sa)}
            res.append(row)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="one or more tables")
    ap.add_argument("--out", required=True, help="profile JSON to write")
    ap.add_argument("--project", help="project dir (for the audit log)")
    a = ap.parse_args(argv)
    project = Project(a.project).ensure() if a.project else None
    with audited(project, "profile_table.py", argv):
        profiles = [profile_file(p) for p in a.input]
        result = {"sources": profiles, "id_overlap": overlap(profiles) if len(profiles) > 1 else []}
        write_json(a.out, result)
        for p in profiles:
            say(f"\n{p['path']}: {p['n_rows']} rows x {p['n_columns']} columns, id guess: {p['id_column_guess']}")
            counts: dict[str, list[str]] = {}
            for c in p["columns"]:
                counts.setdefault(c["role"], []).append(c["column"])
            for role, names in sorted(counts.items()):
                shown = ", ".join(names[:8]) + (f", ... (+{len(names) - 8})" if len(names) > 8 else "")
                say(f"  {role:<22} {len(names):>5}  {shown}")
            for c in p["columns"]:
                if c.get("unit_warning") or c.get("needs_recode"):
                    say(f"  NOTE {c['column']}: {c['reason']}")
        for o in result["id_overlap"]:
            say(f"\nID overlap {Path(o['a']).name} <-> {Path(o['b']).name}: " +
                "; ".join(f"{m}: {o[m]['matched']} matched, {o[m]['only_a']}+{o[m]['only_b']} unmatched"
                          for m in ("strip", "alnum", "alnum_lower")))
        say(f"\nprofile written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
