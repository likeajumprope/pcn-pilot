#!/usr/bin/env python3
"""DETECT half of pcn-qc-skill: grade every response variable of a run in five steps.

    python qc_detect.py --project P --run R [--plots flagged|all|none] [--thresholds file.json]

    S1  completion and convergence     did it fit, are the z-scores finite, did MCMC converge
    S2  calibration                    held-out reference z-scores should look like N(0, 1)
    S3  fit relative to the cohort     outliers among this run's variables + absolute floors
    S4  batch effects                  no site / sex level left with shifted or rescaled z-scores
    S5  covariate structure, centiles  no trend or spread left along the covariate, centiles ordered

Writes <project>/qc/<run>/ metrics.csv, grades.json and plots/. It reads results and
never changes them: detection and fixing stay in separate tools.

A check that cannot be computed is reported as SKIPPED with the reason. SKIPPED is not PASS.
Needs pandas / numpy / scipy / matplotlib (arviz only for HBR convergence).
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (Project, audited, die, modified_z, now, read_json, safe_name, say, worst,  # noqa: E402
                     write_json)

ID = "subject_id"
STEP_TITLES = {"S1": "Completion and convergence", "S2": "Calibration of z-scores",
               "S3": "Fit relative to the cohort", "S4": "Batch effects",
               "S5": "Covariate structure and centiles"}


# --------------------------------------------------------------------------
# reading PCNtoolkit result files
# --------------------------------------------------------------------------
def read_z(rd: Path, name: str) -> pd.DataFrame | None:
    p = rd / "results" / f"Z_{name}.csv"
    if not p.is_file():
        return None
    z = pd.read_csv(p, dtype={"subject_ids": str})       # IDs such as 000123 must not become the number 123
    if "observations" in z.columns:
        z = z.sort_values("observations", key=lambda c: pd.to_numeric(c, errors="coerce")).reset_index(drop=True)
    return z


def read_stats(rd: Path, name: str) -> pd.DataFrame | None:
    """statistics_<name>.csv -> DataFrame indexed by response variable, one column per statistic."""
    p = rd / "results" / f"statistics_{name}.csv"
    if not p.is_file():
        return None
    s = pd.read_csv(p, index_col=0)
    known = {"EXPV", "MSLL", "SMSE", "RMSE", "Rho", "Rho_p", "R2", "MACE", "MAPE", "MLL", "ShapiroW",
             "Skewness", "Kurtosis"}                 # the rows PCNtoolkit 1.3.0 writes
    if len(known & set(map(str, s.index))) >= len(known & set(map(str, s.columns))):
        s = s.T                                   # rows were statistics (PCNtoolkit's layout)
    return s.apply(pd.to_numeric, errors="coerce")


def align(z: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    """Attach covariates / batch effects to z-score rows. Returns (table, how)."""
    if "subject_ids" not in z.columns:
        return (split, "by row order (no subject_ids column)") if len(z) == len(split) else (None, "row counts differ")
    zi = z["subject_ids"].astype(str).str.replace(r"\.0$", "", regex=True).to_numpy()
    si = split[ID].astype(str).to_numpy()
    if len(zi) == len(si) and (zi == si).all():
        return split.reset_index(drop=True), "by row order (subject IDs verified)"
    if len(set(zi)) == len(zi) and len(set(si)) == len(si) and set(zi) <= set(si):
        return split.set_index(split[ID].astype(str)).loc[zi].reset_index(drop=True), "by subject ID"
    return None, "z-score rows could not be matched to the split table"


def centile_checks(rd: Path, name: str, rv: str, cache: dict) -> tuple[float, float] | None:
    """(share of subjects whose centile curves cross, share of centile values that are not finite)."""
    if "df" not in cache:
        p = rd / "results" / f"centiles_{name}.csv"
        cache["df"] = pd.read_csv(p) if p.is_file() else None
    df = cache["df"]
    if df is None or rv not in df.columns or not {"observations", "centile"} <= set(df.columns):
        return None
    try:
        wide = df.pivot(index="observations", columns="centile", values=rv)
        wide = wide[sorted(wide.columns, key=float)]
        v = wide.to_numpy(float)
        if v.shape[1] < 2:
            return None
        bad = ~np.isfinite(v)
        fin = np.where(bad, np.nan, v)
        tol = 1e-9 * (np.nanstd(fin) or 1.0) if (~bad).any() else 0.0
        with np.errstate(invalid="ignore"):
            cross = (np.diff(fin, axis=1) < -tol).any(axis=1)
        return float(cross.mean()), float(bad.mean())
    except Exception:
        return None


def hbr_convergence(rd: Path, rv: str) -> dict | None:
    p = rd / "model" / rv / "idata.nc"
    if not p.is_file():
        return None
    try:
        import warnings
        import arviz as az
        idata = az.from_netcdf(str(p))
        # Zero-size variables (a zero-sum offset over a single batch level, as in a model transferred to
        # one site) and constants (R-hat undefined) are not sampled quantities: leave them out.
        names = [v for v in idata.posterior.data_vars if "per_subject" not in v and idata.posterior[v].size > 0]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            rhat = az.rhat(idata, var_names=names)
            ess = az.ess(idata, var_names=names)
        r_all, e_all = [], []
        for v in names:
            r, e = np.asarray(rhat[v].values, float).ravel(), np.asarray(ess[v].values, float).ravel()
            keep = np.isfinite(r) & np.isfinite(e)
            r_all.append(r[keep])
            e_all.append(e[keep])
        r_all, e_all = np.concatenate(r_all or [np.array([])]), np.concatenate(e_all or [np.array([])])
        if not len(r_all):
            return {"error": "no sampled parameter with a finite R-hat in idata.nc"}
        out = {"rhat_max": float(r_all.max()), "ess_min": float(e_all.min())}
        # PCNtoolkit 1.3.0 saves only the posterior group, so sampler diagnostics are normally absent
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
            dv = idata.sample_stats["diverging"]
            out["divergent_frac"] = float(dv.mean())
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            got = find_key(v, key)
            if got is not None:
                return got
    return None


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------
class Step:
    def __init__(self):
        self.flags: list[dict] = []
        self.skipped: list[str] = []
        self.ran = False

    def flag(self, code: str, severity: str, message: str, value=None) -> None:
        self.flags.append({"code": code, "severity": severity, "message": message, "value": value})

    def result(self) -> dict:
        if self.flags:
            status = worst(f["severity"] for f in self.flags)
        elif self.ran:
            status = "PASS"
        else:
            status = "SKIPPED"
        return {"status": status, "flags": self.flags, "skipped": self.skipped}


def band(v: float, cfg: dict, se: float) -> str | None:
    """PASS/WARN/FAIL for a statistic that should be 0, requiring it to be both large and significant."""
    if abs(v) <= cfg["warn"] or abs(v) < cfg.get("z_stat", 3.0) * se:
        return None
    return "FAIL" if abs(v) > cfg["fail"] else "WARN"


def grade_feature(rv: str, ctx: dict) -> dict:
    T, rd = ctx["T"], ctx["rd"]
    m: dict = {"response_var": rv}
    steps = {k: Step() for k in STEP_TITLES}
    z_all = ctx["z_test"]
    have_z = z_all is not None and rv in z_all.columns
    model_file = (rd / "model" / rv / "regression_model.json").is_file()

    # ---- S1 completion and convergence ---------------------------------------
    s = steps["S1"]
    s.ran = True
    st = read_json(rd / "status" / f"{rv}.json") or {}
    if not model_file or not have_z:
        s.flag("NOT_COMPLETED", "FAIL", "no saved model and/or no test z-scores"
               + (f" ({st.get('error')})" if st.get("error") else ""))
    z = pd.to_numeric(z_all[rv], errors="coerce").to_numpy(float) if have_z else np.array([])
    if have_z and not np.isfinite(z).all():
        s.flag("NONFINITE_Z", "FAIL", f"{int((~np.isfinite(z)).sum())} of {len(z)} test z-scores are NaN or infinite")
    conv = hbr_convergence(rd, rv) if model_file else None
    if conv is None:
        if ctx["is_hbr"] and model_file:
            s.skipped.append("MCMC convergence: no idata.nc found for this variable")
        else:
            nlz = find_key(read_json(rd / "model" / rv / "regression_model.json") or {}, "nlZ")
            if isinstance(nlz, (int, float)) and not np.isfinite(nlz):
                s.flag("NONFINITE_Z", "FAIL", "BLR marginal likelihood is not finite: the optimiser did not converge")
    elif "error" in conv:
        s.skipped.append(f"MCMC convergence: could not read idata.nc ({conv['error']})")
    else:
        m.update(rhat_max=conv["rhat_max"], ess_min=conv["ess_min"], divergent_frac=conv.get("divergent_frac"))
        H = T["hbr"]
        if conv["rhat_max"] > H["rhat_warn"]:
            s.flag("HBR_RHAT", "FAIL" if conv["rhat_max"] > H["rhat_fail"] else "WARN",
                   f"max R-hat {conv['rhat_max']:.3f} (chains disagree)", conv["rhat_max"])
        if conv["ess_min"] < H["ess_warn"]:
            s.flag("HBR_ESS", "FAIL" if conv["ess_min"] < H["ess_fail"] else "WARN",
                   f"min effective sample size {conv['ess_min']:.0f}", conv["ess_min"])
        div = conv.get("divergent_frac")
        if div is None and isinstance(st.get("sampler"), dict) and st.get("state") == "ok":
            div = st["sampler"].get("divergent_frac")        # recorded by run_model.py when it fitted locally
        m["divergent_frac"] = div
        if div is None:
            s.skipped.append("divergent transitions: PCNtoolkit saves only the posterior draws in idata.nc. "
                             "Local runs record the sampler's divergence count at fit time; this run "
                             "(cluster jobs, or a model fitted elsewhere) has no such record")
        elif div > 0:
            s.flag("HBR_DIVERGENT", "FAIL" if div > H["div_fail_frac"] else "WARN",
                   f"{100 * div:.2f}% divergent transitions", div)

    usable = have_z and np.isfinite(z).sum() >= 8
    if not usable:
        for k in ("S2", "S3", "S4", "S5"):
            steps[k].skipped.append("no usable test z-scores")
        return finish(rv, m, steps, ctx)
    ok = np.isfinite(z)
    # One corrupt value (unit error, failed segmentation) should produce one clear flag, not a dozen:
    # extreme z-scores are reported on their own and left out of the distribution statistics below.
    ext = ok & (np.abs(z) > T["extreme_abs_z"])
    s = steps["S2"]
    s.ran = True
    if ext.any():
        ids = z_all["subject_ids"].astype(str).to_numpy()[ext] if "subject_ids" in z_all.columns else np.where(ext)[0]
        m["n_extreme_z"] = int(ext.sum())
        s.flag("EXTREME_Z", "FAIL",
               f"{int(ext.sum())} held-out reference subject(s) with |z| > {T['extreme_abs_z']:g} "
               f"({', '.join(f'{i} (z={v:+.0f})' for i, v in list(zip(ids, z[ext]))[:5])}): almost certainly a data "
               "error rather than a model problem. They are excluded from the statistics below.", int(ext.sum()))
        ok = ok & ~ext
    z = np.where(ok, z, np.nan)
    zf = z[ok]
    n = len(zf)
    low_n = n < T["min_n_reliable"]

    # ---- S2 calibration ------------------------------------------------------
    mean, sd = float(zf.mean()), float(zf.std(ddof=1))
    skew, kurt = float(stats.skew(zf)), float(stats.kurtosis(zf))
    w = float(stats.shapiro(zf if n <= 5000 else np.random.default_rng(0).choice(zf, 5000, replace=False))[0])
    tail = float((np.abs(zf) > 1.96).mean())
    cdf = stats.norm.cdf(zf)
    qs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    mace = float(np.mean([abs((cdf < q).mean() - q) for q in qs]))
    m.update(n_test=n, z_mean=mean, z_sd=sd, z_skew=skew, z_kurt=kurt, shapiro_w=w,
             frac_abs_z_gt_1_96=tail, frac_abs_z_gt_3=float((np.abs(zf) > 3).mean()), mace_z=mace)
    sev = band(mean, T["z_mean"], sd / np.sqrt(n))
    if sev:
        s.flag("Z_MEAN", sev, f"mean z = {mean:+.2f} (should be 0): the model is biased for held-out reference subjects", mean)
    lo_w, hi_w = T["z_sd"]["warn"]
    lo_f, hi_f = T["z_sd"]["fail"]
    if not (lo_w <= sd <= hi_w) and abs(sd - 1) > T["z_sd"]["z_stat"] / np.sqrt(2 * n):
        s.flag("Z_SD", "FAIL" if not (lo_f <= sd <= hi_f) else "WARN",
               f"SD of z = {sd:.2f} (should be 1): predicted variance is too "
               + ("small, deviations are overstated" if sd > 1 else "large, deviations are understated"), sd)
    sev = band(skew, T["z_skew"], np.sqrt(6 / n))
    if sev:
        s.flag("Z_SKEW", sev, f"skewness of z = {skew:+.2f}: the response distribution is asymmetric and the model's is not", skew)
    sev = band(kurt, T["z_kurt"], np.sqrt(24 / n))
    if sev:
        s.flag("Z_KURT", sev, f"excess kurtosis of z = {kurt:+.2f}: tails are "
               + ("heavier" if kurt > 0 else "lighter") + " than the model assumes", kurt)
    tl = T["tail_196"]
    p_tail = stats.binomtest(int((np.abs(zf) > 1.96).sum()), n, tl["expected"]).pvalue
    if (tail < tl["warn_low"] or tail > tl["warn_high"]) and p_tail < tl["p"]:
        s.flag("Z_TAILS", "WARN", f"{100 * tail:.1f}% of reference subjects have |z| > 1.96 (expected 5%)", tail)
    if mace > T["mace"]["warn"] and n >= T["min_n_reliable"]:
        s.flag("MACE", "FAIL" if mace > T["mace"]["fail"] else "WARN",
               f"mean absolute centile error {mace:.3f}: centiles do not contain the stated share of subjects", mace)

    # ---- S3 fit relative to the cohort ---------------------------------------
    s = steps["S3"]
    pst = ctx["stats_test"]
    if pst is not None and rv in pst.index:
        s.ran = True
        row = pst.loc[rv]
        for k in ("EXPV", "MSLL", "SMSE", "Rho", "RMSE", "R2", "MACE"):
            if k in row.index and pd.notna(row[k]):
                m[k] = float(row[k])
        if m.get("MSLL", -1) > T["msll_floor"]:
            s.flag("WORSE_THAN_BASELINE", "WARN",
                   f"MSLL = {m['MSLL']:+.3f} > 0: no better than a constant mean-and-variance baseline", m["MSLL"])
        if m.get("EXPV", 1) < T["expv_floor"]:
            s.flag("NEG_EXPV", "FAIL", f"explained variance {m['EXPV']:.3f} < 0: predictions are worse than the mean",
                   m["EXPV"])
        for k, bad_side in (("MSLL", +1), ("EXPV", -1), ("MACE", +1)):
            cz = ctx["cohort_z"].get(k)
            if cz is not None and rv in cz.index and bad_side * cz[rv] > T["cohort_modified_z"]:
                s.flag("COHORT_OUTLIER", "WARN",
                       f"{k} = {m[k]:.3f} is an outlier among this run's {len(cz)} variables "
                       f"(modified Z = {cz[rv]:+.1f})", float(cz[rv]))
    else:
        s.skipped.append("PCNtoolkit statistics file has no row for this variable")

    # ---- S4 / S5 need covariates and batch labels of the test rows ---------------
    tab = ctx["test_tab"]
    if tab is None:
        steps["S4"].skipped.append(ctx["align_how"])
        steps["S5"].skipped.append(ctx["align_how"])
    else:
        s = steps["S4"]
        S = T["site"]
        levels = []
        for b in ctx["batch"]:
            s.ran = True
            for lvl, idx in tab.groupby(b).indices.items():
                zz = z[idx]
                zz = zz[np.isfinite(zz)]
                if len(zz) < S["min_n"]:
                    continue
                mu, sdl = float(zz.mean()), float(zz.std(ddof=1))
                levels.append({"dim": b, "level": str(lvl), "n": len(zz), "mean": mu, "sd": sdl})
                if abs(mu) > S["mean_warn"] and abs(mu) * np.sqrt(len(zz)) > S["z_stat"]:
                    s.flag("SITE_MEAN", "FAIL" if abs(mu) > S["mean_fail"] else "WARN",
                           f"{b}={lvl}: mean z = {mu:+.2f} over {len(zz)} subjects (batch effect not removed)", mu)
                if len(zz) >= S["sd_min_n"] and not (S["sd_ratio"][0] <= sdl <= S["sd_ratio"][1]) \
                        and abs(sdl - 1) > 3 / np.sqrt(2 * len(zz)):
                    s.flag("SITE_SD", "WARN", f"{b}={lvl}: SD of z = {sdl:.2f} over {len(zz)} subjects "
                                              "(variance differs by batch)", sdl)
        m["batch_levels"] = levels
        if not ctx["batch"]:
            s.skipped.append("the dataset has no batch effects")

        s = steps["S5"]
        C = T["cov"]
        for cv in ctx["cov"]:
            s.ran = True
            x = tab[cv].to_numpy(float)[ok]
            if np.ptp(x) == 0:
                continue
            rho, p = stats.spearmanr(zf, x)
            rho2, p2 = stats.spearmanr(np.abs(zf), x)
            m[f"rho_z_{cv}"], m[f"rho_absz_{cv}"] = float(rho), float(rho2)
            if abs(rho) > C["rho_warn"] and p < C["p"]:
                s.flag("COV_TREND", "WARN", f"z still trends with {cv} (Spearman {rho:+.2f}): the mean curve is too rigid", rho)
            if abs(rho2) > C["rho_warn"] and p2 < C["p"]:
                s.flag("COV_SPREAD", "WARN", f"spread of z changes with {cv} (Spearman {rho2:+.2f} for |z|): "
                                             "variance should depend on the covariate", rho2)
            try:
                bins = pd.qcut(x, C["bins"], duplicates="drop")
                bm = pd.Series(zf).groupby(bins.codes).agg(["mean", "size"])
                bm = bm[bm["size"] >= C["bin_min_n"]]
                if len(bm):
                    worst_bin = float(bm["mean"].abs().max())
                    m[f"max_bin_mean_z_{cv}"] = worst_bin
                    sig = (bm["mean"].abs() * np.sqrt(bm["size"]) > 3.5) & (bm["mean"].abs() > C["bin_mean_warn"])
                    if sig.any():
                        s.flag("COV_BINS", "WARN", f"in part of the {cv} range the mean z reaches {worst_bin:.2f}: "
                                                   "local misfit of the mean curve", worst_bin)
            except Exception:
                pass
        cc = centile_checks(rd, "test", rv, ctx["centile_cache"])
        if cc is None:
            s.skipped.append("centile ordering: centiles_test.csv not available or not parseable")
        else:
            s.ran = True
            cross, nonfinite = cc
            m["centile_cross_frac"], m["centile_nonfinite_frac"] = cross, nonfinite
            CC = T["centile_cross"]
            if nonfinite > 0:
                s.flag("CENTILE_NONFINITE", "FAIL",
                       f"{100 * nonfinite:.1f}% of the saved centile values are infinite or missing: the centile "
                       "file cannot be used (with PCNtoolkit up to 1.3.0 this is what y_transform produces)",
                       nonfinite)
            if cross > CC["warn"]:
                s.flag("CENTILE_CROSS", "FAIL" if cross > CC["fail"] else "WARN",
                       f"centile curves cross for {100 * cross:.1f}% of test subjects", cross)
        zt = ctx["z_train"]
        if zt is not None and rv in zt.columns:
            ztr = pd.to_numeric(zt[rv], errors="coerce").to_numpy(float)
            ztr = ztr[np.isfinite(ztr)]
            if len(ztr) > 8:
                s.ran = True
                gap = sd - float(ztr.std(ddof=1))
                m["z_sd_train"], m["z_sd_gap"] = float(ztr.std(ddof=1)), gap
                if gap > T["overfit"]["sd_gap"] and gap > 3 / np.sqrt(2 * n):
                    s.flag("OVERFIT", "WARN", f"z-scores are wider on test (SD {sd:.2f}) than on train "
                                              f"(SD {ztr.std(ddof=1):.2f}): the model is overfitting", gap)
        else:
            s.skipped.append("train/test comparison: no train z-scores (run `run_model.py predict --data train`)")

    out = finish(rv, m, steps, ctx)
    if low_n:      # small test sets are too noisy to condemn a model on distribution shape alone
        for k in ("S2", "S4", "S5"):
            stp = out["steps"][k]
            for f in stp["flags"]:
                if f["severity"] == "FAIL" and f["code"] not in ("CENTILE_CROSS", "CENTILE_NONFINITE", "EXTREME_Z"):
                    f["severity"] = "WARN"
                    f["message"] += f" [downgraded: only {n} test subjects]"
            if stp["flags"]:
                stp["status"] = worst(f["severity"] for f in stp["flags"])
        out["grade"] = worst(v["status"] for v in out["steps"].values())
        out["notes"].append(f"only {n} held-out reference subjects: distribution checks are noisy")
    return out


def finish(rv: str, m: dict, steps: dict, ctx: dict) -> dict:
    res = {k: v.result() for k, v in steps.items()}
    codes = [f["code"] for v in res.values() for f in v["flags"]]
    return {"response_var": rv, "grade": worst(v["status"] for v in res.values()), "steps": res,
            "metrics": m, "fix_candidates": fix_candidates(codes, ctx["recipe"], m), "notes": [],
            "plot": None}


# --------------------------------------------------------------------------
# PROPOSE: ranked repair candidates. Every fix is a recipe change that pcn-model-skill
# refits under a new run name; QC never edits z-scores or model files itself.
# --------------------------------------------------------------------------
def fix_candidates(codes: list[str], recipe: dict, m: dict) -> list[dict]:
    if not codes:
        return []
    c = set(codes)
    alg = recipe.get("algorithm", "blr")
    blr, hbr = recipe.get("blr", {}), recipe.get("hbr", {})
    basis = recipe.get("basis", {})
    out: list[dict] = []

    def add(label: str, change: dict | None, why: str) -> None:
        if not any(o["label"] == label for o in out):
            out.append({"label": label, "change": change, "why": why})

    shape = c & {"Z_SKEW", "Z_KURT", "Z_TAILS", "MACE"}
    spread = c & {"COV_SPREAD", "Z_SD"}
    trend = c & {"COV_TREND", "COV_BINS"}
    wiggly = c & {"OVERFIT", "CENTILE_CROSS"}
    broken = c & {"NOT_COMPLETED", "NONFINITE_Z"}
    if "CENTILE_NONFINITE" in c and recipe.get("y_transform"):
        add("Refit without y_transform", {"y_transform": None},
            "PCNtoolkit up to 1.3.0 inverts the transform repeatedly when it saves centiles; "
            "model the skew with a warp or a SHASH likelihood instead")

    if alg == "blr":
        if broken:
            if str(blr.get("optimizer", "l-bfgs-b")).lower() == "cg":       # the only optimiser that reads n_iter / tol
                add("Refit with more optimiser iterations", {"blr": {"n_iter": 2000, "tol": 1e-8}},
                    "conjugate gradients may have stopped early")
            if str(blr.get("optimizer", "l-bfgs-b")).lower() != "powell":
                add("Refit with the Powell optimiser", {"blr": {"optimizer": "powell"}},
                    "gradient-free; more robust when L-BFGS-B diverges, but slower")
            if blr.get("warp_name"):
                add("Refit without the warp", {"blr": {"warp_name": None, "warp_reparam": False}},
                    "warp parameters are the usual cause of non-finite likelihoods")
        if shape and not blr.get("warp_name"):
            add("Add a sinh-arcsinh warp", {"blr": {"warp_name": "warpsinharcsinh", "warp_reparam": True}},
                "models skewness and tail weight of the response")
        if spread and not blr.get("heteroskedastic"):
            add("Let the variance depend on the covariate", {"blr": {"heteroskedastic": True}},
                "noise level changes along the covariate")
        if "SITE_MEAN" in c:
            if not blr.get("fixed_effect"):
                add("Add fixed batch effects to the mean", {"blr": {"fixed_effect": True}}, "site offsets are unmodelled")
            elif not blr.get("fixed_effect_slope"):
                add("Let the covariate slope differ by batch", {"blr": {"fixed_effect_slope": True}},
                    "offsets alone did not remove the site shift")
        if "SITE_SD" in c and not blr.get("fixed_effect_var"):
            add("Add a batch effect to the variance", {"blr": {"fixed_effect_var": True, "heteroskedastic": True}},
                "sites differ in spread, not only in level")
        if c & {"SITE_MEAN", "SITE_SD"}:
            add("Switch to HBR with random site effects",
                {"algorithm": "hbr", "hbr": {"likelihood": "SHASHb" if shape else "Normal", "random_intercept_mu": True,
                                             "draws": 1500, "tune": 500, "chains": 4, "cores": 4,
                                             "nuts_sampler": "nutpie"}},
                "partial pooling handles small or unusual sites better than fixed effects; much slower")
    elif alg == "hbr":
        if c & {"HBR_RHAT", "HBR_ESS", "HBR_DIVERGENT"} or broken:
            add("Sample longer", {"hbr": {"tune": max(1500, int(hbr.get("tune", 500)) * 3),
                                          "draws": max(2000, int(hbr.get("draws", 1500)))}},
                "more warm-up usually resolves R-hat and ESS warnings")
            if hbr.get("random_slope_mu") or hbr.get("random_intercept_sigma"):
                add("Simplify the hierarchy", {"hbr": {"random_slope_mu": False, "random_intercept_sigma": False}},
                    "fewer random effects give an easier posterior")
            add("Fall back to BLR", {"algorithm": "blr", "blr": {"fixed_effect": True, "heteroskedastic": True}},
                "no MCMC: use when the sampler will not converge")
        if shape and str(hbr.get("likelihood", "Normal")).lower() == "normal":
            add("Use the SHASH likelihood", {"hbr": {"likelihood": "SHASHb"}},
                "models skewness and tail weight; slower to sample")
        if "SITE_MEAN" in c and not hbr.get("random_intercept_mu", True):
            add("Add a random site intercept", {"hbr": {"random_intercept_mu": True}}, "site offsets are unmodelled")
        if "SITE_SD" in c and not hbr.get("random_intercept_sigma"):
            add("Add a random site effect on the variance", {"hbr": {"random_intercept_sigma": True}},
                "sites differ in spread")
        if spread and not hbr.get("linear_sigma", True):
            add("Let the variance depend on the covariate", {"hbr": {"linear_sigma": True}},
                "noise level changes along the covariate")
    else:                                         # transferred / extended from a reference model
        if c & {"Z_MEAN", "SITE_MEAN", "Z_SD", "SITE_SD"}:
            if recipe.get("mode") != "extend_predict":
                add("Use extend instead of transfer", {"mode": "extend_predict", "transfer_kwargs": None},
                    "extend refits the full model on reference-synthesised plus local data")
            add("Fit from scratch on your own reference data",
                {"mode": "fit_predict", "reference_model": None, "algorithm": "blr", "transfer_kwargs": None,
                 "n_synth_samples": None},
                "only sensible with a few hundred local reference subjects; needs a full recipe from choose_recipe.py")
    if alg in ("blr", "hbr"):
        if trend:
            if basis.get("type") != "bspline":
                add("Use a cubic B-spline for the mean", {"basis": {"type": "bspline", "nknots": 5, "degree": 3,
                                                                    "basis_column": 0}},
                    "a straight line cannot follow the covariate effect")
            else:
                add("Add spline knots", {"basis": {**basis, "nknots": int(basis.get("nknots", 5)) + 2}},
                    "the mean curve is too stiff")
        if wiggly and basis.get("type") == "bspline" and int(basis.get("nknots", 5)) > 3:
            add("Remove spline knots", {"basis": {**basis, "nknots": max(3, int(basis.get("nknots", 5)) - 2)}},
                "a simpler curve generalises better and keeps centiles ordered")
    if c & {"EXTREME_Z", "Z_TAILS", "Z_KURT", "NOT_COMPLETED", "NONFINITE_Z"}:
        add("Review this variable's input data", None,
            "go back to pcn-data-skill: check outliers and units, dismiss corrupt subjects, rebuild, then refit")
    if "EXTREME_Z" in c:                          # a data error is fixed in the data, not by a more flexible model
        out.sort(key=lambda o: o["change"] is not None)
    for i, o in enumerate(out):
        o["id"] = "ABCDEFGHIJ"[i] if i < 10 else str(i)
    return out


# --------------------------------------------------------------------------
# plots (one figure per variable, four single-axis panels)
# --------------------------------------------------------------------------
INK, MUTED, GRID, BLUE, GREY, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#b9b8b2", "#fcfcfb"


def plot_feature(rv: str, feat: dict, ctx: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = pd.to_numeric(ctx["z_test"][rv], errors="coerce").to_numpy(float)
    extreme = np.isfinite(z) & (np.abs(z) > ctx["T"]["extreme_abs_z"])
    ok = np.isfinite(z) & ~extreme           # extreme values would flatten every panel; they are counted instead
    tab = ctx["test_tab"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    lab = dict(color=MUTED, fontsize=9)
    ttl = dict(color=INK, fontsize=10, loc="left")
    cv = ctx["cov"][0] if ctx["cov"] else None

    ax = axes[0, 0]                                                    # z against the covariate
    if tab is not None and cv:
        x = tab[cv].to_numpy(float)
        ax.scatter(x[ok], z[ok], s=9, color=BLUE, alpha=0.55, linewidths=0)
        for y0, ls in ((0, "-"), (1.96, "--"), (-1.96, "--")):
            ax.axhline(y0, color=MUTED, linewidth=0.8, linestyle=ls)
        try:
            bins = pd.qcut(x[ok], 8, duplicates="drop")
            g = pd.DataFrame({"x": x[ok], "z": z[ok]}).groupby(bins.codes).mean()
            ax.plot(g["x"], g["z"], color=INK, linewidth=2)
        except Exception:
            pass
        ax.set_xlabel(cv, **lab)
        ax.set_title("z-scores along the covariate (black: binned mean)", **ttl)
    else:
        ax.text(0.5, 0.5, "covariates not matched to z-scores", ha="center", color=MUTED, transform=ax.transAxes)
    ax.set_ylabel("z", **lab)

    ax = axes[0, 1]                                                    # QQ
    zs = np.sort(z[ok])
    th = stats.norm.ppf((np.arange(1, len(zs) + 1) - 0.5) / len(zs))
    lim = max(3.5, float(np.abs(zs).max()) if len(zs) else 3.5)
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=0.8)
    ax.scatter(th, zs, s=9, color=BLUE, alpha=0.7, linewidths=0)
    ax.set_xlabel("standard normal quantile", **lab)
    ax.set_ylabel("observed z quantile", **lab)
    ax.set_title("QQ plot of held-out reference z-scores", **ttl)

    ax = axes[1, 0]                                                    # batch levels
    lv = sorted(feat["metrics"].get("batch_levels", []), key=lambda r: -abs(r["mean"]))[:14][::-1]
    if lv:
        ypos = np.arange(len(lv))
        means = np.array([r["mean"] for r in lv])
        se = np.array([r["sd"] / np.sqrt(r["n"]) for r in lv])
        ax.hlines(ypos, means - 1.96 * se, means + 1.96 * se, color=BLUE, linewidth=2)
        ax.scatter(means, ypos, s=36, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{r['dim']}={r['level']} (n={r['n']})" for r in lv], fontsize=7.5, color=MUTED)
        ax.set_xlabel("mean z with 95% interval", **lab)
        ax.set_title("Batch levels with the largest mean shift", **ttl)
    else:
        ax.text(0.5, 0.5, "no batch level with enough test subjects", ha="center", color=MUTED,
                transform=ax.transAxes)

    ax = axes[1, 1]                                                    # raw data
    if tab is not None and cv and rv in tab.columns:
        tr = ctx["train_tab"]
        if tr is not None and rv in tr.columns:
            ax.scatter(tr[cv], tr[rv], s=6, color=GREY, alpha=0.5, linewidths=0, label="train")
        ax.scatter(tab[cv], tab[rv], s=9, color=BLUE, alpha=0.7, linewidths=0, label="test")
        lo, hi = np.nanpercentile(tab[rv].to_numpy(float)[ok], [0.5, 99.5]) if ok.any() else (None, None)
        if lo is not None and hi > lo:
            ax.set_ylim(lo - 0.25 * (hi - lo), hi + 0.25 * (hi - lo))
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED, loc="best")
        ax.set_xlabel(cv, **lab)
        ax.set_ylabel(rv[:40], **lab)
        ax.set_title("Raw response values", **ttl)
    note = f"   {int(extreme.sum())} extreme value(s) with |z| > {ctx['T']['extreme_abs_z']:g} not shown" if extreme.any() else ""
    fig.suptitle(f"{rv}   agent grade: {feat['grade']}{note}", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------
def deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--plots", choices=["flagged", "all", "none"], default="flagged")
    ap.add_argument("--thresholds", help="defaults to the skill's assets/qc_thresholds.json")
    a = ap.parse_args(argv)
    project = Project(a.project)
    with audited(project, "qc_detect.py", argv):
        rd, qd = project.run_dir(a.run), project.qc_dir(a.run)
        plan, recipe = read_json(rd / "plan.json"), read_json(rd / "recipe.json")
        man = read_json(project.data / "build_manifest.json")
        if plan is None or recipe is None or man is None:
            die(f"run '{a.run}' or the dataset is incomplete (plan.json / recipe.json / build_manifest.json)", 2)
        tpath = Path(a.thresholds) if a.thresholds else Path(__file__).resolve().parent.parent / "assets" / "qc_thresholds.json"
        T = read_json(tpath)
        if T is None:
            die(f"thresholds file not found: {tpath}", 2)
        str_cols = {c: str for c in [ID] + man["batch_effects"]}
        test = pd.read_csv(project.data / "test.csv", dtype=str_cols)
        train = pd.read_csv(project.data / "train.csv", dtype=str_cols)
        z_test = read_z(rd, "test")
        test_tab, how = align(z_test, test) if z_test is not None else (None, "no test z-scores")
        stats_test = read_stats(rd, "test")
        cohort_z = {}
        if stats_test is not None and len(stats_test) >= T["cohort_min_features"]:
            for k in ("MSLL", "EXPV", "MACE"):
                if k in stats_test.columns and stats_test[k].notna().sum() >= T["cohort_min_features"]:
                    col = stats_test[k].dropna()
                    cohort_z[k] = pd.Series(modified_z(col.to_numpy()), index=col.index)
        is_hbr = recipe.get("algorithm") == "hbr" or (plan.get("reference") or {}).get("kind") == "HBR"
        ctx = {"T": T, "rd": rd, "recipe": recipe, "algorithm": recipe.get("algorithm", "blr"), "is_hbr": is_hbr,
               "z_test": z_test, "z_train": read_z(rd, "train"), "stats_test": stats_test, "cohort_z": cohort_z,
               "test_tab": test_tab, "train_tab": train, "align_how": how, "batch": man["batch_effects"],
               "cov": man["covariates"], "centile_cache": {},
               "train_ranges": {c: (float(train[c].min()), float(train[c].max())) for c in man["covariates"]}}
        features = [grade_feature(rv, ctx) for rv in plan["response_vars"]]
        dataset_notes = []
        if test_tab is not None:
            for cv, (lo, hi) in ctx["train_ranges"].items():
                n_out = int(((test_tab[cv] < lo) | (test_tab[cv] > hi)).sum())
                if n_out:
                    dataset_notes.append(f"{n_out} test subject(s) lie outside the training range of {cv} "
                                         f"[{lo:g}, {hi:g}]: their z-scores are extrapolations for every variable")

        (qd / "plots").mkdir(parents=True, exist_ok=True)
        if a.plots != "none" and z_test is not None:
            for f in features:
                rv = f["response_var"]
                if rv in z_test.columns and (a.plots == "all" or f["grade"] != "PASS"):
                    name = f"{safe_name(rv)}.png"
                    try:
                        plot_feature(rv, f, ctx, qd / "plots" / name)
                        f["plot"] = f"plots/{name}"
                    except Exception as e:  # noqa: BLE001
                        f["notes"].append(f"plot failed: {type(e).__name__}: {e}")
        rows = []
        for f in features:
            flat = {k: v for k, v in f["metrics"].items() if not isinstance(v, (list, dict))}
            flat.update(grade=f["grade"], **{k: f["steps"][k]["status"] for k in STEP_TITLES},
                        flags=";".join(fl["code"] for s in f["steps"].values() for fl in s["flags"]))
            rows.append(flat)
        pd.DataFrame(rows).to_csv(qd / "metrics.csv", index=False)
        counts = {g: sum(f["grade"] == g for f in features) for g in ("PASS", "WARN", "FAIL")}
        write_json(qd / "grades.json", {
            "run": a.run, "graded_at": now(), "thresholds_file": str(tpath), "thresholds": T,
            "alignment": how, "dataset_notes": dataset_notes, "step_titles": STEP_TITLES, "counts": counts, "recipe": recipe,
            "covariates": man["covariates"], "batch_effects": man["batch_effects"], "features": features})
        say(f"graded {len(features)} response variables: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
        say(f"z-scores matched to the test table {how}")
        for note in dataset_notes:
            say(f"NOTE: {note}")
        for k, title in STEP_TITLES.items():
            n_bad = sum(f["steps"][k]["status"] in ("WARN", "FAIL") for f in features)
            n_skip = sum(f["steps"][k]["status"] == "SKIPPED" for f in features)
            say(f"  {k} {title:<34} flagged {n_bad:>5}   skipped {n_skip:>5}")
        say(f"outputs: {qd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
