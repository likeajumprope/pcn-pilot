#!/usr/bin/env python3
"""Route a validated dataset to a modelling recipe (the analogue of modality-driven pipeline selection).

    python choose_recipe.py --project <project> --out recipe.json
                            [--reference <saved model dir> --goal transfer|extend]
                            [--prefer blr|hbr]

Reads the training split, measures a few data traits (size, sites, shape of the
residuals) and writes a recipe with the reasoning for every choice plus ranked
alternatives. The recipe is a proposal: a human approves it before anything is fitted.

Needs only pandas / numpy / scipy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Project, audited, die, read_json, say, write_json  # noqa: E402

ID = "subject_id"
MAX_VARS_FOR_TRAITS = 400


def traits(train: pd.DataFrame, man: dict, seed: int = 0) -> dict:
    cov, batch, rvs = man["covariates"], man["batch_effects"], man["response_vars"]
    x = train[cov[0]].to_numpy(float)
    rng = np.random.default_rng(seed)
    sample = rvs if len(rvs) <= MAX_VARS_FOR_TRAITS else list(rng.choice(rvs, MAX_VARS_FOR_TRAITS, replace=False))
    xs = (x - x.mean()) / (x.std() or 1.0)
    design_lin = np.column_stack([np.ones_like(xs), xs])
    design_cub = np.column_stack([np.ones_like(xs), xs, xs ** 2, xs ** 3])
    if batch:
        dummies = pd.get_dummies(train[batch].astype(str), drop_first=True).to_numpy(float)
        design_site = np.column_stack([design_cub, dummies])
    else:
        design_site = design_cub
    n_skew = n_kurt = n_het = n_nonlin = n_site = n_sitevar = 0
    site_col = max(batch, key=lambda b: train[b].nunique()) if batch else None
    for r in sample:
        y = train[r].to_numpy(float)
        sd = y.std() or 1.0
        ys = (y - y.mean()) / sd

        def resid(design):
            beta, *_ = np.linalg.lstsq(design, ys, rcond=None)
            return ys - design @ beta
        e_lin, e_cub, e_site = resid(design_lin), resid(design_cub), resid(design_site)
        r2_lin, r2_cub, r2_site = 1 - e_lin.var(), 1 - e_cub.var(), 1 - e_site.var()
        n_nonlin += (r2_cub - r2_lin) > 0.01
        n_site += (r2_site - r2_cub) > 0.02
        n_skew += abs(stats.skew(e_site)) > 0.5
        n_kurt += abs(stats.kurtosis(e_site)) > 1.0
        rho, p = stats.spearmanr(np.abs(e_site), xs)
        n_het += (abs(rho) > 0.1) and (p < 0.001)
        if site_col:
            sds = pd.Series(e_site).groupby(train[site_col].to_numpy()).agg(["std", "size"])
            sds = sds[sds["size"] >= 20]["std"]
            n_sitevar += len(sds) >= 2 and (sds.max() / max(sds.min(), 1e-9)) > 1.5
    k = len(sample)
    vals = train[rvs].to_numpy(float)
    levels = {b: train[b].value_counts() for b in batch}
    return {
        "n_train": int(len(train)), "n_response_vars": len(rvs), "n_vars_examined": k,
        "covariate_span": float(x.max() - x.min()),
        "n_batch_levels": {b: int(v.size) for b, v in levels.items()},
        "min_level_n": int(min((int(v.min()) for v in levels.values()), default=len(train))),
        "frac_skewed": round(n_skew / k, 3), "frac_heavy_tailed": round(n_kurt / k, 3),
        "frac_heteroskedastic": round(n_het / k, 3), "frac_nonlinear": round(n_nonlin / k, 3),
        "frac_site_shift": round(n_site / k, 3), "frac_site_variance": round(n_sitevar / k, 3),
        "all_in_unit_interval": bool((vals > 0).all() and (vals < 1).all()),
        "all_positive": bool((vals > 0).all()),
        "repeated_measures": bool(train[ID].duplicated().any()),
    }


def propose(t: dict, has_batch: bool, prefer: str | None, reference: str | None, goal: str | None) -> dict:
    why: list[str] = []
    alternatives: list[dict] = []
    nongauss = max(t["frac_skewed"], t["frac_heavy_tailed"]) > 0.2

    # ---- reference-model routes
    if reference:
        mode = {"transfer": "transfer_predict", "extend": "extend_predict"}.get(goal or "transfer")
        if mode is None:
            die("--goal must be transfer or extend", 2)
        why.append("transfer: re-estimates the reference model for the new site(s); centiles then cover only the "
                   "covariate range of YOUR data, and a model may be transferred only once" if mode == "transfer_predict"
                   else "extend: refits on your data plus data synthesised from the reference model, keeping the "
                        "reference sites and covariate range; can be repeated site after site")
        if t["min_level_n"] < 20:
            why.append(f"WARNING: the smallest batch level has {t['min_level_n']} adaptation observations; "
                       "PCNtoolkit's guidance is 20 to 100 healthy controls per new site")
        other = "extend" if mode == "transfer_predict" else "transfer"
        alternatives.append({"change": {"mode": f"{other}_predict"},
                             "when": f"use {other} instead if " + (
                                 "you need the reference sites and full covariate range kept, or will add more sites later"
                                 if other == "extend" else "you only need scores for this site and want the fastest adaptation")})
        alternatives.append({"change": {"mode": "fit_predict", "reference_model": None},
                             "when": "fit from scratch if you have several hundred reference subjects of your own"})
        return {"algorithm": "from_reference", "mode": mode, "reference_model": str(Path(reference).resolve()),
                "rationale": why, "alternatives": alternatives}

    # ---- basis for the mean
    if t["covariate_span"] < 10 and t["frac_nonlinear"] <= 0.2:
        basis = {"type": "linear", "basis_column": 0}
        why.append(f"linear basis: the first covariate spans only {t['covariate_span']:.1f} units and "
                   f"{100 * t['frac_nonlinear']:.0f}% of variables show curvature")
    else:
        basis = {"type": "bspline", "nknots": 5, "degree": 3, "basis_column": 0}
        why.append(f"cubic B-spline (5 knots): {100 * t['frac_nonlinear']:.0f}% of variables show non-linear "
                   f"covariate effects over a span of {t['covariate_span']:.1f}")

    # ---- algorithm
    many_vars = t["n_response_vars"] > 50
    small_sites = has_batch and t["min_level_n"] < 30 and max(t["n_batch_levels"].values(), default=0) >= 5
    if prefer:
        alg = prefer
        why.append(f"algorithm {alg.upper()}: requested by the user")
    elif t["all_in_unit_interval"]:
        alg = "hbr"
        why.append("HBR with a Beta likelihood: every response value lies strictly between 0 and 1")
    elif many_vars or t["n_train"] > 20000:
        alg = "blr"
        why.append(f"BLR: {t['n_response_vars']} response variables x {t['n_train']} observations; BLR fits each in "
                   "seconds, HBR runs MCMC per variable")
    elif small_sites:
        alg = "hbr"
        why.append(f"HBR: {max(t['n_batch_levels'].values())} batch levels with as few as {t['min_level_n']} "
                   "observations; hierarchical priors pool information across small sites")
    else:
        alg = "blr"
        why.append("BLR: fast, and sites are large enough for fixed batch effects")

    recipe: dict = {"algorithm": alg, "mode": "fit_predict", "reference_model": None, "basis": basis,
                    "inscaler": "standardize", "outscaler": "standardize", "y_transform": None}
    if alg == "blr":
        blr = {"n_iter": 1000, "tol": 1e-8, "optimizer": "l-bfgs-b", "l_bfgs_b_epsilon": 0.1, "l_bfgs_b_l": 0.1,
               "l_bfgs_b_norm": "l2"}
        if has_batch:
            blr["fixed_effect"] = True
            why.append(f"fixed batch effects in the mean: {100 * t['frac_site_shift']:.0f}% of variables shift by batch")
            if t["frac_site_variance"] > 0.2:
                blr["fixed_effect_var"] = True
                why.append(f"batch effect in the variance: {100 * t['frac_site_variance']:.0f}% of variables differ in "
                           "spread between sites")
        if t["frac_heteroskedastic"] > 0.2 or basis["type"] == "bspline":
            blr["heteroskedastic"] = True
            why.append(f"heteroskedastic noise: {100 * t['frac_heteroskedastic']:.0f}% of variables change spread with "
                       "the covariate (and lifespan data usually do)")
        if nongauss:
            blr.update(warp_name="warpsinharcsinh", warp_reparam=True)
            why.append(f"sinh-arcsinh warp: {100 * t['frac_skewed']:.0f}% skewed, "
                       f"{100 * t['frac_heavy_tailed']:.0f}% heavy-tailed residuals")
        else:
            alternatives.append({"change": {"blr": {"warp_name": "warpsinharcsinh", "warp_reparam": True}},
                                 "when": "if QC flags skewed or heavy-tailed z-scores"})
        recipe["blr"] = blr
        alternatives.append({"change": {"algorithm": "hbr", "hbr": {"likelihood": "SHASHb" if nongauss else "Normal"}},
                             "when": "if QC shows residual site effects in small sites that fixed effects cannot absorb"})
    else:
        lik = "Beta" if t["all_in_unit_interval"] else ("SHASHb" if nongauss else "Normal")
        recipe["hbr"] = {"likelihood": lik, "random_intercept_mu": has_batch, "random_slope_mu": False,
                         "linear_sigma": True, "random_intercept_sigma": False,
                         "draws": 1500, "tune": 500, "chains": 4, "cores": 4, "nuts_sampler": "nutpie"}
        why.append(f"{lik} likelihood: " + {
            "Beta": "bounded responses (min-max scaling is applied, as the Beta likelihood requires)",
            "SHASHb": f"{100 * t['frac_skewed']:.0f}% skewed / {100 * t['frac_heavy_tailed']:.0f}% heavy-tailed residuals",
            "Normal": "residuals look close to Gaussian"}[lik])
        if lik == "Beta":
            recipe["inscaler"] = recipe["outscaler"] = "minmax"
        if lik == "Normal":
            alternatives.append({"change": {"hbr": {"likelihood": "SHASHb"}},
                                 "when": "if QC flags skewed or heavy-tailed z-scores (slower to sample)"})
        alternatives.append({"change": {"algorithm": "blr"},
                             "when": "if sampling is too slow or does not converge"})
    if t["repeated_measures"]:
        why.append("NOTE: repeated measures present. BLR and HBR treat observations as independent; "
                   "uncertainty will be somewhat too narrow")
    recipe["rationale"] = why
    recipe["alternatives"] = alternatives
    return recipe


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reference", help="saved PCNtoolkit model directory (pretrained or from another run)")
    ap.add_argument("--goal", choices=["transfer", "extend"])
    ap.add_argument("--prefer", choices=["blr", "hbr"])
    a = ap.parse_args(argv)
    project = Project(a.project)
    with audited(project, "choose_recipe.py", argv):
        man = read_json(project.data / "build_manifest.json")
        v = read_json(project.validation)
        if man is None or v is None:
            die("dataset not built/validated yet: run pcn-data-skill first", 2)
        if v["overall"] == "FAIL":
            die("data validation has FAIL checks; fix them before choosing a model")
        train = pd.read_csv(project.data / "train.csv", dtype={c: str for c in [ID] + man["batch_effects"]})
        t = traits(train, man)
        recipe = propose(t, bool(man["batch_effects"]), a.prefer, a.reference, a.goal)
        recipe["traits"] = t
        write_json(a.out, recipe)
        say("Data traits:")
        for k, val in t.items():
            say(f"  {k:<24} {val}")
        say(f"\nProposed route: {recipe['mode']} / {recipe['algorithm']}")
        for line in recipe["rationale"]:
            say(f"  - {line}")
        if recipe["alternatives"]:
            say("\nRanked alternatives:")
            for i, alt in enumerate(recipe["alternatives"], 1):
                say(f"  {i}. {alt['change']}  ({alt['when']})")
        say(f"\nrecipe written: {a.out}\nShow this to the user and get approval before `run_model.py plan`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
