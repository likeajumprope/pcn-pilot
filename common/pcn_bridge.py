"""The only place where PCN-Pilot touches the PCNtoolkit API (v1.x).

Keeping every toolkit call in one file means a future API change is fixed in
one place. Written against PCNtoolkit 1.3 (NormData / NormativeModel / BLR /
HBR / Runner). Synced into each skill by tools/sync_common.py.

Design rule: be strict. If a recipe asks for a setting that the installed
PCNtoolkit does not accept, stop with a clear message. A model must never be
fitted with a silently different configuration from the one that was approved.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

ID, GROUP, DUMMY_BATCH = "subject_id", "pcn_group", "pcn_all"


class RecipeError(ValueError):
    pass


def import_pcn():
    try:
        import pcntoolkit
    except ImportError as e:
        raise SystemExit(
            "ERROR: pcntoolkit is not importable in this interpreter "
            f"({sys.executable}).\nSet PCN_PYTHON in pipeline.env to the interpreter of the environment "
            "where PCNtoolkit >= 1.0 is installed (pip install pcntoolkit; Python 3.11 or 3.12).\n"
            f"Import error: {e}")
    for needed in ("NormData", "NormativeModel", "BLR", "HBR"):
        if not hasattr(pcntoolkit, needed):
            raise SystemExit(
                f"ERROR: this pcntoolkit has no '{needed}'. PCN-Pilot targets the v1.x API; "
                "legacy 0.x installs (estimate()/predict() functions) are not supported.")
    return pcntoolkit


def quiet(show: bool = False) -> None:
    try:
        import pcntoolkit.util.output as out
        out.Output.set_show_messages(show)
    except Exception:
        pass
    import logging
    import warnings
    lg = logging.getLogger("pymc")
    lg.setLevel(logging.WARNING)
    lg.propagate = False
    warnings.simplefilter(action="ignore", category=FutureWarning)


def call_strict(fn, **kwargs) -> Any:
    """Call fn(**kwargs); refuse (rather than drop) arguments it does not declare."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    named = {n for n, p in sig.parameters.items()
             if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
    # Names swallowed by a **kwargs catch-all are refused too: the toolkit would ignore them without a word.
    unknown = sorted(k for k in kwargs if k not in named)
    if unknown:
        name = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
        raise RecipeError(
            f"{name} in the installed PCNtoolkit does not declare {unknown}. "
            f"Declared: {sorted(n for n in named if n != 'self')}. "
            "Change the recipe or upgrade PCNtoolkit; the setting was NOT dropped silently.")
    return fn(**kwargs)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def read_split(csv_path: str | Path, manifest: dict):
    import pandas as pd
    batch = manifest["batch_effects"]
    df = pd.read_csv(csv_path, dtype={c: str for c in [ID] + batch})
    if not batch:
        df[DUMMY_BATCH] = "0"
    return df.reset_index(drop=True)


def batch_columns(manifest: dict) -> list[str]:
    return list(manifest["batch_effects"]) or [DUMMY_BATCH]


def make_normdata(name: str, df, manifest: dict, response_vars: list[str] | None = None):
    """DataFrame (one of the standardized splits) -> NormData.

    Row i of the DataFrame is observation i, and real subject IDs are passed on,
    so result files can be joined back to the split tables.
    """
    pcn = import_pcn()
    rvs = list(response_vars) if response_vars is not None else list(manifest["response_vars"])
    return call_strict(
        pcn.NormData.from_dataframe, name=name, dataframe=df.reset_index(drop=True),
        covariates=list(manifest["covariates"]), batch_effects=batch_columns(manifest),
        response_vars=rvs, subject_ids=ID)


def select(norm_data, response_vars: list[str]):
    return norm_data.sel({"response_vars": list(response_vars)})


# --------------------------------------------------------------------------
# recipe -> model
# --------------------------------------------------------------------------
RECIPE_DEFAULTS = {
    "algorithm": "blr", "mode": "fit_predict", "reference_model": None,
    "basis": {"type": "bspline", "nknots": 5, "degree": 3, "basis_column": 0},
    "inscaler": "standardize", "outscaler": "standardize", "y_transform": None, "saveplots": False,
}


def normalise_recipe(recipe: dict) -> dict:
    r = {**RECIPE_DEFAULTS, **recipe}
    r["algorithm"] = str(r["algorithm"]).lower()
    if r["mode"] not in ("fit_predict", "transfer_predict", "extend_predict"):
        raise RecipeError("recipe.mode must be fit_predict, transfer_predict or extend_predict")
    if r["mode"] == "fit_predict" and r["algorithm"] not in ("blr", "hbr"):
        raise RecipeError("recipe.algorithm must be 'blr' or 'hbr'")
    if r["mode"] != "fit_predict":
        r["algorithm"] = "from_reference"      # the reference model decides the algorithm
    if r["mode"] != "fit_predict" and not r.get("reference_model"):
        raise RecipeError(f"recipe.mode={r['mode']} needs recipe.reference_model (a saved model directory)")
    return r


def make_basis(cfg: dict | None):
    pcn = import_pcn()
    cfg = dict(cfg or {"type": "linear"})
    kind = str(cfg.pop("type", "linear")).lower()
    classes = {"bspline": "BsplineBasisFunction", "linear": "LinearBasisFunction",
               "polynomial": "PolynomialBasisFunction"}
    if kind not in classes:
        raise RecipeError(f"basis.type must be one of {sorted(classes)}")
    cls = getattr(pcn, classes[kind], None)
    if cls is None:
        raise RecipeError(f"installed PCNtoolkit has no {classes[kind]}")
    cfg.setdefault("basis_column", 0)
    return cls(**cfg)


def _hbr_likelihood(recipe: dict):
    pcn = import_pcn()
    mp = pcn.make_prior
    h = recipe.get("hbr", {})
    kind = str(h.get("likelihood", "Normal")).lower()
    basis = lambda: make_basis(recipe["basis"])  # noqa: E731  (a fresh object per parameter)

    def rand(mu, sig):
        return mp(random=True, mu=mp(dist_name="Normal", dist_params=mu),
                  sigma=mp(dist_name="Normal", dist_params=sig, mapping="softplus", mapping_params=(0.0, 3.0)))

    def linear(slope_sd, random_intercept, random_slope, intercept_mu=(0.0, 1.0), intercept_fixed=None, **extra):
        kw = dict(linear=True, basis_function=basis(), **extra)
        kw["slope"] = rand((0.0, slope_sd), (0.0, 1.0)) if random_slope else \
            mp(dist_name="Normal", dist_params=(0.0, slope_sd))
        if random_intercept:
            kw["intercept"] = rand(intercept_mu, (0.0, 1.0))
        elif intercept_fixed is not None:
            kw["intercept"] = mp(dist_name="Normal", dist_params=intercept_fixed)
        return mp(**kw)

    if kind == "beta":
        def shape():
            return mp(linear=True, slope=mp(dist_name="Normal", dist_params=(0.0, 10.0)),
                      intercept=mp(random=True, mu=mp(dist_name="Normal", dist_params=(10.0, 3.0)),
                                   sigma=mp(dist_name="Normal", dist_params=(0.0, 3.0), mapping="softplus",
                                            mapping_params=(0.0, 3.0))),
                      mapping="softplus", mapping_params=(0.0, 3.0), basis_function=basis())
        return pcn.BetaLikelihood(shape(), shape())

    mu = linear(5.0 if kind == "normal" else 10.0, h.get("random_intercept_mu", True),
                h.get("random_slope_mu", False))
    if h.get("linear_sigma", True):
        sigma = linear(1.0 if kind == "normal" else 2.0, h.get("random_intercept_sigma", False),
                       h.get("random_slope_sigma", False), intercept_mu=(1.0, 1.0), intercept_fixed=(1.0, 1.0),
                       mapping="softplus", mapping_params=(0.0, 2.0 if kind == "normal" else 3.0))
    else:
        sigma = mp(dist_name="Normal", dist_params=(1.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0))
    if kind == "normal":
        return pcn.NormalLikelihood(mu, sigma)
    if kind in ("shashb", "shash"):
        epsilon = mp(dist_name="Normal", dist_params=(0.0, 1.0))
        delta = mp(dist_name="Normal", dist_params=(1.0, 1.0), mapping="softplus", mapping_params=(0.0, 3.0, 0.6))
        return pcn.SHASHbLikelihood(mu, sigma, epsilon, delta)
    raise RecipeError("hbr.likelihood must be Normal, SHASHb or Beta")


HBR_SAMPLER_KEYS = ("draws", "tune", "chains", "cores", "nuts_sampler", "init")


def make_template(recipe: dict):
    pcn = import_pcn()
    recipe = normalise_recipe(recipe)
    if recipe["algorithm"] == "blr":
        cfg = dict(recipe.get("blr", {}))
        kw = dict(name="template", basis_function_mean=make_basis(recipe["basis"]), **cfg)
        if cfg.get("heteroskedastic"):
            kw["basis_function_var"] = make_basis(recipe.get("basis_var") or recipe["basis"])
        return call_strict(pcn.BLR, **kw)
    h = recipe.get("hbr", {})
    kw = {k: h[k] for k in HBR_SAMPLER_KEYS if k in h}
    return call_strict(pcn.HBR, name="template", progressbar=False, likelihood=_hbr_likelihood(recipe), **kw)


def make_model(recipe: dict, save_dir: str | Path):
    pcn = import_pcn()
    recipe = normalise_recipe(recipe)
    kw = dict(template_regression_model=make_template(recipe), savemodel=True, evaluate_model=True,
              saveresults=True, saveplots=bool(recipe.get("saveplots", False)), save_dir=str(save_dir),
              inscaler=recipe["inscaler"], outscaler=recipe["outscaler"])
    if recipe.get("y_transform"):
        kw["y_transform"] = recipe["y_transform"]
    return call_strict(pcn.NormativeModel, **kw)


def load_model(path: str | Path):
    pcn = import_pcn()
    p = Path(path)
    if not (p / "model" / "normative_model.json").is_file():
        raise SystemExit(f"ERROR: {p} is not a saved PCNtoolkit model (no model/normative_model.json)")
    return pcn.NormativeModel.load(str(p))


def fitted_response_vars(model) -> list[str]:
    out = []
    for rv in list(getattr(model, "response_vars", []) or []):
        try:
            if getattr(model[rv], "is_fitted", True):
                out.append(rv)
        except Exception:
            pass
    return out


# --------------------------------------------------------------------------
# what the file system says (never the scheduler's or the agent's word)
# --------------------------------------------------------------------------
def results_columns(save_dir: str | Path, name: str = "test") -> set[str]:
    import pandas as pd
    p = Path(save_dir) / "results" / f"Z_{name}.csv"
    if not p.is_file():
        return set()
    try:
        return set(pd.read_csv(p, nrows=0).columns) - {"observations", "subject_ids"}
    except Exception:
        return set()


def feature_done(save_dir: str | Path, rv: str, z_cols: set[str] | None = None) -> bool:
    """Success sentinel for one response variable: saved model AND test z-scores exist."""
    save_dir = Path(save_dir)
    if z_cols is None:
        z_cols = results_columns(save_dir, "test")
    return (save_dir / "model" / rv / "regression_model.json").is_file() and rv in z_cols
