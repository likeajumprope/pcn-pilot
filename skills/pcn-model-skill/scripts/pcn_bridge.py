"""The only place where PCN-Pilot touches the PCNtoolkit API (v1.x).

Keeping every toolkit call in one file means a future API change is fixed in
one place. Verified against an installed PCNtoolkit 1.3.0 (NormData /
NormativeModel / BLR / HBR / Runner): every call below was run, and every file
read by the skills was written, by that version. Synced into each skill by
tools/sync_common.py.

Design rule: be strict. If a recipe asks for a setting that the installed
PCNtoolkit does not accept, stop with a clear message. A model must never be
fitted with a silently different configuration from the one that was approved.

What 1.3.0 does differently from its documentation, and what this file does about it
(each one is exercised by selftest/run_selftest.py in real mode):

  BLR transfer without a warp   BLR.transfer only defines its target when the model is warped;
                                an unwarped reference raises UnboundLocalError (all 1.0-1.3.0).
                                -> refused at plan time (run_model.py check R7); use extend.
  extend                        NormativeModel.extend pools data with NormData.merge, which keeps
                                the first dataset's batch-effect registry, so any new site raises
                                KeyError (1.2.0-1.3.0). It also inverts y_transform twice.
                                -> extend_fit_data() does the same three steps with public calls:
                                synthesize from the reference, pool, refit a model built from the
                                reference's template. The pooled fit then runs like any other fit.
  y_transform                   z-scores are right, but predict() inverts the transform once per
                                compute step, so centiles come out as exp(exp(...)) = inf.
                                -> refused at plan time on affected versions.
  plots in transfer             transfer() hard-codes saveplots=True and saves it in the model.
                                A reloaded multi-variable model then fails in plot_centiles when it
                                predicts a subset. -> load_model(..., own=True) sets saveplots from
                                the recipe; PCNtoolkit's own plots during transfer cannot be turned off.
  save_dir of a loaded model    load() restores the absolute save_dir stored at fit time; after a
                                project folder is moved, predict() would write to the old place.
                                -> load_model(..., own=True) points it at the folder it was loaded from.
  Runner job status             check_jobs_status() returns (running, finished, failed), and
                                load_from_state() already consumed it. -> scheduler_view() reads the
                                three attributes load_from_state() sets.
  Runner on Torque              the job writer reads Runner.random_sleep_scale, which the constructor
                                no longer sets (1.1-1.3.0) -> set on the instance before submitting;
                                the value is passed to the job script and ignored there.
  **kwargs catch-alls           basis functions and transfer() swallow unknown names -> call_strict()
                                and TRANSFER_KEYS refuse them.
  result files                  merged on write under a file lock; columns ending in "_old" are
                                dropped by that merge -> pcn-data-skill refuses such variable names.
  idata.nc (HBR)                only the posterior group is saved: R-hat and ESS can be recomputed,
                                divergences cannot -> sampler_diagnostics() reads them from the model
                                in memory right after a local fit (run_model.py stores them in the
                                status file); for cluster jobs pcn-qc-skill reports the check as SKIPPED.
  BLR optimiser                 n_iter and tol are read by "cg" only, and "cg" is silently replaced by
                                L-BFGS-B when the model has a warp or a variance model -> that
                                combination is refused; the router no longer writes n_iter / tol.
"""
from __future__ import annotations

import copy
import inspect
import re
import sys
import zlib
from pathlib import Path
from typing import Any

ID, GROUP, DUMMY_BATCH = "subject_id", "pcn_group", "pcn_all"
VERIFIED = (1, 3, 0)                # the PCNtoolkit release these calls were run against
EXTEND_FIT_NAME = "extend_fit"      # name of the pooled (local + synthetic) table an extend run is fitted on
SYNTH_ID_PREFIX = "synth_"
RESERVED_RESULT_COLUMNS = ("observations", "subject_ids", "centile", "statistic")


class RecipeError(ValueError):
    pass


def import_pcn():
    try:
        import pcntoolkit
    except ImportError as e:
        raise SystemExit(
            "ERROR: pcntoolkit is not importable in this interpreter "
            f"({sys.executable}).\nSet PCN_PYTHON in pipeline.env to the interpreter of the environment "
            "where PCNtoolkit 1.3.0 is installed (pip install pcntoolkit==1.3.0; Python 3.11 or 3.12).\n"
            f"Import error: {e}")
    for needed in ("NormData", "NormativeModel", "BLR", "HBR"):
        if not hasattr(pcntoolkit, needed):
            raise SystemExit(
                f"ERROR: this pcntoolkit has no '{needed}'. PCN-Pilot targets the v1.x API; "
                "legacy 0.x installs (estimate()/predict() functions) are not supported.")
    return pcntoolkit


def toolkit_version() -> tuple[int, ...] | None:
    """Installed version as a tuple of ints, e.g. (1, 3, 0). None if it cannot be read."""
    pcn = import_pcn()
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", str(getattr(pcn, "__version__", "")))
    return tuple(int(x or 0) for x in m.groups()) if m else None


def affected() -> bool:
    """True when the installed PCNtoolkit is a release in which the issues listed above were observed.

    Newer releases are not assumed to be broken: there the toolkit gets the benefit of the doubt and
    the real-mode self-test says whether the behaviour changed.
    """
    v = toolkit_version()
    return v is not None and v <= VERIFIED


def version_note() -> str | None:
    v = toolkit_version()
    if v is None or v == VERIFIED:
        return None
    return (f"installed PCNtoolkit is {'.'.join(map(str, v))}; PCN-Pilot was verified against "
            f"{'.'.join(map(str, VERIFIED))}. Run selftest/run_selftest.py with this interpreter before relying on it.")


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


def unstorable_names(names) -> dict[str, str]:
    """Response-variable names PCNtoolkit's result files cannot hold, with the reason.

    Pure string logic (no toolkit import) so the data stage can use it anywhere.
    """
    bad = {}
    for n in map(str, names):
        if n.endswith("_old"):
            bad[n] = ("ends in '_old': PCNtoolkit merges result files with an '_old' suffix for the previous "
                      "copy and then drops every column that ends in it, so this variable's z-scores vanish "
                      "as soon as another variable is written")
        elif n in RESERVED_RESULT_COLUMNS:
            bad[n] = "is the name of an index column in PCNtoolkit's result files"
        elif "/" in n or "\\" in n:
            bad[n] = "contains a path separator: PCNtoolkit saves each model in a folder named after the variable"
        elif n != n.strip() or not n:
            bad[n] = "is empty or has leading/trailing whitespace"
    return bad


# --------------------------------------------------------------------------
# recipe -> model
# --------------------------------------------------------------------------
RECIPE_DEFAULTS = {
    "algorithm": "blr", "mode": "fit_predict", "reference_model": None,
    "basis": {"type": "bspline", "nknots": 5, "degree": 3, "basis_column": 0},
    "inscaler": "standardize", "outscaler": "standardize", "y_transform": None, "saveplots": False,
    "seed": 20260101,
}
RECIPE_KEYS = set(RECIPE_DEFAULTS) | {"blr", "hbr", "basis_var", "transfer_kwargs", "n_synth_samples"}
DOC_KEYS = ("rationale", "alternatives", "traits")          # documentation only, never reach the toolkit
SCALERS = ("standardize", "minmax", "robminmax", "none", "id")
Y_TRANSFORMS = ("log", "log1p")

HBR_SAMPLER_KEYS = ("draws", "tune", "chains", "cores", "nuts_sampler", "init")
HBR_MODEL_KEYS = ("likelihood", "random_intercept_mu", "random_slope_mu", "linear_sigma",
                  "random_intercept_sigma", "random_slope_sigma")
# What <RegressionModel>.transfer(**kwargs) actually reads in 1.3.0. Everything else is swallowed.
TRANSFER_KEYS = {"BLR": (), "HBR": ("freedom", "draws", "tune", "cores", "chains", "nuts_sampler")}

Y_TRANSFORM_MSG = (
    "y_transform is refused with PCNtoolkit <= 1.3.0. Its z-scores are correct, but predict() applies the "
    "inverse transform once per compute step, so the saved centiles are exponentiated repeatedly (they come out "
    "as inf) and extend inverts the data twice. Model the skew instead: a BLR warp "
    "(\"warp_name\": \"warpsinharcsinh\" or \"warplog\") or an HBR SHASHb likelihood.")


def normalise_recipe(recipe: dict) -> dict:
    unknown = sorted(k for k in recipe if k not in RECIPE_KEYS and k not in DOC_KEYS)
    if unknown:
        raise RecipeError(f"recipe has keys PCN-Pilot does not know: {unknown}. Known: {sorted(RECIPE_KEYS)}. "
                          "A misspelt key would otherwise be ignored and the default used in its place.")
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
    for key in ("inscaler", "outscaler"):
        if r[key] not in SCALERS:
            raise RecipeError(f"recipe.{key} must be one of {list(SCALERS)}")
    if r.get("y_transform") is not None:
        if r["y_transform"] not in Y_TRANSFORMS:
            raise RecipeError(f"recipe.y_transform must be null or one of {list(Y_TRANSFORMS)}")
        if affected():
            raise RecipeError(Y_TRANSFORM_MSG)
    if not isinstance(r["seed"], int) or isinstance(r["seed"], bool) or r["seed"] < 0:
        raise RecipeError("recipe.seed must be a non-negative integer")
    n = r.get("n_synth_samples")
    if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
        raise RecipeError("recipe.n_synth_samples must be null or a positive integer")
    if r["mode"] != "transfer_predict" and r.get("transfer_kwargs"):
        raise RecipeError("recipe.transfer_kwargs only applies to mode transfer_predict")
    if r["mode"] != "extend_predict" and r.get("n_synth_samples"):
        raise RecipeError("recipe.n_synth_samples only applies to mode extend_predict")
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
    return call_strict(cls, **cfg)            # the constructors take **kwargs: a typo would be swallowed


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


def make_template(recipe: dict):
    pcn = import_pcn()
    recipe = normalise_recipe(recipe)
    if recipe["algorithm"] == "blr":
        cfg = dict(recipe.get("blr", {}))
        kw = dict(name="template", basis_function_mean=make_basis(recipe["basis"]), **cfg)
        if cfg.get("heteroskedastic"):
            kw["basis_function_var"] = make_basis(recipe.get("basis_var") or recipe["basis"])
        if str(cfg.get("optimizer", "")).lower() == "cg" and (
                cfg.get("warp_name") or cfg.get("heteroskedastic") or cfg.get("fixed_effect_var")):
            raise RecipeError("blr.optimizer 'cg' is refused together with a warp or a variance model "
                              "(heteroskedastic, fixed_effect_var): PCNtoolkit would switch to L-BFGS-B by itself "
                              "and fit a different configuration from the one written in the recipe.")
        return call_strict(pcn.BLR, **kw)
    h = recipe.get("hbr", {})
    unknown = sorted(k for k in h if k not in HBR_SAMPLER_KEYS + HBR_MODEL_KEYS)
    if unknown:
        raise RecipeError(f"recipe.hbr has keys PCN-Pilot does not know: {unknown}. "
                          f"Known: {sorted(HBR_SAMPLER_KEYS + HBR_MODEL_KEYS)}. They were NOT dropped silently.")
    if str(h.get("likelihood", "Normal")).lower() == "beta" and recipe["outscaler"] in ("minmax", "robminmax"):
        raise RecipeError(
            "a Beta likelihood with outscaler 'minmax' is refused: the scaler maps the training minimum and "
            "maximum to the edges of (0, 1), so every held-out value beyond them gets z = +/-inf. Responses "
            "that already lie strictly in (0, 1) need \"outscaler\": \"none\".")
    kw = {k: h[k] for k in HBR_SAMPLER_KEYS if k in h}
    return call_strict(pcn.HBR, name="template", progressbar=False, likelihood=_hbr_likelihood(recipe), **kw)


def _normative_model(template, save_dir: str | Path, inscaler: str, outscaler: str, y_transform, saveplots: bool):
    pcn = import_pcn()
    kw = dict(template_regression_model=template, savemodel=True, evaluate_model=True, saveresults=True,
              saveplots=bool(saveplots), save_dir=str(save_dir), inscaler=inscaler, outscaler=outscaler)
    if y_transform:
        kw["y_transform"] = y_transform
    return call_strict(pcn.NormativeModel, **kw)


def make_model(recipe: dict, save_dir: str | Path):
    recipe = normalise_recipe(recipe)
    return _normative_model(make_template(recipe), save_dir, recipe["inscaler"], recipe["outscaler"],
                            recipe.get("y_transform"), recipe.get("saveplots", False))


def load_model(path: str | Path, own: bool = False, saveplots: bool = False):
    """Load a saved model.

    own=False  a reference model: left exactly as saved, nothing is ever written into its folder.
    own=True   a PCN-Pilot run folder that later predictions write into: results go to the folder the
               model was loaded from (not the absolute path stored at fit time), and PCNtoolkit's own
               plotting follows `saveplots` (a transferred model is always saved with saveplots=True).
    """
    pcn = import_pcn()
    p = Path(path)
    if not (p / "model" / "normative_model.json").is_file():
        raise SystemExit(f"ERROR: {p} is not a saved PCNtoolkit model (no model/normative_model.json)")
    model = pcn.NormativeModel.load(str(p))
    if own:
        model.saveplots = bool(saveplots)
        model.save_dir = str(p)
    return model


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
# reference models: what they are, and which adaptation the installed toolkit can do
# --------------------------------------------------------------------------
def reference_info(model) -> dict:
    """Algorithm, warp and transform of a loaded reference model."""
    tmpl = model.template_regression_model
    return {"kind": type(tmpl).__name__,                              # "BLR" or "HBR"
            "warp": getattr(tmpl, "warp_name", None) or None,
            "y_transform": getattr(model, "y_transform", None) or None,
            "unique_batch_effects": {k: [str(x) for x in v] for k, v in (model.unique_batch_effects or {}).items()},
            "covariates": list(model.covariates or []),
            "can_synthesize": bool(getattr(model, "batch_effect_counts", None)
                                   and getattr(model, "batch_effect_covariate_ranges", None)
                                   and getattr(model, "covariate_ranges", None))}


def route_problems(info: dict, mode: str) -> list[str]:
    """Reasons the installed PCNtoolkit cannot run `mode` from this reference model (empty = it can)."""
    out = []
    if mode == "extend_predict" and not info["can_synthesize"]:
        out.append("the reference model does not store the batch-effect counts and per-batch covariate ranges "
                   "that synthesising data from it needs (it was saved by an older PCNtoolkit); extend is not "
                   "possible from this model")
    if not affected():
        return out
    if mode == "transfer_predict" and info["kind"] == "BLR" and not info["warp"]:
        out.append("the reference is a BLR model without a warp, and PCNtoolkit <= 1.3.0 cannot transfer it "
                   "(BLR.transfer only computes its target for warped models and fails with UnboundLocalError). "
                   "Use extend, which works for every BLR and HBR reference, or a reference fitted with a warp")
    if info["y_transform"]:
        out.append(f"the reference model was fitted with y_transform='{info['y_transform']}', which the adapted "
                   "model inherits. " + Y_TRANSFORM_MSG)
    return out


def check_transfer_kwargs(info: dict, kwargs: dict | None) -> dict:
    kwargs = dict(kwargs or {})
    allowed = TRANSFER_KEYS.get(info["kind"], ())
    unknown = sorted(k for k in kwargs if k not in allowed)
    if unknown:
        raise RecipeError(
            f"recipe.transfer_kwargs {unknown} are not read by {info['kind']}.transfer in PCNtoolkit 1.3.0 "
            f"(it reads {list(allowed) or 'no options'}) and would be ignored without a word. Remove them.")
    if info["kind"] == "HBR":
        kwargs["progressbar"] = False         # otherwise inherited from however the reference was saved
    return kwargs


def transfer(ref, train, test, save_dir: str | Path, info: dict, kwargs: dict | None, saveplots: bool = False):
    """ref.transfer_predict with checked options. Returns the adapted model.

    PCNtoolkit builds the adapted model with saveplots=True, so its QQ and centile plots for the
    adaptation and test data are always written to <save_dir>/plots. Later predictions with the
    returned object follow `saveplots`.
    """
    kw = check_transfer_kwargs(info, kwargs)
    model = ref.transfer_predict(train, test, save_dir=str(save_dir), **kw)
    model.saveplots = bool(saveplots)
    return model


def _seed_for(seed: int, names: list[str]) -> int:
    return (int(seed) + zlib.crc32("\x1f".join(names).encode())) % (2 ** 32)


def extend_fit_data(ref, df, manifest: dict, response_vars: list[str], n_synth_samples: int | None, seed: int):
    """The training data of an extend run: the local rows pooled with data synthesised from `ref`.

    These are the steps of NormativeModel.extend, done with public calls because extend itself fails
    on new batch levels in 1.2.0-1.3.0 (see the module docstring):

      1. synthesize covariates, batch labels and responses from the reference model, with the
         covariate range of each of its batch levels (n_synth_samples; default: as many rows as
         the reference was fitted on)
      2. pool them with the local rows
      3. the caller fits a fresh model built by model_like(ref, ...) on the pooled table

    Only the requested response variables are synthesised (a shallow view of the reference restricted
    to them), so a per-variable loop costs N synthesis passes and not N x N. The synthetic rows get
    subject IDs 'synth_000000', ...; numpy's global random state is seeded from `seed` and the variable
    names for the call and restored afterwards, so the same call gives the same rows.
    """
    import numpy as np
    import pandas as pd
    rvs = list(response_vars)
    view = copy.copy(ref)                     # shares the fitted models; only the list of variables differs
    view.response_vars = rvs
    state = np.random.get_state()
    try:
        np.random.seed(_seed_for(seed, rvs))
        synth = view.synthesize(n_samples=n_synth_samples, covariate_range_per_batch_effect=True)
    finally:
        np.random.set_state(state)
    cov, batch = list(manifest["covariates"]), batch_columns(manifest)
    sdf = pd.DataFrame({c: np.asarray(synth.X.sel(covariates=c).values, dtype=float) for c in cov})
    for b in batch:
        sdf[b] = np.asarray(synth.batch_effects.sel(batch_effect_dims=b).values).astype(str)
    for rv in rvs:
        sdf[rv] = np.asarray(synth.Y.sel(response_vars=rv).values, dtype=float)
    if not np.isfinite(sdf[cov + rvs].to_numpy(float)).all():
        raise RuntimeError("the reference model synthesised non-finite values; it cannot be extended as it is")
    sdf[ID] = [f"{SYNTH_ID_PREFIX}{i:06d}" for i in range(len(sdf))]
    cols = [ID] + cov + batch + rvs
    pooled = pd.concat([df[cols], sdf[cols]], ignore_index=True)
    return make_normdata(EXTEND_FIT_NAME, pooled, manifest, rvs)


def model_like(ref, save_dir: str | Path, saveplots: bool = False):
    """A fresh, unfitted model with the reference's template, scalers and transform (as extend builds it)."""
    return _normative_model(copy.deepcopy(ref.template_regression_model), save_dir, ref.inscaler, ref.outscaler,
                            getattr(ref, "y_transform", None), saveplots)


def sampler_diagnostics(model, rv: str) -> dict | None:
    """Divergent transitions of the MCMC run that has just finished, read from the model in memory.

    None for BLR, and for a model loaded from disk: PCNtoolkit saves the posterior draws only.
    """
    try:
        stats = getattr(getattr(model[rv], "idata", None), "sample_stats", None)
        if stats is None or "diverging" not in stats:
            return None
        d = stats["diverging"].values
        return {"n_divergent": int(d.sum()), "n_draws": int(d.size), "divergent_frac": float(d.mean())}
    except Exception:
        return None


def unknown_batch_levels(model, df, manifest: dict) -> dict[str, list[str]]:
    """Batch-effect labels in `df` the model has no parameter for (predict() refuses such data)."""
    known = {k: {str(x) for x in v} for k, v in (model.unique_batch_effects or {}).items()}
    out = {}
    for b in batch_columns(manifest):
        new = sorted(set(df[b].astype(str)) - known.get(b, set()))
        if new:
            out[b] = new
    return out


# --------------------------------------------------------------------------
# cluster execution through PCNtoolkit's Runner
# --------------------------------------------------------------------------
def make_runner(job_type: str, n_batches: int, resources: dict, log_dir: str | Path, temp_dir: str | Path):
    pcn = import_pcn()
    runner = call_strict(
        pcn.Runner, cross_validate=False, parallelize=True, environment=resources["conda_env"],
        job_type=job_type, n_batches=n_batches, time_limit=resources["time_limit"], memory=resources["memory"],
        n_cores=resources["n_cores"], max_retries=resources["max_retries"], preamble=resources["preamble"],
        log_dir=str(log_dir), temp_dir=str(temp_dir))
    if job_type == "torque" and not hasattr(runner, "random_sleep_scale"):
        runner.random_sleep_scale = 0.1       # read by the Torque job writer, never set by the constructor
    return runner


def submit(runner, mode: str, model, train, test, save_dir: str | Path, kwargs: dict | None = None) -> dict:
    """Submit and return at once. `model` is a fresh model (fit, extend) or the reference (transfer)."""
    if mode == "transfer_predict":
        runner.transfer_predict(model, train, test, save_dir=str(save_dir), observe=False, **(kwargs or {}))
    else:
        runner.fit_predict(model, train, test, save_dir=str(save_dir), observe=False)
    state = Path(str(getattr(runner, "unique_temp_dir", "") or "")) / "runner_state.json"
    return {"state_file": str(state) if state.is_file() else "",
            "log_dir": str(getattr(runner, "unique_log_dir", "") or ""),
            "jobs": dict(getattr(runner, "active_jobs", {}) or {})}


def scheduler_view(state_file: str | Path) -> dict:
    """Running / finished / failed jobs of one submission, as PCNtoolkit's Runner sees them."""
    pcn = import_pcn()
    runner = pcn.Runner.load_from_state(str(state_file))     # asks the scheduler and sorts the jobs
    failed = dict(getattr(runner, "failed_jobs", {}) or {})
    return {"running": len(getattr(runner, "active_jobs", {}) or {}),
            "finished": len(getattr(runner, "finished_jobs", {}) or {}),
            "failed": len(failed),
            "failed_detail": {str(k): str(v)[-300:] for k, v in failed.items()}}


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


def results_subject_ids(save_dir: str | Path, name: str) -> list[str] | None:
    """Subject IDs of an existing Z_<name>.csv in observation order (None if there is no such file)."""
    import pandas as pd
    p = Path(save_dir) / "results" / f"Z_{name}.csv"
    if not p.is_file():
        return None
    z = pd.read_csv(p, usecols=lambda c: c in ("observations", "subject_ids"), dtype=str)
    if "subject_ids" not in z.columns:
        return None
    if "observations" in z.columns:
        z = z.sort_values("observations", key=lambda c: pd.to_numeric(c, errors="coerce"))
    return z["subject_ids"].astype(str).tolist()


def feature_done(save_dir: str | Path, rv: str, z_cols: set[str] | None = None) -> bool:
    """Success sentinel for one response variable: saved model AND test z-scores exist."""
    save_dir = Path(save_dir)
    if z_cols is None:
        z_cols = results_columns(save_dir, "test")
    return (save_dir / "model" / rv / "regression_model.json").is_file() and rv in z_cols
