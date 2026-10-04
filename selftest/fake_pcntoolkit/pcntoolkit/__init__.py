"""A stand-in for PCNtoolkit 1.x, used ONLY to test PCN-Pilot's orchestration offline.

It mimics the public surface PCN-Pilot calls (signatures, file layout, merge-on-write
result files, per-chunk Runner behaviour) with a trivial regression underneath.
It is NOT a normative model and must never be used for analysis.

Mirrored from an installed PCNtoolkit 1.3.0, including the places where that release does not do
what its documentation says (pcn_bridge.py lists them), so that fake mode exercises the same
guards as real mode:
  NormData.from_dataframe / .sel / result CSV layouts and their merge-on-write
  NormativeModel(...).fit_predict / .predict / .save / .load / .synthesize
  .transfer_predict   raises UnboundLocalError for a BLR model without a warp; saves saveplots=True
  .extend_predict     raises KeyError when the new data has a batch level the reference lacks
  .predict            asserts that every batch level is known; with saveplots=True it fails when the
                      model holds more response variables than the data
  basis functions     accept and ignore unknown keyword arguments
  save_dir/model/normative_model.json, save_dir/model/<rv>/regression_model.json, save_dir/results/*.csv
  Runner(...)         checks <environment>/bin/python; check_jobs_status() -> (running, finished, failed);
                      load_from_state() sets active_jobs / finished_jobs / failed_jobs; the Torque job
                      writer needs Runner.random_sleep_scale, which the constructor does not set

Fault injection for tests: PCN_FAKE_FAIL="rv1,rv2" makes fitting those variables raise.
"""
from __future__ import annotations

import copy
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

__version__ = "1.3.0+fake"
CENTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


# ---------------------------------------------------------------- building blocks
class _Basis:
    def __init__(self, basis_column: int = 0, **kw):
        self.basis_column = basis_column
        self.kw = kw

    def to_dict(self):
        return {"type": type(self).__name__, "basis_column": self.basis_column, **self.kw}


class LinearBasisFunction(_Basis):
    def __init__(self, basis_column: int = 0, **kwargs):       # like PCNtoolkit: unknown names are swallowed
        super().__init__(basis_column)


class PolynomialBasisFunction(_Basis):
    def __init__(self, basis_column: int = 0, degree: int = 3, **kwargs):
        super().__init__(basis_column, degree=degree)


class BsplineBasisFunction(_Basis):
    def __init__(self, basis_column: int = 0, degree: int = 3, nknots: int = 5, left_expand: float = 0.05,
                 right_expand: float = 0.05, knot_method: str = "uniform", knots=None, **kwargs):
        super().__init__(basis_column, degree=degree, nknots=nknots)


def make_prior(name: str = "theta", **kwargs):
    return {"name": name, **{k: (v if not isinstance(v, _Basis) else v.to_dict()) for k, v in kwargs.items()}}


class _Likelihood:
    def __init__(self, *params):
        self.params = params
        self.basis = next((p.get("basis_function") for p in params if isinstance(p, dict) and p.get("basis_function")), None)


class NormalLikelihood(_Likelihood):
    def __init__(self, mu, sigma):
        super().__init__(mu, sigma)


class SHASHbLikelihood(_Likelihood):
    def __init__(self, mu, sigma, epsilon, delta):
        super().__init__(mu, sigma, epsilon, delta)


class BetaLikelihood(_Likelihood):
    def __init__(self, alpha, beta):
        super().__init__(alpha, beta)


class BLR:
    def __init__(self, name: str = "template", n_iter: int = 100, tol: float = 1e-3, ard: bool = False,
                 optimizer: str = "l-bfgs-b", l_bfgs_b_l: float = 0.1, l_bfgs_b_epsilon: float = 0.1,
                 l_bfgs_b_norm: str = "l2", fixed_effect: bool = False, fixed_effect_slope: bool = False,
                 fixed_effect_slope_indices=None, heteroskedastic: bool = False, fixed_effect_var: bool = False,
                 fixed_effect_var_slope: bool = False, fixed_effect_var_slope_indices=None,
                 warp_name=None, warp_reparam: bool = False, basis_function_mean=None, basis_function_var=None,
                 hyp0=None, is_fitted: bool = False, is_from_dict: bool = False):
        self.name, self.warp_name = name, warp_name
        self.cfg = dict(kind="blr", fixed_effect=fixed_effect, heteroskedastic=heteroskedastic,
                        gaussianise=bool(warp_name), fixed_effect_var=fixed_effect_var,
                        degree=_degree(basis_function_mean), optimizer=optimizer, warp_name=warp_name)


class HBR:
    def __init__(self, name: str = "template", draws: int = 1500, tune: int = 500, cores: int = 4, chains: int = 4,
                 nuts_sampler: str = "nutpie", init: str = "jitter+adapt_diag", progressbar: bool = True,
                 likelihood=None, is_fitted: bool = False, is_from_dict: bool = False):
        self.name = name
        lik = likelihood or NormalLikelihood({}, {})
        self.cfg = dict(kind="hbr", fixed_effect=True, heteroskedastic=True,
                        gaussianise=isinstance(lik, SHASHbLikelihood), fixed_effect_var=False,
                        degree=3 if (lik.basis or {}).get("type") == "BsplineBasisFunction" else 1, tune=tune)


def _degree(basis) -> int:
    if isinstance(basis, BsplineBasisFunction):
        return 2 + max(1, (basis.kw.get("nknots", 5) - 3) // 2)
    if isinstance(basis, PolynomialBasisFunction):
        return basis.kw.get("degree", 3)
    return 1


# ---------------------------------------------------------------- NormData
class NormData:
    def __init__(self, name, df, covariates, batch_effects, response_vars, ids):
        self.name, self.df = name, df
        self.covariates, self.batch_effect_dims, self.response_vars = list(covariates), list(batch_effects), list(response_vars)
        self.ids = ids
        self.Z = self.centiles = self.statistics = None

    @classmethod
    def from_dataframe(cls, name, dataframe, covariates=None, batch_effects=None, response_vars=None,
                       subject_ids=None, remove_Nan=False, remove_outliers=False, z_threshold=3.0, attrs=None):
        df = dataframe.reset_index(drop=True).copy()
        ids = df[subject_ids].astype(str).to_numpy() if subject_ids else np.arange(len(df))
        for b in batch_effects or []:
            df[b] = df[b].astype(str)
        if df[list(covariates) + list(response_vars)].isna().any().any():
            raise ValueError("NaN in data")
        return cls(name, df, covariates or [], batch_effects or [], response_vars or [], ids)

    def sel(self, indexers=None, **kw):
        rvs = (indexers or kw)["response_vars"]
        rvs = [rvs] if isinstance(rvs, str) else list(rvs)
        return NormData(self.name, self.df, self.covariates, self.batch_effect_dims, rvs, self.ids)

    def chunk(self, n_chunks):
        for i in range(n_chunks):
            yield self.sel({"response_vars": self.response_vars[i::n_chunks]})

    @property
    def unique_batch_effects(self):
        return {b: sorted(self.df[b].unique()) for b in self.batch_effect_dims}

    @property
    def batch_effect_counts(self):
        return {b: {str(k): int(v) for k, v in self.df[b].value_counts(sort=False).items()} for b in self.batch_effect_dims}

    @property
    def batch_effect_covariate_ranges(self):
        return {b: {str(lv): {c: {"min": float(g[c].min()), "max": float(g[c].max())} for c in self.covariates}
                    for lv, g in self.df.groupby(b)} for b in self.batch_effect_dims}

    def check_compatibility(self, other):
        return self.covariates == other.covariates and self.batch_effect_dims == other.batch_effect_dims

    # result files: same names, columns and merge-on-write behaviour as PCNtoolkit
    def _merge_write(self, path: Path, new: pd.DataFrame, keys: list[str]) -> None:
        if path.exists() and path.stat().st_size > 0:
            old = pd.read_csv(path)
            for k in keys:
                old[k] = old[k].astype(str)
            old = old.set_index(keys)
            merged = old.merge(new, on=keys, how="outer", suffixes=("_old", ""))
            merged = merged.loc[:, ~merged.columns.str.endswith("_old")]
        else:
            merged = new
        merged = merged.sort_index(level=0, key=lambda i: pd.to_numeric(i, errors="coerce"), kind="stable")
        merged.to_csv(path)

    def save_results(self, save_dir):
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        obs = pd.Index(np.arange(len(self.df)).astype(str), name="observations")
        z = pd.DataFrame(self.Z, index=obs)
        z.insert(0, "subject_ids", self.ids)
        self._merge_write(d / f"Z_{self.name}.csv", z, ["observations"])
        frames = []
        for c in CENTILES:
            f = pd.DataFrame({rv: self.centiles[rv][c] for rv in self.response_vars})
            f.insert(0, "subject_ids", self.ids)
            f["observations"], f["centile"] = obs.to_numpy(), str(c)
            frames.append(f)
        cen = pd.concat(frames).set_index(["observations", "centile"])
        self._merge_write(d / f"centiles_{self.name}.csv", cen, ["observations", "centile"])
        st = pd.DataFrame(self.statistics)
        st.index.name = "statistic"
        p = d / f"statistics_{self.name}.csv"
        if p.exists() and p.stat().st_size > 0:
            old = pd.read_csv(p, index_col=0)
            st = old.merge(st, on="statistic", how="outer", suffixes=("_old", ""))
            st = st.loc[:, ~st.columns.str.endswith("_old")]
        st.to_csv(p)


# ---------------------------------------------------------------- the toy regression
class _Reg:
    def __init__(self, cfg):
        self.cfg, self.is_fitted = dict(cfg), False
        self.transfered = False

    def _design(self, df, cov, batch):
        x = (df[cov[0]].to_numpy(float) - self.x_mu) / self.x_sd
        cols = [np.ones_like(x)] + [x ** k for k in range(1, self.cfg["degree"] + 1)]
        if self.cfg["fixed_effect"]:
            for b, levels in self.levels.items():
                for lv in levels[1:]:
                    cols.append((df[b].to_numpy() == lv).astype(float))
        return np.column_stack(cols), x

    def fit(self, df, rv, cov, batch):
        if rv in [s for s in os.environ.get("PCN_FAKE_FAIL", "").split(",") if s]:
            raise RuntimeError(f"injected failure for {rv}")
        x = df[cov[0]].to_numpy(float)
        self.x_mu, self.x_sd = float(x.mean()), float(x.std() or 1.0)
        self.levels = {b: sorted(df[b].unique()) for b in batch}
        X, xs = self._design(df, cov, batch)
        y = df[rv].to_numpy(float)
        self.beta = np.linalg.lstsq(X, y, rcond=None)[0]
        e = y - X @ self.beta
        if self.cfg["heteroskedastic"]:
            self.s = np.polyfit(xs, np.abs(e) * 1.2533, 1)
        else:
            self.s = np.array([0.0, e.std()])
        r = e / self._sd(xs)
        self.q = np.quantile(r, np.linspace(0.001, 0.999, 201)) if self.cfg["gaussianise"] else None
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())
        self.shift = {}
        self.is_fitted = True

    def _sd(self, xs):
        return np.maximum(np.polyval(self.s, xs), 1e-6 + 0.05 * abs(self.s[1]))

    def predict(self, df, rv, cov, batch):
        X, xs = self._design(df, cov, batch)
        yhat = X @ self.beta
        if self.shift:
            site = df[self.shift["dim"]].to_numpy()
            yhat = yhat + np.array([self.shift["by"].get(s, self.shift["all"]) for s in site])
        sd = self._sd(xs)
        out = {"yhat": yhat, "sd": sd}
        if rv in df.columns:
            r = (df[rv].to_numpy(float) - yhat) / sd
            if self.q is not None:
                p = np.interp(r, self.q, np.linspace(0.001, 0.999, 201), left=0.0005, right=0.9995)
                r = stats.norm.ppf(p)
            out["z"] = r
        cen = {}
        for c in CENTILES:
            zq = stats.norm.ppf(c) if self.q is None else np.interp(c, np.linspace(0.001, 0.999, 201), self.q)
            cen[c] = yhat + zq * sd
        out["centiles"] = cen
        return out

    def to_dict(self):
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d["nlZ"] = 123.4
        return d

    @classmethod
    def from_dict(cls, d):
        r = cls(d["cfg"])
        for k, v in d.items():
            if k != "nlZ":
                setattr(r, k, np.array(v) if k in ("beta", "s", "q") and v is not None else v)
        return r


class NormativeModel:
    def __init__(self, template_regression_model, savemodel: bool = True, evaluate_model: bool = True,
                 saveresults: bool = True, saveplots: bool = True, save_dir=None, inscaler: str = "standardize",
                 outscaler: str = "standardize", y_transform=None, name=None):
        self.template_regression_model = template_regression_model
        self.savemodel, self.evaluate_model, self.saveresults, self.saveplots = savemodel, evaluate_model, saveresults, saveplots
        self.save_dir, self.inscaler, self.outscaler, self.name = save_dir, inscaler, outscaler, name
        self.y_transform = y_transform
        self.regression_models: dict[str, _Reg] = {}
        self.response_vars: list[str] = []      # a plain attribute, as in PCNtoolkit
        self.covariates, self.unique_batch_effects, self.covariate_ranges = [], {}, {}
        self.batch_effect_counts, self.batch_effect_covariate_ranges = {}, {}
        self.is_fitted = False

    def __getitem__(self, rv):
        return self.regression_models[rv]

    def set_save_dir(self, d):
        self.save_dir = d

    def _register(self, data: NormData):
        self.covariates = list(data.covariates)
        self.unique_batch_effects = data.unique_batch_effects
        self.batch_effect_counts = data.batch_effect_counts
        self.batch_effect_covariate_ranges = data.batch_effect_covariate_ranges
        self.covariate_ranges = {c: {"min": float(data.df[c].min()), "max": float(data.df[c].max())}
                                 for c in data.covariates}

    def fit(self, data: NormData):
        self._register(data)
        self.response_vars = list(data.response_vars)
        for rv in data.response_vars:          # like PCNtoolkit: no per-variable try/except
            reg = _Reg(self.template_regression_model.cfg)
            reg.fit(data.df, rv, data.covariates, data.batch_effect_dims)
            self.regression_models[rv] = reg
        self.is_fitted = True
        if self.savemodel:
            self.save()
        self.predict(data)                    # PCNtoolkit also scores the fit data

    def synthesize(self, data=None, n_samples=None, covariate_range_per_batch_effect=False):
        """Draw batch labels by their training frequency, covariates within the ranges seen for those
        labels, and responses from the fitted models. Uses numpy's global random state, like PCNtoolkit."""
        assert self.is_fitted
        dims = list(self.unique_batch_effects)
        n = n_samples or sum(self.batch_effect_counts[dims[0]].values())
        cols = {}
        for b in dims:
            levels = list(self.batch_effect_counts[b])
            w = np.array([self.batch_effect_counts[b][lv] for lv in levels], float)
            cols[b] = np.random.choice(levels, size=n, p=w / w.sum())
        for c in self.covariates:
            lo = np.full(n, self.covariate_ranges[c]["min"])
            hi = np.full(n, self.covariate_ranges[c]["max"])
            if covariate_range_per_batch_effect:
                for b in dims:
                    r = self.batch_effect_covariate_ranges[b]
                    lo = np.maximum(lo, [r[lv][c]["min"] for lv in cols[b]])
                    hi = np.minimum(hi, [r[lv][c]["max"] for lv in cols[b]])
            cols[c] = np.random.uniform(lo, np.maximum(hi, lo))
        df = pd.DataFrame(cols)
        ys = {}
        for rv in self.response_vars:
            out = self.regression_models[rv].predict(df, rv, self.covariates, dims)
            ys[rv] = out["yhat"] + out["sd"] * np.random.randn(n)

        def arr(names):
            src = ys if names is self.response_vars else cols
            return types.SimpleNamespace(sel=lambda **kw: types.SimpleNamespace(values=np.asarray(src[next(iter(kw.values()))])))
        return types.SimpleNamespace(X=arr(self.covariates), batch_effects=arr(dims), Y=arr(self.response_vars),
                                     name="synthesized")

    def predict(self, data: NormData):
        unknown = {b: [u for u in lv if u not in self.unique_batch_effects.get(b, [])]
                   for b, lv in data.unique_batch_effects.items()}
        assert not any(unknown.values()), "Data is not compatible with the model!"
        if self.saveplots and set(self.response_vars) - set(data.response_vars):
            raise KeyError("not all values found in index 'response_vars'")      # plot_centiles in 1.3.0
        data.Z, data.centiles, data.statistics = {}, {}, {}
        for rv in data.response_vars:
            out = self.regression_models[rv].predict(data.df, rv, data.covariates, data.batch_effect_dims)
            y, z = data.df[rv].to_numpy(float), out["z"]
            data.Z[rv], data.centiles[rv] = z, out["centiles"]
            var = y.var() or 1.0
            mse = float(np.mean((y - out["yhat"]) ** 2))
            nll = 0.5 * np.log(2 * np.pi * out["sd"] ** 2) + 0.5 * z ** 2
            base = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y - y.mean()) ** 2 / var
            rho, p = stats.spearmanr(y, out["yhat"])
            cdf = stats.norm.cdf(z)
            data.statistics[rv] = {
                "Rho": float(rho), "Rho_p": float(p), "R2": 1 - mse / var, "RMSE": mse ** 0.5, "SMSE": mse / var,
                "MSLL": float(np.mean(nll - base)), "MLL": float(np.mean(nll)),
                "ShapiroW": float(stats.shapiro(z)[0]), "MACE": float(np.mean([abs((cdf < c).mean() - c) for c in CENTILES])),
                "MAPE": float(np.mean(np.abs((y - out["yhat"]) / np.where(y == 0, 1, y)))),
                "EXPV": 1 - float(np.var(y - out["yhat"])) / var,
                "Skewness": float(stats.skew(z)), "Kurtosis": float(stats.kurtosis(z))}
        if self.saveresults:
            data.save_results(os.path.join(self.save_dir, "results"))
        return data

    def fit_predict(self, fit_data, predict_data):
        self.fit(fit_data)
        self.predict(predict_data)
        return predict_data

    def save(self, path=None):
        root = Path(path or self.save_dir) / "model"
        root.mkdir(parents=True, exist_ok=True)
        cfg = self.template_regression_model.cfg
        meta = {"name": self.name, "save_dir": str(self.save_dir), "ptk_version": __version__,
                "covariates": self.covariates, "unique_batch_effects": self.unique_batch_effects,
                "covariate_ranges": self.covariate_ranges, "batch_effect_counts": self.batch_effect_counts,
                "batch_effect_covariate_ranges": self.batch_effect_covariate_ranges,
                "is_fitted": True, "inscaler": self.inscaler, "outscaler": self.outscaler,
                "saveplots": self.saveplots, "y_transform": self.y_transform, "template": cfg,
                # the two fields choose_recipe.py reads from a real model
                "template_regression_model": {"type": cfg["kind"].upper(), "warp_name": cfg.get("warp_name")}}
        (root / "normative_model.json").write_text(json.dumps(meta, indent=1))
        for rv, reg in self.regression_models.items():
            (root / rv).mkdir(exist_ok=True)
            (root / rv / "regression_model.json").write_text(json.dumps({"model": reg.to_dict()}, indent=1))

    @classmethod
    def load(cls, path, into=None):
        root = Path(path) / "model"
        meta = json.loads((root / "normative_model.json").read_text())
        tmpl = HBR() if meta["template"]["kind"] == "hbr" else BLR(warp_name=meta["template"].get("warp_name"))
        tmpl.cfg = meta["template"]
        # like PCNtoolkit: the stored save_dir and saveplots come back, wherever the folder is now
        m = into or cls(tmpl, save_dir=meta["save_dir"], inscaler=meta["inscaler"], outscaler=meta["outscaler"],
                        saveplots=meta.get("saveplots", True), y_transform=meta.get("y_transform"))
        m.covariates, m.unique_batch_effects = meta["covariates"], meta["unique_batch_effects"]
        m.covariate_ranges, m.is_fitted = meta["covariate_ranges"], True
        m.batch_effect_counts = meta.get("batch_effect_counts", {})
        m.batch_effect_covariate_ranges = meta.get("batch_effect_covariate_ranges", {})
        for d in sorted(root.glob("*")):
            if (d / "regression_model.json").is_file():
                m.regression_models[d.name] = _Reg.from_dict(json.loads((d / "regression_model.json").read_text())["model"])
        m.response_vars = list(m.regression_models)
        return m

    def _adapt(self, data, save_dir, refit: bool):
        cfg = self.template_regression_model.cfg
        if not refit and cfg["kind"] == "blr" and not cfg.get("warp_name"):
            # BLR.transfer in 1.0-1.3.0 only defines `y` inside `if self.warp:`
            raise UnboundLocalError("cannot access local variable 'y' where it is not associated with a value")
        if refit:
            for b, levels in data.unique_batch_effects.items():     # NormData.merge keeps a stale registry
                for lv in self.unique_batch_effects.get(b, []):
                    if lv not in levels:
                        raise KeyError(lv)
        new = NormativeModel(copy.deepcopy(self.template_regression_model), save_dir=save_dir or self.save_dir + "_transfer",
                             inscaler=self.inscaler, outscaler=self.outscaler, saveplots=True,
                             y_transform=self.y_transform)
        new._register(data)
        site = max(data.batch_effect_dims, key=lambda b: data.df[b].nunique())
        for rv in [v for v in data.response_vars if v in self.regression_models]:
            reg = copy.deepcopy(self.regression_models[rv])
            reg.shift = {}
            yhat = reg.predict(data.df.drop(columns=[rv]), rv, data.covariates, data.batch_effect_dims)["yhat"]
            e = data.df[rv].to_numpy(float) - yhat
            reg.shift = {"dim": site, "all": float(e.mean()),
                         "by": {k: float(v) for k, v in pd.Series(e).groupby(data.df[site].to_numpy()).mean().items()}}
            reg.transfered = not refit
            new.regression_models[rv] = reg
        new.response_vars = list(new.regression_models)
        new.is_fitted = True
        new.save()
        new.predict(data)                     # like PCNtoolkit: the adaptation data are scored too
        return new

    def transfer(self, transfer_data, save_dir=None, **kwargs):
        return self._adapt(transfer_data, save_dir, refit=False)

    def transfer_predict(self, transfer_data, predict_data, save_dir=None, **kwargs):
        new = self._adapt(transfer_data, save_dir, refit=False)
        new.predict(predict_data)
        return new

    def extend(self, data, save_dir=None, n_synth_samples=None):
        return self._adapt(data, save_dir, refit=True)

    def extend_predict(self, extend_data, predict_data, save_dir=None, n_synth_samples=None):
        new = self._adapt(extend_data, save_dir, refit=True)
        new.predict(predict_data)
        return new


# ---------------------------------------------------------------- Runner ("jobs" run inline)
class Runner:
    def __init__(self, parallelize: bool = False, job_type: str = "slurm", n_batches=None, batch_size=None,
                 n_cores: int = 1, time_limit="00:05:00", memory: str = "5GB", max_retries: int = 3,
                 environment=None, cross_validate: bool = False, cv_folds: int = 5,
                 preamble: str = "module load anaconda3", log_dir=None, temp_dir=None):
        if parallelize and not (environment and os.path.exists(os.path.join(environment, "bin", "python"))):
            raise ValueError(f"invalid environment: {environment}")
        self.n_batches, self.log_dir, self.temp_dir, self.job_type = n_batches or 1, log_dir, temp_dir, job_type
        self.active_jobs, self.finished_jobs, self.failed_jobs = {}, {}, {}

    def _submit(self, fn, fit_data, predict_data):
        if self.job_type == "torque":
            self.random_sleep_scale           # AttributeError unless the caller set it, as in 1.1-1.3.0
        task = "fake_" + time.strftime("%Y-%m-%d_%H:%M:%S") + f"_{time.time_ns() % 1000}"
        self.unique_temp_dir = os.path.join(self.temp_dir, task)
        self.unique_log_dir = os.path.join(self.log_dir, task)
        os.makedirs(self.unique_temp_dir, exist_ok=True)
        os.makedirs(self.unique_log_dir, exist_ok=True)
        for i, (a, b) in enumerate(zip(fit_data.chunk(self.n_batches), predict_data.chunk(self.n_batches))):
            job = f"job_{i}"
            try:
                fn(a, b)
                Path(self.unique_log_dir, f"{job}.success").touch()
                self.finished_jobs[job] = str(1000 + i)
            except Exception as e:  # a whole batch dies with its first failing variable
                self.failed_jobs[job] = f"{type(e).__name__}: {e}"
        with open(os.path.join(self.unique_temp_dir, "runner_state.json"), "w") as f:
            json.dump({"failed_jobs": self.failed_jobs, "finished_jobs": self.finished_jobs, "log_dir": self.log_dir,
                       "temp_dir": self.temp_dir, "unique_temp_dir": self.unique_temp_dir,
                       "unique_log_dir": self.unique_log_dir}, f)

    def fit_predict(self, model, fit_data, predict_data=None, save_dir=None, observe=True):
        def fn(a, b):
            m = copy.deepcopy(model)
            m.set_save_dir(save_dir)
            m.fit_predict(a, b)
        self._submit(fn, fit_data, predict_data)

    def transfer_predict(self, model, fit_data, predict_data=None, save_dir=None, observe=True, **kwargs):
        self._submit(lambda a, b: model.transfer_predict(a, b, save_dir=save_dir, **kwargs), fit_data, predict_data)

    def extend_predict(self, model, fit_data, predict_data=None, save_dir=None, observe=True, **kwargs):
        self._submit(lambda a, b: model.extend_predict(a, b, save_dir=save_dir), fit_data, predict_data)

    def check_jobs_status(self):
        """(running, finished, failed), computed from active_jobs only, like PCNtoolkit."""
        return dict(self.active_jobs), {}, {}

    @classmethod
    def load_from_state(cls, runner_file):
        st = json.load(open(runner_file))
        r = cls(log_dir=st["log_dir"], temp_dir=st["temp_dir"])
        # PCNtoolkit sorts the jobs here; calling check_jobs_status() again afterwards sees only running ones
        r.active_jobs, r.finished_jobs, r.failed_jobs = {}, st["finished_jobs"], st["failed_jobs"]
        return r
