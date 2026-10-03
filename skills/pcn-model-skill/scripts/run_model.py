#!/usr/bin/env python3
"""Fit, transfer, extend and predict normative models with PCNtoolkit v1.x.

    $PCN_PYTHON run_model.py plan    --project P --run R --recipe recipe.json [--response-vars a,b] [--backend local|slurm|torque]
    $PCN_PYTHON run_model.py run     --project P --run R --approved-by "Name"
    $PCN_PYTHON run_model.py status  --project P --run R
    $PCN_PYTHON run_model.py predict --project P --run R --data clinical|train|test|<csv> [--name NAME]

plan     computes nothing: it freezes the recipe, lists what is still pending, checks a reference
         model for compatibility and states how many jobs with which resources would start
run      executes the approved plan. Re-running is always safe: response variables that already
         have a saved model and test z-scores are skipped; only missing ones are (re)done
status   reports what is on disk, plus the scheduler's view for cluster runs
predict  scores another table (clinical cohort, train split, new data) with the fitted models

One response variable failing never stops the others.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (EXIT_PARTIAL, Project, append_jsonl, audited, die, dismissed_ids, load_env, now,  # noqa: E402
                     pcntoolkit_version, read_json, read_jsonl, say, write_json)
import pcn_bridge as B  # noqa: E402

IGNORED_RECIPE_KEYS = ("rationale", "alternatives", "traits")


def recipe_core(recipe: dict) -> dict:
    return {k: v for k, v in recipe.items() if k not in IGNORED_RECIPE_KEYS}


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_context(a):
    project = Project(a.project)
    man = read_json(project.data / "build_manifest.json")
    if man is None:
        die("dataset not built: run pcn-data-skill first", 2)
    return project, man, load_env(project.root, getattr(a, "env", None)), project.run_dir(a.run)


def pending_vars(rd: Path, rvs: list[str]) -> list[str]:
    zc = B.results_columns(rd, "test")
    return [v for v in rvs if not B.feature_done(rd, v, zc)]


# --------------------------------------------------------------------------
# reference-model compatibility (transfer / extend)
# --------------------------------------------------------------------------
def reference_compat(ref_dir: str, mode: str, man: dict, project: Project) -> tuple[list[dict], list[str]]:
    import pandas as pd
    out: list[dict] = []

    def add(cid, status, detail):
        out.append({"id": cid, "status": status, "detail": detail})

    model = B.load_model(ref_dir)
    train = pd.read_csv(project.data / "train.csv", dtype={c: str for c in [B.ID] + man["batch_effects"]})

    m_cov = list(getattr(model, "covariates", []) or [])
    add("R1", "PASS" if m_cov == man["covariates"] else "FAIL",
        f"covariates: model {m_cov} vs data {man['covariates']} (names and order must match)")

    m_be = dict(getattr(model, "unique_batch_effects", {}) or {})
    d_be = B.batch_columns(man)
    add("R2", "PASS" if set(m_be) == set(d_be) else "FAIL",
        f"batch-effect dimensions: model {sorted(m_be)} vs data {sorted(d_be)}")

    for dim in sorted(set(m_be) & set(man["batch_effects"])):
        m_lv = {str(x) for x in m_be[dim]}
        d_lv = set(train[dim].astype(str))
        new = sorted(d_lv - m_lv)
        if not (d_lv & m_lv) and len(m_lv) <= 3:
            add("R3", "FAIL", f"'{dim}' is coded {sorted(d_lv)} in the data but {sorted(m_lv)} in the model: "
                              "recode it in the data spec (e.g. 0/1 -> F/M)")
        elif new:
            small = {lv: int((train[dim] == lv).sum()) for lv in new}
            few = {lv: n for lv, n in small.items() if n < 20}
            add("R3", "WARN" if few else "PASS",
                f"'{dim}': {len(new)} level(s) new to the model {new[:6]} (expected for a new site)"
                + (f"; fewer than 20 adaptation observations for {few} (guidance: 20 to 100 healthy controls per site)"
                   if few else ""))
        else:
            add("R3", "PASS", f"'{dim}': all levels known to the model")

    fitted = set(B.fitted_response_vars(model))
    shared = [v for v in man["response_vars"] if v in fitted]
    lost = [v for v in man["response_vars"] if v not in fitted]
    add("R4", "FAIL" if not shared else ("WARN" if lost else "PASS"),
        f"response variables: {len(shared)} shared with the model, {len(lost)} in the data have no fitted "
        f"counterpart {lost[:8]}. Names must match the model's exactly (same atlas, same measure).")

    try:
        ranges = getattr(model, "covariate_ranges", None) or {}
        msgs = []
        for cv in man["covariates"]:
            rg = ranges.get(cv) if hasattr(ranges, "get") else None
            if rg is None:
                continue
            lo, hi = (rg.get("min"), rg.get("max")) if hasattr(rg, "get") else (rg[0], rg[1])
            x = train[cv].astype(float)
            if x.min() < float(lo) or x.max() > float(hi):
                msgs.append(f"{cv}: data [{x.min():g}, {x.max():g}] exceeds model range [{float(lo):g}, {float(hi):g}]")
        add("R5", "WARN" if msgs else "PASS", "; ".join(msgs) or "data covariates lie within the model's range")
    except Exception as e:  # noqa: BLE001
        add("R5", "SKIPPED", f"could not read the model's covariate ranges ({type(e).__name__})")

    lineage = read_json(Path(ref_dir) / "lineage.json") or {}
    chain = [lineage.get("mode")] + [x.get("mode") for x in lineage.get("ancestors", [])]
    if mode == "transfer_predict" and "transfer_predict" in chain:
        add("R6", "FAIL", "the reference model is itself the product of a transfer. A model should be transferred "
                          "only once: transfer from the original reference instead, or use extend")
    else:
        add("R6", "PASS", "lineage: " + (" <- ".join(str(c) for c in chain if c) or "original model (no PCN-Pilot lineage file)"))
    return out, shared


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def cmd_plan(a) -> int:
    project, man, env, rd = load_context(a)
    recipe = read_json(a.recipe)
    if recipe is None:
        die(f"recipe not found: {a.recipe}", 2)
    try:
        recipe = B.normalise_recipe(recipe)
    except B.RecipeError as e:
        die(str(e), 2)
    frozen = read_json(rd / "recipe.json")
    if frozen is not None and digest(recipe_core(frozen)) != digest(recipe_core(recipe)):
        die(f"run '{a.run}' already exists with a different recipe. Results are never overwritten: "
            f"choose a new --run name (for example {a.run}_v2).", 2)

    old_plan = read_json(rd / "plan.json")
    if old_plan and old_plan.get("data_built_at") != man["built_at"] and \
            len(pending_vars(rd, old_plan["response_vars"])) < len(old_plan["response_vars"]):
        die(f"run '{a.run}' holds models fitted on an earlier build of the dataset. Models from two builds are "
            f"never mixed: choose a new --run name.", 2)

    rvs = list(man["response_vars"])
    if a.response_vars:
        wanted = [v.strip() for v in a.response_vars.split(",") if v.strip()]
    elif a.response_vars_file:
        wanted = [ln.strip() for ln in Path(a.response_vars_file).read_text().splitlines() if ln.strip()]
    else:
        wanted = None
    if wanted is not None:
        unknown = [v for v in wanted if v not in rvs]
        if unknown:
            die(f"not in the dataset: {unknown[:10]}", 2)
        rvs = wanted
    dismissed = dismissed_ids(project, "response_vars")
    rvs = [v for v in rvs if v not in dismissed]

    compat: list[dict] = []
    if recipe["mode"] != "fit_predict":
        B.quiet()
        compat, shared = reference_compat(recipe["reference_model"], recipe["mode"], man, project)
        rvs = [v for v in rvs if v in shared]
    else:
        try:                                  # fail now, not in job 37 of 200
            B.quiet()
            B.make_template(recipe)
        except B.RecipeError as e:
            die(str(e), 2)
    if not rvs:
        die("no response variable left to model")

    backend = (a.backend or env["PCN_BACKEND"]).lower()
    if backend not in ("local", "slurm", "torque"):
        die("backend must be local, slurm or torque", 2)
    pending = pending_vars(rd, rvs)
    n_batches = int(a.n_batches or env["PCN_N_BATCHES"])
    n_jobs = 0 if not pending else (1 if backend == "local" else min(n_batches, len(pending)))
    parent = read_json(Path(recipe["reference_model"]) / "lineage.json") if recipe.get("reference_model") else None
    plan = {
        "run": a.run, "planned_at": now(), "mode": recipe["mode"], "algorithm": recipe["algorithm"],
        "backend": backend, "response_vars": rvs, "n_pending": len(pending), "n_jobs": n_jobs,
        "resources": {"time_limit": env["PCN_TIME_LIMIT"], "memory": env["PCN_MEMORY"],
                      "n_cores": int(env["PCN_N_CORES"]), "max_retries": int(env["PCN_MAX_RETRIES"]),
                      "conda_env": env["PCN_CONDA_ENV"], "preamble": env["PCN_PREAMBLE"],
                      "partition": env["PCN_SLURM_PARTITION"], "account": env["PCN_SLURM_ACCOUNT"],
                      "qos": env["PCN_SLURM_QOS"]} if backend != "local" else {},
        "data_built_at": man["built_at"], "recipe_digest": digest(recipe_core(recipe)),
        "compatibility": compat, "pcntoolkit": pcntoolkit_version(),
    }
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "recipe.json", recipe)
    write_json(rd / "plan.json", plan)
    write_json(rd / "lineage.json", {
        "run": a.run, "mode": recipe["mode"], "reference_model": recipe.get("reference_model"),
        "ancestors": ([{"mode": parent.get("mode"), "run": parent.get("run")}] + parent.get("ancestors", []))
        if parent else [], "created": now()})

    say(f"PLAN for run '{a.run}'  (nothing has been computed)")
    say(f"  route        {recipe['mode']} / {recipe['algorithm']}"
        + (f"  from {recipe['reference_model']}" if recipe.get("reference_model") else ""))
    say(f"  variables    {len(rvs)} total, {len(rvs) - len(pending)} already complete, {len(pending)} pending")
    say(f"  backend      {backend}" + (f": {n_jobs} job(s), {plan['resources']['time_limit']} / "
                                        f"{plan['resources']['memory']} / {plan['resources']['n_cores']} core(s) each"
                                        if backend != "local" else ": sequential in this process"))
    say(f"  output       {rd}")
    for c in compat:
        say(f"  [{c['status']:<7}] {c['id']} {c['detail']}")
    if any(c["status"] == "FAIL" for c in compat):
        say("\nThe reference model is NOT compatible. Fix the FAIL items (usually in the data spec) and re-plan.")
        raise SystemExit(1)
    say(f"\nShow this plan to the user. After explicit approval:\n"
        f"  run_model.py run --project {project.root} --run {a.run} --approved-by \"<name>\"")
    return 0


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------
def write_status(rd: Path, rv: str, state: str, attempt: int, seconds: float, error: str = "", tb: str = "") -> None:
    write_json(rd / "status" / f"{rv}.json", {
        "response_var": rv, "state": state, "attempt": attempt, "seconds": round(seconds, 2), "at": now(),
        "error": error, "traceback": tb, "pcntoolkit": pcntoolkit_version()})


def fit_one(recipe: dict, rd: Path, ref, tr, te):
    mode = recipe["mode"]
    if mode == "fit_predict":
        model = B.make_model(recipe, rd)
        model.fit_predict(tr, te)
        return model
    kw = dict(recipe.get("transfer_kwargs") or {}) if mode == "transfer_predict" else \
        ({"n_synth_samples": recipe["n_synth_samples"]} if recipe.get("n_synth_samples") else {})
    fn = ref.transfer_predict if mode == "transfer_predict" else ref.extend_predict
    return fn(tr, te, save_dir=str(rd), **kw)


def run_local(project: Project, man: dict, recipe: dict, rd: Path, pending: list[str], verbose: bool) -> None:
    B.quiet(show=verbose)
    d = project.data
    train_df, test_df = B.read_split(d / "train.csv", man), B.read_split(d / "test.csv", man)
    clin_df = B.read_split(d / "clinical.csv", man) if (d / "clinical.csv").exists() else None
    ref = B.load_model(recipe["reference_model"]) if recipe["mode"] != "fit_predict" else None
    n_ok = 0
    for i, rv in enumerate(pending, 1):
        prev = read_json(rd / "status" / f"{rv}.json") or {}
        attempt = int(prev.get("attempt", 0)) + 1
        t0 = time.time()
        try:
            # fresh NormData per variable: a failure can never contaminate the next one
            tr = B.make_normdata("train", train_df, man, [rv])
            te = B.make_normdata("test", test_df, man, [rv])
            fitted = fit_one(recipe, rd, ref, tr, te)
            if not B.feature_done(rd, rv):
                raise RuntimeError("PCNtoolkit returned without writing the model and test z-scores")
            if rv not in B.results_columns(rd, "train"):
                fitted.predict(B.make_normdata("train", train_df, man, [rv]))
            if clin_df is not None and len(clin_df):
                fitted.predict(B.make_normdata("clinical", clin_df, man, [rv]))
            write_status(rd, rv, "ok", attempt, time.time() - t0)
            n_ok += 1
            say(f"[{i}/{len(pending)}] ok      {rv}  ({time.time() - t0:.1f}s)")
        except KeyboardInterrupt:
            write_status(rd, rv, "failed", attempt, time.time() - t0, "interrupted")
            raise
        except BaseException as e:  # noqa: BLE001  (one variable must not stop the cohort)
            write_status(rd, rv, "failed", attempt, time.time() - t0, f"{type(e).__name__}: {e}",
                         traceback.format_exc(limit=12))
            say(f"[{i}/{len(pending)}] FAILED  {rv}: {type(e).__name__}: {str(e)[:200]}")
    say(f"\n{n_ok}/{len(pending)} pending variables completed in this pass")


def scheduler_status(rd: Path) -> dict | None:
    """The scheduler's view of the most recent submission (None if unavailable)."""
    subs = read_jsonl(rd / "submissions.jsonl")
    if not subs or not subs[-1].get("state_file") or not Path(subs[-1]["state_file"]).is_file():
        return None
    try:
        pcn = B.import_pcn()
        runner = pcn.Runner.load_from_state(subs[-1]["state_file"])
        running, failed, finished = runner.check_jobs_status()
        return {"running": len(running), "failed": len(failed), "finished": len(finished),
                "failed_detail": {str(k): str(v)[:300] for k, v in dict(failed).items()}}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def run_cluster(project: Project, man: dict, recipe: dict, rd: Path, plan: dict, pending: list[str],
                env: dict, force: bool) -> None:
    import shutil
    exe = "sbatch" if plan["backend"] == "slurm" else "qsub"
    if not shutil.which(exe) and not os.environ.get("PCN_SKIP_SCHEDULER_CHECK"):
        die(f"backend {plan['backend']} needs `{exe}` on PATH. Run this from a login node, or re-plan with --backend local.")
    pcn = B.import_pcn()
    B.quiet()
    sched = scheduler_status(rd)
    if sched and sched.get("running") and not force:
        die(f"{sched['running']} job(s) of the previous submission are still running. Wait for them "
            f"(`run_model.py status`) or pass --force to submit anyway.")
    attempt = len(read_jsonl(rd / "submissions.jsonl")) + 1
    # a retry isolates each remaining variable in its own job, so one bad variable cannot sink a batch again
    n_batches = len(pending) if attempt > 1 else min(int(plan["n_jobs"]) or 1, len(pending))
    res = plan["resources"]
    for var, key in (("SBATCH_PARTITION", "partition"), ("SBATCH_ACCOUNT", "account"), ("SBATCH_QOS", "qos")):
        if res.get(key):
            os.environ[var] = res[key]       # read natively by sbatch
    d = project.data
    train = B.make_normdata("train", B.read_split(d / "train.csv", man), man, pending)
    test = B.make_normdata("test", B.read_split(d / "test.csv", man), man, pending)
    runner = B.call_strict(
        pcn.Runner, cross_validate=False, parallelize=True, environment=res["conda_env"],
        job_type=plan["backend"], n_batches=n_batches, time_limit=res["time_limit"], memory=res["memory"],
        n_cores=res["n_cores"], max_retries=res["max_retries"], preamble=res["preamble"],
        log_dir=str(rd / "logs"), temp_dir=str(rd / "tmp"))
    mode = recipe["mode"]
    if mode == "fit_predict":
        runner.fit_predict(B.make_model(recipe, rd), train, test, save_dir=str(rd), observe=False)
    else:
        ref = B.load_model(recipe["reference_model"])
        fn = runner.transfer_predict if mode == "transfer_predict" else runner.extend_predict
        fn(ref, train, test, save_dir=str(rd), observe=False)
    state = Path(str(getattr(runner, "unique_temp_dir", "") or "")) / "runner_state.json"
    append_jsonl(rd / "submissions.jsonl", {
        "at": now(), "attempt": attempt, "n_jobs": n_batches, "n_variables": len(pending),
        "state_file": str(state) if state.is_file() else "", "log_dir": str(getattr(runner, "unique_log_dir", ""))})
    say(f"submitted {n_batches} job(s) for {len(pending)} variables (attempt {attempt}).")
    say(f"logs: {getattr(runner, 'unique_log_dir', rd / 'logs')}")
    say("Check progress with `run_model.py status`; when jobs finish, run `check_status.py post model`.")


def cmd_run(a) -> int:
    project, man, env, rd = load_context(a)
    plan, recipe = read_json(rd / "plan.json"), read_json(rd / "recipe.json")
    if plan is None or recipe is None:
        die(f"run '{a.run}' has no plan: run `run_model.py plan` first", 2)
    if not (a.approved_by or "").strip():
        die("--approved-by is required: the plan must be shown to and approved by a named person", 2)
    if any(c["status"] == "FAIL" for c in plan.get("compatibility", [])):
        die("the plan has failing compatibility checks; fix them and re-plan")
    if plan["data_built_at"] != man["built_at"]:
        die("the dataset was rebuilt after this plan was made. Re-run `run_model.py plan` "
            "(and use a new --run name if models were already fitted on the old data).")
    pending = pending_vars(rd, plan["response_vars"])
    append_jsonl(rd / "approvals.jsonl", {"by": a.approved_by.strip(), "at": now(), "n_pending": len(pending),
                                          "plan_digest": digest(plan), "backend": plan["backend"]})
    if not pending:
        say("nothing to do: every response variable already has a saved model and test z-scores")
        return 0
    say(f"{len(plan['response_vars']) - len(pending)} complete, {len(pending)} pending -> running "
        f"({plan['backend']})")
    if plan["backend"] == "local":
        run_local(project, man, recipe, rd, pending, a.verbose)
        left = pending_vars(rd, plan["response_vars"])
        if left:
            say(f"{len(left)} variable(s) still incomplete; see {rd / 'status'}")
            raise SystemExit(EXIT_PARTIAL)
    else:
        run_cluster(project, man, recipe, rd, plan, pending, env, a.force)
    return 0


# --------------------------------------------------------------------------
# status / predict
# --------------------------------------------------------------------------
def cmd_status(a) -> int:
    project, man, env, rd = load_context(a)
    plan = read_json(rd / "plan.json")
    if plan is None:
        die(f"run '{a.run}' has no plan", 2)
    rvs = plan["response_vars"]
    pending = pending_vars(rd, rvs)
    failed = [v for v in pending if (read_json(rd / "status" / f"{v}.json") or {}).get("state") == "failed"]
    say(f"run '{a.run}': {len(rvs) - len(pending)}/{len(rvs)} complete on disk, {len(failed)} failed, "
        f"{len(pending) - len(failed)} without outputs")
    if plan["backend"] != "local":
        s = scheduler_status(rd)
        if s is None:
            say("scheduler: no submission recorded")
        elif "error" in s:
            say(f"scheduler: status unavailable ({s['error']}); trust the on-disk counts above")
        else:
            say(f"scheduler: {s['running']} running, {s['finished']} finished, {s['failed']} failed")
            for job, err in list(s["failed_detail"].items())[:5]:
                say(f"  job {job}: {err}")
    for v in failed[:10]:
        say(f"  FAILED {v}: {(read_json(rd / 'status' / f'{v}.json') or {}).get('error', '')[:200]}")
    raise SystemExit(0 if not pending else EXIT_PARTIAL)


def cmd_predict(a) -> int:
    project, man, env, rd = load_context(a)
    B.quiet()
    named = {"clinical": project.data / "clinical.csv", "train": project.data / "train.csv",
             "test": project.data / "test.csv"}
    path = named.get(a.data, Path(a.data))
    name = a.name or (a.data if a.data in named else Path(a.data).stem)
    if not path.is_file():
        die(f"table not found: {path}", 2)
    df = B.read_split(path, man)
    need = [B.ID] + man["covariates"] + B.batch_columns(man)
    missing = [c for c in need if c not in df.columns]
    if missing:
        die(f"{path} lacks standardized columns {missing}; build it with pcn-data-skill", 2)
    model = B.load_model(rd)
    done = [v for v in B.fitted_response_vars(model) if v in df.columns]
    have = set() if a.force else B.results_columns(rd, name)
    todo = [v for v in done if v not in have]
    if not todo:
        say(f"nothing to do: '{name}' predictions exist for all {len(done)} fitted variables")
        return 0
    n_ok = 0
    for rv in todo:
        try:
            model.predict(B.make_normdata(name, df, man, [rv]))
            n_ok += 1
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            say(f"FAILED {rv}: {type(e).__name__}: {str(e)[:200]}")
    say(f"predicted '{name}' for {n_ok}/{len(todo)} variables -> {rd / 'results' / f'Z_{name}.csv'}")
    if n_ok < len(todo):
        raise SystemExit(EXIT_PARTIAL)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--project", required=True)
        p.add_argument("--run", required=True, help="run name: one folder under <project>/models/")
        p.add_argument("--env", help="explicit pipeline.env path")
        return p

    p = common(sub.add_parser("plan"))
    p.add_argument("--recipe", required=True)
    p.add_argument("--response-vars", help="comma-separated subset")
    p.add_argument("--response-vars-file", help="one response variable per line")
    p.add_argument("--backend", choices=["local", "slurm", "torque"])
    p.add_argument("--n-batches", type=int)
    p = common(sub.add_parser("run"))
    p.add_argument("--approved-by", help="name of the person who approved the plan")
    p.add_argument("--force", action="store_true", help="submit even if earlier jobs still appear to run")
    p.add_argument("--verbose", action="store_true", help="show PCNtoolkit's own progress messages")
    common(sub.add_parser("status"))
    p = common(sub.add_parser("predict"))
    p.add_argument("--data", required=True)
    p.add_argument("--name")
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    project = Project(a.project)
    with audited(project, f"run_model.py {a.cmd}", argv):
        return {"plan": cmd_plan, "run": cmd_run, "status": cmd_status, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
