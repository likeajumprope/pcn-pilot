"""Shared helpers for the PCN-Pilot skills.

This file is the single source of truth; `tools/sync_common.py` copies it into
each skill's `scripts/` folder so every skill stays self-contained.

Nothing here imports pcntoolkit, so the data and QC stages can run on a
machine that only has pandas / numpy / scipy.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import getpass
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

try:                                    # `script | head` must not end in a BrokenPipe traceback
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
except (ImportError, AttributeError, ValueError):
    pass

SCHEMA_VERSION = 1
GRADES = ("PASS", "WARN", "FAIL")          # ordered best -> worst
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_PARTIAL = 0, 1, 2, 3


# --------------------------------------------------------------------------
# time / json
# --------------------------------------------------------------------------
def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | os.PathLike, obj: Any) -> None:
    """Atomic write: a crashed run never leaves a half-written state file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")
    os.replace(tmp, p)


def _json_default(o: Any) -> Any:
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, (Path, set)):
        return str(o) if isinstance(o, Path) else sorted(o)
    return str(o)


def append_jsonl(path: str | os.PathLike, row: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# pipeline.env  (the one site-specific file)
# --------------------------------------------------------------------------
ENV_DEFAULTS = {
    "PCN_PYTHON": sys.executable,
    "PCN_BACKEND": "local",            # local | slurm | torque
    "PCN_CONDA_ENV": "",               # conda env *path* handed to pcntoolkit's Runner
    "PCN_PREAMBLE": "module load anaconda3",
    "PCN_TIME_LIMIT": "02:00:00",
    "PCN_MEMORY": "8GB",
    "PCN_N_CORES": "1",
    "PCN_N_BATCHES": "10",
    "PCN_MAX_RETRIES": "3",
    "PCN_SLURM_PARTITION": "",
    "PCN_SLURM_ACCOUNT": "",
    "PCN_SLURM_QOS": "",
    "PCN_QC_PORT": "8765",
}


def find_env_file(project: str | os.PathLike | None, explicit: str | None = None) -> Path | None:
    candidates = [explicit, os.environ.get("PCN_PIPELINE_ENV")]
    if project:
        candidates.append(str(Path(project) / "pipeline.env"))
    candidates.append(str(Path.home() / ".config" / "pcnpilot" / "pipeline.env"))
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def load_env(project: str | os.PathLike | None = None, explicit: str | None = None) -> dict[str, str]:
    """KEY=VALUE parser (no shell evaluation). Real environment variables win."""
    env = dict(ENV_DEFAULTS)
    path = find_env_file(project, explicit)
    if path:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, val = line.split("=", 1)
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                val = val[1:-1]
            else:
                val = val.split(" #", 1)[0].strip()
            env[key.strip()] = val
    for k in list(env):
        if k in os.environ:
            env[k] = os.environ[k]
    env["_ENV_FILE"] = str(path) if path else ""
    return env


# --------------------------------------------------------------------------
# project layout
# --------------------------------------------------------------------------
class Project:
    """All paths of a PCN-Pilot project in one place."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root).expanduser().resolve()

    # stage folders
    @property
    def data(self) -> Path: return self.root / "data"
    @property
    def audit(self) -> Path: return self.root / "audit"
    @property
    def commands_log(self) -> Path: return self.audit / "commands.jsonl"
    @property
    def ledger(self) -> Path: return self.audit / "ledger.jsonl"
    @property
    def dismissed(self) -> Path: return self.root / "dismissed.json"
    @property
    def spec(self) -> Path: return self.data / "spec.json"
    @property
    def validation(self) -> Path: return self.data / "validation.json"

    def run_dir(self, run: str) -> Path: return self.root / "models" / run
    def qc_dir(self, run: str) -> Path: return self.root / "qc" / run
    def export_dir(self, name: str) -> Path: return self.root / "export" / name

    def ensure(self) -> "Project":
        for d in (self.root, self.data, self.audit):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def runs(self) -> list[str]:
        d = self.root / "models"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def safe_name(name: str) -> str:
    """Response-variable name -> file-system safe token (stable, reversible via manifest)."""
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(name))
    return keep or "unnamed"


# --------------------------------------------------------------------------
# audit trail: every script invocation becomes one row
# --------------------------------------------------------------------------
def pcntoolkit_version() -> str | None:
    try:
        from importlib import metadata
        return metadata.version("pcntoolkit")
    except Exception:
        try:
            import pcntoolkit  # noqa: F401
            return getattr(sys.modules["pcntoolkit"], "__version__", "unknown")
        except Exception:
            return None


@contextlib.contextmanager
def audited(project: Project | None, script: str, argv: list[str] | None = None):
    """Wrap a script's main(): logs command, versions, exit code and duration."""
    t0 = time.time()
    row = {
        "at": now(), "script": script, "argv": list(argv if argv is not None else sys.argv[1:]),
        "user": _whoami(), "host": socket.gethostname(), "python": platform.python_version(),
        "pcntoolkit": pcntoolkit_version(),
    }
    code = EXIT_OK
    try:
        yield row
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (EXIT_OK if e.code is None else EXIT_FAIL)
        raise
    except BaseException as e:  # noqa: BLE001
        code = EXIT_FAIL
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc(limit=6)
        raise
    finally:
        row["exit_code"] = code
        row["seconds"] = round(time.time() - t0, 2)
        if project is not None:
            try:
                append_jsonl(project.commands_log, row)
            except OSError:
                pass


def _whoami() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown")


# --------------------------------------------------------------------------
# dismissal list: one shared file that every later step filters on
# --------------------------------------------------------------------------
DISMISS_KINDS = ("subjects", "response_vars", "batch_levels")


def load_dismissed(project: Project) -> dict[str, list[dict]]:
    d = read_json(project.dismissed, {}) or {}
    return {k: list(d.get(k, [])) for k in DISMISS_KINDS}


def dismissed_ids(project: Project, kind: str) -> set[str]:
    return {str(r["id"]) for r in load_dismissed(project)[kind]}


def add_dismissal(project: Project, kind: str, ident: str, reason: str, by: str, step: str) -> bool:
    if kind not in DISMISS_KINDS:
        raise ValueError(f"kind must be one of {DISMISS_KINDS}")
    d = load_dismissed(project)
    if any(str(r["id"]) == str(ident) for r in d[kind]):
        return False
    d[kind].append({"id": str(ident), "reason": reason, "by": by, "step": step, "at": now()})
    write_json(project.dismissed, d)
    return True


# --------------------------------------------------------------------------
# sign-off ledger: append-only, hash-chained (tamper evident)
# --------------------------------------------------------------------------
def _row_hash(row: dict) -> str:
    body = {k: row[k] for k in sorted(row) if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def ledger_append(project: Project, by: str, reviewer_id: str, step: str, summary: str,
                  payload: dict | None = None) -> dict:
    if not by.strip() or not reviewer_id.strip():
        raise ValueError("A sign-off needs both a reviewer name and an institutional ID.")
    rows = read_jsonl(project.ledger)
    row = {"by": by.strip(), "id": reviewer_id.strip(), "step": step, "at": now(),
           "summary": summary, "payload": payload or {},
           "prev_hash": rows[-1]["hash"] if rows else "GENESIS"}
    row["hash"] = _row_hash(row)
    append_jsonl(project.ledger, row)
    return row


def ledger_verify(project: Project) -> tuple[bool, str]:
    rows = read_jsonl(project.ledger)
    prev = "GENESIS"
    for i, row in enumerate(rows):
        if row.get("prev_hash") != prev:
            return False, f"row {i}: chain broken (a row was removed, reordered or inserted)"
        if _row_hash(row) != row.get("hash"):
            return False, f"row {i}: content was modified after signing"
        prev = row["hash"]
    return True, f"{len(rows)} ledger rows verified"


# --------------------------------------------------------------------------
# small numeric helpers used by both data validation and QC
# --------------------------------------------------------------------------
def modified_z(values: Iterable[float]):
    """Iglewicz-Hoaglin modified Z: 0.6745 * (x - median) / MAD.

    Falls back to the mean absolute deviation when MAD is 0 (heavily tied data),
    as recommended by Iglewicz & Hoaglin (1993).
    """
    import numpy as np
    x = np.asarray(list(values), dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad > 0:
        return 0.6745 * (x - med) / mad
    meanad = np.nanmean(np.abs(x - med))
    if meanad > 0:
        return (x - med) / (1.253314 * meanad)
    return np.zeros_like(x)


def worst(grades: Iterable[str]) -> str:
    rank = {g: i for i, g in enumerate(GRADES)}
    best = "PASS"
    for g in grades:
        if g in rank and rank[g] > rank[best]:
            best = g
    return best


def say(msg: str) -> None:
    print(msg, flush=True)


def die(msg: str, code: int = EXIT_FAIL) -> None:
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)
