"""Shared runtime state + HP-grid / optimizer / scheduler / session helpers.

This module holds the values that ``main()`` / ``__main__`` mutate at runtime
(CLI overrides for data dirs, batch size, epochs, seed, HP-grid combos, the
eval-only checkpoint, and the deferred-log queue). Every other training module
reads them as ATTRIBUTES on this module (``run_state.MODELS_DIR``, never
``from run_state import MODELS_DIR``) so a reassignment here is seen everywhere.

``DEVICE`` is a constant computed once at import — safe to import by value.

Module-level env setup (telemetry off, HF cache dir, mlflow artifact patch)
lives here because this is the first training module imported by the rest, so
it runs before ``mlflow`` is used anywhere downstream.
"""

import os
import json
import logging
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
# Cache HuggingFace model weights persistently so they are not re-downloaded each run
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".hf_cache"))

import torch
import mlflow
from mlflow.tracking import MlflowClient

from utils.mlflow_utils import patch_artifact_logging
patch_artifact_logging()

from config import (
    ARCH_CFG,
    WARMUP_EPOCHS, SCHED_POWER, WARMUP_START_FACTOR,
    S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_BY_YEAR, CDL_TRAIN,
    MODELS_DIR, FIGURES_DIR, PRELOAD_CACHE_DIR,
    BATCH_SIZE, MAX_EPOCHS, SEED,
)
from geoai.geoai.utils.device import get_device

log = logging.getLogger(__name__)

DEVICE = "cpu" if os.environ.get("FORCE_CPU") else get_device()

# ── Runtime-overridable config mirrors ────────────────────────────────────────
# Initialised from config defaults; reassigned by main() for --data-dir /
# --batch-size / --epochs / --seed. Read cross-module as run_state.<NAME>.
# (BATCH_SIZE, MAX_EPOCHS, SEED, and the data-dir paths above are the mutable set.)

# ── Hyperparameter-grid overrides ─────────────────────────────────────────────
# Set per-combo by main() when --hp-grid is used. None = use config/ARCH_CFG
# defaults. lr/weight_decay override ARCH_CFG per-arch values uniformly across
# all archs in the run; warmup_epochs/sched_power override config defaults.
HP_OVERRIDE: dict | None = None   # {lr, weight_decay, warmup_epochs, sched_power}
HP_TAG: str = ""                  # short run-name suffix, e.g. "lr1e-04_wd1e-02_wu5_pw0.9"
SEED_TAG: str = ""                # seed suffix appended to run names when --seed-grid is used
SESSION_LOG_PATH: str | None = None  # top-level session .log file (LOGS_DIR)
EVAL_ONLY_CKPT: str | None = None    # path to a checkpoint to evaluate instead of training
# (run_id, per_run_log_path) captured per finished run; logs uploaded to MLflow
# only AFTER the whole session ends (avoids HTTP errors from uploading the
# still-growing session log mid-training).
_DEFERRED_LOG_RUNS: list[tuple] = []


# Recognised HP-grid keys (validated on load).
HP_KEYS = {
    "lr", "weight_decay", "warmup_epochs", "sched_power",
    "scheduler", "optimizer", "grad_clip", "batch_size", "momentum",
}
_OPTIMIZERS  = {"adamw", "adam", "sgd"}
_SCHEDULERS  = {"polynomial", "cosine"}


def _resolve_hp(cfg: dict) -> dict:
    """Merge HP_OVERRIDE over a per-arch ARCH_CFG entry + config defaults.

    batch_size=None → use the module BATCH_SIZE (CLI/config). grad_clip=0 → off.
    """
    o = HP_OVERRIDE or {}
    optimizer = str(o.get("optimizer", cfg.get("optimizer", "adamw"))).lower()
    scheduler = str(o.get("scheduler", cfg.get("scheduler", "polynomial"))).lower()
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"--hp-grid optimizer '{optimizer}' invalid; choose {sorted(_OPTIMIZERS)}")
    if scheduler not in _SCHEDULERS:
        raise ValueError(f"--hp-grid scheduler '{scheduler}' invalid; choose {sorted(_SCHEDULERS)}")
    return {
        "lr":            float(o.get("lr",            cfg["lr"])),
        "weight_decay":  float(o.get("weight_decay",  cfg["weight_decay"])),
        "warmup_epochs": int(o.get("warmup_epochs",   cfg.get("warmup_epochs", WARMUP_EPOCHS))),
        "sched_power":   float(o.get("sched_power",   cfg.get("sched_power", SCHED_POWER))),
        "scheduler":     scheduler,
        "optimizer":     optimizer,
        "momentum":      float(o.get("momentum", cfg.get("momentum", 0.9))),   # SGD only
        "grad_clip":     float(o.get("grad_clip", 0.0)),  # 0 = disabled
        "batch_size":    int(o["batch_size"]) if o.get("batch_size") else None,
    }


def _build_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float):
    """AdamW (default) / Adam / SGD(+momentum, nesterov)."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum,
                               weight_decay=weight_decay, nesterov=momentum > 0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _build_scheduler(optimizer, max_epochs: int, power: float, warmup_epochs: int,
                     kind: str = "polynomial"):
    """LR decay (PolynomialLR or CosineAnnealingLR) with optional linear warmup.

    Stepped per-epoch. warmup_epochs=0 → plain decay. Otherwise LinearLR
    (start_factor → 1.0 over warmup_epochs) chained into the decay over the
    remaining epochs via SequentialLR.
    """
    def _decay(iters):
        if kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)
        return torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=iters, power=power)

    if warmup_epochs and warmup_epochs > 0:
        decay_iters = max(1, max_epochs - warmup_epochs)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=WARMUP_START_FACTOR, end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, _decay(decay_iters)], milestones=[warmup_epochs],
        )
    return _decay(max_epochs)


def _expand_grid_block(block: dict) -> list[dict]:
    """Expand one {'grid':{...}} / {'combos':[...]} / bare-dict-of-lists block."""
    import itertools

    valid = HP_KEYS

    if isinstance(block, dict) and "combos" in block:
        combos = block["combos"]
        if not isinstance(combos, list) or not combos:
            raise ValueError("--hp-grid 'combos' must be a non-empty list of dicts")
        for c in combos:
            bad = set(c) - valid
            if bad:
                raise ValueError(f"--hp-grid combo has unknown keys {bad}; valid={sorted(valid)}")
        return [dict(c) for c in combos]

    grid = block.get("grid", block) if isinstance(block, dict) else None
    if not isinstance(grid, dict) or not grid:
        raise ValueError("--hp-grid block must be a 'grid' dict, 'combos' list, or a bare dict of lists")
    bad = set(grid) - valid
    if bad:
        raise ValueError(f"--hp-grid has unknown keys {bad}; valid={sorted(valid)}")

    keys = list(grid)
    values = [v if isinstance(v, list) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _load_hp_grid(path: str) -> list[tuple]:
    """Expand an HP-grid JSON into a list of (arch_or_None, combo) tuples.

    Two schemas:

      Shared (applies to every --arch uniformly):
        {"grid": {"lr": [...], "weight_decay": [...], ...}}
        {"combos": [{...}, ...]}
        → [(None, combo), ...]

      Per-arch (separate search space per architecture — recommended, since
      CNN vs. transformer encoders want different lr/wd regimes):
        {"deeplabv3plus_cbam": {"grid": {...}},
         "segformer":          {"combos": [...]}}
        → [(arch, combo), ...]

    arch=None means "use the run's --arch matrix"; an arch string pins the
    combo to that single architecture.
    """
    with open(path) as f:
        spec = json.load(f)

    if not isinstance(spec, dict):
        raise ValueError("--hp-grid must be a JSON object")

    # Per-arch when top-level keys are architecture names (ignore _comment etc.).
    arch_keys = {k for k in spec if k in ARCH_CFG}
    non_arch  = {k for k in spec if not k.startswith("_") and k not in ARCH_CFG}
    if arch_keys and not non_arch:
        out: list[tuple] = []
        for arch in spec:
            if arch.startswith("_"):
                continue
            for combo in _expand_grid_block(spec[arch]):
                out.append((arch, combo))
        if not out:
            raise ValueError("--hp-grid per-arch spec expanded to zero combos")
        return out

    return [(None, combo) for combo in _expand_grid_block(spec)]


def _hp_tag(combo: dict) -> str:
    """Short, filename-safe run-name suffix for an HP combo."""
    parts = []
    if "optimizer" in combo:     parts.append(str(combo["optimizer"]))
    if "lr" in combo:            parts.append(f"lr{float(combo['lr']):.0e}")
    if "weight_decay" in combo:  parts.append(f"wd{float(combo['weight_decay']):.0e}")
    if "batch_size" in combo:    parts.append(f"bs{int(combo['batch_size'])}")
    if "scheduler" in combo:     parts.append(str(combo["scheduler"])[:3])
    if "warmup_epochs" in combo: parts.append(f"wu{int(combo['warmup_epochs'])}")
    if "sched_power" in combo:   parts.append(f"pw{float(combo['sched_power']):g}")
    if "grad_clip" in combo and float(combo["grad_clip"]) > 0:
        parts.append(f"gc{float(combo['grad_clip']):g}")
    return "_".join(parts)


def _combo_done(exp_name: str) -> bool:
    """True if a finished run dir exists for this combo (a `.done` marker).

    Run dirs are `{exp_name}_{timestamp}/`; exp_name is deterministic per combo
    (exp_key + selection + hp_tag + arch, no timestamp). A `.done` file is
    written only after a run fully completes — enables resuming a grid sweep
    that died mid-way (skips finished combos, reruns the rest).
    """
    if not MODELS_DIR.exists():
        return False
    return any(
        (d / ".done").exists()
        for d in MODELS_DIR.glob(f"{exp_name}_*")
        if d.is_dir()
    )


def _flush_deferred_logs() -> None:
    """Upload per-run + session logs to MLflow AFTER the session ends.

    Deferred so the still-growing session log is never uploaded mid-training
    (that triggered HTTP errors). Uses MlflowClient to attach to each already
    closed run by id. Best-effort per run.
    """
    if not _DEFERRED_LOG_RUNS:
        return
    for _h in logging.root.handlers:
        try:
            _h.flush()
        except Exception:
            pass
    client = MlflowClient()
    sess = SESSION_LOG_PATH if (SESSION_LOG_PATH and Path(SESSION_LOG_PATH).exists()) else None
    log.info(f"Uploading logs for {len(_DEFERRED_LOG_RUNS)} run(s) → MLflow logs/ …")
    for run_id, run_log in _DEFERRED_LOG_RUNS:
        try:
            if run_log and Path(run_log).exists():
                client.log_artifact(run_id, run_log, artifact_path="logs")
            if sess:
                client.log_artifact(run_id, sess, artifact_path="logs")
        except Exception as e:
            log.warning(f"  Could not upload logs for run {run_id}: {e}")
    _DEFERRED_LOG_RUNS.clear()


def _device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"
