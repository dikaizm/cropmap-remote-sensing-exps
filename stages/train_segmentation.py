"""
Stage 3 — Full Model Validation.

Six experiment configurations × 2 architectures = up to 12 training runs.

| Config             | Dates               | Band selection | Purpose                      |
|--------------------|---------------------|----------------|------------------------------|
| single_date        | peak NDVI           | none (all bands)| Baseline (isolates temporal) |
| mt_base            | 4 calendar dates     | none           | Multi-temporal baseline + GSI bands   |
| gsi                | multi-temporal      | GSI-direct     | Proposed method + RF bands    |
| gsi                | GSI-direct          | GSI-direct     | GSI spectral-temporal        |
| rf                 | RF-direct           | RF-direct      | RF spectral-temporal         |

Usage:
    python stages/train_segmentation.py                       # run all 6 experiments
    python stages/train_segmentation.py --exp single_date     # only single-date baseline
    python stages/train_segmentation.py --exp gsi --arch segformer
    python stages/train_segmentation.py --force               # re-run even if ckpt exists
    python stages/train_segmentation.py --data-dir /mnt/data
"""

import os
import re
import sys
import time
import json
import tempfile
import hashlib
import argparse
import logging
from glob import glob
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import rasterio

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
# Cache HuggingFace model weights persistently so they are not re-downloaded each run
os.environ.setdefault("HF_HOME", str(Path(__file__).parent.parent / ".hf_cache"))
import mlflow
from mlflow.tracking import MlflowClient

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.utils.mlflow_utils import patch_artifact_logging
patch_artifact_logging()

from crop_mapping_pipeline.config import (
    S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_BY_YEAR, CDL_TRAIN, MODELS_DIR, FIGURES_DIR, LOGS_DIR,
    PROCESSED_DIR, PRELOAD_CACHE_DIR, GDRIVE_PRELOAD_CACHE_FOLDER_ID,
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS,
    KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, S2_NODATA, S2_MIN_VALID_FRAC,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE,
    TRAIN_YEARS, TEST_YEAR,
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC, BATCH_SIZE, MAX_EPOCHS, EARLY_STOP, EARLY_STOP_DELTA,
    VAL_FRAC, TEST_FRAC, SEED, ARCH_CFG,
    BLOCK_SIZE, MIN_CLASS_FRAC,
    SCHED_POWER, WARMUP_EPOCHS, WARMUP_START_FACTOR,
    GDRIVE_OAUTH_TOKEN, GDRIVE_MODELS_FOLDER_ID,
    SELECT_TOP_K_PER_CROP,
)
from crop_mapping_pipeline.utils.constants import USDA_CDL_COLORS
from geoai.geoai.train import RasterPatchDataset, train_semantic_one_epoch
from crop_mapping_pipeline.stages.losses import (
    build_wce, build_focal_tversky, build_dynamic_balanced,
)
from geoai.geoai.utils.device import get_device
from crop_mapping_pipeline.models import DeepLabV3PlusCBAM, build_segformer
from crop_mapping_pipeline.stages.spatial_split import (
    _block_spatial_split, _save_block_split_artifacts,
)

log = logging.getLogger(__name__)
DEVICE = "cpu" if os.environ.get("FORCE_CPU") else get_device()

# ── Hyperparameter-grid overrides ─────────────────────────────────────────────
# Set per-combo by main() when --hp-grid is used. None = use config/ARCH_CFG
# defaults. lr/weight_decay override ARCH_CFG per-arch values uniformly across
# all archs in the run; warmup_epochs/sched_power override config defaults.
HP_OVERRIDE: dict | None = None   # {lr, weight_decay, warmup_epochs, sched_power}
HP_TAG: str = ""                  # short run-name suffix, e.g. "lr1e-04_wd1e-02_wu5_pw0.9"
SESSION_LOG_PATH: str | None = None  # top-level session .log file (LOGS_DIR)
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
    optimizer = str(o.get("optimizer", "adamw")).lower()
    scheduler = str(o.get("scheduler", "polynomial")).lower()
    if optimizer not in _OPTIMIZERS:
        raise ValueError(f"--hp-grid optimizer '{optimizer}' invalid; choose {sorted(_OPTIMIZERS)}")
    if scheduler not in _SCHEDULERS:
        raise ValueError(f"--hp-grid scheduler '{scheduler}' invalid; choose {sorted(_SCHEDULERS)}")
    return {
        "lr":            float(o.get("lr",            cfg["lr"])),
        "weight_decay":  float(o.get("weight_decay",  cfg["weight_decay"])),
        "warmup_epochs": int(o.get("warmup_epochs",   WARMUP_EPOCHS)),
        "sched_power":   float(o.get("sched_power",    SCHED_POWER)),
        "scheduler":     scheduler,
        "optimizer":     optimizer,
        "momentum":      float(o.get("momentum", 0.9)),   # SGD only
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


def _check_gdrive_token() -> None:
    """Attempt to refresh the OAuth token; log warning if unavailable or expired."""
    import pickle
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        log.warning(f"GDrive token not found at {GDRIVE_OAUTH_TOKEN} — artifact upload will fail")
        return
    try:
        with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
            creds = pickle.load(f)
        if creds.expired and creds.refresh_token:
            log.info("GDrive token expired — refreshing...")
            creds.refresh(Request())
            with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
                pickle.dump(creds, f)
            log.info("GDrive token refreshed and saved.")
        elif creds.expired:
            log.warning("GDrive token expired and no refresh_token — artifact upload will fail")
        else:
            log.info("GDrive token valid.")
    except Exception as e:
        log.warning(f"GDrive token check failed ({e}) — artifact upload will fail")


def _device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _s2_for_year(s2_processed, yr):
    # Flat train/ dir — all files belong to the single training year
    return sorted(s2_processed)


def _valid_global_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return the subset of band_indices that are in range for s2_paths."""
    if band_indices is None:
        return set()
    needed = sorted({gi // n_bands_per_file for gi in band_indices
                     if gi // n_bands_per_file < len(s2_paths)})
    new_idx_map = set()
    for fi in needed:
        for local in range(n_bands_per_file):
            new_idx_map.add(fi * n_bands_per_file + local)
    return set(gi for gi in band_indices if gi in new_idx_map)


def _filter_s2_by_band_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return (filtered_paths, remapped_indices) keeping only TIF files that
    contribute at least one channel in band_indices, with indices remapped to
    their positions in the reduced stack.

    Example: 25 files × 10 bands = 250 channels.  single_date selects bands [140..149]
    (file 14 only) → returns [s2_paths[14]], remapped to [0..9].
    """
    if band_indices is None:
        return s2_paths, None
    # Which file indices (0-based) are needed?
    needed_file_idxs = sorted({gi // n_bands_per_file for gi in band_indices
                                if gi // n_bands_per_file < len(s2_paths)})
    filtered_paths = [s2_paths[i] for i in needed_file_idxs]
    # Build global-index → new-stacked-index map for every band in kept files
    new_idx_map = {}
    stacked = 0
    for fi in needed_file_idxs:
        for local in range(n_bands_per_file):
            new_idx_map[fi * n_bands_per_file + local] = stacked
            stacked += 1
    skipped = [gi for gi in band_indices if gi not in new_idx_map]
    if skipped:
        log.warning("  Dropping %d channel(s) from excluded/empty S2 files: %s",
                    len(skipped), skipped)
    remapped = [new_idx_map[gi] for gi in band_indices if gi in new_idx_map]
    return filtered_paths, remapped


from crop_mapping_pipeline.stages.experiments import (
    parse_date,
    build_local_band_map,
    build_single_date_indices,
    build_naive_multitemporal_indices,
    build_naive_multitemporal_selected_indices,
    build_registry,
    expand_exp_keys,
)
from crop_mapping_pipeline.stages.experiments.exp_select_direct import build_direct_indices
from crop_mapping_pipeline.stages.selections.rf_band_only import run_rf_band_only, save_rf_band_json
from crop_mapping_pipeline.stages.ndvi_disagreement_analysis import (
    run_ndvi_disagreement, score_patch_verdict, B4_IDX, B8_IDX,
)
from crop_mapping_pipeline.config import (
    SELECT_GSI_DIRECT_JSON,
    SELECT_RF_DIRECT_JSON,
    GSI_CANDIDATES_JSON,
    PROCESSED_DIR,
)


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(cdl_path=None, return_counts=False):
    """Inverse-frequency weights from CDL (train area). Caches result alongside CDL.

    If return_counts=True, returns (weights_tensor, counts_array).
    """
    ref_cdl   = Path(cdl_path) if cdl_path else CDL_TRAIN
    cache_key = {"cdl": str(ref_cdl), "keep_classes": KEEP_CLASSES, "num_classes": NUM_CLASSES}
    cache_h   = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = ref_cdl.parent / f"class_weights_{cache_h}.json"

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                d = json.load(f)
            w = d["weights"]
            c = d.get("class_counts")
            log.info(f"Class weights cache hit → {cache_path.name}")
            wt = torch.tensor(w, dtype=torch.float32)
            if return_counts and c is not None:
                return wt, np.asarray(c, dtype=np.float64)
            if not return_counts:
                return wt
        except Exception:
            pass

    with rasterio.open(ref_cdl) as src:
        cdl_arr = src.read(1).astype(np.int32)

    class_counts      = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_counts[0]   = (cdl_arr == 0).sum()
    for cdl_id, model_id in CLASS_REMAP.items():
        class_counts[model_id] += (cdl_arr == cdl_id).sum()

    freq    = class_counts / (class_counts.sum() + 1e-9)
    weights = 1.0 / (freq + 1e-9)
    weights /= weights.sum()

    with open(cache_path, "w") as f:
        json.dump({"weights": weights.tolist(), "class_counts": class_counts.tolist()}, f)
    log.info(f"Class weights cached → {cache_path.name}")

    wt = torch.tensor(weights, dtype=torch.float32)
    if return_counts:
        return wt, class_counts
    return wt


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(logits, labels, num_classes):
    preds  = logits.argmax(dim=1).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    ious   = []
    for cls in range(1, num_classes):
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum()
        union = (p | l).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_per_class_iou(logits, labels, num_classes):
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    ious   = {}
    for cls in range(1, num_classes):
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum()
        union = (p | l).sum()
        ious[cls] = float(inter / union) if union > 0 else float("nan")
    return ious


def compute_per_class_f1(logits, labels, num_classes):
    """Per-class F1 via precision × recall. Excludes background (class 0)."""
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    f1s = {}
    for cls in range(1, num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not (np.isnan(prec) or np.isnan(recall)) and (prec + recall) > 0:
            f1s[cls] = float(2 * prec * recall / (prec + recall))
        else:
            f1s[cls] = float("nan")
    return f1s


def compute_per_class_oa(logits, labels, num_classes):
    """Per-class OA = recall = TP / (TP + FN). Excludes background (class 0)."""
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    oas = {}
    for cls in range(1, num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        oas[cls] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    return oas


def mean_f1(f1_dict):
    vals = [v for v in f1_dict.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs        = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
        logits      = model(imgs)
        loss        = criterion(logits, masks)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(masks.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=1)
    oa              = (preds == all_labels).float().mean().item()
    miou            = compute_miou(all_logits, all_labels, num_classes)
    per_class_iou   = compute_per_class_iou(all_logits, all_labels, num_classes)
    per_class_f1    = compute_per_class_f1(all_logits, all_labels, num_classes)
    per_class_oa    = compute_per_class_oa(all_logits, all_labels, num_classes)
    mf1             = mean_f1(per_class_f1)
    return {
        "loss": total_loss / len(loader), "miou": miou, "oa": oa,
        "mf1": mf1, "per_class_iou": per_class_iou, "per_class_f1": per_class_f1,
        "per_class_oa": per_class_oa,
    }


@torch.no_grad()
def _get_hardware_info() -> dict:
    """CPU/GPU/RAM identity for mlflow params — static per-machine, not a metric."""
    import platform

    cpu_name = platform.processor()
    if not cpu_name and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    info = {
        "cpu_name":  cpu_name or "unknown",
        "cpu_cores": os.cpu_count(),
    }

    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["ram_total_gb"] = None

    if torch.cuda.is_available():
        info["gpu_name"]      = torch.cuda.get_device_name(0)
        info["gpu_count"]     = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    else:
        info["gpu_name"]      = "none"
        info["gpu_count"]     = 0
        info["gpu_memory_gb"] = None

    return info


def evaluate_test_set(model, loader, num_classes, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=5.0, neginf=-5.0)
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(masks.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=1)
    per_class_f1 = compute_per_class_f1(all_logits, all_labels, num_classes)
    return {
        "miou":          compute_miou(all_logits, all_labels, num_classes),
        "oa":            (preds == all_labels).float().mean().item(),
        "mf1":           mean_f1(per_class_f1),
        "per_class_iou": compute_per_class_iou(all_logits, all_labels, num_classes),
        "per_class_f1":  per_class_f1,
        "preds":         preds,
        "labels":        all_labels,
    }


def benchmark_inference_latency(model, loader, device, run_id):
    """Time inference one patch at a time (excludes data loading/metric compute).

    Logs per-patch latency (ms) to mlflow via log_batch (chunked, avoids one
    HTTP call per patch) plus avg/std/min/max summary metrics.
    """
    from mlflow.entities import Metric

    model.eval()
    client  = MlflowClient()
    metrics = []
    latencies_ms = []
    is_cuda = torch.cuda.is_available() and str(device) != "cpu"
    idx = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=5.0, neginf=-5.0)
            for b in range(imgs.shape[0]):
                patch = imgs[b:b + 1].to(device)
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(patch)
                if is_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                metrics.append(Metric(
                    key="inference_time_ms_patch", value=elapsed_ms,
                    timestamp=int(time.time() * 1000), step=idx,
                ))
                idx += 1

    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])

    lat = np.array(latencies_ms)
    summary = {
        "inference_time_ms_avg":       float(lat.mean()),
        "inference_time_ms_std":       float(lat.std()),
        "inference_time_ms_min":       float(lat.min()),
        "inference_time_ms_max":       float(lat.max()),
        "inference_patches_benchmarked": len(lat),
    }
    mlflow.log_metrics(summary)
    log.info(
        f"  Inference latency: avg={summary['inference_time_ms_avg']:.2f}ms "
        f"std={summary['inference_time_ms_std']:.2f}ms "
        f"(min={summary['inference_time_ms_min']:.2f} max={summary['inference_time_ms_max']:.2f}) "
        f"over {len(lat)} patches"
    )
    return summary


# ── Model builder ─────────────────────────────────────────────────────────────

def build_model(arch, in_channels, num_classes):
    cfg = ARCH_CFG[arch]
    if arch == "deeplabv3plus_cbam":
        model = DeepLabV3PlusCBAM(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=in_channels,
            num_classes=num_classes,
        )
    elif arch == "segformer":
        model = build_segformer(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=in_channels,
            num_classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    n = sum(p.numel() for p in model.parameters())
    log.info(f"  {arch} ({cfg['encoder']}): {n:,} params")
    model._n_params = n
    return model.to(DEVICE)


# ── Confusion matrix ──────────────────────────────────────────────────────────

def _plot_confusion_matrix(preds, labels, save_path):
    """
    Normalized (row-wise) confusion matrix over all NUM_CLASSES classes.
    Rows = ground truth, columns = predicted.
    """
    p = preds.view(-1).numpy()
    l = labels.view(-1).numpy()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, pred in zip(l, p):
        if 0 <= t < NUM_CLASSES and 0 <= pred < NUM_CLASSES:
            cm[t, pred] += 1

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm.astype(float), row_sums,
                         out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_LABELS, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalized)", fontsize=12, fontweight="bold")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


# ── Class-weighted patch sampler ──────────────────────────────────────────────

def _patch_weights(datasets: list) -> np.ndarray:
    """
    Compute a weight per patch across a list of RasterPatchDataset objects.
    Weight = sum over classes of (patch_pixel_count[c] / global_pixel_count[c]).
    Rare-class patches get higher weight → balanced mini-batches.
    Uses the in-memory _cdl array — no S2 I/O.
    """
    ps = datasets[0].patch_size

    # Pass 1: global class pixel counts
    global_counts: dict[int, int] = {}
    for ds in datasets:
        cdl = ds._cdl
        remap = ds._remap_lut
        for r, c in ds.patches:
            patch_cdl = cdl[r:r + ps, c:c + ps]
            remapped  = remap[np.clip(patch_cdl, 0, 255)]
            for cls_id in np.unique(remapped):
                if cls_id == 0:
                    continue
                global_counts[int(cls_id)] = global_counts.get(int(cls_id), 0) + int((remapped == cls_id).sum())

    if not global_counts:
        # Fallback: uniform weights
        return np.ones(sum(len(ds.patches) for ds in datasets), dtype=np.float32)

    # Pass 2: per-patch weight
    weights = []
    for ds in datasets:
        cdl   = ds._cdl
        remap = ds._remap_lut
        for r, c in ds.patches:
            patch_cdl = cdl[r:r + ps, c:c + ps]
            remapped  = remap[np.clip(patch_cdl, 0, 255)]
            w = 0.0
            for cls_id in np.unique(remapped):
                if cls_id == 0:
                    continue
                cnt = int((remapped == cls_id).sum())
                w  += cnt / global_counts[int(cls_id)]
            weights.append(w if w > 0 else 1e-6)

    return np.array(weights, dtype=np.float64)


# ── Augmentation wrapper ───────────────────────────────────────────────────────

class AugmentedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies geometric + spectral augmentations to (img, mask).

    Spectral augmentation is *per-band* (B1..B12) not per-channel. All channels
    belonging to the same S2 band (e.g., all B4 dates) share the same scale and
    offset within one augmentation. This is physically faithful — atmospheric
    scattering, sensor calibration, and BRDF effects are band-specific but
    time-consistent for a fixed sensor.

    Designed to improve spatial generalisation by simulating cross-area
    reflectance variation (different atmospheric/illumination conditions in
    held-out areas).

    Args:
        subset:        underlying torch Dataset / Subset producing (img, mask).
        band_indices:  global band index per channel of img — used to map each
                       channel to its S2 band (B1..B12). If None, falls back to
                       per-channel augmentation.
        band_scale:    per-band multiplicative scale range (default ±15%).
        band_offset:   per-band additive offset range on normalised reflectance
                       (default ±5%, simulates haze/aerosol).
        brightness:    global multiplicative scale applied to all channels
                       (default ±10%, simulates illumination).
        gamma:         per-band gamma correction range (1±0.15), simulates
                       nonlinear sensor/atmosphere response.
        noise_std:     additive Gaussian noise sigma (default 0.05).
        drop_p:        per-channel random dropout probability (default 0.05).
        erase_p:       random erasing probability (default 0.5).
    """

    def __init__(self, subset, band_indices=None,
                 band_scale=0.15, band_offset=0.05, brightness=0.10,
                 gamma=0.15, noise_std=0.05, drop_p=0.05, erase_p=0.5):
        self.subset      = subset
        self.band_scale  = band_scale
        self.band_offset = band_offset
        self.brightness  = brightness
        self.gamma_range = gamma
        self.noise_std   = noise_std
        self.drop_p      = drop_p
        self.erase_p     = erase_p

        # Pre-compute per-channel → per-band lookup (LongTensor, K,)
        if band_indices is not None:
            ch2band = np.asarray(
                [int(bi) % N_BANDS_PER_DATE for bi in band_indices],
                dtype=np.int64,
            )
            self.ch2band = torch.from_numpy(ch2band)   # (K,)
            self.n_bands = N_BANDS_PER_DATE
        else:
            self.ch2band = None
            self.n_bands = None

    def __len__(self):
        return len(self.subset)

    def _per_channel_from_band(self, per_band_vals):
        """Expand per-band values (n_bands,) → per-channel (K, 1, 1) via lookup."""
        if self.ch2band is None:
            raise RuntimeError("band_indices not set; per-band augmentation unavailable")
        return per_band_vals[self.ch2band].view(-1, 1, 1)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]   # img: (C,H,W) float [0,1] (percentile-normalised), mask: (H,W)

        # ── Geometric ────────────────────────────────────────────────────────
        if torch.rand(1).item() > 0.5:
            img  = torch.flip(img,  [-1])
            mask = torch.flip(mask, [-1])
        if torch.rand(1).item() > 0.5:
            img  = torch.flip(img,  [-2])
            mask = torch.flip(mask, [-2])
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img  = torch.rot90(img,  k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])

        C = img.shape[0]

        # ── Per-band spectral augmentation ──────────────────────────────────
        if self.ch2band is not None:
            # Per-band scale: simulates atmospheric/sensor variation per wavelength
            band_scale  = 1.0 + (torch.rand(self.n_bands) - 0.5) * 2.0 * self.band_scale
            band_offset = (torch.rand(self.n_bands) - 0.5) * 2.0 * self.band_offset
            scale       = self._per_channel_from_band(band_scale)
            offset      = self._per_channel_from_band(band_offset)
            img = img * scale + offset

            # Per-band gamma: nonlinear response variation
            band_gamma = 1.0 + (torch.rand(self.n_bands) - 0.5) * 2.0 * self.gamma_range
            gamma      = self._per_channel_from_band(band_gamma)
            img = img.clamp(min=0.0).pow(gamma)
        else:
            # Fallback per-channel (no band_indices supplied)
            scale  = 1.0 + (torch.rand(C, 1, 1) - 0.5) * 2.0 * self.band_scale
            offset = (torch.rand(C, 1, 1) - 0.5) * 2.0 * self.band_offset
            img = img * scale + offset

        # ── Global brightness (illumination simulation) ─────────────────────
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 2.0 * self.brightness
        img = img * brightness

        # ── Per-channel random dropout ──────────────────────────────────────
        if self.drop_p > 0:
            drop_mask = (torch.rand(C, 1, 1) > self.drop_p).float()
            img = img * drop_mask

        # ── Gaussian noise ──────────────────────────────────────────────────
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std

        # ── Random erasing ──────────────────────────────────────────────────
        if torch.rand(1).item() < self.erase_p:
            H, W = img.shape[-2], img.shape[-1]
            rh   = int(H * (0.1 + 0.1 * torch.rand(1).item()))
            rw   = int(W * (0.1 + 0.1 * torch.rand(1).item()))
            r0   = torch.randint(0, H - rh + 1, (1,)).item()
            c0   = torch.randint(0, W - rw + 1, (1,)).item()
            img[:, r0:r0 + rh, c0:c0 + rw] = 0.0

        return img, mask


# ── Per-band percentile normalisation ─────────────────────────────────────────
# Compute 1st/99th percentile per S2 band (B1-B12) from training pixels.
# Sentinel-Hub finding: percentile clipping outperforms z-score and min/max for
# long-tailed S2 reflectance. Per-band stats are area-invariant when computed
# from a representative sample across the training region.

NORM_MODES = ("percentile", "minmax", "zscore")


def _sample_per_band(s2_paths, n_samples_per_file=50_000, seed=42):
    """Return list[np.ndarray] — one array of valid samples per S2 band."""
    rng = np.random.default_rng(seed)
    samples: list = [[] for _ in range(N_BANDS_PER_DATE)]
    for path in s2_paths:
        try:
            with rasterio.open(path) as src:
                h, w, nb = src.height, src.width, src.count
                if nb != N_BANDS_PER_DATE:
                    continue
                n_pick = min(n_samples_per_file, h * w)
                ys = rng.integers(0, h, n_pick)
                xs = rng.integers(0, w, n_pick)
                for b in range(1, nb + 1):
                    arr = src.read(b)
                    vals = arr[ys, xs].astype(np.float32)
                    vals = vals[np.isfinite(vals) & (vals != S2_NODATA)]
                    samples[b - 1].append(vals)
        except Exception as e:
            log.warning(f"  [norm] {Path(path).name}: read failed — {e}")
    return [np.concatenate(s) if s else np.array([0.0, 10000.0], dtype=np.float32)
            for s in samples]


def compute_per_band_percentiles(s2_paths, n_samples_per_file=50_000,
                                  percentiles=(2.0, 98.0), seed=42):
    """Compute (p_lo, p_hi) per S2 band. Default P2/P98 (ablation baseline).

    Returns: (lo, hi), each shape (N_BANDS_PER_DATE,) float32.
    """
    log.info(f"  [norm:percentile] P{percentiles[0]}/P{percentiles[1]}  "
             f"{n_samples_per_file} px/file × {len(s2_paths)} files …")
    samples = _sample_per_band(s2_paths, n_samples_per_file, seed)
    lo  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    hi  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    for b in range(N_BANDS_PER_DATE):
        lo[b], hi[b] = np.percentile(samples[b], percentiles)
        log.info(f"    {S2_BAND_NAMES[b]}: lo={lo[b]:.1f}  hi={hi[b]:.1f}")
    return lo, hi


def compute_per_band_minmax(s2_paths, n_samples_per_file=50_000, seed=42):
    """Compute (min, max) per S2 band for min-max normalization → [0, 1].

    Returns: (lo, hi), each shape (N_BANDS_PER_DATE,) float32.
    """
    log.info(f"  [norm:minmax] {n_samples_per_file} px/file × {len(s2_paths)} files …")
    samples = _sample_per_band(s2_paths, n_samples_per_file, seed)
    lo  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    hi  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    for b in range(N_BANDS_PER_DATE):
        lo[b] = float(samples[b].min())
        hi[b] = float(samples[b].max())
        log.info(f"    {S2_BAND_NAMES[b]}: min={lo[b]:.1f}  max={hi[b]:.1f}")
    return lo, hi


def compute_per_band_zscore(s2_paths, n_samples_per_file=50_000, seed=42):
    """Compute (mean, std) per S2 band for z-score normalization: (x - mean) / std.

    Returns: (mean, std), each shape (N_BANDS_PER_DATE,) float32.
    """
    log.info(f"  [norm:zscore] {n_samples_per_file} px/file × {len(s2_paths)} files …")
    samples = _sample_per_band(s2_paths, n_samples_per_file, seed)
    lo  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    hi  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    for b in range(N_BANDS_PER_DATE):
        lo[b] = float(samples[b].mean())
        hi[b] = float(samples[b].std()) or 1.0
        log.info(f"    {S2_BAND_NAMES[b]}: mean={lo[b]:.1f}  std={hi[b]:.1f}")
    return lo, hi


def load_or_compute_norm_stats(norm_mode, s2_paths, cache_dir):
    """Load or compute (lo, hi) normalization stats for the given norm_mode.

    Returns (lo, hi) each shape (N_BANDS_PER_DATE,) float32.
    Cache keyed by norm_mode — different modes never share a cache file.
    """
    assert norm_mode in NORM_MODES, f"norm_mode must be one of {NORM_MODES}"
    cache_path = Path(cache_dir) / f"norm_stats_{norm_mode}.npz"
    if cache_path.exists():
        d = np.load(str(cache_path))
        log.info(f"  [norm:{norm_mode}] Loaded from cache → {cache_path.name}")
        return d["lo"].astype(np.float32), d["hi"].astype(np.float32)
    if norm_mode == "percentile":
        lo, hi = compute_per_band_percentiles(s2_paths)
    elif norm_mode == "minmax":
        lo, hi = compute_per_band_minmax(s2_paths)
    else:  # zscore
        lo, hi = compute_per_band_zscore(s2_paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), lo=lo, hi=hi)
    log.info(f"  [norm:{norm_mode}] Cached → {cache_path.name}")
    return lo, hi


# Keep old name as alias for backward compat
def load_or_compute_band_percentiles(s2_paths, cache_path):
    return load_or_compute_norm_stats("percentile", s2_paths, Path(cache_path).parent)


def _channel_to_band_idx(dataset_band_indices):
    """Map each selected channel → its S2 band index (0..N_BANDS_PER_DATE-1)."""
    if dataset_band_indices is None:
        # All channels of all files — band cycles every N_BANDS_PER_DATE channels
        return None
    return np.asarray([bi % N_BANDS_PER_DATE for bi in dataset_band_indices], dtype=np.int64)


def _per_channel_percentiles(band_indices, plo_per_band, phi_per_band):
    """Expand (N_BANDS,) per-band (lo, hi) stats to (n_ch,) per-channel via band lookup.

    lo/hi are the norm_mode stats per band — for percentile mode that's P2/P98
    """
    band_idx_per_ch = _channel_to_band_idx(band_indices)
    if band_idx_per_ch is None:
        raise ValueError("band_indices required for per-band percentile lookup")
    return plo_per_band[band_idx_per_ch], phi_per_band[band_idx_per_ch]


# ── In-memory dataset cache ───────────────────────────────────────────────────

class PreloadedDataset(torch.utils.data.Dataset):
    """Builds a persistent disk cache of all patches; loads imgs via memory-map.

    Reads each TIF file once in full (parallel threads) instead of per-patch
    window reads → ~30–60s instead of 15+ min for large datasets.
    Cache key covers s2_paths/cdl_path/bands/patch_size.

    Imgs stored as float16 .npy → loaded with mmap_mode='r' so the OS pages in
    only what each minibatch needs.  Peak RAM = model + batch, not full dataset.
    Masks stored as int64 .pt (typically <1 GB, always in RAM).
    """

    def __init__(self, dataset, desc="preload", cache_dir=None, n_threads=None,
                 channel_stats=None, band_percentiles=None, norm_mode="percentile"):
        """Normalisation: per-band stats → normalised values stored as float16.

        band_percentiles: (lo, hi) each shape (N_BANDS_PER_DATE,). Required.
          Semantics depend on norm_mode:
            percentile → (p2, p98) clip to [0,1]
            minmax     → (min, max) clip to [0,1]
            zscore     → (mean, std) no clip
        channel_stats: deprecated, kept for API compat (ignored).
        norm_mode: one of NORM_MODES ("percentile", "minmax", "zscore").
        """
        assert band_percentiles is not None, "band_percentiles (lo, hi) required"
        assert norm_mode in NORM_MODES, f"norm_mode must be one of {NORM_MODES}"
        self._norm_mode = norm_mode
        imgs_path, masks_path = self._cache_paths(dataset, cache_dir, norm_mode) if cache_dir else (None, None)

        if imgs_path and imgs_path.exists() and masks_path and masks_path.exists():
            log.info(f"  [{desc}] Cache hit → mmap {imgs_path.name}")
            t0 = time.time()
            self._imgs  = np.load(str(imgs_path), mmap_mode="r")
            self._masks = torch.load(masks_path, map_location="cpu", weights_only=True)
            gb_disk = imgs_path.stat().st_size / 1e9
            log.info(f"  [{desc}] mmap ready in {time.time()-t0:.1f}s ({gb_disk:.2f} GB on disk)")
            return

        log.info(f"  [{desc}] Cache miss → preloading from {len(dataset._s2_srcs)} TIF files …")
        t0 = time.time()

        n  = len(dataset)
        ps = dataset.patch_size
        band_indices  = dataset.band_indices
        n_ch_per_file = [src.count for src in dataset._s2_srcs]
        ch_offsets    = np.cumsum([0] + n_ch_per_file).tolist()
        n_ch          = len(band_indices) if band_indices is not None else ch_offsets[-1]

        # file_extraction[fi] = [(output_col, local_band_idx_1based), ...]
        file_extraction: dict = {}
        targets = band_indices if band_indices is not None else list(range(ch_offsets[-1]))
        for out_pos, gi in enumerate(targets):
            for fi in range(len(n_ch_per_file)):
                if ch_offsets[fi] <= gi < ch_offsets[fi + 1]:
                    file_extraction.setdefault(fi, []).append((out_pos, gi - ch_offsets[fi] + 1))
                    break

        patches = dataset.patches
        nodata  = dataset.nodata

        # Allocate buf as a disk-backed float16 memmap — never occupies RAM regardless
        # of channel count. 70ch × 1800 patches × 256² × float32 ≈ 33 GB; float16
        # memmap keeps peak RAM to ~O(one TIF file) during the fill loop.
        _buf_path = (imgs_path.with_suffix(".tmp.npy") if imgs_path
                     else Path(PRELOAD_CACHE_DIR) / f"_tmp_{os.getpid()}.npy")
        _buf_path.parent.mkdir(parents=True, exist_ok=True)
        buf = np.lib.format.open_memmap(
            str(_buf_path), mode="w+", dtype=np.float16, shape=(n, n_ch, ps, ps)
        )
        gb_alloc = buf.nbytes / 1e9
        log.info(f"  [{desc}] Buf: {n}×{n_ch}×{ps}×{ps} float16 = {gb_alloc:.1f} GB on disk")

        def _read_one_file(fi):
            extractions = file_extraction[fi]
            local_idxs  = [e[1] for e in extractions]
            out_cols    = [e[0] for e in extractions]
            try:
                with rasterio.open(dataset.s2_paths[fi]) as src:
                    arr = src.read(indexes=local_idxs).astype(np.float32)
                arr[arr == nodata]      = 0.0
                arr[~np.isfinite(arr)]  = 0.0
                return fi, arr, out_cols
            except Exception as e:
                log.warning(f"  [{desc}] read failed file {fi}: {e}")
                return fi, None, out_cols

        # Single-threaded write to memmap — concurrent writes to overlapping patches
        # cause data races; read threads are fine, write serialised via main thread.
        _n_threads = n_threads or min(len(file_extraction), os.cpu_count() or 8)
        log.info(f"  [{desc}] Using {_n_threads} read threads for {len(file_extraction)} files")
        with ThreadPoolExecutor(max_workers=_n_threads) as pool:
            for fi, arr, out_cols in pool.map(_read_one_file, list(file_extraction.keys())):
                if arr is None:
                    continue
                for ci, out_pos in enumerate(out_cols):
                    band_plane = arr[ci]
                    for pi, (r, c) in enumerate(patches):
                        buf[pi, out_pos, :, :] = band_plane[r:r+ps, c:c+ps]
                del arr

        # Per-band normalisation using norm_mode stats.
        lo_per_ch, hi_per_ch = _per_channel_percentiles(band_indices, *band_percentiles)
        denom = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
        lo_b  = lo_per_ch[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)
        d_b   = denom[np.newaxis, :, np.newaxis, np.newaxis]
        CHUNK = 128
        log.info(f"  [{desc}] Normalising with norm_mode={norm_mode} …")
        for start in range(0, n, CHUNK):
            end   = min(start + CHUNK, n)
            chunk = buf[start:end].astype(np.float32)
            chunk = (chunk - lo_b) / d_b
            if norm_mode != "zscore":
                chunk = np.clip(chunk, 0.0, 1.0)
            buf[start:end] = chunk.astype(np.float16)
        buf.flush()

        masks = [
            torch.from_numpy(
                dataset._remap_lut[np.clip(dataset._cdl[r:r+ps, c:c+ps], 0, 255)].astype(np.int64)
            )
            for r, c in patches
        ]
        self._masks = torch.stack(masks)

        elapsed = time.time() - t0
        log.info(f"  [{desc}] Preloaded in {elapsed:.1f}s — {gb_alloc:.1f} GB float16 on disk")

        if imgs_path:
            del buf  # close write-mode memmap before rename (WSL/NTFS: open handle blocks rename+reopen)
            _buf_path.rename(imgs_path)
            torch.save(self._masks, masks_path)
            log.info(f"  [{desc}] Cached → {imgs_path.name} + {masks_path.name}")
            self._imgs = np.load(str(imgs_path), mmap_mode="r")
        else:
            self._imgs = np.array(buf)   # no cache dir: load into RAM
            del buf
            _buf_path.unlink(missing_ok=True)

    @staticmethod
    def _cache_paths(dataset, cache_dir, norm_mode):
        key = {
            "s2":             sorted(os.path.basename(str(p)) for p in dataset.s2_paths),
            "cdl":            os.path.basename(str(dataset.cdl_path)),
            "ps":             dataset.patch_size,
            "bands":          list(dataset.band_indices) if dataset.band_indices is not None else None,
            "stride":         getattr(dataset, "stride", None),
            "min_valid_frac": getattr(dataset, "min_valid_frac", None),
            "n_patches":      len(dataset.patches),
            "norm":           f"norm_v2_{norm_mode}",  # invalidates pre-norm_mode caches
        }
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        base = Path(cache_dir) / f"preload_{h}"
        return base.with_suffix(".npy"), base.with_name(base.name + "_masks.pt")

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, idx):
        # np array (memmap or plain) → float32 tensor; .copy() required for mmap slices
        img = torch.tensor(self._imgs[idx], dtype=torch.float32)
        return img, self._masks[idx]


# ── No-preload: on-the-fly normalisation ─────────────────────────────────────

class NormalizedDataset(torch.utils.data.Dataset):
    """Per-band normalisation wrapper — on-the-fly, no disk cache.

    norm_mode: "percentile" (P2/P98, clip [0,1]) | "minmax" (clip [0,1]) | "zscore" (no clip).
    band_percentiles: (lo, hi) per-band stats (semantics depend on norm_mode).
    """

    def __init__(self, dataset, channel_stats=None, band_percentiles=None,
                 norm_mode="percentile"):
        assert band_percentiles is not None, "band_percentiles (lo, hi) required"
        assert norm_mode in NORM_MODES, f"norm_mode must be one of {NORM_MODES}"
        self.dataset   = dataset
        self.norm_mode = norm_mode
        lo_per_ch, hi_per_ch = _per_channel_percentiles(dataset.band_indices, *band_percentiles)
        denom = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
        self.lo    = torch.tensor(lo_per_ch.astype(np.float32)).view(-1, 1, 1)
        self.denom = torch.tensor(denom).view(-1, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        img = (img.float() - self.lo) / self.denom
        if self.norm_mode != "zscore":
            img = img.clamp(0.0, 1.0)
        return img, mask


def _compute_channel_stats_full(dataset, n_threads=None):
    """Full-pass per-channel mean/std — same math as PreloadedDataset, no memmap.

    Reads each TIF file once (parallel), extracts patches in chunks, accumulates
    ch_sums / ch_sums2 in float64. Peak RAM = one TIF in memory + ~64 MB per chunk.
    Identical stats to PreloadedDataset; avoids the disk-backed memmap allocation
    that causes OOM at high channel counts.
    """
    n   = len(dataset)
    ps  = dataset.patch_size
    band_indices  = dataset.band_indices
    n_ch_per_file = [src.count for src in dataset._s2_srcs]
    ch_offsets    = np.cumsum([0] + n_ch_per_file).tolist()
    n_ch          = len(band_indices) if band_indices is not None else ch_offsets[-1]
    patches       = dataset.patches
    nodata        = dataset.nodata

    # Same file_extraction map as PreloadedDataset
    file_extraction: dict = {}
    targets = band_indices if band_indices is not None else list(range(ch_offsets[-1]))
    for out_pos, gi in enumerate(targets):
        for fi in range(len(n_ch_per_file)):
            if ch_offsets[fi] <= gi < ch_offsets[fi + 1]:
                file_extraction.setdefault(fi, []).append((out_pos, gi - ch_offsets[fi] + 1))
                break

    ch_sums  = np.zeros(n_ch, dtype=np.float64)
    ch_sums2 = np.zeros(n_ch, dtype=np.float64)
    ch_cnt   = n * ps * ps

    log.info(f"  [stats] Full-pass channel stats: {n} patches, {len(file_extraction)} TIF files …")

    def _read_one_file(fi):
        extractions = file_extraction[fi]
        local_idxs  = [e[1] for e in extractions]
        out_cols    = [e[0] for e in extractions]
        try:
            with rasterio.open(dataset.s2_paths[fi]) as src:
                arr = src.read(indexes=local_idxs).astype(np.float32)
            arr[arr == nodata]     = 0.0
            arr[~np.isfinite(arr)] = 0.0
            return fi, arr, out_cols
        except Exception as e:
            log.warning(f"  [stats] read failed file {fi}: {e}")
            return fi, None, out_cols

    PCHUNK = 64  # patches per accumulation chunk; peak RAM = PCHUNK × ps² × float64 ≈ 67 MB
    _n_threads = n_threads or min(len(file_extraction), os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=_n_threads) as pool:
        for fi, arr, out_cols in pool.map(_read_one_file, list(file_extraction.keys())):
            if arr is None:
                continue
            for ci, out_pos in enumerate(out_cols):
                plane = arr[ci]  # (H, W) float32
                for start in range(0, len(patches), PCHUNK):
                    pslice = patches[start:start + PCHUNK]
                    batch  = np.stack([plane[r:r + ps, c:c + ps] for r, c in pslice])
                    flat   = batch.astype(np.float64).ravel()
                    ch_sums[out_pos]  += flat.sum()
                    ch_sums2[out_pos] += (flat * flat).sum()
            del arr

    means = (ch_sums / ch_cnt).astype(np.float32)
    stds  = np.sqrt(np.maximum(ch_sums2 / ch_cnt - means.astype(np.float64) ** 2, 0)).astype(np.float32)
    stds  = np.where(stds < 1.0, 1.0, stds)
    log.info(f"  [stats]  mean [{means.min():.1f}, {means.max():.1f}]"
             f"  std [{stds.min():.1f}, {stds.max():.1f}]")
    return means, stds


# ── Spatial test area evaluation ─────────────────────────────────────────────

def _evaluate_spatial_area(
    model,
    area: dict,
    band_names: list,
    exp_name: str,
    exp_dir: Path,
    skip_viz: bool = False,
    channel_stats: "tuple | None" = None,  # kept for API compat, unused
    band_percentiles: "tuple | None" = None,
    no_preload: bool = False,
    norm_mode: str = "percentile",
) -> "dict | None":
    """Evaluate model on one held-out spatial test area.

    area: {"name": str, "s2_dir": Path, "cdl": Path}
    band_names: channel names from experiment (e.g. ["B4_20240730", ...]).
    Returns evaluate_test_set result dict, or None if area data missing.
    """
    import glob as _glob

    area_name = area["name"]
    s2_dir    = Path(area["s2_dir"])
    cdl_path  = Path(area["cdl"])

    area_s2 = sorted(f for f in _glob.glob(str(s2_dir / "*.tif")) if not Path(f).name.startswith("._"))
    if not area_s2:
        log.warning(f"  Spatial test {area_name}: no S2 files in {s2_dir} — skipping")
        return None
    if not cdl_path.exists():
        log.warning(f"  Spatial test {area_name}: CDL not found at {cdl_path} — skipping")
        return None

    log.info(f"  Spatial test [{area_name}]: {len(area_s2)} S2 files, CDL={cdl_path.name}")

    _, area_band_to_idx, _, _ = build_local_band_map(area_s2)

    area_global_indices = []
    skipped_bands = []
    for bname in band_names:
        idx = area_band_to_idx.get(bname)
        if idx is not None:
            area_global_indices.append(idx)
        else:
            skipped_bands.append(bname)

    if skipped_bands:
        log.warning(f"  Spatial test {area_name}: {len(skipped_bands)} band(s) not found in area files (date mismatch?): {skipped_bands[:3]}...")
    if not area_global_indices:
        log.error(f"  Spatial test {area_name}: no matching bands — skipping")
        return None

    area_s2_filtered, area_idx_local = _filter_s2_by_band_indices(area_s2, area_global_indices)

    area_ds = RasterPatchDataset(
        s2_paths=area_s2_filtered, cdl_path=str(cdl_path),
        patch_size=PATCH_SIZE, stride=STRIDE,
        keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
        min_valid_frac=MIN_VALID_FRAC, band_indices=area_idx_local,
    )
    if no_preload:
        area_norm = NormalizedDataset(area_ds, band_percentiles=band_percentiles,
                                      norm_mode=norm_mode)
    else:
        area_norm = PreloadedDataset(area_ds, desc=area_name, cache_dir=PRELOAD_CACHE_DIR,
                                     band_percentiles=band_percentiles, norm_mode=norm_mode)
    area_dl = DataLoader(area_norm, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    area_r = evaluate_test_set(model, area_dl, NUM_CLASSES, DEVICE)
    log.info(f"  [{area_name}] mIoU={area_r['miou']:.4f}  OA={area_r['oa']:.4f}")
    log.info(f"  {'Class':<20} {'IoU':>7}")
    for cls_id, iou in area_r["per_class_iou"].items():
        cdl_id = KEEP_CLASSES[cls_id - 1]
        name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
        log.info(f"  {name:<20} {iou:.4f}" if not np.isnan(iou) else f"  {name:<20}     nan")

    # MLflow metrics prefixed with area name
    mlflow.log_metrics({
        f"{area_name}_miou": area_r["miou"],
        f"{area_name}_mf1":  area_r["mf1"],
        f"{area_name}_oa":   area_r["oa"],
    })
    for cls_id, iou in area_r["per_class_iou"].items():
        if not np.isnan(iou):
            cdl_id = KEEP_CLASSES[cls_id - 1]
            cname  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
            mlflow.log_metric(
                f"{area_name}_iou_{cname.lower().replace('/', '_').replace(' ', '_')}",
                iou,
            )
    for cls_id, f1v in area_r["per_class_f1"].items():
        if not np.isnan(f1v):
            cdl_id = KEEP_CLASSES[cls_id - 1]
            cname  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
            mlflow.log_metric(
                f"{area_name}_f1_{cname.lower().replace('/', '_').replace(' ', '_')}",
                f1v,
            )

    # Per-class metrics CSV (IoU + F1)
    iou_rows = [
        {
            "class_id":   cls_id,
            "cdl_id":     KEEP_CLASSES[cls_id - 1],
            "class_name": CDL_CLASS_NAMES.get(KEEP_CLASSES[cls_id - 1], f"cls{cls_id}"),
            "iou":        round(iou, 4) if not np.isnan(iou) else float("nan"),
            "f1":         round(area_r["per_class_f1"].get(cls_id, float("nan")), 4)
                          if not np.isnan(area_r["per_class_f1"].get(cls_id, float("nan")))
                          else float("nan"),
        }
        for cls_id, iou in area_r["per_class_iou"].items()
    ]
    iou_csv = exp_dir / f"{area_name}_per_class_iou.csv"
    pd.DataFrame(iou_rows).to_csv(iou_csv, index=False)
    mlflow.log_artifact(str(iou_csv))

    # Confusion matrix
    cm_path = exp_dir / f"{area_name}_confusion_matrix.png"
    _plot_confusion_matrix(area_r["preds"], area_r["labels"], str(cm_path))
    mlflow.log_artifact(str(cm_path))

    # Segmentation map
    if not skip_viz:
        gt_map, _   = load_gt_remap(str(cdl_path))
        pred_map, _ = run_full_inference(
            model, area_s2_filtered, area_idx_local,
            patch_size=PATCH_SIZE, stride=PATCH_SIZE,
            channel_stats=None, band_percentiles=band_percentiles,
        )
        seg_path = exp_dir / f"{area_name}_segmentation_map.png"
        save_segmentation_map(
            pred_map, gt_map,
            title=f"{exp_name} — {area_name}",
            save_path=str(seg_path),
        )
        mlflow.log_artifact(str(seg_path))
        del pred_map, gt_map

    return area_r


# ── Main experiment runner ────────────────────────────────────────────────────

# Set by --eval-only in __main__: path to a checkpoint to evaluate instead of training.
EVAL_ONLY_CKPT = None


def run_experiment(
    exp_name,
    arch,
    band_indices,           # list[int]  OR  dict{yr: (list[int], list[str])}
    band_names_list,        # list[str]  (reference year; used for logging/metadata)
    description,
    s2_processed,
    class_weights_tensor,
    class_counts=None,      # required for focal_tversky effective-number weights
    loss="wce",             # "wce" | "focal_tversky" | "dynamic_balanced"
    force=False,
    skip_viz=False,
    no_preload=False,       # skip disk preload cache; use on-the-fly normalisation
    cache_only=False,       # build PreloadedDataset cache then exit without training
    norm_mode="percentile", # "percentile" | "minmax" | "zscore"
    skip_ndvi=False,        # skip NDVI GT-vs-pred disagreement analysis (CalCROP21 method)
):
    """band_indices: list[int] same for all years, or dict{yr: (idx, names)} per-year."""
    cfg           = ARCH_CFG[arch]
    hp            = _resolve_hp(cfg)
    bs            = hp["batch_size"] or BATCH_SIZE   # per-combo batch size override
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_only     = EVAL_ONLY_CKPT is not None
    if eval_only:
        # Write outputs (patch PNGs + test_patch_metrics.csv) to a local dir;
        # load weights from the provided checkpoint path.
        exp_dir   = MODELS_DIR / f"{exp_name}_evalonly_{run_timestamp}"
        best_ckpt = Path(EVAL_ONLY_CKPT)
        last_ckpt = best_ckpt
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Resume support: skip combos already finished (have a `.done` marker)
        # unless --force. Checked before creating a new dir so skips leave no litter.
        if not force and _combo_done(exp_name):
            log.info(f"✓ already done — skipping {exp_name}  (use --force to re-run)")
            return None
        exp_dir   = MODELS_DIR / f"{exp_name}_{run_timestamp}"
        best_ckpt = exp_dir / "best_model.pth"
        last_ckpt = exp_dir / "last_model.pth"
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Per-run log file — captured from start of training; uploaded as MLflow artifact at end
    run_log_path    = exp_dir / f"{exp_name}_train.log"
    run_log_handler = logging.FileHandler(run_log_path, mode="w")
    run_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(run_log_handler)

    per_year = isinstance(band_indices, dict)

    def _yr_idx(yr):
        """Return (idx_list, names_list) for a given year."""
        if per_year:
            if yr in band_indices:
                return band_indices[yr]
            # fallback: use the first available year's indices
            fallback_yr = next(iter(band_indices))
            log.warning(
                f"Exp C projected: year {yr} not in projected map — "
                f"falling back to {fallback_yr} indices"
            )
            return band_indices[fallback_yr]
        return band_indices, band_names_list

    # Pre-pass: find globally consistent band indices available in ALL years.
    # Prevents channel-count mismatch when some years lack a file (e.g. excluded empty date).
    if not per_year:
        base_idx = band_indices  # same list applied to every year
        all_years = list(TRAIN_YEARS) + [TEST_YEAR]
        valid_sets = []
        for yr in all_years:
            yr_s2_all = _s2_for_year(s2_processed, yr)
            valid_sets.append(_valid_global_indices(yr_s2_all, base_idx))
        consistent = sorted(set.intersection(*valid_sets))
        dropped = len(base_idx) - len(consistent)
        if dropped:
            log.warning(
                f"  Dropping {dropped} channel(s) not available in all years "
                f"({', '.join(all_years)}) — keeping {len(consistent)} consistent channels"
            )
        consistent_set  = set(consistent)
        band_names_list = [name for gi, name in zip(base_idx, band_names_list) if gi in consistent_set]
        band_indices    = consistent

    in_channels = len(_yr_idx(TRAIN_YEARS[0])[0])
    log.info(f"\n{'='*65}")
    log.info(f" {exp_name}")
    log.info(f"  arch={arch}  in_channels={in_channels}  per_year_indices={per_year}")
    log.info(f"  {description}")
    log.info(f"{'='*65}\n")

    # ── Per-band normalisation stats (computed once from all training files) ────
    _stats_cache_dir = Path(s2_processed[0]).parent
    _all_train_s2 = []
    for yr in TRAIN_YEARS:
        _all_train_s2.extend(_s2_for_year(s2_processed, yr))
    band_percentiles = load_or_compute_norm_stats(norm_mode, _all_train_s2, _stats_cache_dir)
    log.info(f"  norm_mode={norm_mode}")

    # ── Year-based dataset split ──────────────────────────────────────────────
    train_year_datasets_raw = []   # RasterPatchDataset — for _patch_weights (needs _cdl etc.)
    train_year_datasets     = []   # PreloadedDataset  — for DataLoader
    primary_s2_filtered = None     # S2 paths for primary year (used for segmentation map)
    primary_idx_local   = None     # band indices for primary year
    for yr in TRAIN_YEARS:
        yr_s2  = _s2_for_year(s2_processed, yr)
        yr_cdl = CDL_TRAIN
        if not yr_s2 or not yr_cdl.exists():
            log.warning(f"Skipping train year {yr}: {'no S2' if not yr_s2 else 'CDL missing'}")
            continue
        yr_idx, _ = _yr_idx(yr)
        yr_s2_filtered, yr_idx_local = _filter_s2_by_band_indices(yr_s2, yr_idx)
        if primary_s2_filtered is None:
            primary_s2_filtered = yr_s2_filtered
            primary_idx_local   = yr_idx_local
        ds_raw = RasterPatchDataset(
            s2_paths=yr_s2_filtered, cdl_path=str(yr_cdl),
            patch_size=PATCH_SIZE, stride=STRIDE,
            keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
            min_valid_frac=MIN_VALID_FRAC, band_indices=yr_idx_local,
        )
        log.info(f"  [{yr}] {len(ds_raw):,} patches  ({len(yr_idx)} channels, {len(yr_s2_filtered)}/{len(yr_s2)} files)")
        train_year_datasets_raw.append(ds_raw)
        if no_preload:
            train_year_datasets.append(NormalizedDataset(ds_raw, band_percentiles=band_percentiles,
                                                          norm_mode=norm_mode))
        else:
            preloaded = PreloadedDataset(ds_raw, desc=yr, cache_dir=PRELOAD_CACHE_DIR,
                                         band_percentiles=band_percentiles, norm_mode=norm_mode)
            train_year_datasets.append(preloaded)

    assert train_year_datasets, "No training data for any TRAIN_YEAR"

    if cache_only:
        log.info(f"  [--build-cache-only] Cache built for {exp_name} — skipping training")
        log.removeHandler(run_log_handler)
        run_log_handler.close()
        return None

    train_val_ds = ConcatDataset(train_year_datasets)

    # Split: train / val / test — spatial block (grid) split; whole blocks per
    # split, prevents patch-adjacency spatial leakage. When TEST_FRAC=0: test
    # evaluation is skipped.
    n_total   = len(train_val_ds)
    tr_idx, va_idx, te_idx, split_info = _block_spatial_split(
        train_year_datasets_raw, BLOCK_SIZE, VAL_FRAC, TEST_FRAC,
        NUM_CLASSES, SEED, min_class_frac=MIN_CLASS_FRAC, log=log,
    )
    train_ds = torch.utils.data.Subset(train_val_ds, tr_idx)
    val_ds   = torch.utils.data.Subset(train_val_ds, va_idx)
    test_ds  = torch.utils.data.Subset(train_val_ds, te_idx) if te_idx else None
    n_train, n_val, n_test = len(tr_idx), len(va_idx), len(te_idx)
    split_label = f"block_spatial_{int(round((1-VAL_FRAC-TEST_FRAC)*100))}_{int(round(VAL_FRAC*100))}_{int(round(TEST_FRAC*100))}"
    split_artifacts = _save_block_split_artifacts(
        split_info, exp_dir, exp_name,
        class_names=[CDL_CLASS_NAMES[c] for c in KEEP_CLASSES], log=log,
    )
    test_s2_filtered = None
    test_idx_local   = None

    # Class-weighted sampler: rare-class patches sampled more frequently
    log.info("  Computing patch weights for class-balanced sampling...")
    all_weights = _patch_weights(train_year_datasets_raw)
    train_weights = all_weights[train_ds.indices]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_weights).double(),
        num_samples=n_train,
        replacement=True,
    )
    # Band indices threaded through to enable per-band (vs per-channel) spectral
    # augmentation. For per-year dict, use the primary year's indices.
    _aug_bi = primary_idx_local if isinstance(band_indices, dict) else band_indices
    aug_train_ds = AugmentedSubset(train_ds, band_indices=_aug_bi)
    # In --eval-only with on-the-fly (no-preload) datasets, workers can't pickle open
    # rasterio handles under macOS spawn; use 0 workers (single test pass, speed is fine).
    _nw = 0 if eval_only else 4
    train_dl = DataLoader(aug_train_ds, batch_size=bs, sampler=sampler, num_workers=_nw, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,       batch_size=bs, shuffle=False,   num_workers=_nw, pin_memory=True)
    test_dl  = DataLoader(test_ds,      batch_size=bs, shuffle=False,   num_workers=_nw, pin_memory=True) if test_ds is not None else None
    if n_test > 0:
        log.info(f"  Patches: {n_train:,} train / {n_val:,} val / {n_test:,} test ({split_label})")
    else:
        log.info(f"  Patches: {n_train:,} train / {n_val:,} val  (no test split — TEST_FRAC=0)")

    # ── Model + optimiser + scheduler + loss ──────────────────────────────────
    model     = build_model(arch, in_channels, NUM_CLASSES)
    grad_clip = hp["grad_clip"]
    optimizer = _build_optimizer(
        hp["optimizer"], model.parameters(),
        lr=hp["lr"], weight_decay=hp["weight_decay"], momentum=hp["momentum"],
    )
    scheduler = _build_scheduler(
        optimizer, MAX_EPOCHS, power=hp["sched_power"],
        warmup_epochs=hp["warmup_epochs"], kind=hp["scheduler"],
    )
    _decay_label = (f"CosineAnnealingLR" if hp["scheduler"] == "cosine"
                    else f"PolynomialLR(power={hp['sched_power']:g})")
    _sched_label = _decay_label + (f"+LinearWarmup({hp['warmup_epochs']}ep)" if hp["warmup_epochs"] else "")
    _opt_label   = {"adamw": "AdamW", "adam": "Adam", "sgd": f"SGD(m={hp['momentum']:g})"}[hp["optimizer"]]
    if HP_OVERRIDE:
        log.info(
            f"  HP override: opt={_opt_label} lr={hp['lr']:.2e} wd={hp['weight_decay']:.2e} "
            f"bs={bs} sched={_sched_label} grad_clip={grad_clip or 'off'}"
        )

    # ── Loss function (named) ──────────────────────────────────────────────
    if loss == "focal_tversky":
        criterion = build_focal_tversky(
            class_counts=class_counts,
            tv_alpha=0.7, tv_beta=0.3, tv_gamma=0.75,
        ).to(DEVICE)
        log.info("  Loss=focal_tversky — Focal Tversky (median-freq weighted class-mean)")
    elif loss == "dynamic_balanced":
        criterion = build_dynamic_balanced(
            num_classes=NUM_CLASSES, beta=0.9999, fallback_weight=2.0,
        ).to(DEVICE)
        log.info("  Loss=dynamic_balanced — Dynamic Effective Class Balanced (per-batch, β=0.9999)")
    else:
        criterion = build_wce(class_weights_tensor.to(DEVICE))
        log.info("  Loss=wce — WeightedCrossEntropy")

    # ── MLflow run (child — nested under parent created in main()) ────────────

    with mlflow.start_run(run_name=exp_name, nested=True, log_system_metrics=True) as run:
        mlflow.log_params({
            "experiment":     exp_name,
            "architecture":   arch,
            "encoder":        cfg["encoder"],
            "in_channels":    in_channels,
            "num_classes":    NUM_CLASSES,
            "patch_size":     PATCH_SIZE,
            "stride":         STRIDE,
            "batch_size":     bs,
            "max_epochs":     MAX_EPOCHS,
            "early_stopping": EARLY_STOP,
            "learning_rate":  hp["lr"],
            "weight_decay":   hp["weight_decay"],
            "warmup_epochs":  hp["warmup_epochs"],
            "sched_power":    hp["sched_power"],
            "grad_clip":      grad_clip,
            "optimizer":      _opt_label,
            "lr_scheduler":   _sched_label,
            "loss":           loss,
            "norm_mode":      norm_mode,
            "train_years":    str(TRAIN_YEARS),
            "test_year":      TEST_YEAR,
            "train_patches":  n_train,
            "val_patches":    n_val,
            "test_patches":   n_test,
            "split":          split_label,
            "block_size":     BLOCK_SIZE,
            "n_blocks":       (split_info or {}).get("n_blocks"),
            "description":    description,
            "keep_classes":   str(KEEP_CLASSES),
            "model_params":   getattr(model, "_n_params", None),
            **_get_hardware_info(),
        })
        mlflow.set_tag("band_names", str(band_names_list))
        mlflow.set_tag("n_bands",    str(in_channels))
        mlflow.set_tag(
            "mlflow.note.content",
            f"{description}. Arch={arch} ({cfg['encoder']}), {in_channels} input "
            f"channels, loss={loss}. Trained on {TRAIN_YEARS}, tested on {TEST_YEAR} "
            f"({split_label}: {n_train} train / {n_val} val / {n_test} test patches).",
        )


        # ── Training loop ─────────────────────────────────────────────────────
        best_miou              = 0.0
        best_val_mf1           = 0.0
        best_val_oa            = 0.0
        best_val_per_class_iou = {}
        best_val_per_class_f1  = {}
        no_improve             = 0
        history                = []
        t_start    = time.time()

        if eval_only:
            log.info(f"  [--eval-only] Skipping training — evaluating checkpoint {best_ckpt}")

        for epoch in ([] if eval_only else range(MAX_EPOCHS)):
            t_ep = time.time()

            model.train()
            train_loss_acc, n_batches = 0.0, 0
            _logged_vram = epoch > 0   # log VRAM once on first batch of epoch 0
            for imgs, masks in train_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                imgs        = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = criterion(logits, masks)

                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss_acc += loss.item()
                n_batches += 1

                if not _logged_vram and torch.cuda.is_available():
                    alloc  = torch.cuda.memory_allocated()  / 1024**3
                    reserv = torch.cuda.memory_reserved()   / 1024**3
                    log.info(f"  [VRAM] allocated={alloc:.2f} GB  reserved={reserv:.2f} GB")
                    _logged_vram = True

            train_loss = train_loss_acc / n_batches
            val_m = validate_one_epoch(model, val_dl, criterion, DEVICE, NUM_CLASSES)
            scheduler.step()

            ep_t = time.time() - t_ep
            per_cls_metrics = {}
            for cls_id, iou in val_m["per_class_iou"].items():
                if not np.isnan(iou):
                    cdl_id = KEEP_CLASSES[cls_id - 1]
                    name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                    slug   = name.lower().replace('/', '_').replace(' ', '_')
                    per_cls_metrics[f"val_iou_{slug}"] = iou
            for cls_id, f1v in val_m["per_class_f1"].items():
                if not np.isnan(f1v):
                    cdl_id = KEEP_CLASSES[cls_id - 1]
                    name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                    slug   = name.lower().replace('/', '_').replace(' ', '_')
                    per_cls_metrics[f"val_f1_{slug}"] = f1v
            for cls_id, oav in val_m["per_class_oa"].items():
                if not np.isnan(oav):
                    cdl_id = KEEP_CLASSES[cls_id - 1]
                    name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                    slug   = name.lower().replace('/', '_').replace(' ', '_')
                    per_cls_metrics[f"val_oa_{slug}"] = oav
            mlflow.log_metrics({
                "train_loss":   train_loss,
                "val_loss":     val_m["loss"],
                "val_miou":     val_m["miou"],
                "val_mf1":      val_m["mf1"],
                "val_oa":       val_m["oa"],
                "lr":           scheduler.get_last_lr()[0],
                "epoch_time_s": ep_t,
                **per_cls_metrics,
            }, step=epoch)

            history.append({
                "epoch":      epoch + 1,
                "train_loss": round(train_loss,       4),
                "val_loss":   round(val_m["loss"],    4),
                "val_miou":   round(val_m["miou"],    4),
                "val_mf1":    round(val_m["mf1"],     4),
                "val_oa":     round(val_m["oa"],      4),
                "epoch_t_s":  round(ep_t,              1),
            })

            if val_m["miou"] > best_miou + EARLY_STOP_DELTA:
                best_miou              = val_m["miou"]
                best_val_mf1           = val_m["mf1"]
                best_val_oa            = val_m["oa"]
                best_val_per_class_iou = val_m["per_class_iou"]
                best_val_per_class_f1  = val_m["per_class_f1"]
                no_improve = 0
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_miou":        best_miou,
                    "band_indices":     band_indices,
                    "band_names":       band_names_list,
                    "in_channels":      in_channels,
                    "num_classes":      NUM_CLASSES,
                    "architecture":     arch,
                }, best_ckpt)
            else:
                no_improve += 1

            total_min = (time.time() - t_start) / 60
            log.info(
                f"  Ep {epoch+1:3d}/{MAX_EPOCHS} "
                f"loss={train_loss:.4f} val={val_m['loss']:.4f} "
                f"mIoU={val_m['miou']:.4f} mF1={val_m['mf1']:.4f} OA={val_m['oa']:.4f} "
                f"best={best_miou:.4f} patience={no_improve}/{EARLY_STOP} "
                f"{ep_t:.0f}s  {total_min:.1f}min"
            )
            _iou_parts, _f1_parts, _oa_parts = [], [], []
            for cls_id, iou in val_m["per_class_iou"].items():
                cdl_id = KEEP_CLASSES[cls_id - 1]
                short  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}").replace(" ", "")
                _iou_parts.append(f"{short}={iou:.3f}" if not np.isnan(iou) else f"{short}=  nan")
                f1v = val_m["per_class_f1"].get(cls_id, float("nan"))
                _f1_parts.append(f"{short}={f1v:.3f}" if not np.isnan(f1v) else f"{short}=  nan")
                oav = val_m["per_class_oa"].get(cls_id, float("nan"))
                _oa_parts.append(f"{short}={oav:.3f}" if not np.isnan(oav) else f"{short}=  nan")
            log.info("    IoU: " + "  ".join(_iou_parts))
            log.info("     F1: " + "  ".join(_f1_parts))
            log.info("     OA: " + "  ".join(_oa_parts))

            # Save last checkpoint every epoch (overwrites previous)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_miou":         val_m["miou"],
                "band_indices":     band_indices,
                "band_names":       band_names_list,
                "in_channels":      in_channels,
                "num_classes":      NUM_CLASSES,
                "architecture":     arch,
            }, last_ckpt)

            if no_improve >= EARLY_STOP:
                log.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # Training time only — measured up to here, excludes test/inference below.
        train_time_total_s = time.time() - t_start
        mlflow.log_metrics({
            "train_time_total_s":   train_time_total_s,
            "train_time_total_min": train_time_total_s / 60,
        })
        log.info(f"  Training time: {train_time_total_s:.1f}s ({train_time_total_s / 60:.1f}min)")

        # ── Test evaluation (held-out same-area split, only when TEST_FRAC > 0) ─
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if eval_only:
            best_miou = ckpt.get("best_miou", best_miou)
            log.info(f"  [--eval-only] Loaded checkpoint (reported best_val_miou={best_miou:.4f})")

        if test_dl is not None:
            log.info("  Evaluating on held-out test set (same area, random split)...")
            test_r = evaluate_test_set(model, test_dl, NUM_CLASSES, DEVICE)
            log.info("  Benchmarking per-patch inference latency...")
            benchmark_inference_latency(model, test_dl, DEVICE, run.info.run_id)
        else:
            log.info("  No same-area test split — TEST_FRAC=0; skipping test evaluation")
            test_r = None

        _base_metrics = {
            "best_val_miou": best_miou,
            "best_val_mf1":  best_val_mf1,
            "best_val_oa":   best_val_oa,
            "total_epochs":  len(history),
        }
        if test_r is not None:
            _base_metrics.update({
                "test_miou": test_r["miou"],
                "test_mf1":  test_r["mf1"],
                "test_oa":   test_r["oa"],
            })
        mlflow.log_metrics(_base_metrics)

        for cls_id, iou in best_val_per_class_iou.items():
            if not np.isnan(iou):
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                slug   = name.lower().replace('/', '_').replace(' ', '_')
                mlflow.log_metric(f"best_val_iou_{slug}", iou)
        for cls_id, f1v in best_val_per_class_f1.items():
            if not np.isnan(f1v):
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                slug   = name.lower().replace('/', '_').replace(' ', '_')
                mlflow.log_metric(f"best_val_f1_{slug}", f1v)

        if test_r is not None:
            for cls_id, iou in test_r["per_class_iou"].items():
                if not np.isnan(iou):
                    cdl_id = KEEP_CLASSES[cls_id - 1]
                    name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                    mlflow.log_metric(
                        f"test_iou_{name.lower().replace('/', '_').replace(' ', '_')}",
                        iou,
                    )
            for cls_id, f1v in test_r["per_class_f1"].items():
                if not np.isnan(f1v):
                    cdl_id = KEEP_CLASSES[cls_id - 1]
                    name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                    mlflow.log_metric(
                        f"test_f1_{name.lower().replace('/', '_').replace(' ', '_')}",
                        f1v,
                    )

            # ── Log per-class IoU table to console ───────────────────────────
            log.info(f"  Test results  mIoU={test_r['miou']:.4f}  mF1={test_r['mf1']:.4f}  OA={test_r['oa']:.4f}")
            log.info(f"  {'Class':<20} {'CDL ID':>6}  {'IoU':>7}")
            log.info(f"  {'-'*38}")
            for cls_id, iou in test_r["per_class_iou"].items():
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                iou_s  = f"{iou:.4f}" if not np.isnan(iou) else "    nan"
                log.info(f"  {name:<20} {cdl_id:>6}  {iou_s:>7}")
            log.info(f"  {'-'*38}")
            log.info(f"  {'mIoU':<20} {'':>6}  {test_r['miou']:>7.4f}")

        # ── eval-only: write per-patch viz + metrics CSV, then stop ────────────
        # (skips training-only artifacts: history/curve/seg-map regen/gdrive upload)
        if eval_only:
            if test_r is not None and test_dl is not None:
                log.info(f"  [--eval-only] Saving per-patch test visualizations + metrics CSV for {exp_name}...")
                save_test_patch_visualizations(
                    test_dl, test_r["preds"], test_r["labels"],
                    s2_processed, test_ds, train_year_datasets_raw,
                    band_percentiles, exp_dir, exp_name,
                )
                log.info(f"  [--eval-only] Outputs written to {exp_dir}")
            else:
                log.warning("  [--eval-only] No test split available — nothing to write")
            _eval_run_id = run.info.run_id
            run_log_handler.flush()
            log.removeHandler(run_log_handler)
            run_log_handler.close()
            _DEFERRED_LOG_RUNS.append((_eval_run_id, str(run_log_path)))
            return None

        # ── Artifacts ─────────────────────────────────────────────────────────

        # Training history CSV
        hist_df  = pd.DataFrame(history)
        hist_csv = exp_dir / "training_history.csv"
        hist_df.to_csv(hist_csv, index=False)

        # Training curve PNG
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(hist_df["epoch"], hist_df["train_loss"], "--", label="Train")
        ax1.plot(hist_df["epoch"], hist_df["val_loss"],         label="Val")
        ax1.set(xlabel="Epoch", ylabel="Loss", title=f"{exp_name} — Loss")
        ax1.legend(); ax1.grid(True)
        ax2.plot(hist_df["epoch"], hist_df["val_miou"], color="green", label="Val mIoU")
        ax2.plot(hist_df["epoch"], hist_df["val_mf1"],  color="blue",  label="Val mF1", alpha=0.7)
        ax2.axhline(best_miou, linestyle="--", color="gray", label=f"Best mIoU={best_miou:.4f}")
        ax2.set(xlabel="Epoch", ylabel="Score", title=f"{exp_name} — mIoU / mF1")
        ax2.legend(); ax2.grid(True)
        plt.tight_layout()
        curve_path = exp_dir / "training_curve.png"
        plt.savefig(curve_path, dpi=150)
        plt.close()

        # Per-class metrics CSV + confusion matrix (only when same-area test exists)
        iou_csv = exp_dir / "test_per_class_iou.csv"
        cm_path = exp_dir / "confusion_matrix.png"
        if test_r is not None:
            iou_rows = []
            for cls_id, iou in test_r["per_class_iou"].items():
                cdl_id = KEEP_CLASSES[cls_id - 1]
                f1v    = test_r["per_class_f1"].get(cls_id, float("nan"))
                iou_rows.append({
                    "class_id":   cls_id,
                    "cdl_id":     cdl_id,
                    "class_name": CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}"),
                    "iou":        round(iou, 4) if not np.isnan(iou) else float("nan"),
                    "f1":         round(f1v, 4) if not np.isnan(f1v) else float("nan"),
                })
            pd.DataFrame(iou_rows).to_csv(iou_csv, index=False)
            if "preds" in test_r and "labels" in test_r:
                _plot_confusion_matrix(test_r["preds"], test_r["labels"], str(cm_path))

        # Segmentation map PNG (full-tile inference)
        seg_path = None
        ndvi_char_matrix = ndvi_char_var = None
        ndvi_s2_paths    = None
        if not skip_viz and test_s2_filtered is not None:
            log.info(f"  Running full-image inference for {exp_name}...")
            gt_map, _    = load_gt_remap(str(test_cdl))
            pred_map, _  = run_full_inference(
                model, test_s2_filtered, test_idx_local, patch_size=PATCH_SIZE, stride=PATCH_SIZE,
                channel_stats=None, band_percentiles=band_percentiles,
                norm_mode=norm_mode,
            )
            seg_path = exp_dir / "test_segmentation_map.png"
            rgb_img = _load_rgb_for_viz(test_s2_filtered, band_percentiles, downsample=4)
            save_segmentation_map(
                pred_map, gt_map,
                title=f"{exp_name} — Test Segmentation ({TEST_YEAR})",
                save_path=str(seg_path),
                rgb_img=rgb_img,
            )
            # Persisted full-res arrays — input to ndvi_disagreement_analysis.py
            # (Ghosh et al. 2021 CalCROP21 NDVI-based GT-vs-pred resolution).
            np.save(exp_dir / "test_pred_map.npy", pred_map)
            np.save(exp_dir / "test_gt_map.npy", gt_map)
            if not skip_ndvi:
                ndvi_char_matrix, ndvi_char_var = _run_ndvi_disagreement_and_log(
                    pred_map, gt_map, test_s2_filtered, exp_dir, label="test")
                ndvi_s2_paths = test_s2_filtered
            del pred_map, gt_map
        elif not skip_viz and primary_s2_filtered is not None:
            log.info(f"  Running full-image inference on training area for {exp_name}...")
            gt_map, _   = load_gt_remap(str(CDL_TRAIN))
            pred_map, _ = run_full_inference(
                model, primary_s2_filtered, primary_idx_local,
                patch_size=PATCH_SIZE, stride=PATCH_SIZE,
                channel_stats=None, band_percentiles=band_percentiles,
                norm_mode=norm_mode,
            )
            seg_path = exp_dir / "test_segmentation_map.png"
            rgb_img = _load_rgb_for_viz(primary_s2_filtered, band_percentiles, downsample=4)
            save_segmentation_map(
                pred_map, gt_map,
                title=f"{exp_name} — Segmentation Map ({TRAIN_YEARS[0]})",
                save_path=str(seg_path),
                rgb_img=rgb_img,
            )
            np.save(exp_dir / "test_pred_map.npy", pred_map)
            np.save(exp_dir / "test_gt_map.npy", gt_map)
            if not skip_ndvi:
                ndvi_char_matrix, ndvi_char_var = _run_ndvi_disagreement_and_log(
                    pred_map, gt_map, primary_s2_filtered, exp_dir, label="train_area")
                ndvi_s2_paths = primary_s2_filtered
            del pred_map, gt_map

        # Per-patch test visualizations
        if not skip_viz and test_r is not None and test_dl is not None:
            log.info(f"  Saving per-patch test visualizations for {exp_name}...")
            patch_dir = save_test_patch_visualizations(
                test_dl, test_r["preds"], test_r["labels"],
                s2_processed, test_ds, train_year_datasets_raw,
                band_percentiles, exp_dir, exp_name,
            )
            mlflow.log_artifacts(str(patch_dir), artifact_path="test_patches")

            # NDVI disagreement, per patch — own artifact folder (CalCROP21 method).
            if not skip_ndvi and ndvi_char_matrix is not None:
                log.info(f"  Saving NDVI disagreement patch visualizations for {exp_name}...")
                ndvi_patch_dir = save_ndvi_patch_visualizations(
                    test_dl, test_r["preds"], test_r["labels"],
                    ndvi_s2_paths, test_ds, train_year_datasets_raw,
                    ndvi_char_matrix, ndvi_char_var, exp_dir, exp_name,
                )
                try:
                    mlflow.log_artifacts(str(ndvi_patch_dir), artifact_path="ndvi_patches")
                except Exception as e:
                    log.warning(f"  Could not log ndvi_patches to MLflow: {e}")

        gdrive_links = upload_models_to_gdrive(
            run_name=f"{exp_name}_{run_timestamp}",
            model_files=[best_ckpt, last_ckpt],
        )
        for fname, link in gdrive_links.items():
            mlflow.set_tag(f"gdrive_{fname}", link)
        mlflow.log_artifact(str(hist_csv))
        mlflow.log_artifact(str(curve_path))
        if split_artifacts is not None:
            for p in split_artifacts.values():
                if Path(p).exists():
                    mlflow.log_artifact(str(p), artifact_path="split")
        if iou_csv.exists():
            mlflow.log_artifact(str(iou_csv))
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))
        if seg_path is not None:
            mlflow.log_artifact(str(seg_path))

        run_id = run.info.run_id

    # Logs uploaded after the whole session ends (see _flush_deferred_logs).
    run_log_handler.flush()
    run_log_handler.close()
    log.removeHandler(run_log_handler)
    _DEFERRED_LOG_RUNS.append((run_id, str(run_log_path)))

    summary = {
        "exp_name":      exp_name,
        "arch":          arch,
        "in_channels":   in_channels,
        "best_val_miou": round(best_miou, 4),
        "total_epochs":  len(history),
        "run_id":        run_id,
        "ckpt":          str(best_ckpt),
    }
    if test_r is not None:
        summary["test_miou"] = round(test_r["miou"], 4) if not np.isnan(test_r["miou"]) else float("nan")
        summary["test_mf1"]  = round(test_r["mf1"],  4) if not np.isnan(test_r["mf1"])  else float("nan")
        summary["test_oa"]   = round(test_r["oa"],   4) if not np.isnan(test_r["oa"])   else float("nan")
    if test_r is not None:
        spatial_str = f"test_mIoU={test_r['miou']:.4f}"
    else:
        spatial_str = "(no test set)"
    log.info(f"\n✅ {exp_name}  val_mIoU={best_miou:.4f}  {spatial_str}  run={run_id}")

    # Resume marker — written last, so a crashed run is NOT marked done and reruns.
    try:
        (exp_dir / ".done").write_text(f"{run_timestamp}\trun_id={run_id}\tval_miou={best_miou:.4f}\n")
    except Exception as e:
        log.warning(f"  Could not write .done marker: {e}")
    return summary


# ── Full-image inference & visualization ─────────────────────────────────────

# Derived from KEEP_CLASSES (config.py) — stays in sync if the class set changes,
# unlike a hardcoded list which silently desyncs (IndexError once len < NUM_CLASSES).
CROP_COLORS  = ["#000000"] + [USDA_CDL_COLORS[c] for c in KEEP_CLASSES]
CLASS_LABELS = ["Background"] + [CDL_CLASS_NAMES[c] for c in KEEP_CLASSES]
SEG_CMAP     = ListedColormap(CROP_COLORS)
SEG_NORM     = BoundaryNorm(boundaries=range(NUM_CLASSES + 1), ncolors=NUM_CLASSES)


def _build_drive_service():
    """Authenticate GDrive API v3 using the OAuth token."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally with:\n"
            "  python stages/process_data_v6.py --auth\n"
            "Then copy to the server via scp."
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(service, name, parent_id):
    """Return GDrive folder ID for `name` under `parent_id`, creating it if needed."""
    query  = (f"name='{name}' and '{parent_id}' in parents "
              f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        return result["files"][0]["id"]
    meta   = {"name": name, "mimeType": "application/vnd.google-apps.folder",
              "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def _upload_file_gdrive(service, local_path, folder_id):
    """Upload a single file to a GDrive folder (resumable). Skips if already exists."""
    from googleapiclient.http import MediaFileUpload

    fname  = os.path.basename(local_path)
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        log.info(f"  GDrive: already exists — {fname}")
        return result["files"][0]["id"]

    size  = os.path.getsize(local_path)
    log.info(f"  GDrive: uploading {fname}  ({size/1e6:.0f} MB)")
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)
    meta  = {"name": fname, "parents": [folder_id]}
    req   = service.files().create(body=meta, media_body=media, fields="id")
    resp  = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            log.info(f"    {int(status.progress() * 100)}%")
    log.info(f"  GDrive: uploaded {fname}  (id={resp['id']})")
    return resp["id"]


def upload_models_to_gdrive(run_name, model_files):
    """
    Upload model checkpoint files to GDrive under:
      <GDRIVE_MODELS_FOLDER_ID>/runs/<run_name>/

    Creates the `runs/` and `<run_name>/` folders if they don't exist.
    Returns dict {filename: gdrive_view_link} for MLflow tag logging.
    """
    try:
        service   = _build_drive_service()
        runs_id   = _get_or_create_folder(service, "runs", GDRIVE_MODELS_FOLDER_ID)
        run_id    = _get_or_create_folder(service, run_name, runs_id)
        links = {}
        for path in model_files:
            file_id = _upload_file_gdrive(service, str(path), run_id)
            links[os.path.basename(path)] = f"https://drive.google.com/file/d/{file_id}/view"
        log.info(f"  GDrive upload complete for {run_name}")
        return links
    except Exception as e:
        log.warning(f"  GDrive upload failed ({e}) — models kept locally only")
        return {}


def run_full_inference(model, s2_paths, band_indices, patch_size=256, stride=256,
                       channel_stats=None, band_percentiles=None, norm_mode="percentile"):
    """Tiled inference — reads one window at a time, never loads full rasters."""
    assert band_percentiles is not None, "band_percentiles required"
    with rasterio.open(s2_paths[0]) as src:
        H, W    = src.height, src.width
        profile = dict(src.profile)

    srcs     = [rasterio.open(p) for p in s2_paths]
    pred_map = np.zeros((H, W), dtype=np.uint8)
    n_rows   = (H + stride - 1) // stride
    n_cols   = (W + stride - 1) // stride
    total    = n_rows * n_cols
    K        = len(band_indices)

    lo_per_ch, hi_per_ch = _per_channel_percentiles(band_indices, *band_percentiles)
    denom_per_ch = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
    lo_per_ch    = lo_per_ch.astype(np.float32)

    model.eval()
    done = 0
    try:
        with torch.no_grad():
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    ph  = min(patch_size, H - y)
                    pw  = min(patch_size, W - x)
                    win = rasterio.windows.Window(x, y, pw, ph)

                    # Read only this window from each file
                    bands = []
                    for src in srcs:
                        try:
                            arr = src.read(window=win).astype(np.float32)
                        except Exception:
                            arr = np.zeros((src.count, ph, pw), dtype=np.float32)
                        arr[arr == S2_NODATA] = 0.0
                        arr[~np.isfinite(arr)] = 0.0
                        bands.append(arr)

                    patch = np.concatenate(bands, axis=0)[band_indices]  # (K, ph, pw)
                    patch = (patch - lo_per_ch[:, None, None]) / denom_per_ch[:, None, None]
                    if norm_mode != "zscore":
                        patch = np.clip(patch, 0.0, 1.0)

                    # Pad to patch_size if at border
                    if ph < patch_size or pw < patch_size:
                        padded = np.zeros((K, patch_size, patch_size), dtype=np.float32)
                        padded[:, :ph, :pw] = patch
                        patch = padded

                    t   = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                    out = model(t).argmax(dim=1).squeeze().cpu().numpy()
                    pred_map[y:y + ph, x:x + pw] = out[:ph, :pw]
                    done += 1
                    if done % 200 == 0 or done == total:
                        log.info(f"  {done}/{total} tiles")
    finally:
        for src in srcs:
            src.close()

    return pred_map, profile


def load_gt_remap(cdl_path):
    with rasterio.open(cdl_path) as src:
        cdl     = src.read(1).astype(np.int32)
        profile = dict(src.profile)
    gt = REMAP_LUT[np.clip(cdl, 0, 255)]
    return gt.astype(np.uint8), profile


def save_test_patch_visualizations(
    test_dl,
    preds_tensor,
    labels_tensor,
    s2_processed,
    test_ds,
    raw_datasets,
    band_percentiles,
    exp_dir,
    exp_name,
):
    """Save individual test patch PNGs: Median Composite / Ground Truth / Prediction / Correct-Incorrect.

    RGB (B4/B3/B2) is loaded directly from raw S2 tifs as a pixel-wise median
    across all dates — independent of which bands were selected for the model.
    """
    import rasterio.windows as _rwin

    patch_dir = exp_dir / "test_patches"
    patch_dir.mkdir(exist_ok=True)

    b4_rast      = S2_BAND_NAMES.index("B4") + 1   # rasterio 1-based
    b3_rast      = S2_BAND_NAMES.index("B3") + 1
    b2_rast      = S2_BAND_NAMES.index("B2") + 1
    norm_indices = [S2_BAND_NAMES.index("B4"), S2_BAND_NAMES.index("B3"), S2_BAND_NAMES.index("B2")]
    p1_arr, p99_arr = band_percentiles

    # Build patch (row, col) list in test_dl iteration order
    cum_sizes = [0]
    for ds_raw in raw_datasets:
        cum_sizes.append(cum_sizes[-1] + len(ds_raw.patches))

    patch_coords = []
    for j in range(len(test_ds)):
        flat_idx = test_ds.indices[j]
        for i in range(len(raw_datasets)):
            if cum_sizes[i] <= flat_idx < cum_sizes[i + 1]:
                row, col = raw_datasets[i].patches[flat_idx - cum_sizes[i]]
                patch_coords.append((row, col, raw_datasets[i].patch_size))
                break

    n_patches = len(patch_coords)
    ps        = patch_coords[0][2] if patch_coords else PATCH_SIZE

    # Cache: keyed on seed + n_patches (split is deterministic — same for every experiment)
    _cache_dir  = Path(s2_processed[0]).parent
    _cache_path = _cache_dir / f"rgb_median_patches_seed{SEED}_n{n_patches}.npy"

    if _cache_path.exists():
        log.info(f"  Patch RGB cache hit → {_cache_path.name}")
        rgb_medians = np.load(str(_cache_path))
    else:
        log.info(f"  Building RGB median for {n_patches} test patches across {len(s2_processed)} dates...")
        rgb_stack = np.full((n_patches, len(s2_processed), 3, ps, ps), np.nan, dtype=np.float16)
        for fi, path in enumerate(s2_processed):
            try:
                with rasterio.open(path) as src:
                    for pi, (row, col, _) in enumerate(patch_coords):
                        win = _rwin.Window(col, row, ps, ps)
                        arr = src.read([b4_rast, b3_rast, b2_rast], window=win).astype(np.float16)
                        arr[arr == S2_NODATA] = np.nan
                        rgb_stack[pi, fi] = arr
            except Exception as e:
                log.warning(f"  RGB skip {Path(path).name}: {e}")

        rgb_medians = np.nanmedian(rgb_stack.astype(np.float32), axis=1)  # (n, 3, ps, ps)
        del rgb_stack

        for ci, bi in enumerate(norm_indices):
            lo, hi = float(p1_arr[bi]), float(p99_arr[bi])
            if hi > lo:
                rgb_medians[:, ci] = (rgb_medians[:, ci] - lo) / (hi - lo)
        rgb_medians = np.nan_to_num(rgb_medians, nan=0.0)
        rgb_medians = np.clip(rgb_medians, 0, 1)

        np.save(str(_cache_path), rgb_medians)
        log.info(f"  Patch RGB cached → {_cache_path.name}")

    n_panels = 4

    error_cmap = ListedColormap(["#d0d0d0", "#22cc44", "#ee2222"])
    error_norm = BoundaryNorm([0, 1, 2, 3], error_cmap.N)
    crop_legend = [mpatches.Patch(color=CROP_COLORS[i], label=CLASS_LABELS[i])
                   for i in range(1, NUM_CLASSES)]
    error_legend = [
        mpatches.Patch(color="#22cc44", label="Correct"),
        mpatches.Patch(color="#ee2222", label="Incorrect"),
        mpatches.Patch(color="#d0d0d0", label="Background"),
    ]

    patch_metrics = []

    patch_idx = 0
    for imgs_batch, _ in test_dl:
        for b in range(imgs_batch.shape[0]):
            pred = preds_tensor[patch_idx].numpy()    # (H, W)
            gt   = labels_tensor[patch_idx].numpy()   # (H, W)
            rgb  = np.transpose(rgb_medians[patch_idx], (1, 2, 0))  # (H, W, 3)

            error = np.zeros_like(gt, dtype=np.uint8)
            crop_mask = gt > 0
            error[crop_mask & (pred == gt)] = 1
            error[crop_mask & (pred != gt)] = 2

            # ── Per-patch metrics (exact, from pred vs gt arrays) ──────────────
            n_fg        = int(crop_mask.sum())
            n_correctfg = int((crop_mask & (pred == gt)).sum())
            fg_acc      = (n_correctfg / n_fg) if n_fg > 0 else float("nan")
            overall_acc = float((pred == gt).mean())
            # mean IoU over foreground classes present in gt or pred
            ious = []
            for cls in range(1, NUM_CLASSES):
                gt_c, pr_c = (gt == cls), (pred == cls)
                union = int((gt_c | pr_c).sum())
                if union == 0:
                    continue
                ious.append(int((gt_c & pr_c).sum()) / union)
            patch_miou  = float(np.mean(ious)) if ious else float("nan")
            present     = sorted(int(c) for c in np.unique(gt) if c > 0)
            patch_metrics.append({
                "patch_idx":     patch_idx,
                "fg_pixel_acc":  round(fg_acc, 6),
                "overall_acc":   round(overall_acc, 6),
                "patch_miou":    round(patch_miou, 6),
                "n_fg_pixels":   n_fg,
                "classes_present": "|".join(CLASS_LABELS[c] for c in present),
            })

            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

            axes[0].imshow(rgb)
            axes[0].set_title("Median Composite\n(B4/B3/B2, 2024)", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gt,    cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
            axes[1].set_title("Ground Truth",    fontsize=11, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(pred,  cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
            axes[2].set_title("Prediction",      fontsize=11, fontweight="bold")
            axes[2].axis("off")

            axes[3].imshow(error, cmap=error_cmap, norm=error_norm, interpolation="nearest")
            axes[3].set_title("Correct / Incorrect", fontsize=11, fontweight="bold")
            axes[3].axis("off")

            fig.legend(handles=crop_legend + error_legend, loc="lower center",
                       ncol=min(NUM_CLASSES + 2, 9), fontsize=9,
                       bbox_to_anchor=(0.5, -0.02), frameon=True)
            plt.suptitle(f"{exp_name} — Test Patch {patch_idx:04d}", fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(str(patch_dir / f"patch_{patch_idx:04d}.png"), dpi=100, bbox_inches="tight")
            plt.close()
            patch_idx += 1

    log.info(f"  Saved {patch_idx} test patch PNGs → {patch_dir}")

    # ── Dump per-patch metrics CSV (sorted best→worst by fg_pixel_acc) ─────────
    if patch_metrics:
        import csv as _csv
        patch_metrics.sort(key=lambda r: (r["fg_pixel_acc"] != r["fg_pixel_acc"],
                                          -(r["fg_pixel_acc"] if r["fg_pixel_acc"] == r["fg_pixel_acc"] else 0)))
        csv_path = exp_dir / "test_patch_metrics.csv"
        with open(csv_path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(patch_metrics[0].keys()))
            w.writeheader()
            w.writerows(patch_metrics)
        log.info(f"  Saved per-patch metrics → {csv_path}")
        try:
            mlflow.log_artifact(str(csv_path))
        except Exception as e:
            log.warning(f"  Could not log patch metrics to MLflow: {e}")
    return patch_dir


def _load_rgb_for_viz(s2_paths, band_percentiles, downsample=4):
    """Pixel-wise median composite of B4/B3/B2. Cached to disk — computed once per data dir."""
    cache_path = Path(s2_paths[0]).parent / f"rgb_median_composite_ds{downsample}.npy"
    if cache_path.exists():
        log.info(f"  RGB composite cache hit → {cache_path.name}")
        return np.load(str(cache_path))

    log.info(f"  Building RGB median composite from {len(s2_paths)} dates...")
    b4 = S2_BAND_NAMES.index("B4") + 1
    b3 = S2_BAND_NAMES.index("B3") + 1
    b2 = S2_BAND_NAMES.index("B2") + 1
    band_norm_idx = [S2_BAND_NAMES.index("B4"), S2_BAND_NAMES.index("B3"), S2_BAND_NAMES.index("B2")]

    stack = []
    for path in s2_paths:
        try:
            with rasterio.open(path) as src:
                arr = src.read([b4, b3, b2]).astype(np.float32)
            arr[arr == S2_NODATA] = np.nan
            arr[~np.isfinite(arr)] = np.nan
            stack.append(arr)
        except Exception as e:
            log.warning(f"  RGB skip {Path(path).name}: {e}")

    if not stack:
        return None

    composite = np.nanmedian(np.stack(stack, axis=0), axis=0)   # (3, H, W)
    p1, p99 = band_percentiles
    for ci, bi in enumerate(band_norm_idx):
        lo, hi = float(p1[bi]), float(p99[bi])
        if hi > lo:
            composite[ci] = (composite[ci] - lo) / (hi - lo)
    composite = np.nan_to_num(composite, nan=0.0)
    composite = np.clip(composite, 0, 1)
    composite = composite[:, ::downsample, ::downsample]
    result = np.transpose(composite, (1, 2, 0))   # (H, W, 3)

    np.save(str(cache_path), result)
    log.info(f"  RGB composite cached → {cache_path.name}")
    return result


def save_segmentation_map(pred_map, gt_map, title, save_path, downsample=4, rgb_img=None):
    pred_ds = pred_map[::downsample, ::downsample]
    gt_ds   = gt_map[::downsample, ::downsample]

    error = np.zeros_like(gt_ds, dtype=np.uint8)
    crop_mask = gt_ds > 0
    error[crop_mask & (pred_ds == gt_ds)] = 1
    error[crop_mask & (pred_ds != gt_ds)] = 2
    error_cmap = ListedColormap(["#d0d0d0", "#22cc44", "#ee2222"])
    error_norm = BoundaryNorm([0, 1, 2, 3], error_cmap.N)

    n_panels = 4 if rgb_img is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8))

    panel = 0
    if rgb_img is not None:
        axes[panel].imshow(rgb_img)
        axes[panel].set_title("Median Composite\n(B4/B3/B2, 2024)", fontsize=12, fontweight="bold")
        axes[panel].axis("off")
        panel += 1

    axes[panel].imshow(gt_ds,   cmap=SEG_CMAP,   norm=SEG_NORM,   interpolation="nearest")
    axes[panel].set_title("Ground Truth (CDL)", fontsize=12, fontweight="bold")
    axes[panel].axis("off")
    panel += 1
    axes[panel].imshow(pred_ds, cmap=SEG_CMAP,   norm=SEG_NORM,   interpolation="nearest")
    axes[panel].set_title("Prediction",         fontsize=12, fontweight="bold")
    axes[panel].axis("off")
    panel += 1
    axes[panel].imshow(error,   cmap=error_cmap, norm=error_norm, interpolation="nearest")
    axes[panel].set_title("Correct / Incorrect", fontsize=12, fontweight="bold")
    axes[panel].axis("off")

    crop_patches = [mpatches.Patch(color=CROP_COLORS[i], label=CLASS_LABELS[i])
                    for i in range(1, NUM_CLASSES)]
    error_patches = [
        mpatches.Patch(color="#22cc44", label="Correct"),
        mpatches.Patch(color="#ee2222", label="Incorrect"),
        mpatches.Patch(color="#d0d0d0", label="Background"),
    ]
    fig.legend(handles=crop_patches + error_patches, loc="lower center",
               ncol=min(NUM_CLASSES + 2, 9), fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True)
    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")


def _run_ndvi_disagreement_and_log(pred_map, gt_map, s2_paths, exp_dir, label):
    """Run CalCROP21 (Ghosh et al. 2021) NDVI GT-vs-pred disagreement analysis
    and log results to the active MLflow run. Non-fatal — a failure here must
    not take down a completed training run.

    Returns (char_matrix, char_var) on success, (None, None) on failure — these
    are reused by save_ndvi_patch_visualizations to avoid rebuilding the
    per-class characteristic series at patch scale.
    """
    ndvi_out_dir = exp_dir / "ndvi_disagreement"
    try:
        overall, char_matrix, char_var = run_ndvi_disagreement(pred_map, gt_map, s2_paths, ndvi_out_dir)
    except Exception as e:
        log.warning(f"  NDVI disagreement analysis failed ({label}): {e}")
        return None, None

    if overall.get("pred_win_rate_overall") is not None:
        mlflow.log_metric("ndvi_pred_win_rate", overall["pred_win_rate_overall"])
    mlflow.log_metric("ndvi_n_disagreement_px", overall["n_disagreement_px"])
    mlflow.log_metric("ndvi_n_scored", overall["n_scored"])

    summary_csv = ndvi_out_dir / "ndvi_disagreement_summary.csv"
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        for _, row in summary_df.iterrows():
            name = str(row["class_name"]).lower().replace("/", "_").replace(" ", "_")
            mlflow.log_metric(f"ndvi_pred_win_rate_{name}", row["pred_win_rate"])
        mlflow.log_artifact(str(summary_csv))
    overall_json = ndvi_out_dir / "ndvi_disagreement_overall.json"
    if overall_json.exists():
        mlflow.log_artifact(str(overall_json))

    log.info(f"  NDVI disagreement ({label}): logged to MLflow")
    return char_matrix, char_var


def save_ndvi_patch_visualizations(
    test_dl,
    preds_tensor,
    labels_tensor,
    s2_processed,
    test_ds,
    raw_datasets,
    char_matrix,
    char_var,
    exp_dir,
    exp_name,
):
    """Per-patch NDVI GT-vs-pred disagreement overlay (Ghosh et al. 2021 CalCROP21).

    Separate artifact folder from test_patches/ — colors disagreement pixels by
    which label (CDL ground truth or model prediction) the pixel's own NDVI
    series actually follows, using the characteristic series already built by
    _run_ndvi_disagreement_and_log over the full raster.
    """
    import rasterio.windows as _rwin

    patch_dir = exp_dir / "ndvi_patches"
    patch_dir.mkdir(exist_ok=True)

    b4_rast, b8_rast = B4_IDX + 1, B8_IDX + 1   # rasterio 1-based

    cum_sizes = [0]
    for ds_raw in raw_datasets:
        cum_sizes.append(cum_sizes[-1] + len(ds_raw.patches))

    patch_coords = []
    for j in range(len(test_ds)):
        flat_idx = test_ds.indices[j]
        for i in range(len(raw_datasets)):
            if cum_sizes[i] <= flat_idx < cum_sizes[i + 1]:
                row, col = raw_datasets[i].patches[flat_idx - cum_sizes[i]]
                patch_coords.append((row, col, raw_datasets[i].patch_size))
                break

    n_patches = len(patch_coords)
    ps        = patch_coords[0][2] if patch_coords else PATCH_SIZE

    # Cache keyed on seed + n_patches + n_dates — split is deterministic, same every
    # experiment, but the NDVI date stack (s2_processed) varies per experiment's
    # filtered S2 file list, so n_dates must be part of the key or experiments with
    # different date counts collide on a stale cache (shape mismatch downstream).
    _cache_dir  = Path(s2_processed[0]).parent
    _cache_path = _cache_dir / f"ndvi_patches_seed{SEED}_n{n_patches}_d{len(s2_processed)}.npy"

    if _cache_path.exists():
        log.info(f"  NDVI patch cache hit → {_cache_path.name}")
        ndvi_patches = np.load(str(_cache_path))
    else:
        log.info(f"  Building NDVI series for {n_patches} test patches across {len(s2_processed)} dates...")
        ndvi_patches = np.full((n_patches, len(s2_processed), ps, ps), np.nan, dtype=np.float16)
        for fi, path in enumerate(s2_processed):
            try:
                with rasterio.open(path) as src:
                    for pi, (row, col, _) in enumerate(patch_coords):
                        win = _rwin.Window(col, row, ps, ps)
                        b4, b8 = src.read([b4_rast, b8_rast], window=win).astype(np.float32)
                        invalid = (b4 == S2_NODATA) | (b8 == S2_NODATA) | ~np.isfinite(b4) | ~np.isfinite(b8)
                        denom = b4 + b8
                        with np.errstate(divide="ignore", invalid="ignore"):
                            ndvi = (b8 - b4) / denom
                        ndvi[invalid | (denom == 0)] = np.nan
                        ndvi_patches[pi, fi] = ndvi.astype(np.float16)
            except Exception as e:
                log.warning(f"  NDVI skip {Path(path).name}: {e}")
        np.save(str(_cache_path), ndvi_patches)
        log.info(f"  NDVI patches cached → {_cache_path.name}")

    verdict_cmap = ListedColormap(["#d0d0d0", "#22cc44", "#ee2222", "#f5a623"])
    verdict_norm = BoundaryNorm([0, 1, 2, 3, 4], verdict_cmap.N)
    verdict_legend = [
        mpatches.Patch(color="#d0d0d0", label="Agreement / background"),
        mpatches.Patch(color="#22cc44", label="Prediction wins (NDVI)"),
        mpatches.Patch(color="#ee2222", label="CDL wins (NDVI)"),
        mpatches.Patch(color="#f5a623", label="Disagreement, unscored"),
    ]

    patch_ndvi_metrics = []
    patch_idx = 0
    for imgs_batch, _ in test_dl:
        for b in range(imgs_batch.shape[0]):
            pred = preds_tensor[patch_idx].numpy()
            gt   = labels_tensor[patch_idx].numpy()
            ndvi_patch = ndvi_patches[patch_idx].astype(np.float32)   # (n_dates, ps, ps)

            verdict = score_patch_verdict(ndvi_patch, gt, pred, char_matrix, char_var)

            n_pred_win = int((verdict == 1).sum())
            n_gt_win   = int((verdict == 2).sum())
            n_unscored = int((verdict == 3).sum())
            patch_ndvi_metrics.append({
                "patch_idx":       patch_idx,
                "n_disagreement":  int((gt != pred).sum()),
                "pred_wins":       n_pred_win,
                "gt_cdl_wins":     n_gt_win,
                "unscored":        n_unscored,
                "pred_win_rate":   round(n_pred_win / (n_pred_win + n_gt_win), 4)
                                   if (n_pred_win + n_gt_win) > 0 else float("nan"),
            })

            fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 5))
            axes[0].imshow(gt,      cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
            axes[0].set_title("Ground Truth (CDL)", fontsize=11, fontweight="bold"); axes[0].axis("off")
            axes[1].imshow(pred,    cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
            axes[1].set_title("Prediction",         fontsize=11, fontweight="bold"); axes[1].axis("off")
            axes[2].imshow(verdict, cmap=verdict_cmap, norm=verdict_norm, interpolation="nearest")
            axes[2].set_title("NDVI Disagreement Verdict", fontsize=11, fontweight="bold"); axes[2].axis("off")

            fig.legend(handles=verdict_legend, loc="lower center", ncol=4, fontsize=9,
                       bbox_to_anchor=(0.5, -0.02), frameon=True)
            plt.suptitle(f"{exp_name} — NDVI Patch {patch_idx:04d}", fontsize=12, y=1.02)
            plt.tight_layout()
            plt.savefig(str(patch_dir / f"ndvi_patch_{patch_idx:04d}.png"), dpi=100, bbox_inches="tight")
            plt.close()
            patch_idx += 1

    log.info(f"  Saved {patch_idx} NDVI patch PNGs → {patch_dir}")

    if patch_ndvi_metrics:
        csv_path = exp_dir / "ndvi_patch_metrics.csv"
        pd.DataFrame(patch_ndvi_metrics).to_csv(csv_path, index=False)
        log.info(f"  Saved per-patch NDVI metrics → {csv_path}")
        try:
            mlflow.log_artifact(str(csv_path))
        except Exception as e:
            log.warning(f"  Could not log ndvi patch metrics to MLflow: {e}")

    return patch_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    exps=None,
    archs=None,
    loss="wce",
    force=False,
    data_dir=None,
    phenol_dates=None,
    skip_viz=False,
    skip_ndvi=False,
    top_k=None,
    percentile=None,
    score_threshold=None,
    batch_size=None,
    epochs=None,
    no_preload=False,
    cache_only=False,
    norm_mode="percentile",
    hp=None,
):
    global BATCH_SIZE, MAX_EPOCHS, HP_OVERRIDE, HP_TAG
    if batch_size:
        BATCH_SIZE = batch_size
        log.info(f"Batch size overridden: {BATCH_SIZE}")
    if epochs:
        MAX_EPOCHS = epochs
        log.info(f"Max epochs overridden: {MAX_EPOCHS}")

    HP_OVERRIDE = hp or None
    HP_TAG = _hp_tag(hp) if hp else ""
    if HP_OVERRIDE:
        log.info(f"HP grid combo: {HP_OVERRIDE}  (tag={HP_TAG})")

    _check_gdrive_token()

    # Override data directories
    # Use `global` so all module-level functions pick up the new paths at call time.
    if data_dir:
        global S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_BY_YEAR, CDL_TRAIN, MODELS_DIR, FIGURES_DIR, PRELOAD_CACHE_DIR
        data_dir = Path(data_dir)
        S2_TRAIN_DIR      = data_dir / "s2" / "2024"
        S2_PROCESSED_DIR  = S2_TRAIN_DIR
        CDL_TRAIN         = data_dir / "cdl" / "cdl_2024_study_area_filtered.tif"
        CDL_BY_YEAR       = {"2024": CDL_TRAIN}
        MODELS_DIR        = data_dir / "models"
        FIGURES_DIR       = data_dir / "figures"
        PRELOAD_CACHE_DIR = data_dir / "preload_cache"   # cache lives under the data dir
        PRELOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Data dir overridden to {data_dir}  (preload_cache={PRELOAD_CACHE_DIR})")

    # ── Cloud preload cache — download a prebuilt cache instead of rebuilding ──
    # Filenames are content-hash keyed by PreloadedDataset, so a matching file is a
    # cache hit at train time. Skipped under --no-preload (no cache is consulted).
    _pc_gdrive = (GDRIVE_PRELOAD_CACHE_FOLDER_ID or None) if (
        "args" in globals() and getattr(args, "use_cloud_preload", False)) else None
    if _pc_gdrive and not getattr(args, "no_preload", False):
        PRELOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from crop_mapping_pipeline.stages.fetch_data_v6 import fetch_preload_cache
        log.info(f"Fetching cloud preload cache from GDrive folder {_pc_gdrive} → {PRELOAD_CACHE_DIR}")
        got = fetch_preload_cache(_pc_gdrive, str(PRELOAD_CACHE_DIR), overwrite=False)
        log.info(f"Cloud preload cache: {len(got)} file(s) ready in {PRELOAD_CACHE_DIR}")

    s2_processed = sorted(
        glob(str(S2_TRAIN_DIR / "*_processed.tif")) +
        glob(str(S2_TRAIN_DIR / "S2H_*.tif"))
    )
    seen = set()
    s2_processed = [p for p in s2_processed if not (p in seen or seen.add(p))
                    and not Path(p).name.startswith("._")]
    if not s2_processed:
        raise FileNotFoundError(f"No processed S2 files in {S2_TRAIN_DIR}")

    # ── Validate TIF files — drop corrupt, empty, and low-validity dates ──────
    # Drops acquisitions whose valid-pixel fraction < S2_MIN_VALID_FRAC (config),
    # e.g. high-cloud/partial-capture dates. Cache invalidated when the file set
    # OR the threshold changes.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    MIN_VALID_FRAC_FILE = S2_MIN_VALID_FRAC
    VALIDATION_WIN      = 512

    _val_cache_path = S2_TRAIN_DIR / "s2_validation_cache.json"
    _val_cache_key  = sorted(Path(p).name for p in s2_processed)

    def _load_validation_cache():
        if not _val_cache_path.exists():
            return None
        try:
            with open(_val_cache_path) as f:
                c = json.load(f)
            if c.get("files_key") == _val_cache_key and c.get("threshold") == MIN_VALID_FRAC_FILE:
                return c
        except Exception:
            pass
        return None

    _cached = _load_validation_cache()
    if _cached:
        log.info(f"S2 validation cache hit ({len(_cached['valid'])} valid files)")
        valid_s2  = [p for p in s2_processed if Path(p).name in set(_cached["valid"])]
        _corrupt_names = set(_cached.get("corrupt", []))
        _nodata_names  = {r[0] for r in _cached.get("no_data", [])}
        corrupt  = [(p, "") for p in s2_processed if Path(p).name in _corrupt_names]
        no_data  = [(p, r[1]) for p in s2_processed
                    for r in _cached.get("no_data", []) if r[0] == Path(p).name]
    else:
        corrupt  = []
        no_data  = []
        valid_s2 = []

        def _check_file(path):
            try:
                with rasterio.open(path) as src:
                    h, w      = src.height, src.width
                    n_bands   = src.count
                    sz        = min(VALIDATION_WIN, w // 4, h // 4)
                    valid_px, total_px = 0, 0
                    check_bands = sorted({1, n_bands // 2, n_bands})
                    for band in check_bands:
                        for gy in range(3):
                            for gx in range(3):
                                ox = int((gx + 0.5) * w / 3) - sz // 2
                                oy = int((gy + 0.5) * h / 3) - sz // 2
                                ox = max(0, min(ox, w - sz))
                                oy = max(0, min(oy, h - sz))
                                win  = rasterio.windows.Window(ox, oy, sz, sz)
                                data = src.read(band, window=win).astype(np.float32)
                                ok   = (data != S2_NODATA) & np.isfinite(data)
                                valid_px += ok.sum()
                                total_px += ok.size
                return path, valid_px / total_px, None
            except Exception as e:
                return path, 0.0, str(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_check_file, p): p for p in s2_processed}
            for fut in as_completed(futures):
                path, frac, err = fut.result()
                if err:
                    corrupt.append((path, err))
                elif frac < MIN_VALID_FRAC_FILE:
                    no_data.append((path, frac))
                else:
                    valid_s2.append(path)

        valid_s2.sort()

        try:
            with open(_val_cache_path, "w") as f:
                json.dump({
                    "files_key": _val_cache_key,
                    "threshold": MIN_VALID_FRAC_FILE,
                    "valid":     [Path(p).name for p in valid_s2],
                    "corrupt":   [Path(p).name for p, _ in corrupt],
                    "no_data":   [[Path(p).name, frac] for p, frac in no_data],
                }, f)
            log.info(f"S2 validation cached → {_val_cache_path.name}")
        except Exception as e:
            log.warning(f"Could not write validation cache: {e}")

    if corrupt:
        log.error(f"Found {len(corrupt)} corrupt S2 file(s) — re-download before training:")
        for p, err in corrupt:
            log.error(f"  {Path(p).name}  ({err})")
        raise RuntimeError(
            f"{len(corrupt)} corrupt S2 file(s) detected. "
            "Re-download:  python stages/fetch_data_v6.py --folder-id FOLDER_ID --years <year> --overwrite"
        )
    if no_data:
        log.warning(f"Excluding {len(no_data)} date(s) below {MIN_VALID_FRAC_FILE*100:.0f}% valid pixels (high cloud / partial capture):")
        for p, frac in no_data:
            log.warning(f"  {Path(p).name}  ({frac*100:.2f}% valid)")
    s2_processed = valid_s2
    log.info(f"{len(s2_processed)} S2 dates valid for training ({len(no_data)} low-validity excluded, threshold={MIN_VALID_FRAC_FILE*100:.0f}%)")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Build local band map (reference year) ──────────────────────────────
    (local_band_names, local_band_to_idx,
     local_date_to_idx, mmdd_to_date) = build_local_band_map(s2_processed)

    # ── Build experiment channel sets ─────────────────────────────────────
    _ref_year_s2  = _s2_for_year(s2_processed, TRAIN_YEARS[0])
    _ref_year_cdl = CDL_TRAIN

    _base_dir = Path(data_dir) if data_dir else PROCESSED_DIR

    # ── Base domain channels (all 9 VEGE_BANDS, no band selection) ─────────
    needs_sd  = not exps or "single_date" in exps
    needs_nmt = not exps or "mt_base" in exps

    sd_base_idx = sd_base_names = sd_date_key = None
    nmt_base_idx = nmt_base_names = phenol_map_base = None

    if needs_sd:
        sd_base = build_single_date_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
        )
        sd_base_idx, sd_base_names, sd_date_key = sd_base

    if needs_nmt:
        nmt_base = build_naive_multitemporal_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
            phenol_json=phenol_dates,
        )
        nmt_base_idx, nmt_base_names, phenol_map_base = nmt_base

    # ── single_date (peak NDVI date × ALL bands — conventional baseline) ──
    # No band selection: isolates the temporal variable against the multi-temporal
    # configurations. GSI/RF are not applied here.
    single_date_idx = single_date_names = single_date_key = None
    if not exps or "single_date" in exps:
        single_date_idx, single_date_names, single_date_key = sd_base_idx, sd_base_names, sd_date_key

    # ── mt_base (4 calendar dates × ALL VEGE_BANDS — no selection) ──
    mt_base_idx = mt_base_names = phenol_map = None
    if not exps or "mt_base" in exps:
        mt_base_idx, mt_base_names, phenol_map = nmt_base_idx, nmt_base_names, phenol_map_base

    def _find_direct_json(selector: str) -> Path:
        """Return JSON path for a direct selector.

        Score-threshold mode → select_{selector}_s{T}.json (Wei et al. 2023, no subset).
        Percentile mode      → select_{selector}_p{P}.json (final selection, no subset).
        Top-K mode           → select_{selector}_k{K}.json (falls back to largest k + subset).
        """
        base = Path(data_dir) if data_dir else SELECT_GSI_DIRECT_JSON.parent
        if score_threshold is not None:
            return base / f"select_{selector}_s{score_threshold:g}.json"
        if percentile is not None:
            return base / f"select_{selector}_p{percentile:g}.json"
        if top_k:
            exact = base / f"select_{selector}_k{top_k}.json"
            if exact.exists():
                return exact
            candidates = sorted(base.glob(f"select_{selector}_k*.json"))
            if candidates:
                log.info(f"  {selector}: k={top_k} JSON not found, using {candidates[-1].name} with subset_k")
                return candidates[-1]
        return base / f"select_{selector}_k{SELECT_TOP_K_PER_CROP}.json"

    # score_threshold and percentile modes: JSON union is already the final selection → no subset_k.
    _subset = None if (score_threshold is not None or percentile is not None) else top_k

    gsi_idx = gsi_names = None
    if not exps or "gsi" in exps:
        gsi_json = _find_direct_json("gsi_direct")
        gsi_idx, gsi_names = build_direct_indices(
            gsi_json, mmdd_to_date, local_band_to_idx,
            selector_name="gsi", subset_k=_subset,
        )
        _gsi_mode = (f"s={score_threshold:g}" if score_threshold is not None
                     else f"P{percentile:g}" if percentile is not None
                     else f"k={top_k or 'all'}")
        log.info(f"gsi ({_gsi_mode}): {len(gsi_idx)} channels")

    rf_idx = rf_names = None
    if not exps or "rf" in exps:
        rf_json = _find_direct_json("rf_direct")
        rf_idx, rf_names = build_direct_indices(
            rf_json, mmdd_to_date, local_band_to_idx,
            selector_name="rf", subset_k=_subset,
        )
        _rf_mode = (f"s={score_threshold:g}" if score_threshold is not None
                    else f"P{percentile:g}" if percentile is not None
                    else f"k={top_k or 'all'}")
        log.info(f"rf ({_rf_mode}): {len(rf_idx)} channels")

    # ── Class weights ──────────────────────────────────────────────────────
    cw_tensor, cw_counts = compute_class_weights(return_counts=True)
    log.info("Class weights computed")

    # ── Build experiment registry & plan ───────────────────────────────────
    all_archs = list(ARCH_CFG.keys())
    run_exps  = exps  or ["single_date", "mt_base", "gsi", "rf"]
    run_archs = archs or all_archs

    registry = build_registry(
        single_date_idx=single_date_idx, single_date_names=single_date_names, single_date_key=sd_date_key,
        mt_base_idx=mt_base_idx,         mt_base_names=mt_base_names,         phenol_map=phenol_map,
        gsi_idx=gsi_idx,                 gsi_names=gsi_names,
        rf_idx=rf_idx,                   rf_names=rf_names,
    )

    expanded_exps = expand_exp_keys(run_exps, registry)
    log.info(f"Selected experiments: {expanded_exps}")

    plan = []
    for exp_key in expanded_exps:
        cfg = registry.get(exp_key)
        if cfg is None:
            log.warning(f"Experiment '{exp_key}' not in registry — skipping")
            continue
        if cfg.band_indices is None:
            raise RuntimeError(
                f"Exp {exp_key}: band indices are None — required feature-selection output is missing."
            )
        for arch in run_archs:
            plan.append((exp_key, arch, cfg.band_indices, cfg.band_names,
                         f"{cfg.description}, {arch}", cfg.extra_kw))

    log.info(f"Planned {len(plan)} run(s): {[(e, a) for e, a, *_ in plan]}")

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ── Run experiments — one top-level run per exp_key, nested run per arch ─
    all_results = []
    exp_groups: dict = {}
    for exp_key, arch, band_idx, band_names, description, extra_kw in plan:
        exp_groups.setdefault(exp_key, []).append((arch, band_idx, band_names, description, extra_kw))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for exp_key, arch_runs in exp_groups.items():
        cfg_entry  = registry[exp_key]
        experiment = mlflow.set_experiment(cfg_entry.mlflow_experiment)
        MlflowClient().set_experiment_tag(
            experiment.experiment_id, "mlflow.note.content",
            "Segmentation training — 8-crop CalCROP21-style class selection "
            "(>=1M px threshold), block spatial split "
            f"({int(round((1-VAL_FRAC-TEST_FRAC)*100))}/{int(round(VAL_FRAC*100))}/{int(round(TEST_FRAC*100))}; "
            "block split groups whole grid cells per split to avoid patch-adjacency "
            "leakage). Compares band-selection "
            "experiments (single-date / naive multi-temporal / GSI / RF direct-K) "
            f"across architectures. train_years={TRAIN_YEARS}, test_year={TEST_YEAR}.",
        )
        n_ch = len(arch_runs[0][1]) if arch_runs[0][1] else 0
        _sel_sfx = (f"_p{percentile:g}" if percentile is not None
                    else (f"_k{top_k}" if top_k else ""))
        if HP_TAG:
            _sel_sfx += f"_{HP_TAG}"
        parent_run_name = f"exp_{exp_key}{_sel_sfx}_{timestamp}"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_params({
                "experiment":   f"exp_{exp_key}",
                "n_channels":   n_ch,
                "train_years":  str(TRAIN_YEARS),
                "test_year":    TEST_YEAR,
                "description":  cfg_entry.description,
                "loss":         loss,
                **({"top_k": top_k} if top_k else {}),
                **({"percentile": percentile} if percentile is not None else {}),
                **({f"hp_{k}": v for k, v in HP_OVERRIDE.items()} if HP_OVERRIDE else {}),
                **_get_hardware_info(),
            })
            mlflow.set_tag(
                "mlflow.note.content",
                f"Parent run grouping all architectures for experiment '{exp_key}': "
                f"{cfg_entry.description}. {n_ch} input channels, trained on "
                f"{TRAIN_YEARS} and tested on {TEST_YEAR}.",
            )
            log.info(f"Parent MLflow run: {parent_run_name}  (id={parent_run.info.run_id})")
            for arch, band_idx, band_names, description, extra_kw in arch_runs:
                exp_name = f"exp_{exp_key}{_sel_sfx}_{arch}"
                result = run_experiment(
                    exp_name=exp_name,
                    arch=arch,
                    band_indices=band_idx,
                    band_names_list=band_names,
                    description=description,
                    s2_processed=s2_processed,
                    class_weights_tensor=cw_tensor,
                    class_counts=cw_counts,
                    loss=loss,
                    force=force,
                    skip_viz=skip_viz,
                    skip_ndvi=skip_ndvi,
                    no_preload=no_preload,
                    cache_only=cache_only,
                    norm_mode=norm_mode,
                    **extra_kw,
                )
                if result is not None:
                    all_results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        summary_df  = pd.DataFrame(all_results)
        sort_col = next(
            (c for c in ("test_miou", "best_val_miou") if c in summary_df.columns),
            None,
        )
        if sort_col:
            summary_df = summary_df.sort_values(sort_col, ascending=False)
        summary_csv = MODELS_DIR / "experiment_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        log.info("\n=== Experiment Summary ===")
        cols = [c for c in [
            "exp_name", "arch", "in_channels",
            "best_val_miou", "test_miou", "test_mf1", "test_oa",
            "total_epochs",
        ] if c in summary_df.columns]
        log.info("\n" + summary_df[cols].to_string(index=False))
        log.info(f"Saved: {summary_csv}")

    log.info("All experiments done — segmentation maps, confusion matrices, and IoU CSVs logged to MLflow.")


def _upload_existing_models(filter_exps=None, filter_archs=None):
    """Upload best_model.pth + last_model.pth for all existing run dirs.

    Scans MODELS_DIR for subdirectories that contain at least one of the two
    checkpoint files and uploads them to GDrive under runs/<run_dir_name>/.

    filter_exps  — optional list of exp shorthand keys (e.g. ["C_v3", "A_v2"]).
                   Run dir must contain any of the keys as a substring.
    filter_archs — optional list of arch names to further filter.
    """
    import re as _re

    def _matches(run_dir_name):
        if filter_exps:
            if not any(
                _re.search(r"(?i)" + _re.escape(e.lower()), run_dir_name.lower())
                for e in filter_exps
            ):
                return False
        if filter_archs:
            if not any(arch.lower() in run_dir_name.lower() for arch in filter_archs):
                return False
        return True

    candidates = sorted(MODELS_DIR.iterdir()) if MODELS_DIR.exists() else []
    run_dirs = [
        d for d in candidates
        if d.is_dir() and _matches(d.name)
        and (
            (d / "best_model.pth").exists()
            or (d / "last_model.pth").exists()
        )
    ]

    if not run_dirs:
        log.warning("No matching run dirs with model checkpoints found under %s", MODELS_DIR)
        return

    log.info("Uploading models for %d run(s)…", len(run_dirs))
    for run_dir in run_dirs:
        model_files = [
            f for f in [run_dir / "best_model.pth", run_dir / "last_model.pth"]
            if f.exists()
        ]
        log.info("  %s: %s", run_dir.name, [f.name for f in model_files])
        links = upload_models_to_gdrive(run_name=run_dir.name, model_files=model_files)
        if links:
            for fname, link in links.items():
                log.info("    %s → %s", fname, link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models for band selection comparison")
    parser.add_argument(
        "--exp", nargs="+",
        choices=["single_date", "mt_base", "gsi", "rf"],
        default=["single_date", "mt_base", "gsi", "rf"],
        help=(
            "Experiments to run (default: all four). "
            "single_date=peak NDVI date + ALL bands (single-date baseline), "
            "mt_base=4 calendar dates + ALL S2_BAND_NAMES (multi-temporal baseline, no selection), "
            "gsi=GSI-direct top-K, rf=RF-direct top-K (multi-class MDI)."
        ),
    )
    parser.add_argument(
        "--arch", nargs="+", choices=list(ARCH_CFG.keys()),
        default=None,
        help="Which architectures to run (default: all)",
    )
    parser.add_argument(
        "--loss",
        choices=["wce", "focal_tversky", "dynamic_balanced"],
        default="wce",
        help=(
            "Loss function: wce (default, WeightedCrossEntropy), "
            "focal_tversky (Focal Tversky, median-freq weighted class-mean), "
            "dynamic_balanced (per-batch Cui+2019 weights; thesis primary, DECB-CE)"
        ),
    )
    parser.add_argument("--force",      action="store_true", help="Re-run even if checkpoint exists")
    parser.add_argument("--skip-viz",   action="store_true", help="Skip full-image visualization")
    parser.add_argument("--ndvi", action="store_true",
                        help="Run NDVI GT-vs-pred disagreement analysis (Ghosh et al. 2021 CalCROP21 method). "
                             "OFF by default. Runs after full-image inference, logs per-class win-rate to MLflow.")
    parser.add_argument("--skip-ndvi", action="store_true",
                        help="Deprecated/no-op — NDVI analysis is off by default now; use --ndvi to enable.")
    parser.add_argument("--no-preload", action="store_true",
                        help="Skip disk preload cache; use on-the-fly normalization. "
                             "Slower per epoch but avoids large disk/RAM allocation — useful for high channel counts.")
    parser.add_argument("--norm", default="percentile", choices=list(NORM_MODES),
                        help="Input normalization strategy for ablation. "
                             "percentile: clip [P2,P98]→[0,1] (default). "
                             "minmax: clip [min,max]→[0,1]. "
                             "zscore: (x-mean)/std, no clip.")
    parser.add_argument("--build-cache-only", action="store_true",
                        help="Build PreloadedDataset cache for all selected experiments then exit without training. "
                             "Transfer the cache dir to another machine and training will use it as a cache hit.")
    parser.add_argument(
        "--use-cloud-preload", action="store_true",
        help="Download the cloud-built portable preload cache (preload_*.npy + *_masks.pt) from "
             "config.GDRIVE_PRELOAD_CACHE_FOLDER_ID into the preload_cache dir before training, "
             "instead of rebuilding locally. Ignored with --no-preload.")
    parser.add_argument(
        "--upload-cache-gdrive", nargs="?", const=GDRIVE_PRELOAD_CACHE_FOLDER_ID or None,
        default=None, metavar="FOLDER_ID",
        help="After --build-cache-only, upload the built preload cache to this GDrive folder. "
             "Bare flag uses config.GDRIVE_PRELOAD_CACHE_FOLDER_ID. With --build-cache-only and a "
             "configured folder id, upload runs automatically.")
    parser.add_argument("--no-upload-cache", action="store_true",
                        help="Disable the automatic preload-cache upload after --build-cache-only.")
    parser.add_argument("--data-dir", default=None, help="Override data/processed directory")
    parser.add_argument("--phenol-dates", default=None, help="Path to pre-computed phenol_dates.json for Exp B multi-temporal baseline")
    parser.add_argument("--shutdown", action="store_true", help="Stop the RunPod pod after training")
    parser.add_argument(
        "--upload-existing", action="store_true",
        help=(
            "Upload best_model.pth and last_model.pth for all existing run dirs under "
            "MODELS_DIR to Google Drive without re-training. "
            "Optionally filter with --exp / --arch."
        ),
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=None, metavar="K",
        help="Top-K value(s) to sweep (loads select_gsi/rf_direct_k{K}.json per k). E.g. --top-k 5 10 15 20 30",
    )
    parser.add_argument(
        "--percentile", type=float, nargs="+", default=None, metavar="P",
        help="Percentile threshold(s) to sweep for gsi/rf direct selection "
             "(loads select_gsi/rf_direct_p{P}.json per P). Pooled-percentile, per-class union. "
             "Mutually exclusive with --top-k and --score-threshold. E.g. --percentile 70 75 80 85 90 95",
    )
    parser.add_argument(
        "--score-threshold", type=float, nargs="+", default=None, metavar="T",
        help="Per-crop normalized-score threshold(s) for gsi/rf direct selection "
             "(loads select_gsi/rf_direct_s{T}.json). Wei et al. 2023 approach: "
             "normalize per crop to [0,1], retain >= T. Mutually exclusive with --top-k and --percentile. "
             "E.g. --score-threshold 0.5",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=f"Override BATCH_SIZE from config (default: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help=f"Override MAX_EPOCHS from config (default: {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--eval-only", metavar="CKPT_PATH",
        help="Skip training — load checkpoint and run spatial test evaluation only.",
    )
    parser.add_argument(
        "--hp-grid", metavar="JSON_PATH", default=None,
        help="Hyperparameter-grid JSON. Per-arch (recommended): top-level keys = "
             "arch names, each {\"grid\":{...}} or {\"combos\":[...]} — separate search "
             "space per architecture. Shared: top-level {\"grid\":{\"lr\":[...], "
             "\"weight_decay\":[...], \"warmup_epochs\":[...], \"sched_power\":[...]}} or "
             "{\"combos\":[...]} applied to every --arch. Tunable keys: lr, weight_decay, "
             "warmup_epochs, sched_power, scheduler(polynomial|cosine), optimizer(adamw|"
             "adam|sgd), momentum, grad_clip(0=off), batch_size. Each combo overrides "
             "ARCH_CFG/config defaults, runs the --exp/--top-k matrix, and logs to MLflow "
             "tagged with the combo. Combos run outermost, nesting with --top-k/--percentile "
             "sweeps. See configs/hp_grid_example.json.",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Suppress GDAL tile-decode noise (LZW/ZIP errors on legacy files).
    # Filters on a Logger only apply at that logger — not on propagation —
    # so we must filter on each Handler after basicConfig creates them.
    class _SuppressGDALFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "GDAL signalled an error" not in msg and "IReadBlock failed" not in msg

    _gdal_filter = _SuppressGDALFilter()
    # Also silence rasterio._err directly (covers worker processes via fork)
    logging.getLogger("rasterio._err").setLevel(logging.ERROR)

    SESSION_LOG_PATH = str(LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(SESSION_LOG_PATH),
        ],
    )
    for _h in logging.root.handlers:
        _h.addFilter(_gdal_filter)

    log.info(f"Device: {_device_label()}  PyTorch: {torch.__version__}")

    if args.upload_existing:
        _upload_existing_models(filter_exps=args.exp, filter_archs=args.arch)
        sys.exit(0)

    if args.eval_only:
        # Route through the normal --exp/--top-k/--arch path so the deterministic
        # same-area split + correct band selection are rebuilt; run_experiment then
        # skips training, loads this checkpoint, and runs test eval + per-patch viz.
        ckpt_path = Path(args.eval_only)
        if not ckpt_path.exists():
            log.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        EVAL_ONLY_CKPT = str(ckpt_path)
        # Keep all MLflow logging local (do not pollute the tracking server with eval runs).
        _eval_mlruns = Path(tempfile.mkdtemp(prefix="evalonly_mlruns_"))
        mlflow.set_tracking_uri(f"file://{_eval_mlruns}")
        log.info(f"--eval-only: MLflow → local {_eval_mlruns} (server untouched)")
        log.info(f"--eval-only: evaluating {ckpt_path}")

    n_sel_modes = sum([bool(args.top_k), bool(args.percentile), bool(args.score_threshold)])
    if n_sel_modes > 1:
        log.error("--top-k, --percentile, and --score-threshold are mutually exclusive — pick one.")
        sys.exit(1)

    if args.percentile:
        sweep = [("percentile", p) for p in args.percentile]
    elif args.top_k:
        sweep = [("top_k", k) for k in args.top_k]
    elif args.score_threshold:
        sweep = [("score_threshold", t) for t in args.score_threshold]
    else:
        sweep = [(None, None)]

    # HP-grid combos run outermost; [(None, None)] = no grid (single default pass).
    # Each entry is (arch_or_None, combo): arch=None → use the --arch matrix;
    # an arch string pins the combo to that single architecture (per-arch grid).
    hp_combos = _load_hp_grid(args.hp_grid) if args.hp_grid else [(None, None)]
    if args.hp_grid:
        log.info(f"HP grid: {len(hp_combos)} combo(s) from {args.hp_grid}")
        for i, (a, c) in enumerate(hp_combos):
            log.info(f"  [{i+1}/{len(hp_combos)}] arch={a or 'ALL'}  {c}")

    for hp_arch, hp in hp_combos:
        # Per-arch grid pinned to an arch excluded by --arch → skip.
        if hp_arch is not None and args.arch and hp_arch not in args.arch:
            log.info(f"Skip HP combo (arch {hp_arch} not in --arch {args.arch})")
            continue
        run_archs = [hp_arch] if hp_arch is not None else args.arch
        if hp is not None:
            log.info(f"{'#'*65}")
            log.info(f"  HP combo: arch={hp_arch or 'ALL'}  {hp}")
            log.info(f"{'#'*65}")
        for mode, val in sweep:
            if mode is not None:
                log.info(f"{'='*65}")
                mode_label = {"percentile": "Percentile", "top_k": "Top-K", "score_threshold": "Score-threshold"}.get(mode, mode)
                log.info(f"  {mode_label} sweep: {mode}={val}")
                log.info(f"{'='*65}")
            main(
                exps=args.exp,
                archs=run_archs,
                loss=args.loss,
                force=args.force,
                data_dir=args.data_dir,
                phenol_dates=args.phenol_dates,
                skip_viz=args.skip_viz,
                skip_ndvi=(not args.ndvi) or args.skip_ndvi,
                top_k=val if mode == "top_k" else None,
                percentile=val if mode == "percentile" else None,
                score_threshold=val if mode == "score_threshold" else None,
                batch_size=args.batch_size,
                epochs=args.epochs,
                no_preload=args.no_preload,
                cache_only=args.build_cache_only,
                norm_mode=args.norm,
                hp=hp,
            )

    # ── Upload all logs once, after the whole session finished ────────────────
    _flush_deferred_logs()

    # ── Auto-upload preload cache after --build-cache-only ────────────────────
    if args.build_cache_only and not args.no_upload_cache:
        _up_folder = args.upload_cache_gdrive or GDRIVE_PRELOAD_CACHE_FOLDER_ID or None
        if _up_folder:
            from crop_mapping_pipeline.stages.fetch_data_v6 import upload_preload_cache
            log.info(f"Uploading built preload cache from {PRELOAD_CACHE_DIR} → GDrive {_up_folder}")
            up = upload_preload_cache(_up_folder, str(PRELOAD_CACHE_DIR), overwrite=False)
            log.info(f"Preload cache upload complete: {len(up)} file(s)")
        else:
            log.info("No upload folder set (config.GDRIVE_PRELOAD_CACHE_FOLDER_ID empty / no --upload-cache-gdrive) — skipping upload.")

    if args.shutdown:
        import urllib.request, urllib.error, json as _json, time as _time
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        pod_id  = os.environ.get("RUNPOD_POD_ID")
        api_key = os.environ.get("RUNPOD_API_KEY")
        delay   = 5   # minutes
        if pod_id and api_key:
            log.warning(f"RunPod pod {pod_id} will stop in {delay} minutes.")
            _time.sleep(delay * 60)
            query = f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) {{ id desiredStatus }} }}"}}'
            req   = urllib.request.Request(
                "https://api.runpod.io/graphql",
                data    = query.encode(),
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    log.info(f"Pod stop response: {_json.loads(resp.read())}")
            except urllib.error.URLError as e:
                log.error(f"Failed to stop pod: {e}")
        else:
            log.warning(f"RUNPOD_POD_ID/RUNPOD_API_KEY not set — falling back to sudo shutdown in {delay} min")
            import subprocess
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay}"], check=False)
