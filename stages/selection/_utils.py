"""Shared pixel-sampling utilities for single-stage direct selectors."""

import json
import logging
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from crop_mapping_pipeline.config import S2_BAND_NAMES, S2_NODATA, KEEP_CLASSES, SAMPLE_FRACTION, CDL_CLASS_NAMES

log = logging.getLogger(__name__)


def build_channel_names(s2_paths: list[str]) -> tuple[list[str], list[str], dict[str, int]]:
    """Return (all_bandnames, all_dates, band_name_to_idx) for a list of S2 files."""
    all_bandnames: list[str] = []
    dates_seen: list[str] = []
    for path in s2_paths:
        fname = os.path.basename(path)
        m = re.search(r"_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$", fname)
        date_str = m.group(1).replace("_", "") if m else fname[:8]
        if date_str not in dates_seen:
            dates_seen.append(date_str)
        all_bandnames.extend([f"{band}_{date_str}" for band in S2_BAND_NAMES])
    all_dates = sorted(dates_seen)
    band_name_to_idx = {name: idx for idx, name in enumerate(all_bandnames)}
    return all_bandnames, all_dates, band_name_to_idx


def sample_pixels(s2_paths: list[str], cdl_path: str,
                  bandnames: list[str]) -> pd.DataFrame:
    """Sample crop pixels from S2 files without loading all files into RAM at once.

    Strategy: read CDL once → determine valid pixel indices → for each S2 file
    read only the sampled rows. Peak RAM = 1 S2 file (11 bands × H × W × 4 bytes ≈ 1 GB)
    instead of all 25 files stacked (≈ 28 GB).
    """
    # Read CDL once to get valid pixel indices and labels
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
        height, width = cdl.shape

    lbl_1d = cdl.flatten()
    del cdl

    valid_mask = np.isin(lbl_1d, KEEP_CLASSES)
    valid_indices = np.where(valid_mask)[0]   # flat pixel indices of crop pixels
    lbl_valid = lbl_1d[valid_mask]
    del lbl_1d

    # Draw sample indices once (same seed → reproducible)
    rng = np.random.default_rng(42)
    n = min(len(valid_indices), max(1000, int(len(valid_indices) * SAMPLE_FRACTION)))
    chosen = rng.choice(len(valid_indices), n, replace=False)
    sample_flat_idx = valid_indices[chosen]   # flat pixel positions to extract
    lbl_sample = lbl_valid[chosen]
    del valid_indices, lbl_valid

    # Pre-allocate output array: (n_samples, n_channels)
    n_channels = len(bandnames)
    data = np.full((n, n_channels), np.nan, dtype=np.float32)

    # Fill columns from each S2 file — one file at a time (11 bands × H × W × 4 bytes)
    col = 0
    for path in s2_paths:
        with rasterio.open(path) as src:
            n_file_bands = src.count
            arr = src.read().astype(np.float32)   # (11, H, W)

        arr[arr == S2_NODATA] = np.nan
        arr_2d = arr.reshape(n_file_bands, -1).T  # (H*W, 11)
        del arr

        data[:, col:col + n_file_bands] = arr_2d[sample_flat_idx]
        del arr_2d
        col += n_file_bands

    df = pd.DataFrame(data, columns=bandnames)
    df.insert(0, "class_label", lbl_sample.astype(int))
    return df


def save_selection(
    per_crop: dict[int, list[str]],
    json_path: Path,
    txt_path: Path,
    selector: str,
    top_k: int,
    meta: dict | None = None,
    percentile: float | None = None,
    score_threshold: float | None = None,
) -> list[str]:
    """Compute union of per-crop channels, save JSON + TXT, return union list."""
    seen: dict[str, None] = {}
    for channels in per_crop.values():
        for ch in channels:
            seen[ch] = None
    union: list[str] = list(seen.keys())

    from crop_mapping_pipeline.config import CDL_CLASS_NAMES
    from datetime import datetime

    if score_threshold is not None:
        sel_mode = "score_threshold"
    elif percentile is not None:
        sel_mode = "percentile"
    else:
        sel_mode = "top_k"

    payload = {
        "run_ts":           datetime.now().strftime("%Y%m%d-%H%M%S"),
        "selector":         selector,
        "top_k":            top_k,
        "percentile":       percentile,
        "score_threshold":  score_threshold,
        "selection_mode":   sel_mode,
        "n_union":          len(union),
        "per_crop":         {str(k): v for k, v in per_crop.items()},
        "union_channels":   union,
        **(meta or {}),
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(union) + "\n")

    return union


def save_per_class_table(
    per_crop: dict,
    save_dir: Path,
    stem: str,
    score_label: str = "Score",
    adjusted_per_crop: "dict | None" = None,
) -> list[Path]:
    """Generate per-class band-date result tables (CSV + PNG) from a per_crop selection dict.

    per_crop: {crop_id (int or str): [channel_name, ...]}  e.g. {"1": ["B5_20240614", ...]}
    adjusted_per_crop: optional {crop_id: pd.Series(index=channel_names)} with numeric scores.
    Returns list of saved paths.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def _parse(ch: str) -> tuple[str, str]:
        parts = ch.rsplit("_", 1)
        if len(parts) == 2:
            band, date8 = parts[0], parts[1]
            return band, f"{date8[4:6]}/{date8[6:8]}" if len(date8) == 8 else date8
        return ch, "?"

    # ── 1. Detail table: one row per selected channel per crop ───────────────
    rows = []
    for crop_id_raw, channels in per_crop.items():
        crop_id = int(crop_id_raw)
        crop_name = CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}")
        scores = adjusted_per_crop.get(crop_id) if adjusted_per_crop else None
        for rank, ch in enumerate(channels, 1):
            band, date_fmt = _parse(ch)
            score = float(scores[ch]) if scores is not None and ch in scores.index else None
            row = {"Crop": crop_name, "Rank": rank, "Band": band, "Date": date_fmt, "Channel": ch}
            if score is not None:
                row[score_label] = round(score, 6)
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    detail_csv = save_dir / f"{stem}_per_class_detail.csv"
    detail_df.to_csv(detail_csv, index=False)
    saved.append(detail_csv)
    log.info(f"  Saved detail table: {detail_csv}")

    # ── 2. Pivot table: crop × date (count of selected bands per date) ───────
    all_dates_sorted = sorted({_parse(ch)[1] for chs in per_crop.values() for ch in chs})
    pivot_rows: dict[str, dict[str, int]] = {}
    for crop_id_raw, channels in per_crop.items():
        crop_id = int(crop_id_raw)
        crop_name = CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}")
        date_counts: dict[str, int] = {d: 0 for d in all_dates_sorted}
        for ch in channels:
            _, date_fmt = _parse(ch)
            date_counts[date_fmt] = date_counts.get(date_fmt, 0) + 1
        pivot_rows[crop_name] = date_counts

    pivot_df = pd.DataFrame(pivot_rows).T.fillna(0).astype(int)
    pivot_df = pivot_df[[c for c in all_dates_sorted if c in pivot_df.columns]]
    pivot_csv = save_dir / f"{stem}_per_class_pivot.csv"
    pivot_df.to_csv(pivot_csv)
    saved.append(pivot_csv)
    log.info(f"  Saved pivot table: {pivot_csv}")

    # ── 3. PNG: styled pivot table ───────────────────────────────────────────
    n_rows, n_cols = pivot_df.shape
    fig_w = max(14, n_cols * 0.55 + 3)
    fig_h = max(3, n_rows * 0.45 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=pivot_df.values,
        rowLabels=list(pivot_df.index),
        colLabels=list(pivot_df.columns),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.5)

    # Highlight non-zero cells
    for (row, col), cell in tbl.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            val = pivot_df.iloc[row - 1, col]
            if val > 0:
                intensity = min(1.0, val / max(pivot_df.values.max(), 1))
                cell.set_facecolor((1.0 - 0.55 * intensity, 1.0 - 0.30 * intensity, 1.0 - 0.55 * intensity))

    ax.set_title(
        f"Per-Class Selected Bands per Date — {stem.replace('_', ' ').title()}",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    pivot_png = save_dir / f"{stem}_per_class_pivot.png"
    fig.savefig(pivot_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(pivot_png)
    log.info(f"  Saved pivot PNG: {pivot_png}")

    # ── 4. PNG: band × crop heatmap (spectral bands as rows) ─────────────────
    all_bands_sorted = sorted({_parse(ch)[0] for chs in per_crop.values() for ch in chs},
                              key=lambda b: S2_BAND_NAMES.index(b) if b in S2_BAND_NAMES else 99)
    band_rows: dict[str, dict[str, int]] = {}
    for crop_id_raw, channels in per_crop.items():
        crop_id = int(crop_id_raw)
        crop_name = CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}")
        band_counts: dict[str, int] = {b: 0 for b in all_bands_sorted}
        for ch in channels:
            band, _ = _parse(ch)
            band_counts[band] = band_counts.get(band, 0) + 1
        band_rows[crop_name] = band_counts

    band_df = pd.DataFrame(band_rows).T.fillna(0).astype(int)
    band_df = band_df[[b for b in all_bands_sorted if b in band_df.columns]]
    band_csv = save_dir / f"{stem}_per_class_band.csv"
    band_df.to_csv(band_csv)
    saved.append(band_csv)

    fig2_w = max(8, len(all_bands_sorted) * 0.7 + 2)
    fig2_h = max(3, n_rows * 0.45 + 1.5)
    fig2, ax2 = plt.subplots(figsize=(fig2_w, fig2_h))
    ax2.axis("off")
    tbl2 = ax2.table(
        cellText=band_df.values,
        rowLabels=list(band_df.index),
        colLabels=list(band_df.columns),
        cellLoc="center",
        loc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(8)
    tbl2.scale(1, 1.5)
    for (row, col), cell in tbl2.get_celld().items():
        if row == 0 or col == -1:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            val = band_df.iloc[row - 1, col]
            if val > 0:
                intensity = min(1.0, val / max(band_df.values.max(), 1))
                cell.set_facecolor((1.0 - 0.55 * intensity, 1.0 - 0.30 * intensity, 1.0 - 0.55 * intensity))
    ax2.set_title(
        f"Per-Class Selected Channels per Band — {stem.replace('_', ' ').title()}",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    band_png = save_dir / f"{stem}_per_class_band.png"
    fig2.savefig(band_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    saved.append(band_png)
    log.info(f"  Saved band PNG: {band_png}")

    return saved


def hardware_info() -> dict:
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

    info = {"cpu_name": cpu_name or "unknown", "cpu_cores": os.cpu_count()}
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["ram_total_gb"] = None

    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_name"]      = torch.cuda.get_device_name(0)
            info["gpu_count"]     = torch.cuda.device_count()
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        else:
            info["gpu_name"], info["gpu_count"], info["gpu_memory_gb"] = "none", 0, None
    except Exception:
        info["gpu_name"], info["gpu_count"], info["gpu_memory_gb"] = "unknown", None, None
    return info


def _metric_safe(name: str) -> str:
    """mlflow metric keys: keep alnum/_-./space — replace anything else."""
    return re.sub(r"[^0-9A-Za-z_\-./ ]", "_", name)


def log_selection_run(
    *,
    selector: str,
    run_name_prefix: str,
    per_crop: dict[int, list[str]],
    union: list[str],
    json_path: Path,
    params: dict,
    duration_s: float,
    threshold: float | None = None,
    extra_metrics: dict | None = None,
    extra_artifacts: "list[Path] | None" = None,
):
    """Log a band-selection run to MLflow: results, per-crop counts, runtime,
    machine identity, and (if available) live system metrics. Non-fatal on error."""
    import shutil
    import tempfile
    from datetime import datetime

    import mlflow
    from crop_mapping_pipeline.config import (
        MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_FEATURE, CDL_CLASS_NAMES,
    )

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_FEATURE)
        run_name = f"{run_name_prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        try:
            run_ctx = mlflow.start_run(run_name=run_name, log_system_metrics=True)
        except TypeError:   # older mlflow without system-metrics kwarg
            run_ctx = mlflow.start_run(run_name=run_name)

        with run_ctx:
            hw = hardware_info()
            mlflow.log_params({
                **params,
                **{f"hw_{k}": v for k, v in hw.items()},
            })
            # Results
            mlflow.log_metric("n_union_channels", len(union))
            mlflow.log_metric("runtime_seconds", round(duration_s, 2))
            mlflow.log_metric("runtime_minutes", round(duration_s / 60.0, 3))
            if threshold is not None:
                mlflow.log_metric("pooled_threshold", float(threshold))
            # Per-crop selected counts
            for cid, chs in per_crop.items():
                cname = CDL_CLASS_NAMES.get(cid, str(cid))
                mlflow.log_metric(_metric_safe(f"n_sel_{cname}"), len(chs))
            for k, v in (extra_metrics or {}).items():
                if v is not None:
                    mlflow.log_metric(_metric_safe(k), float(v))
            # Full selection JSON (per-crop bands + union)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_json = Path(tmp) / Path(json_path).name
                shutil.copy(json_path, tmp_json)
                mlflow.log_artifact(str(tmp_json))
            # Per-class table artifacts (CSV + PNG)
            for art_path in (extra_artifacts or []):
                art_path = Path(art_path)
                if art_path.exists():
                    mlflow.log_artifact(str(art_path))
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")
