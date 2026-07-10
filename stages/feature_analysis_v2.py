"""
Stage 1v3 + 2v2 — Feature Analysis v2 (Date × Band Selection)

Stage versions live under:
  - stages/selections/feature_analysis_v2/stage1/
  - stages/selections/feature_analysis_v2/stage2/
"""

import argparse
import json
import logging
import os
import pathlib
from pathlib import Path
import re
import sys
import time
from datetime import datetime
from glob import glob

import matplotlib
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT.parent))
sys.modules.setdefault("crop_mapping_pipeline.stages.feature_analysis_v2", sys.modules[__name__])

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
import mlflow

from crop_mapping_pipeline.utils.mlflow_utils import patch_artifact_logging
patch_artifact_logging()

from crop_mapping_pipeline.config import (
    BATCH_SIZE,
    CDL_BY_YEAR as _CDL_BY_YEAR,
    CDL_CLASS_NAMES,
    FIGURES_DIR as _FIGURES_DIR,
    KEEP_CLASSES,
    LOGS_DIR as _LOGS_DIR,
    MAX_BANDS_PER_CROP,
    MAX_DATES_PER_CROP,
    MIN_VALID_FRAC,
    MLFLOW_EXPERIMENT_FEATURE,
    MLFLOW_EXPERIMENT_TRAIN_V3,
    MLFLOW_TRACKING_URI,
    PATCH_SIZE,
    PROCESSED_DIR as _PROCESSED_DIR,
    REMAP_LUT,
    RF_IMPORTANCE_THRESH,
    RF_MAX_PIXELS,
    RF_N_ESTIMATORS,
    S2_BAND_NAMES,
    S2_NODATA,
    S2_PROCESSED_DIR as _S2_PROCESSED_DIR,
    SAMPLE_FRACTION,
    SELECT_GSI_DIRECT_JSON,
    SELECT_GSI_DIRECT_BANDS,
    SELECT_RF_DIRECT_JSON,
    SELECT_RF_DIRECT_BANDS,
    SELECT_TOP_K_PER_CROP,
    STRIDE,
    TEST_YEAR,
    TOP_BANDS_PER_CROP,
    TOP_DATES_PER_CROP,
    TRAIN_YEARS,
    VEGE_BANDS,
)

log = logging.getLogger(__name__)

# Stage 2 CNN oracle hyperparameters (removed from config; defined here for fa2.* access)
S2_ENCODER        = "resnet18"
S2_BATCH_SIZE     = BATCH_SIZE
S2_PATCH_SIZE     = 64
S2_STRIDE         = 32
S2_MIN_VALID      = MIN_VALID_FRAC
S2_EPOCHS         = 15
S2_PATIENCE       = 5
S2_BAND_DELTA     = 0.005
S2_BAND_NO_IMPROVE = 5
S2_DATE_DELTA     = 0.005
S2_DATE_NO_IMPROVE = 5
S2_MAX_BANDS_V2   = 20
S2_MAX_DATES      = 10

S2_PROCESSED_DIR = _S2_PROCESSED_DIR
CDL_BY_YEAR = dict(_CDL_BY_YEAR)
PROCESSED_DIR = _PROCESSED_DIR
FIGURES_DIR = _FIGURES_DIR
LOGS_DIR = _LOGS_DIR

# Stage output paths — defaults based on PROCESSED_DIR; overridden by configure_data_dir
STAGE1V3_CANDIDATES_JSON        = _PROCESSED_DIR / "s2" / "2024" / "stage1v3_candidates.json"
STAGE2V3_PER_CROP_JSON          = _PROCESSED_DIR / "stage2v3_per_crop_results.json"
STAGE3_EXP_C_V2_JSON            = _PROCESSED_DIR / "stage3_exp_c_v2.json"
STAGE3_EXP_C_V2_BANDS           = _PROCESSED_DIR / "stage3_exp_c_v2_bands.txt"
STAGE3_EXP_D_JSON               = _PROCESSED_DIR / "stage3_exp_d.json"
STAGE3_EXP_D_BANDS              = _PROCESSED_DIR / "stage3_exp_d_bands.txt"
STAGE2V3_RF_PER_CROP_JSON       = _PROCESSED_DIR / "stage2v3_rf_per_crop_results.json"
STAGE3_EXP_C_V2_RF_JSON         = _PROCESSED_DIR / "stage3_exp_c_v2_rf.json"
STAGE3_EXP_C_V2_RF_BANDS        = _PROCESSED_DIR / "stage3_exp_c_v2_rf_bands.txt"
STAGE3_EXP_C_V2_BANDS_PROJECTED = _PROCESSED_DIR / "stage3_exp_c_v2_bands_projected.json"
STAGE2V3_SWEEP_PER_CROP_JSON    = _PROCESSED_DIR / "stage2v3_sweep_per_crop_results.json"
STAGE3_EXP_C_V3_JSON            = _PROCESSED_DIR / "stage3_exp_c_v3.json"
STAGE3_EXP_C_V3_BANDS           = _PROCESSED_DIR / "stage3_exp_c_v3_bands.txt"


def configure_data_dir(data_dir: str | None) -> None:
    global S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR
    global STAGE1V3_CANDIDATES_JSON, STAGE2V3_PER_CROP_JSON, STAGE3_EXP_C_V2_JSON
    global STAGE3_EXP_C_V2_BANDS, STAGE3_EXP_C_V2_BANDS_PROJECTED
    global STAGE3_EXP_D_JSON, STAGE3_EXP_D_BANDS
    global STAGE2V3_RF_PER_CROP_JSON, STAGE3_EXP_C_V2_RF_JSON, STAGE3_EXP_C_V2_RF_BANDS
    global STAGE2V3_SWEEP_PER_CROP_JSON, STAGE3_EXP_C_V3_JSON, STAGE3_EXP_C_V3_BANDS

    if not data_dir:
        return

    processed = pathlib.Path(data_dir)
    PROCESSED_DIR = processed
    S2_PROCESSED_DIR = processed / "s2" / "2024"
    CDL_BY_YEAR = {"2024": processed / "cdl" / "cdl_2024_study_area_filtered.tif"}
    STAGE1V3_CANDIDATES_JSON = processed / "s2" / "2024" / "stage1v3_candidates.json"
    STAGE2V3_PER_CROP_JSON = processed / "stage2v3_per_crop_results.json"
    STAGE3_EXP_C_V2_JSON = processed / "stage3_exp_c_v2.json"
    STAGE3_EXP_C_V2_BANDS = processed / "stage3_exp_c_v2_bands.txt"
    STAGE3_EXP_C_V2_BANDS_PROJECTED = processed / "stage3_exp_c_v2_bands_projected.json"
    STAGE3_EXP_D_JSON = processed / "stage3_exp_d.json"
    STAGE3_EXP_D_BANDS = processed / "stage3_exp_d_bands.txt"
    STAGE2V3_RF_PER_CROP_JSON = processed / "stage2v3_rf_per_crop_results.json"
    STAGE3_EXP_C_V2_RF_JSON = processed / "stage3_exp_c_v2_rf.json"
    STAGE3_EXP_C_V2_RF_BANDS = processed / "stage3_exp_c_v2_rf_bands.txt"
    STAGE2V3_SWEEP_PER_CROP_JSON = processed / "stage2v3_sweep_per_crop_results.json"
    STAGE3_EXP_C_V3_JSON = processed / "stage3_exp_c_v3.json"
    STAGE3_EXP_C_V3_BANDS = processed / "stage3_exp_c_v3_bands.txt"
    log.info(f"Data dir overridden to {processed}")


def get_stage2_output_path(selector: str) -> pathlib.Path:
    return STAGE3_EXP_C_V2_RF_BANDS if selector == "rf" else STAGE3_EXP_C_V2_BANDS


def _glob_s2_year(yr: str) -> list[str]:
    """Glob S2 files from flat train dir (yr param kept for API compat), then drop
    low-validity dates (same filter as training → standalone selection stays consistent)."""
    from crop_mapping_pipeline.stages.valid_dates import filter_valid_s2_dates
    d = S2_PROCESSED_DIR
    files = sorted(glob(str(d / "*_processed.tif")) + glob(str(d / "S2H_*.tif")))
    seen  = set()
    files = [p for p in files if not (p in seen or seen.add(p))]
    valid, _ = filter_valid_s2_dates(files, cache_dir=d)
    return valid


def build_band_name_to_idx(s2_files: list[str]) -> tuple[list[str], dict[str, int]]:
    all_bandnames = []
    for s2_path in s2_files:
        fname = os.path.basename(s2_path)
        match = re.search(r"_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$", fname)
        date_str = match.group(1).replace("_", "") if match else fname[:8]
        all_bandnames.extend([f"{band}_{date_str}" for band in S2_BAND_NAMES])
    return all_bandnames, {name: idx for idx, name in enumerate(all_bandnames)}


def get_train_year_inputs() -> tuple[str, list[str], str]:
    """Primary training year (2022) — used by Stage 2 and Stage 2v3."""
    s2_year = TRAIN_YEARS[0]
    s2_files = _glob_s2_year(s2_year)
    assert s2_files, f"No S2 files for year {s2_year} in {S2_PROCESSED_DIR}"
    cdl_path = str(CDL_BY_YEAR[s2_year])
    assert os.path.exists(cdl_path), f"CDL not found: {cdl_path}"
    return s2_year, s2_files, cdl_path


def get_stage1_inputs() -> list[tuple[str, list[str], str]]:
    """All training years — used by Stage 1 for more robust GSI band ranking.
    Date candidates always come from TRAIN_YEARS[0] (2022) to stay compatible with Stage 2.
    Returns [(year, s2_files, cdl_path), ...] for each year that has data on disk.
    """
    result = []
    for yr in TRAIN_YEARS:
        s2_files = _glob_s2_year(yr)
        cdl_path = str(CDL_BY_YEAR[yr])
        if not s2_files:
            log.warning(f"Stage 1: no S2 files for year {yr} — skipping")
            continue
        if not os.path.exists(cdl_path):
            log.warning(f"Stage 1: CDL not found for year {yr} ({cdl_path}) — skipping")
            continue
        result.append((yr, s2_files, cdl_path))
    assert result, f"No S2 files found for any training year in {S2_PROCESSED_DIR}"
    return result


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps (Apple Silicon)"
    return "cpu"


DEVICE = _get_device()


class RasterPatchDataset(Dataset):
    def __init__(
        self,
        s2_paths,
        cdl_path,
        patch_size,
        stride,
        min_valid_frac=0.3,
        band_indices=None,
        remap_lut=None,
        target_class_id=None,
    ):
        self.s2_paths = s2_paths
        self.patch_size = patch_size
        self.band_indices = band_indices
        self.remap_lut = remap_lut if remap_lut is not None else REMAP_LUT

        with rasterio.open(cdl_path) as src:
            self._cdl = src.read(1).astype(np.int32)
            self.height = src.height
            self.width = src.width

        self._s2_srcs = [rasterio.open(path) for path in s2_paths]

        ps = patch_size
        self.patches = [
            (row, col)
            for row in range(0, self.height - ps + 1, stride)
            for col in range(0, self.width - ps + 1, stride)
            if (
                np.isin(self._cdl[row : row + ps, col : col + ps], KEEP_CLASSES).mean() >= min_valid_frac
                and (
                    target_class_id is None
                    or (self._cdl[row : row + ps, col : col + ps] == target_class_id).any()
                )
            )
        ]
        tgt = (
            f", require class {target_class_id} ({CDL_CLASS_NAMES.get(target_class_id, '')})"
            if target_class_id is not None
            else ""
        )
        log.info(
            f"  RasterPatchDataset: {len(self.patches)} patches "
            f"(patch={ps}px, stride={stride}px, min_valid={min_valid_frac}{tgt})"
        )

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        row, col = self.patches[idx]
        ps = self.patch_size
        win = rasterio.windows.Window(col, row, ps, ps)

        arrays = [src.read(window=win).astype(np.float32) for src in self._s2_srcs]
        img = np.concatenate(arrays, axis=0)

        if self.band_indices is not None:
            img = img[self.band_indices]

        img[img == S2_NODATA] = 0.0
        for ch in range(img.shape[0]):
            mn, mx = img[ch].min(), img[ch].max()
            img[ch] = (img[ch] - mn) / (mx - mn + 1e-9)

        cdl_patch = self._cdl[row : row + ps, col : col + ps]
        mask = self.remap_lut[np.clip(cdl_patch, 0, 255)]
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64))

    def __del__(self):
        for src in getattr(self, "_s2_srcs", []):
            try:
                src.close()
            except Exception:
                pass


def preload_patches(dataset: RasterPatchDataset) -> TensorDataset:
    n = len(dataset)
    t0 = time.time()
    log.info(f"  Pre-loading {n} patches into RAM...")
    imgs_list, masks_list = [], []
    for idx in range(n):
        img, mask = dataset[idx]
        imgs_list.append(img)
        masks_list.append(mask)
    imgs_t = torch.stack(imgs_list)
    masks_t = torch.stack(masks_list)
    elapsed = time.time() - t0
    mem_mb = (imgs_t.nbytes + masks_t.nbytes) / 1e6
    log.info(f"  Pre-load done: {n} patches  {mem_mb:.1f} MB  ({elapsed:.1f}s)")
    return TensorDataset(imgs_t, masks_t)


def build_unet(in_channels: int) -> nn.Module:
    return smp.Unet(
        encoder_name=S2_ENCODER,
        encoder_weights=None,
        in_channels=in_channels,
        classes=2,
    ).to(DEVICE)


def compute_iou_class1(preds: torch.Tensor, labels: torch.Tensor) -> float:
    pred_mask = (preds.view(-1) == 1).cpu().numpy()
    label_mask = (labels.view(-1) == 1).cpu().numpy()
    inter = (pred_mask & label_mask).sum()
    union = (pred_mask | label_mask).sum()
    return float(inter / union) if union > 0 else 0.0


def split_tensor_dataset(tensor_ds: TensorDataset):
    n_val = max(1, int(0.2 * len(tensor_ds)))
    n_train = len(tensor_ds) - n_val
    return random_split(
        tensor_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )


def build_dataloaders(train_ds, val_ds):
    use_pin = DEVICE.startswith("cuda")
    n_workers = min(4, os.cpu_count() or 1)
    train_dl = DataLoader(
        train_ds,
        batch_size=S2_BATCH_SIZE,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=use_pin,
        persistent_workers=n_workers > 0,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=S2_BATCH_SIZE,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=use_pin,
        persistent_workers=n_workers > 0,
    )
    return train_dl, val_dl, n_workers, use_pin


def dates_to_band_indices(selected_dates, band_name_to_idx, vege_bands=None):
    vege_bands = vege_bands or VEGE_BANDS
    indices = []
    for date in selected_dates:
        for band in vege_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                indices.append(band_name_to_idx[key])
    return indices


def dates_bands_to_indices(selected_dates, selected_bands, band_name_to_idx):
    indices = []
    for date in selected_dates:
        for band in selected_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                indices.append(band_name_to_idx[key])
    return indices


def fmt_date(date_str: str) -> str:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        return f"{months[int(date_str[4:6]) - 1]} {int(date_str[6:8])}"
    except Exception:
        return date_str


def load_stage1_candidates() -> tuple[dict, dict, list]:
    if not os.path.exists(STAGE1V3_CANDIDATES_JSON):
        raise FileNotFoundError(
            f"Stage 1v3 candidates not found: {STAGE1V3_CANDIDATES_JSON}\n"
            "Run Stage 1v3 first:  python feature_analysis_v2.py --stage 1"
        )
    with open(STAGE1V3_CANDIDATES_JSON) as f:
        payload = json.load(f)
    log.info(f"Loaded Stage 1v3 candidates from {STAGE1V3_CANDIDATES_JSON}")
    return payload["date_candidates_per_crop"], payload["band_candidates_per_crop"], payload["all_dates"]


_MLFLOW_EXPERIMENT_OVERRIDE: str | None = None


def mlflow_setup() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT_OVERRIDE or MLFLOW_EXPERIMENT_FEATURE)
    if mlflow.active_run():
        log.warning(f"Closing stale MLflow run: {mlflow.active_run().info.run_id}")
        mlflow.end_run(status="FAILED")


def plot_gsi_heatmaps(gsi_df: pd.DataFrame, all_dates: list, save_dir: pathlib.Path) -> list[pathlib.Path]:
    save_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    all_vals = []
    matrices = {}
    for crop_id in KEEP_CLASSES:
        si_col = gsi_df[crop_id] if crop_id in gsi_df.columns else pd.Series(dtype=float)
        mat = np.zeros((len(all_dates), len(S2_BAND_NAMES)), dtype=np.float32)
        for date_idx, date in enumerate(all_dates):
            for band_idx, band in enumerate(S2_BAND_NAMES):
                key = f"{band}_{date}"
                if key in si_col.index:
                    mat[date_idx, band_idx] = si_col[key]
        matrices[crop_id] = mat
        all_vals.extend(mat.flatten().tolist())
    global_vmax = float(np.nanpercentile(all_vals, 95)) if all_vals else 1.0
    global_vmax = max(global_vmax, 1e-3)
    log.info(f"  GSI heatmap global_vmax (95th pct): {global_vmax:.4f}")

    n_crops = len(KEEP_CLASSES)
    n_cols = 4
    n_rows = (n_crops + n_cols - 1) // n_cols
    fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4.5), constrained_layout=True)
    axes_flat = axes.flatten() if n_crops > 1 else [axes]

    for ax_idx, crop_id in enumerate(KEEP_CLASSES):
        crop_name = CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}")
        ax = axes_flat[ax_idx]
        matrix = matrices[crop_id]

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=global_vmax)
        ax.set_xticks(range(len(S2_BAND_NAMES)))
        ax.set_xticklabels(S2_BAND_NAMES, fontsize=8)
        ax.set_yticks(range(len(all_dates)))
        ax.set_yticklabels([fmt_date(date) for date in all_dates], fontsize=7)
        ax.set_title(crop_name, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        im_single = ax_single.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=global_vmax)
        ax_single.set_xticks(range(len(S2_BAND_NAMES)))
        ax_single.set_xticklabels(S2_BAND_NAMES, fontsize=9)
        ax_single.set_yticks(range(len(all_dates)))
        ax_single.set_yticklabels([fmt_date(date) for date in all_dates], fontsize=8)
        ax_single.set_title(f"SI_global — {crop_name}", fontsize=11)
        ax_single.set_xlabel("Spectral Band")
        ax_single.set_ylabel("Acquisition Date")
        plt.colorbar(im_single, ax=ax_single, label="SI_global")
        plt.tight_layout()
        out = save_dir / f"stage1v3_gsi_{crop_name.lower().replace(' ', '_')}.png"
        fig_single.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig_single)
        saved.append(out)

    for ax in axes_flat[n_crops:]:
        ax.set_visible(False)

    fig_grid.suptitle("SI_global Heatmaps — Date × Band per Crop", fontsize=13)
    grid_path = save_dir / "stage1v3_gsi_heatmaps_all.png"
    fig_grid.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig_grid)
    saved.append(grid_path)
    log.info(f"  Saved {len(saved)} GSI heatmap(s) to {save_dir}")
    return saved


def plot_selection_table(results_per_crop: dict, save_path: pathlib.Path) -> None:
    rows = []
    for crop_id in KEEP_CLASSES:
        result = results_per_crop.get(crop_id, {})
        rows.append(
            {
                "Crop": CDL_CLASS_NAMES.get(crop_id, f"cls{crop_id}"),
                "Key Period": ", ".join(fmt_date(date) for date in result.get("dates", [])) or "—",
                "Selected Bands": ", ".join(result.get("bands", [])) or "—",
                "IoU (bands)": f"{result.get('best_iou_after_bands', 0.0):.4f}",
            }
        )
    df = pd.DataFrame(rows)
    csv_path = save_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    log.info(f"  Saved selection table CSV: {csv_path}")

    fig_h = 0.45 * (len(rows) + 1.5)
    fig, ax = plt.subplots(figsize=(13, max(fig_h, 3)))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.12, 0.38, 0.32, 0.12],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for col_idx in range(len(df.columns)):
        tbl[(0, col_idx)].set_facecolor("#2d6a2d")
        tbl[(0, col_idx)].set_text_props(color="white", fontweight="bold")
    for row_idx in range(1, len(rows) + 1):
        face_color = "#f0f7f0" if row_idx % 2 == 0 else "white"
        for col_idx in range(len(df.columns)):
            tbl[(row_idx, col_idx)].set_facecolor(face_color)
    ax.set_title("Stage 2v2 Per-Crop Feature Selection", fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved selection table PNG: {save_path}")


def save_results_v2(results_per_crop: dict, band_name_to_idx: dict, per_crop_json=None, exp_json=None, exp_bands=None) -> None:
    per_crop_path = per_crop_json or STAGE2V3_PER_CROP_JSON
    exp_json_path = exp_json or STAGE3_EXP_C_V2_JSON
    exp_bands_path = exp_bands or STAGE3_EXP_C_V2_BANDS
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    union_dates = sorted({date for crop_id in KEEP_CLASSES for date in results_per_crop.get(crop_id, {}).get("dates", [])})
    seen_bands, union_bands = set(), []
    for crop_id in KEEP_CLASSES:
        for band in results_per_crop.get(crop_id, {}).get("bands", []):
            if band not in seen_bands:
                seen_bands.add(band)
                union_bands.append(band)

    total_channels = len(union_dates) * len(union_bands)
    per_crop_summary = {}
    for crop_id in KEEP_CLASSES:
        result = results_per_crop.get(crop_id, {})
        per_crop_summary[str(crop_id)] = {
            "crop_name": CDL_CLASS_NAMES[crop_id],
            "dates": result.get("dates", []),
            "bands": result.get("bands", []),
            "k_dates": result.get("k_dates", 0),
            "k_bands": result.get("k_bands", 0),
            "best_iou_after_dates": result.get("best_iou_after_dates", 0.0),
            "best_iou_after_bands": result.get("best_iou_after_bands", 0.0),
            "fallback_dates": result.get("fallback_dates", False),
            "fallback_bands": result.get("fallback_bands", False),
            "mlflow_run_id": result.get("mlflow_run_id", ""),
        }

    with open(per_crop_path, "w") as f:
        json.dump(per_crop_summary, f, indent=2)
    log.info(f"Saved: {per_crop_path}")

    with open(exp_json_path, "w") as f:
        json.dump(
            {
                "union_dates": union_dates,
                "union_bands": union_bands,
                "total_channels": total_channels,
                "per_crop": per_crop_summary,
            },
            f,
            indent=2,
        )
    log.info(f"Saved: {exp_json_path}")

    band_lines = []
    for date in union_dates:
        for band in union_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                band_lines.append(key)
    with open(exp_bands_path, "w") as f:
        f.write("\n".join(band_lines))
    log.info(f"Saved: {exp_bands_path}  ({len(band_lines)} channel entries)")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = FIGURES_DIR / "stage2v2_selection_table.png"
    plot_selection_table(results_per_crop, table_path)
    try:
        mlflow.log_metrics(
            {
                "n_union_dates": len(union_dates),
                "n_union_bands": len(union_bands),
                "total_channels": total_channels,
            }
        )
        mlflow.log_artifact(str(table_path))
        mlflow.log_artifact(str(table_path.with_suffix(".csv")))
    except Exception:
        pass


def save_exp_d_bands(date_candidates_per_crop: dict, band_candidates_per_crop: dict, band_name_to_idx: dict, data_dir=None) -> None:
    d_json = STAGE3_EXP_D_JSON if not data_dir else pathlib.Path(data_dir) / "stage3_exp_d.json"
    d_bands = STAGE3_EXP_D_BANDS if not data_dir else pathlib.Path(data_dir) / "stage3_exp_d_bands.txt"

    seen_dates, union_dates = set(), []
    for crop_id in KEEP_CLASSES:
        for date in date_candidates_per_crop.get(str(crop_id), []):
            if date not in seen_dates:
                seen_dates.add(date)
                union_dates.append(date)

    seen_bands, union_bands = set(), []
    for crop_id in KEEP_CLASSES:
        for band in band_candidates_per_crop.get(str(crop_id), []):
            if band not in seen_bands:
                seen_bands.add(band)
                union_bands.append(band)

    with open(d_json, "w") as f:
        json.dump(
            {
                "union_dates": union_dates,
                "union_bands": union_bands,
                "total_channels": len(union_dates) * len(union_bands),
                "per_crop": {
                    str(crop_id): {
                        "crop_name": CDL_CLASS_NAMES[crop_id],
                        "top_dates": date_candidates_per_crop.get(str(crop_id), []),
                        "top_bands": band_candidates_per_crop.get(str(crop_id), []),
                    }
                    for crop_id in KEEP_CLASSES
                },
            },
            f,
            indent=2,
        )
    band_lines = []
    for date in union_dates:
        for band in union_bands:
            key = f"{band}_{date}"
            if key in band_name_to_idx:
                band_lines.append(key)
    with open(d_bands, "w") as f:
        f.write("\n".join(band_lines))


def run_project_v2() -> None:
    from datetime import date as _date

    if not STAGE3_EXP_C_V2_BANDS.exists():
        raise FileNotFoundError(
            f"Stage 2v2 output not found: {STAGE3_EXP_C_V2_BANDS}\n"
            "Run Stage 2v2 first:  python feature_analysis_v2.py --stage 2"
        )

    with open(STAGE3_EXP_C_V2_BANDS) as f:
        selected_bands = [line.strip() for line in f if line.strip()]
    if not selected_bands:
        raise ValueError(f"{STAGE3_EXP_C_V2_BANDS} is empty — re-run Stage 2v2.")

    band_mmdd = []
    for entry in selected_bands:
        match = re.match(r"(.+)_(\d{4})(\d{2})(\d{2})$", entry)
        if match:
            band_mmdd.append((match.group(1), match.group(3) + match.group(4)))

    projected = {}
    for year in list(dict.fromkeys(list(TRAIN_YEARS) + [TEST_YEAR])):
        year_files = _glob_s2_year(year)
        if not year_files:
            log.warning(f"  {year}: no S2 files found — skipping")
            continue
        year_dates = []
        for path in year_files:
            match = re.search(r"_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$", pathlib.Path(path).name)
            if match:
                year_dates.append(match.group(1).replace("_", ""))
        year_dates = sorted(set(year_dates))

        year_bands = []
        for band, mmdd in band_mmdd:
            month, day = int(mmdd[:2]), int(mmdd[2:])
            try:
                target = _date(int(year), month, day)
            except ValueError:
                target = _date(int(year), month, min(day, 28))
            target_doy = target.timetuple().tm_yday

            best_date, best_dist = None, 999
            for yyyymmdd in year_dates:
                d = _date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
                dist = abs(d.timetuple().tm_yday - target_doy)
                if dist < best_dist:
                    best_dist, best_date = dist, yyyymmdd
            year_bands.append(f"{band}_{best_date}")
        projected[year] = year_bands

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(STAGE3_EXP_C_V2_BANDS_PROJECTED, "w") as f:
        json.dump(projected, f, indent=2)

    mlflow_setup()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=f"stage2v3_project_{ts}"):
        mlflow.set_tag("stage", "project_v2")
        for year, bands in projected.items():
            mlflow.log_param(f"n_bands_{year}", len(bands))
            mlflow.set_tag(f"bands_{year}", str(bands))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_BANDS_PROJECTED))
        mlflow.log_artifact(str(STAGE3_EXP_C_V2_BANDS))


from crop_mapping_pipeline.stages.selections.feature_analysis_v2.stage1.v3 import run_stage1v3
from crop_mapping_pipeline.stages.selections.feature_analysis_v2.stage2.v2 import run_stage2v2
from crop_mapping_pipeline.stages.selections.feature_analysis_v2.stage2.v2_rf import run_stage2v2_rf
from crop_mapping_pipeline.stages.selections.feature_analysis_v2.stage2.v3 import run_stage2v3
from crop_mapping_pipeline.stages.selections.gsi_direct import run_gsi_direct
from crop_mapping_pipeline.stages.selections.rf_direct import run_rf_direct
from crop_mapping_pipeline.stages.selections.single_date_gsi import run_single_date_gsi
from crop_mapping_pipeline.stages.selections.single_date_rf import run_single_date_rf
from crop_mapping_pipeline.stages.selections.naive_mt_gsi import run_naive_mt_gsi
from crop_mapping_pipeline.stages.selections.naive_mt_rf import run_naive_mt_rf

_DIRECT_OUTPUT_MAP = {
    "gsi_direct": (SELECT_GSI_DIRECT_JSON, SELECT_GSI_DIRECT_BANDS),
    "rf_direct":  (SELECT_RF_DIRECT_JSON,  SELECT_RF_DIRECT_BANDS),
}

_DOMAIN_SCOPED_SELECTORS = {"single_date_gsi", "single_date_rf", "naive_mt_gsi", "naive_mt_rf"}


def main(force: bool = False, data_dir: str = None, output_dir: str = None,
         stage: str = "all", selector: str = "cnn",
         mlflow_exp: str | None = None, top_k_values: list[int] | None = None,
         percentile_values: list[float] | None = None,
         score_threshold: float | None = None) -> None:
    global _MLFLOW_EXPERIMENT_OVERRIDE, KEEP_CLASSES, CDL_CLASS_NAMES
    if mlflow_exp == "v3":
        _MLFLOW_EXPERIMENT_OVERRIDE = MLFLOW_EXPERIMENT_TRAIN_V3
    else:
        _MLFLOW_EXPERIMENT_OVERRIDE = None

    configure_data_dir(data_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if stage in ("1", "all"):
        if not force and STAGE1V3_CANDIDATES_JSON.exists():
            log.info(f"Stage 1v3 output already exists: {STAGE1V3_CANDIDATES_JSON}")
            log.info("Use --force to re-run.")
        else:
            log.info(f"Device: {device_label()}")
            years_data = get_stage1_inputs()
            run_stage1v3(years_data, data_dir=data_dir)
            log.info("Stage 1v3 complete.")
        if stage == "1":
            return

    if stage in ("2", "all"):
        output_check = get_stage2_output_path(selector)
        if not force and output_check.exists():
            log.info(f"Stage 2v2-{selector.upper()} output already exists: {output_check}")
            log.info("Use --force to re-run.")
            if stage == "2":
                return
        else:
            date_candidates_per_crop, band_candidates_per_crop, all_dates = load_stage1_candidates()
            _s2_year, s2_files, cdl_path = get_train_year_inputs()
            _all_bandnames, band_name_to_idx = build_band_name_to_idx(s2_files)
            log.info(f"Device: {device_label()}  selector={selector}")
            stage2_fn = run_stage2v2_rf if selector == "rf" else run_stage2v2
            stage2_fn(
                s2_paths=s2_files,
                cdl_path=cdl_path,
                date_candidates_per_crop=date_candidates_per_crop,
                band_candidates_per_crop=band_candidates_per_crop,
                band_name_to_idx=band_name_to_idx,
                all_dates=all_dates,
                data_dir=data_dir,
            )
            log.info(f"Stage 2v2-{selector.upper()} complete.")
        if stage == "2":
            return

    if stage in ("2v3", "all"):
        if not force and STAGE2V3_SWEEP_PER_CROP_JSON.exists():
            log.info(f"Stage 2v3 sweep output already exists: {STAGE2V3_SWEEP_PER_CROP_JSON}")
            log.info("Use --force to re-run.")
            if stage == "2v3":
                return
        else:
            date_candidates_per_crop, band_candidates_per_crop, all_dates = load_stage1_candidates()
            _s2_year, s2_files, cdl_path = get_train_year_inputs()
            _all_bandnames, band_name_to_idx = build_band_name_to_idx(s2_files)
            log.info("Running Stage 2v3 incremental top-K enumeration (no training)")
            run_stage2v3(
                date_candidates_per_crop=date_candidates_per_crop,
                band_candidates_per_crop=band_candidates_per_crop,
                band_name_to_idx=band_name_to_idx,
                data_dir=data_dir,
            )
            log.info("Stage 2v3 sweep complete.")
        if stage == "2v3":
            return

    if stage == "project":
        if not force and STAGE3_EXP_C_V2_BANDS_PROJECTED.exists():
            log.info(f"Projected bands already exist: {STAGE3_EXP_C_V2_BANDS_PROJECTED}")
            log.info("Use --force to re-run.")
            return
        run_project_v2()
        log.info("Band projection complete.")
        return

    if stage == "select":
        if selector in _DOMAIN_SCOPED_SELECTORS:
            _year, s2_files, cdl_path = get_train_year_inputs()
            _rf_dir = str(Path(data_dir) if data_dir else PROCESSED_DIR)
            if selector == "single_date_gsi":
                run_single_date_gsi(s2_files, cdl_path, force=force)
            elif selector == "single_date_rf":
                run_single_date_rf(s2_files, cdl_path, data_dir=_rf_dir, force=force)
            elif selector == "naive_mt_gsi":
                run_naive_mt_gsi(s2_files, cdl_path, force=force)
            elif selector == "naive_mt_rf":
                run_naive_mt_rf(s2_files, cdl_path, data_dir=_rf_dir, force=force)
            log.info(f"Domain-scoped selection ({selector}) complete.")
            return

        if selector not in _DIRECT_OUTPUT_MAP:
            raise ValueError(
                f"--selector must be one of {sorted(list(_DIRECT_OUTPUT_MAP) + list(_DOMAIN_SCOPED_SELECTORS))} "
                f"for --stage select, got {selector!r}"
            )
        fn = run_gsi_direct if selector == "gsi_direct" else run_rf_direct
        years_data = get_stage1_inputs()
        # output_dir overrides where JSONs are written; data_dir controls S2/CDL input only
        out_base = Path(output_dir) if output_dir else (Path(data_dir) if data_dir else _DIRECT_OUTPUT_MAP[selector][0].parent)
        out_base.mkdir(parents=True, exist_ok=True)

        if score_threshold is not None:
            stem     = f"select_{selector}_s{score_threshold:g}"
            json_out = out_base / f"{stem}.json"
            if not force and json_out.exists():
                log.info(f"  score_threshold={score_threshold}: output exists ({json_out.name}) — skipping (--force to re-run)")
            else:
                log.info(f"  Running {selector} score_threshold={score_threshold} ...")
                fn(years_data, score_threshold=score_threshold, data_dir=str(out_base), out_stem=stem)
                log.info(f"  score_threshold={score_threshold} complete → {json_out}")
            log.info(f"Direct selection ({selector}) score_threshold={score_threshold} complete.")
            return

        if percentile_values:
            for p in percentile_values:
                stem     = f"select_{selector}_p{p:g}"
                json_out = out_base / f"{stem}.json"
                if not force and json_out.exists():
                    log.info(f"  P{p:g}: output exists ({json_out.name}) — skipping (--force to re-run)")
                    continue
                log.info(f"  Running {selector} percentile={p:g} ...")
                fn(years_data, percentile=p, data_dir=str(out_base), out_stem=stem)
                log.info(f"  P{p:g} complete → {json_out}")
            log.info(f"Direct selection ({selector}) percentile sweep complete: P={percentile_values}")
            return

        ks = top_k_values or [SELECT_TOP_K_PER_CROP]
        for k in ks:
            stem     = f"select_{selector}_k{k}"
            json_out = out_base / f"{stem}.json"
            if not force and json_out.exists():
                log.info(f"  k={k}: output exists ({json_out.name}) — skipping (--force to re-run)")
                continue
            log.info(f"  Running {selector} top_k={k} ...")
            fn(years_data, top_k=k, data_dir=str(out_base), out_stem=stem)
            log.info(f"  k={k} complete → {json_out}")
        log.info(f"Direct selection ({selector}) sweep complete: k={ks}")
        return

    log.info("Feature analysis v2 complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Feature analysis v2: Stage 1v3 + Stage 2v2/v3")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "2v3", "all", "project", "select"],
        default="all",
        help=(
            "'select' runs a single-stage selector: gsi_direct/rf_direct (full year, all dates) "
            "or single_date_gsi/single_date_rf/naive_mt_gsi/naive_mt_rf (domain-scoped)."
        ),
    )
    parser.add_argument(
        "--selector",
        choices=["cnn", "rf", "gsi_direct", "rf_direct",
                 "single_date_gsi", "single_date_rf", "naive_mt_gsi", "naive_mt_rf"],
        default="cnn",
        help="Selector: cnn/rf for Stage 2v2; others for --stage select.",
    )
    parser.add_argument("--force", "--overwrite", dest="force", action="store_true",
                        help="Re-run even if outputs exist")
    parser.add_argument("--top-k", type=int, nargs="+", default=None, metavar="K",
                        help="Top-K per crop for --stage select sweep (e.g. --top-k 5 10 15 20 30)")
    parser.add_argument("--percentile", type=float, nargs="+", default=None, metavar="P",
                        help="Pooled-percentile threshold(s) for --stage select sweep "
                             "(per-class union, writes select_*_p{P}.json). Mutually exclusive with --top-k and --score-threshold. "
                             "E.g. --percentile 70 75 80 85 90 95")
    parser.add_argument("--score-threshold", type=float, default=None, metavar="T",
                        help="Per-crop normalized-score threshold for --stage select "
                             "(normalize each crop's scores to [0,1], retain >= T, union). "
                             "Follows Wei et al. 2023 (recommended T=0.5). "
                             "Mutually exclusive with --top-k and --percentile.")
    parser.add_argument("--data-dir", type=str, default=None, help="Override processed data directory (S2/CDL input)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for selection output JSONs (--stage select only); defaults to --data-dir")
    parser.add_argument("--mlflow-exp", choices=["v3"], default=None,
                        help="Route MLflow runs to a specific experiment. 'v3' → cropmap_segmentation_s2_v3.")
    return parser


def configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / f"feature_analysis_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    n_modes = sum([bool(args.top_k), bool(args.percentile), args.score_threshold is not None])
    if n_modes > 1:
        build_parser().error("--top-k, --percentile, and --score-threshold are mutually exclusive — pick one.")
    configure_logging()
    main(force=args.force, data_dir=args.data_dir, output_dir=args.output_dir,
         stage=args.stage, selector=args.selector,
         mlflow_exp=args.mlflow_exp, top_k_values=args.top_k,
         percentile_values=args.percentile,
         score_threshold=args.score_threshold)


if __name__ == "__main__":
    cli()
