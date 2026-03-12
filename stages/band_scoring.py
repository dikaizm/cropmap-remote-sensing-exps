"""Band scoring — per-crop GSI and RF importance scoring for band selection comparison.

Produces:
  gsi_candidates.json       — per-crop date/band ranked by GSI (used by single_date + naive_multitemporal)
  select_gsi_direct_k*.json — joint spectral-temporal top-K by GSI (used by gsi experiment)
  select_rf_direct_k*.json  — joint spectral-temporal top-K by RF importance (used by rf experiment)
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
    CDL_BY_YEAR as _CDL_BY_YEAR,
    CDL_TRAIN as _CDL_TRAIN,
    CDL_CLASS_NAMES,
    FIGURES_DIR as _FIGURES_DIR,
    GSI_CANDIDATES_JSON as _GSI_CANDIDATES_JSON,
    KEEP_CLASSES,
    LOGS_DIR as _LOGS_DIR,
    MAX_BANDS_PER_CROP,
    MAX_DATES_PER_CROP,
    MLFLOW_EXPERIMENT_FEATURE,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR as _PROCESSED_DIR,
    REMAP_LUT,
    RF_IMPORTANCE_THRESH,
    RF_MAX_PIXELS,
    RF_N_ESTIMATORS,
    S2_BAND_NAMES,
    S2_NODATA,
    S2_TRAIN_DIR as _S2_TRAIN_DIR,
    SAMPLE_FRACTION,
    SELECT_GSI_DIRECT_JSON,
    SELECT_GSI_DIRECT_BANDS,
    SELECT_RF_DIRECT_JSON,
    SELECT_RF_DIRECT_BANDS,
    SELECT_TOP_K_PER_CROP,
    TEST_YEAR,
    TOP_BANDS_PER_CROP,
    TOP_DATES_PER_CROP,
    TRAIN_YEARS,
    VEGE_BANDS,
)

log = logging.getLogger(__name__)

S2_TRAIN_DIR     = _S2_TRAIN_DIR
S2_PROCESSED_DIR = S2_TRAIN_DIR   # backwards-compat alias
CDL_TRAIN        = _CDL_TRAIN
CDL_BY_YEAR      = dict(_CDL_BY_YEAR)
PROCESSED_DIR    = _PROCESSED_DIR
FIGURES_DIR      = _FIGURES_DIR
LOGS_DIR         = _LOGS_DIR
GSI_CANDIDATES_JSON = _GSI_CANDIDATES_JSON


def configure_data_dir(data_dir: str | None) -> None:
    global S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_TRAIN, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR, GSI_CANDIDATES_JSON

    if not data_dir:
        return

    processed = pathlib.Path(data_dir)
    PROCESSED_DIR    = processed
    S2_TRAIN_DIR     = processed / "s2" / "2024"
    S2_PROCESSED_DIR = S2_TRAIN_DIR
    CDL_TRAIN        = processed / "cdl" / "cdl_2024_study_area_filtered.tif"
    CDL_BY_YEAR      = {"2024": CDL_TRAIN}
    GSI_CANDIDATES_JSON = processed / "s2" / "2024" / "gsi_candidates.json"
    log.info(f"Data dir overridden to {processed}")


def _glob_s2_train() -> list[str]:
    """Glob S2 files from flat train/ dir, then drop low-validity dates
    (same filter as the training pipeline → standalone selection stays consistent)."""
    from crop_mapping_pipeline.stages.valid_dates import filter_valid_s2_dates
    files = sorted(glob(str(S2_TRAIN_DIR / "*_processed.tif")) + glob(str(S2_TRAIN_DIR / "S2H_*.tif")))
    seen  = set()
    files = [p for p in files if not (p in seen or seen.add(p))]
    valid, _ = filter_valid_s2_dates(files, cache_dir=S2_TRAIN_DIR)
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
    """Training data from flat train/ dir."""
    s2_files = _glob_s2_train()
    assert s2_files, f"No S2 files in {S2_TRAIN_DIR}"
    cdl_path = str(CDL_TRAIN)
    assert os.path.exists(cdl_path), f"CDL not found: {cdl_path}"
    return TRAIN_YEARS[0], s2_files, cdl_path


def get_stage1_inputs() -> list[tuple[str, list[str], str]]:
    """Training data for GSI band scoring — flat train/ dir.
    Returns [(year, s2_files, cdl_path)].
    """
    s2_files = _glob_s2_train()
    cdl_path = str(CDL_TRAIN)
    if not s2_files:
        raise FileNotFoundError(f"No S2 files in {S2_TRAIN_DIR}")
    if not os.path.exists(cdl_path):
        raise FileNotFoundError(f"CDL not found: {cdl_path}")
    return [(TRAIN_YEARS[0], s2_files, cdl_path)]


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


def load_gsi_candidates() -> tuple[dict, dict, list]:
    if not os.path.exists(GSI_CANDIDATES_JSON):
        raise FileNotFoundError(
            f"GSI candidates not found: {GSI_CANDIDATES_JSON}\n"
            "Run band scoring first:  python stages/band_scoring.py"
        )
    with open(GSI_CANDIDATES_JSON) as f:
        payload = json.load(f)
    log.info(f"Loaded GSI candidates from {GSI_CANDIDATES_JSON}")
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


_DIRECT_OUTPUT_MAP = {
    "gsi_direct": (SELECT_GSI_DIRECT_JSON, SELECT_GSI_DIRECT_BANDS),
    "rf_direct":  (SELECT_RF_DIRECT_JSON,  SELECT_RF_DIRECT_BANDS),
}


def main(force: bool = False, data_dir: str = None, output_dir: str = None,
         mode: str = "gsi", selector: str = "gsi_direct",
         top_k_values: list[int] | None = None) -> None:
    global _MLFLOW_EXPERIMENT_OVERRIDE, KEEP_CLASSES, CDL_CLASS_NAMES
    _MLFLOW_EXPERIMENT_OVERRIDE = None

    configure_data_dir(data_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "gsi":
        from crop_mapping_pipeline.stages.selections.band_scoring.gsi import run_gsi_scoring
        gsi_path = GSI_CANDIDATES_JSON if not data_dir else Path(data_dir) / "s2" / "2022" / "gsi_candidates.json"
        if not force and gsi_path.exists():
            log.info(f"GSI candidates already exist: {gsi_path}  (--force to re-run)")
        else:
            years_data = get_stage1_inputs()
            run_gsi_scoring(years_data, data_dir=data_dir)
            log.info("GSI scoring complete.")
        return

    if mode == "select":
        from crop_mapping_pipeline.stages.selections.gsi_direct import run_gsi_direct
        from crop_mapping_pipeline.stages.selections.rf_direct import run_rf_direct
        if selector not in _DIRECT_OUTPUT_MAP:
            raise ValueError(
                f"--selector must be 'gsi_direct' or 'rf_direct', got {selector!r}"
            )
        ks = top_k_values or [SELECT_TOP_K_PER_CROP]
        fn = run_gsi_direct if selector == "gsi_direct" else run_rf_direct
        years_data = get_stage1_inputs()
        out_base = Path(output_dir) if output_dir else (Path(data_dir) if data_dir else _DIRECT_OUTPUT_MAP[selector][0].parent)
        out_base.mkdir(parents=True, exist_ok=True)
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

    log.info("Band scoring complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Band scoring: GSI and RF importance for band selection comparison")
    parser.add_argument(
        "--mode",
        choices=["gsi", "select"],
        default="gsi",
        help="'gsi' runs per-crop GSI scoring; 'select' runs joint spectral-temporal direct selection.",
    )
    parser.add_argument(
        "--selector",
        choices=["gsi_direct", "rf_direct"],
        default="gsi_direct",
        help="Direct selector for --mode select.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--top-k", type=int, nargs="+", default=None, metavar="K",
                        help="Top-K per crop sweep (e.g. --top-k 5 10 15 20 30)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override processed data directory (S2/CDL input)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for selection output JSONs (--mode select only)")
    return parser


def configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / f"band_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging()
    main(force=args.force, data_dir=args.data_dir, output_dir=args.output_dir,
         mode=args.mode, selector=args.selector, top_k_values=args.top_k)


if __name__ == "__main__":
    cli()
