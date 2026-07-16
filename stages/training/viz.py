"""Training/eval plots: confusion matrix, segmentation maps, per-patch panels,
RGB median composites, and the training curve.
"""

import os
import logging
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio
import mlflow

from config import (
    NUM_CLASSES, KEEP_CLASSES, CDL_CLASS_NAMES,
    S2_BAND_NAMES, S2_NODATA, PATCH_SIZE,
)
from utils.constants import USDA_CDL_COLORS
from stages.training import run_state

log = logging.getLogger(__name__)

# Derived from KEEP_CLASSES (config.py) — stays in sync if the class set changes,
# unlike a hardcoded list which silently desyncs (IndexError once len < NUM_CLASSES).
CROP_COLORS  = ["#000000"] + [USDA_CDL_COLORS[c] for c in KEEP_CLASSES]
CLASS_LABELS = ["Background"] + [CDL_CLASS_NAMES[c] for c in KEEP_CLASSES]
SEG_CMAP     = ListedColormap(CROP_COLORS)
SEG_NORM     = BoundaryNorm(boundaries=range(NUM_CLASSES + 1), ncolors=NUM_CLASSES)


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


# ── Training curve ────────────────────────────────────────────────────────────

def save_training_curve(hist_df, best_miou, exp_name, save_path):
    """Two-panel training curve: loss (train/val) + val mIoU/mF1 with best-mIoU line."""
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
    plt.savefig(save_path, dpi=150)
    plt.close()


# ── Per-patch test visualizations ─────────────────────────────────────────────

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
    _cache_path = _cache_dir / f"rgb_median_patches_seed{run_state.SEED}_n{n_patches}.npy"

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

            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.6))

            axes[0].imshow(rgb)
            axes[0].set_title("Median Composite\n(B4/B3/B2, 2024)", fontsize=20, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gt,    cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
            axes[1].set_title("Ground Truth",    fontsize=20, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(pred,  cmap=SEG_CMAP, norm=SEG_NORM, interpolation="nearest")
            axes[2].set_title("Prediction",      fontsize=20, fontweight="bold")
            axes[2].axis("off")

            axes[3].imshow(error, cmap=error_cmap, norm=error_norm, interpolation="nearest")
            axes[3].set_title("Correct / Incorrect", fontsize=20, fontweight="bold")
            axes[3].axis("off")

            # tight panel spacing + title close above panels, legend below in one line
            fig.subplots_adjust(left=0.005, right=0.995, top=0.86, bottom=0.14, wspace=0.005)
            fig.legend(handles=crop_legend + error_legend, loc="lower center",
                       ncol=len(crop_legend) + len(error_legend), fontsize=18,
                       columnspacing=0.8, handletextpad=0.35,
                       bbox_to_anchor=(0.5, 0.0), frameon=True)
            fig.suptitle(f"Test Patch {patch_idx:04d}", fontsize=26, fontweight="bold", y=0.97)
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
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8.5))

    panel = 0
    if rgb_img is not None:
        axes[panel].imshow(rgb_img)
        axes[panel].set_title("Median Composite\n(B4/B3/B2, 2024)", fontsize=22, fontweight="bold")
        axes[panel].axis("off")
        panel += 1

    axes[panel].imshow(gt_ds,   cmap=SEG_CMAP,   norm=SEG_NORM,   interpolation="nearest")
    axes[panel].set_title("Ground Truth (CDL)", fontsize=22, fontweight="bold")
    axes[panel].axis("off")
    panel += 1
    axes[panel].imshow(pred_ds, cmap=SEG_CMAP,   norm=SEG_NORM,   interpolation="nearest")
    axes[panel].set_title("Prediction",         fontsize=22, fontweight="bold")
    axes[panel].axis("off")
    panel += 1
    axes[panel].imshow(error,   cmap=error_cmap, norm=error_norm, interpolation="nearest")
    axes[panel].set_title("Correct / Incorrect", fontsize=22, fontweight="bold")
    axes[panel].axis("off")

    crop_patches = [mpatches.Patch(color=CROP_COLORS[i], label=CLASS_LABELS[i])
                    for i in range(1, NUM_CLASSES)]
    error_patches = [
        mpatches.Patch(color="#22cc44", label="Correct"),
        mpatches.Patch(color="#ee2222", label="Incorrect"),
        mpatches.Patch(color="#d0d0d0", label="Background"),
    ]
    # no figure title; tight spacing, legend below in one line
    fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.12, wspace=0.03)
    fig.legend(handles=crop_patches + error_patches, loc="lower center",
               ncol=len(crop_patches) + len(error_patches), fontsize=18,
               columnspacing=0.8, handletextpad=0.35,
               bbox_to_anchor=(0.5, 0.0), frameon=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {save_path}")
