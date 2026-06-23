"""
Generate train/val/test patch-location map for thesis figures.

Usage:
    python stages/visualize_split.py
    python stages/visualize_split.py --out figures/split_map.png --downsample 6

Produces:
  - Split map: CDL background with colored patch rectangles
    (blue=train, green=val, red=test)
  - Split bar chart: patch counts per class per split
"""

import argparse
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import rasterio
import torch
from torch.utils.data import random_split, ConcatDataset

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_TRAIN_DIR, CDL_TRAIN,
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC,
    KEEP_CLASSES, CLASS_REMAP, NUM_CLASSES, CDL_CLASS_NAMES,
    REMAP_LUT, S2_NODATA, SEED,
    VAL_FRAC, TEST_FRAC,
    FIGURES_DIR,
)
from geoai.geoai.train import RasterPatchDataset

SPLIT_COLORS = {
    "train": "#2166ac",   # blue
    "val":   "#4dac26",   # green
    "test":  "#d01c8b",   # magenta
}
SPLIT_ALPHA = 0.55


def _build_cdl_rgb(cdl_path, downsample):
    """Render CDL as RGB using a simple colormap."""
    COLORS = {
        0:  (210, 210, 210),   # background / grey
        3:  ( 70, 163, 213),   # rice / blue
        24: (209, 187, 130),   # winter wheat / tan
        36: (147, 204,  57),   # alfalfa / green
        54: (204,   5,   5),   # tomatoes / red
        75: (209, 187,  28),   # almonds / gold
        76: (148,  98,  50),   # walnuts / brown
    }
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
        transform = src.transform

    h, w = cdl.shape
    rgb = np.full((h, w, 3), 230, dtype=np.uint8)   # default light grey
    for cls_id, (r, g, b) in COLORS.items():
        mask = cdl == cls_id
        rgb[mask] = (r, g, b)

    rgb_ds = rgb[::downsample, ::downsample]
    return rgb_ds, transform, h, w


def plot_split_map(s2_paths, cdl_path, out_path, downsample=6):
    s2_paths = sorted(s2_paths)
    ds = RasterPatchDataset(
        s2_paths=s2_paths, cdl_path=str(cdl_path),
        patch_size=PATCH_SIZE, stride=STRIDE,
        keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
        min_valid_frac=MIN_VALID_FRAC,
    )
    n_total = len(ds)
    n_val   = max(1, int(VAL_FRAC  * n_total))
    n_test  = max(1, int(TEST_FRAC * n_total))
    n_train = n_total - n_val - n_test

    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)

    split_idx = {}
    split_idx["train"] = set(train_ds.indices)
    split_idx["val"]   = set(val_ds.indices)
    split_idx["test"]  = set(test_ds.indices)

    patches = ds.patches   # list of (row, col)

    rgb_ds, transform, H, W = _build_cdl_rgb(cdl_path, downsample)

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(rgb_ds, origin="upper")
    ax.set_title("Train / Val / Test Patch Distribution", fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")

    ps_ds = PATCH_SIZE / downsample   # patch size in display pixels

    for split_name, idxs in split_idx.items():
        color = SPLIT_COLORS[split_name]
        for i in idxs:
            r, c = patches[i]
            x = c / downsample
            y = r / downsample
            rect = mpatches.FancyBboxPatch(
                (x, y), ps_ds, ps_ds,
                boxstyle="square,pad=0",
                linewidth=0,
                facecolor=color,
                alpha=SPLIT_ALPHA,
            )
            ax.add_patch(rect)

    legend_handles = [
        mpatches.Patch(color=SPLIT_COLORS["train"], alpha=0.8,
                       label=f"Train  ({n_train:,} patches, {n_train/n_total*100:.0f}%)"),
        mpatches.Patch(color=SPLIT_COLORS["val"],   alpha=0.8,
                       label=f"Val    ({n_val:,} patches, {n_val/n_total*100:.0f}%)"),
        mpatches.Patch(color=SPLIT_COLORS["test"],  alpha=0.8,
                       label=f"Test   ({n_test:,} patches, {n_test/n_total*100:.0f}%)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10,
              framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved split map → {out_path}")
    return n_train, n_val, n_test, patches, split_idx, ds


def plot_class_distribution(ds, split_idx, patches, out_path):
    """Bar chart: patch count per class per split."""
    from collections import defaultdict

    split_class_counts = {s: defaultdict(int) for s in split_idx}

    for split_name, idxs in split_idx.items():
        for i in idxs:
            r, c = patches[i]
            patch_cdl = ds._cdl[r:r + PATCH_SIZE, c:c + PATCH_SIZE]
            remapped  = REMAP_LUT[np.clip(patch_cdl, 0, 255)]
            for cls_id in range(1, NUM_CLASSES):
                if (remapped == cls_id).any():
                    split_class_counts[split_name][cls_id] += 1

    class_names = [CDL_CLASS_NAMES[k] for k in KEEP_CLASSES]
    n_cls = len(KEEP_CLASSES)
    x = np.arange(n_cls)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, split_name in zip([-width, 0, width], ["train", "val", "test"]):
        counts = [split_class_counts[split_name].get(i + 1, 0) for i in range(n_cls)]
        ax.bar(x + offset, counts, width,
               label=split_name.capitalize(),
               color=SPLIT_COLORS[split_name], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Patches containing class", fontsize=11)
    ax.set_title("Class Distribution Across Train / Val / Test Splits", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved class distribution → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",        default=None,  help="Output path for split map PNG")
    parser.add_argument("--downsample", type=int, default=6, help="Display downsample factor")
    parser.add_argument("--data-dir",   default=None)
    args = parser.parse_args()

    from glob import glob
    s2_dir = Path(args.data_dir) / "s2" / "2024" if args.data_dir else S2_TRAIN_DIR
    cdl    = Path(args.data_dir) / "cdl" / "cdl_train.tif" if args.data_dir else CDL_TRAIN
    s2_paths = sorted(f for f in glob(str(s2_dir / "*.tif")) if not Path(f).name.startswith("._"))

    if not s2_paths:
        print(f"No S2 files found in {s2_dir}"); sys.exit(1)
    if not cdl.exists():
        print(f"CDL not found: {cdl}"); sys.exit(1)

    out_map  = Path(args.out) if args.out else FIGURES_DIR / "split_map.png"
    out_dist = out_map.parent / "split_class_distribution.png"

    n_train, n_val, n_test, patches, split_idx, ds = plot_split_map(
        s2_paths, cdl, out_map, downsample=args.downsample
    )
    print(f"  Total patches: {n_train+n_val+n_test:,}  "
          f"train={n_train:,} val={n_val:,} test={n_test:,}")

    plot_class_distribution(ds, split_idx, patches, out_dist)


if __name__ == "__main__":
    main()
