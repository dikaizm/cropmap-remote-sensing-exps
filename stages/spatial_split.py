"""
Spatial block (grid) train/val/test split — standalone, no torch/mlflow/geoai deps.

The split depends ONLY on the CDL label raster + patch geometry (patch_size, stride,
keep_classes, min_valid_frac) — NOT on Sentinel-2 band selection. It is therefore
identical across every Stage-3 experiment (Exp A/B/C, any architecture), and can be
computed and inspected independently of a training run.

Patches are grouped into BLOCK_SIZE×BLOCK_SIZE px blocks by pixel origin (r, c); each
block is assigned wholly to one split via class-balanced greedy stratification, so no
train patch is ever spatially adjacent to a val/test patch within a block (kills
patch-adjacency spatial leakage).

CLI:
    python stages/spatial_split.py                       # uses config defaults
    python stages/spatial_split.py --year 2024 --block-size 768
    python stages/spatial_split.py --cdl path/to/cdl.tif --out /tmp/split
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

# Split-name → short matrix marker
_SPLIT_CODE = {"train": "TR", "val": "VL", "test": "TS"}


# ── Lightweight patch index (CDL-only; duck-types RasterPatchDataset for the split) ──
class CDLPatchIndex:
    """
    Minimal stand-in for RasterPatchDataset exposing exactly the attributes the
    block splitter needs: ``patches`` (list of (r,c)), ``_cdl`` (full label array),
    ``patch_size``, ``_remap_lut``. Built from the CDL raster alone — no S2 reads.

    The valid-patch construction MUST match RasterPatchDataset.__init__ exactly so the
    standalone split is byte-identical to the split used during training.
    """

    def __init__(self, cdl_path, patch_size, stride, keep_classes,
                 remap_lut=None, min_valid_frac=0.1):
        self.patch_size = patch_size
        self._remap_lut = (remap_lut if remap_lut is not None
                           else np.arange(256, dtype=np.int64))
        with rasterio.open(cdl_path) as src:
            self._cdl = src.read(1).astype(np.int32)
            self.height = src.height
            self.width = src.width
        keep_set = list(set(keep_classes))
        ps = patch_size
        self.patches = [
            (r, c)
            for r in range(0, self.height - ps + 1, stride)
            for c in range(0, self.width - ps + 1, stride)
            if np.isin(self._cdl[r:r + ps, c:c + ps], keep_set).mean() >= min_valid_frac
        ]

    def __len__(self):
        return len(self.patches)


# ── Core split ──────────────────────────────────────────────────────────────────
def _block_spatial_split(datasets_raw, block_size, val_frac, test_frac,
                         num_classes, seed, min_class_frac=0.05, log=None):
    """
    Spatial block split that prevents patch-adjacency leakage.

    Patches are grouped into ``block_size``×``block_size`` px blocks by their pixel
    origin ``(r, c)``. Every patch in a block is assigned to the SAME split, so no
    train patch is spatially adjacent to a val/test patch within a block.

    Blocks are assigned to train/val/test by a deterministic greedy stratifier that
    balances per-class foreground pixel mass toward each split's target fraction,
    then a repair pass enforces a per-class pixel FLOOR: every split must hold at
    least ``min_class_frac`` of each crop's total pixels. This stops a crop from
    appearing in val/test only as a token sliver (meaningless for evaluation) — a
    real risk with large blocks / rare crops. If a crop is too rare to reach the
    floor in every split, the repair does its best and logs a ⚠ warning.

    Args:
        datasets_raw: list of objects exposing ``.patches`` (list of (r,c)), ``._cdl``
            (full label array), ``.patch_size``, ``._remap_lut`` — in ConcatDataset order
            (RasterPatchDataset during training, CDLPatchIndex standalone).
        block_size: block side length in pixels.
        val_frac, test_frac: target fractions (train = 1 - val - test).
        num_classes: total classes incl. background (foreground = 1..num_classes-1).
        seed: deterministic tie-break.
        min_class_frac: per-split, per-crop minimum pixel fraction (of that crop's
            total) the repair pass enforces. 0 disables the floor (presence-only).
        log: optional logger.

    Returns:
        (train_idx, val_idx, test_idx, info) — global ConcatDataset index lists and
        a summary dict (grid shape, per-split block/patch counts, per-class pixel
        fractions, per-block assignment).
    """
    n_fg = num_classes - 1
    block_indices = {}                       # block_key -> list[int] global idx
    block_hist = {}                          # block_key -> np.int64[n_fg]
    offset = 0
    for ds in datasets_raw:
        lut, cdl, ps = ds._remap_lut, ds._cdl, ds.patch_size
        for local_i, (r, c) in enumerate(ds.patches):
            bk = (r // block_size, c // block_size)
            block_indices.setdefault(bk, []).append(offset + local_i)
            lab = lut[np.clip(cdl[r:r + ps, c:c + ps], 0, 255)]
            counts = np.bincount(lab.ravel(), minlength=num_classes)[1:]  # drop bg
            block_hist[bk] = block_hist.get(bk, np.zeros(n_fg, np.int64)) + counts
        offset += len(ds.patches)

    blocks = list(block_indices.keys())
    total = np.sum([block_hist[b] for b in blocks], axis=0).astype(float)  # [n_fg]
    total_safe = np.maximum(total, 1.0)

    # Process rarest-content blocks first (inverse class frequency weighting) so
    # scarce classes get placed before common ones dominate the greedy choice.
    inv_w = 1.0 / total_safe
    blocks.sort(key=lambda b: (-float((block_hist[b] * inv_w).sum()), b))

    fracs = {"train": 1.0 - val_frac - test_frac, "val": val_frac, "test": test_frac}
    fracs = {k: v for k, v in fracs.items() if v > 0}
    target = {s: f * total for s, f in fracs.items()}
    current = {s: np.zeros(n_fg) for s in fracs}
    assign = {}
    for b in blocks:
        h = block_hist[b].astype(float)
        best, best_score = None, -np.inf
        for s in fracs:                       # fill split most deficient in this block's classes
            deficit = np.maximum(target[s] - current[s], 0.0) / total_safe
            score = float((deficit * (h / total_safe)).sum())
            if score > best_score:
                best, best_score = s, score
        assign[b] = best
        current[best] += h

    # Repair: enforce a per-class pixel FLOOR — every split must hold at least
    # `min_class_frac` of each crop's total pixels (so no split gets a crop only as
    # a token sliver). floor[k] = min_class_frac * total[k].
    splits = list(fracs)
    floor = min_class_frac * total                                  # [n_fg]
    for _ in range(n_fg * len(splits) * 2):    # bounded passes
        moved = False
        for s in splits:
            for k in range(n_fg):
                if current[s][k] >= floor[k]:                       # split already meets floor
                    continue
                # Find a donor block rich in class k whose source split keeps its
                # OWN floor after donating (don't rob Peter to pay Paul). Prefer the
                # block contributing the most of k.
                donor = None
                for b in sorted(blocks, key=lambda x: -block_hist[x][k]):
                    if block_hist[b][k] <= 0 or assign[b] == s:
                        continue
                    src = assign[b]
                    if current[src][k] - block_hist[b][k] >= floor[k]:   # src keeps its floor
                        donor = b
                        break
                if donor is not None:
                    src = assign[donor]
                    current[src] -= block_hist[donor].astype(float)
                    current[s] += block_hist[donor].astype(float)
                    assign[donor] = s
                    moved = True
        if not moved:
            break

    out = {s: [] for s in splits}
    for b in blocks:
        out[assign[b]].extend(block_indices[b])
    for s in splits:
        out[s].sort()

    H, W = datasets_raw[0]._cdl.shape
    grid_rows = -(-H // block_size)   # ceil
    grid_cols = -(-W // block_size)
    blocks_meta = [
        {
            "block_row": int(br),
            "block_col": int(bc),
            "split": assign[(br, bc)],
            "n_patches": len(block_indices[(br, bc)]),
            "class_px": block_hist[(br, bc)].astype(int).tolist(),
        }
        for (br, bc) in sorted(blocks)
    ]

    # Per-split per-class pixel counts (absolute) + below-floor flags
    class_px = {s: current[s].astype(int).tolist() for s in splits}
    below_floor = {
        s: [k + 1 for k in range(n_fg) if current[s][k] < floor[k]] for s in splits
    }

    info = {
        "block_size": block_size,
        "grid_shape": [int(grid_rows), int(grid_cols)],
        "n_blocks": len(blocks),
        "val_frac": val_frac,
        "test_frac": test_frac,
        "min_class_frac": min_class_frac,
        "blocks_per_split": {s: sum(1 for b in blocks if assign[b] == s) for s in splits},
        "patches_per_split": {s: len(out[s]) for s in splits},
        "class_pixel_frac": {
            s: (current[s] / total_safe).round(4).tolist() for s in splits
        },
        "class_pixel_count": class_px,
        "below_floor_classes": below_floor,
        "blocks": blocks_meta,
    }
    if log is not None:
        log.info(f"  Block split (size={block_size}px, floor={min_class_frac:.0%}/class): "
                 f"{len(blocks)} valid blocks → "
                 + " / ".join(f"{info['blocks_per_split'][s]} {s}" for s in splits))
        for s in splits:
            bf = below_floor[s]
            warn = f"  ⚠ below {min_class_frac:.0%} floor: classes {bf}" if bf else ""
            log.info(f"    {s:5s}: {info['patches_per_split'][s]:>5d} patches, "
                     f"class-px frac={np.round(current[s] / total_safe, 3).tolist()}{warn}")

    test_idx = out.get("test", [])
    return out["train"], out["val"], test_idx, info


# ── Artifact writers ──────────────────────────────────────────────────────────
_SPLIT_COLOR = {"train": "#1b9e77", "val": "#7570b3", "test": "#d95f02"}


def _plot_class_distribution(info, out_path, exp_name, class_names=None):
    """
    Bar charts of crop-class distribution across the split and across blocks. 3 panels:
      (a) per-crop pixel FRACTION by split (train/val/test) — balance check vs 70/15/15;
      (b) per-crop absolute pixel COUNT by split (log y) — shows class imbalance;
      (c) per-block stacked class composition, blocks ordered & grouped by split.
    """
    n_fg = len(info["class_pixel_count"]["train"])
    names = class_names or [f"C{k+1}" for k in range(n_fg)]
    splits = [s for s in ("train", "val", "test") if s in info["class_pixel_count"]]
    counts = {s: np.array(info["class_pixel_count"][s], float) for s in splits}
    totals = np.sum([counts[s] for s in splits], axis=0)
    totals_safe = np.maximum(totals, 1.0)
    fracs = {s: counts[s] / totals_safe for s in splits}

    x = np.arange(n_fg)
    w = 0.8 / len(splits)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # (a) per-crop fraction by split
    ax = axes[0]
    for i, s in enumerate(splits):
        ax.bar(x + i * w - 0.4 + w / 2, fracs[s], w, label=s, color=_SPLIT_COLOR[s])
    if "val_frac" in info:                               # target ref lines
        ax.axhline(info["val_frac"], ls=":", c="#7570b3", lw=1, alpha=0.6)
        ax.axhline(info["test_frac"], ls=":", c="#d95f02", lw=1, alpha=0.6)
    ax.axhline(info["min_class_frac"], ls="--", c="k", lw=1, alpha=0.5,
               label=f"floor {info['min_class_frac']:.0%}")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("pixel fraction of crop"); ax.set_title("(a) Per-crop split fraction")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (b) per-crop absolute pixel count by split (log)
    ax = axes[1]
    for i, s in enumerate(splits):
        ax.bar(x + i * w - 0.4 + w / 2, counts[s], w, label=s, color=_SPLIT_COLOR[s])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("pixels (log)"); ax.set_title("(b) Per-crop pixel count by split")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # (c) per-block stacked class composition, grouped by split
    ax = axes[2]
    blocks = sorted(info["blocks"], key=lambda b: (splits.index(b["split"]),
                                                   -sum(b["class_px"])))
    cmap = plt.get_cmap("tab10")
    bottoms = np.zeros(len(blocks))
    px = np.array([b["class_px"] for b in blocks], float)
    for k in range(n_fg):
        ax.bar(range(len(blocks)), px[:, k], bottom=bottoms, width=1.0,
               color=cmap(k % 10), label=names[k], edgecolor="white", linewidth=0.2)
        bottoms += px[:, k]
    # split group separators + labels (boxed, above bars; legend moved outside)
    ymax = bottoms.max() if len(bottoms) else 1.0
    start = 0
    for s in splits:
        cnt = sum(1 for b in blocks if b["split"] == s)
        if start > 0:
            ax.axvline(start - 0.5, color="k", lw=1.2, alpha=0.6)
        ax.text(start + cnt / 2 - 0.5, ymax * 1.02, f"{s.upper()} ({cnt})",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", fc=_SPLIT_COLOR[s], ec="none"))
        start += cnt
    ax.set_xlim(-0.5, len(blocks) - 0.5)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_xlabel("block (grouped by split, sorted by size)")
    ax.set_ylabel("pixels"); ax.set_title("(c) Per-block class composition")
    ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="crop")

    fig.suptitle(f"Class distribution — {exp_name}  "
                 f"(block={info['block_size']}px, {info['n_blocks']} blocks)",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_block_split_artifacts(info, out_dir, exp_name, class_names=None, log=None):
    """
    Persist the spatial block split as CSV (matrix of TR/VL/TS markers), JSON
    (full split metadata), PNG (color-coded block grid), and a class-distribution
    bar chart (per-split + per-block).

    Returns dict of written paths: {"csv":..., "json":..., "png":..., "dist":...}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gr, gc = info["grid_shape"]
    bs = info["block_size"]

    # ── Matrix of markers (rows = block_row, cols = block_col) ────────────────
    grid = np.full((gr, gc), "", dtype=object)          # "" = no valid patches
    for b in info["blocks"]:
        grid[b["block_row"], b["block_col"]] = _SPLIT_CODE.get(b["split"], "?")
    df = pd.DataFrame(
        grid,
        index=[f"r{r}" for r in range(gr)],
        columns=[f"c{c}" for c in range(gc)],
    )
    csv_path = out_dir / "split_blocks_matrix.csv"
    df.to_csv(csv_path)

    # ── JSON (full metadata) ──────────────────────────────────────────────────
    json_path = out_dir / "split_blocks.json"
    with open(json_path, "w") as f:
        json.dump(info, f, indent=2)

    # ── PNG (color-coded grid) ────────────────────────────────────────────────
    code_to_int = {"": 0, "TR": 1, "VL": 2, "TS": 3}
    int_grid = np.vectorize(lambda v: code_to_int.get(v, 0))(grid).astype(int)
    cmap = ListedColormap(["#e8e8e8", "#1b9e77", "#7570b3", "#d95f02"])  # none/TR/VL/TS
    fig, ax = plt.subplots(figsize=(max(6, gc * 0.6), max(5, gr * 0.6)))
    ax.imshow(int_grid, cmap=cmap, vmin=0, vmax=3, aspect="equal")
    for r in range(gr):
        for c in range(gc):
            txt = grid[r, c]
            if txt:
                ax.text(c, r, txt, ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
    ax.set_xticks(range(gc)); ax.set_xticklabels([f"c{c}" for c in range(gc)], fontsize=7)
    ax.set_yticks(range(gr)); ax.set_yticklabels([f"r{r}" for r in range(gr)], fontsize=7)
    ax.set_xticks(np.arange(-.5, gc, 1), minor=True)
    ax.set_yticks(np.arange(-.5, gr, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    pps = info["patches_per_split"]; bpsplit = info["blocks_per_split"]
    legend = [
        mpatches.Patch(color="#1b9e77", label=f"TR train ({bpsplit.get('train',0)} blk / {pps.get('train',0)} patches)"),
        mpatches.Patch(color="#7570b3", label=f"VL val ({bpsplit.get('val',0)} blk / {pps.get('val',0)})"),
        mpatches.Patch(color="#d95f02", label=f"TS test ({bpsplit.get('test',0)} blk / {pps.get('test',0)})"),
        mpatches.Patch(color="#e8e8e8", label="— no valid patches"),
    ]
    ax.legend(handles=legend, bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=8, frameon=False)
    ax.set_title(f"Spatial block split — {exp_name}\nblock={bs}px, {info['n_blocks']} valid blocks")
    plt.tight_layout()
    png_path = out_dir / "split_blocks_map.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Class-distribution bar chart (per-split + per-block) ──────────────────
    dist_path = out_dir / "split_class_distribution.png"
    _plot_class_distribution(info, dist_path, exp_name, class_names=class_names)

    if log is not None:
        log.info(f"  Saved split artifacts → {csv_path.name}, {json_path.name}, "
                 f"{png_path.name}, {dist_path.name}")
    return {"csv": csv_path, "json": json_path, "png": png_path, "dist": dist_path}


# ── Standalone convenience runner ───────────────────────────────────────────────
def compute_split_from_cdl(cdl_paths, patch_size, stride, keep_classes, remap_lut,
                           min_valid_frac, num_classes, block_size, val_frac,
                           test_frac, seed, min_class_frac=0.05, log=None):
    """Build CDL patch indices for each year and run the block split."""
    idx = [
        CDLPatchIndex(p, patch_size, stride, keep_classes, remap_lut, min_valid_frac)
        for p in cdl_paths
    ]
    if log is not None:
        for p, ix in zip(cdl_paths, idx):
            log.info(f"  {Path(p).name}: {len(ix)} valid patches")
    return _block_spatial_split(idx, block_size, val_frac, test_frac,
                                num_classes, seed, min_class_frac=min_class_frac, log=log)


def _main():
    _ROOT = Path(__file__).parent.parent          # crop_mapping_pipeline/
    sys.path.insert(0, str(_ROOT.parent))
    from crop_mapping_pipeline.config import (
        PATCH_SIZE, STRIDE, MIN_VALID_FRAC, KEEP_CLASSES, REMAP_LUT, NUM_CLASSES,
        VAL_FRAC, TEST_FRAC, SEED, BLOCK_SIZE, MIN_CLASS_FRAC, CDL_BY_YEAR, CDL_TRAIN,
        TRAIN_YEARS, PROCESSED_DIR, CDL_CLASS_NAMES,
    )
    class_names = [CDL_CLASS_NAMES.get(c, f"CDL{c}") for c in KEEP_CLASSES]

    ap = argparse.ArgumentParser(description="Compute spatial block split (standalone)")
    ap.add_argument("--year", nargs="*", default=None,
                    help=f"Train year(s) to build the split from (default: {TRAIN_YEARS})")
    ap.add_argument("--cdl", nargs="*", default=None,
                    help="Explicit CDL raster path(s) (overrides --year)")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--min-class-frac", type=float, default=MIN_CLASS_FRAC,
                    help="Per-split per-crop min pixel fraction floor (0 disables)")
    ap.add_argument("--out", default=str(PROCESSED_DIR / "spatial_split"),
                    help="Output directory for CSV/JSON/PNG")
    ap.add_argument("--name", default="spatial_block_split",
                    help="Label used in PNG title / output")
    args = ap.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("spatial_split")

    if args.cdl:
        cdl_paths = args.cdl
    else:
        years = args.year or TRAIN_YEARS
        cdl_paths = [str(CDL_BY_YEAR.get(str(y), CDL_TRAIN)) for y in years]
    missing = [p for p in cdl_paths if not Path(p).exists()]
    if missing:
        log.error(f"CDL file(s) not found: {missing}")
        sys.exit(1)

    log.info(f"Spatial block split  block={args.block_size}px  "
             f"ratio={1-args.val_frac-args.test_frac:.2f}/{args.val_frac:.2f}/{args.test_frac:.2f}  "
             f"seed={args.seed}  floor={args.min_class_frac:.0%}/class")
    log.info(f"CDL: {cdl_paths}")
    _, _, _, info = compute_split_from_cdl(
        cdl_paths, PATCH_SIZE, STRIDE, KEEP_CLASSES, REMAP_LUT, MIN_VALID_FRAC,
        NUM_CLASSES, args.block_size, args.val_frac, args.test_frac, args.seed,
        min_class_frac=args.min_class_frac, log=log,
    )
    paths = _save_block_split_artifacts(info, args.out, args.name,
                                        class_names=class_names, log=log)
    log.info(f"\nDone. Artifacts in {args.out}:")
    for k, p in paths.items():
        log.info(f"  {k}: {p}")


if __name__ == "__main__":
    _main()
