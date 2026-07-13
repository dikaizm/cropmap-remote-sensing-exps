"""Materialize the spatial block split to disk for demonstration / inspection.

Runs the deterministic block split from a CDL raster, then crops each block's
window from the CDL + all valid S2 dates into per-block folders:

    <out>/{train,val,test}/block_r{R}_c{C}/
        s2.tif    # (n_valid_dates × 10 bands) date-major stack, float32, nodata=-9999
        cdl.tif   # uint8 CDL crop (same grid/window)

Plus split artifacts (map PNG, matrix CSV/JSON) and blocks_manifest.json at <out>/.
Channel band descriptions are `{band}_{YYYYMMDD}` so they map directly to the
selection JSON union_channels used by notebook 06.

Usage:
    python stages/tools/materialize_spatial_split.py \\
        --s2-dir /Volumes/T7/research-crop-mapping-geoai/data/raw_v6/s2/2024 \\
        --cdl    /Volumes/T7/research-crop-mapping-geoai/data/raw_v6/cdl/cdl_train_reprojected.tif \\
        --out    /Volumes/T7/research-crop-mapping-geoai/data/spatial_split
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, transform as win_transform

_ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT.parent))
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

from cropmap_pipeline.config import (
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC, KEEP_CLASSES, REMAP_LUT, NUM_CLASSES,
    VAL_FRAC, TEST_FRAC, SEED, BLOCK_SIZE, MIN_CLASS_FRAC, CDL_CLASS_NAMES,
    S2_BAND_NAMES, S2_NODATA,
)
from cropmap_pipeline.stages.data.spatial_split import (
    compute_split_from_cdl, _save_block_split_artifacts,
)
from cropmap_pipeline.stages.data.valid_dates import filter_valid_s2_dates

log = logging.getLogger("materialize_split")


def _date(p):
    m = re.search(r"_(\d{4})_(\d{2})_(\d{2})\.tif$", os.path.basename(p))
    return "".join(m.groups()) if m else os.path.basename(p)[:8]


def main():
    ap = argparse.ArgumentParser(description="Materialize spatial block split to disk")
    ap.add_argument("--s2-dir", required=True, help="Directory of S2 date TIFs")
    ap.add_argument("--cdl", required=True, help="CDL raster on the S2 grid (reprojected)")
    ap.add_argument("--out", required=True, help="Output directory for split folders + artifacts")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--val-frac", type=float, default=VAL_FRAC)
    ap.add_argument("--test-frac", type=float, default=TEST_FRAC)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s2_dir, cdl_path, out = Path(args.s2_dir), Path(args.cdl), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # valid S2 dates (same filter as training/selection)
    all_s2 = sorted(str(p) for p in s2_dir.glob("S2H_*.tif") if not p.name.startswith("._"))
    valid_s2, dropped = filter_valid_s2_dates(all_s2, cache_dir=s2_dir)
    valid_s2 = sorted(valid_s2)
    dates = [_date(p) for p in valid_s2]
    band_desc = [f"{b}_{d}" for d in dates for b in S2_BAND_NAMES]
    n_ch = len(band_desc)
    log.info(f"{len(valid_s2)} valid S2 dates ({len(dropped)} dropped) → {n_ch} channels")

    tr, va, te, info = compute_split_from_cdl(
        [str(cdl_path)], PATCH_SIZE, STRIDE, KEEP_CLASSES, REMAP_LUT, MIN_VALID_FRAC,
        NUM_CLASSES, args.block_size, args.val_frac, args.test_frac, args.seed,
        min_class_frac=MIN_CLASS_FRAC, log=log,
    )
    class_names = [CDL_CLASS_NAMES.get(c, f"CDL{c}") for c in KEEP_CLASSES]
    _save_block_split_artifacts(info, str(out), "spatial_block_split", class_names=class_names, log=log)

    with rasterio.open(cdl_path) as src:
        H, W, ref_crs, ref_tf = src.height, src.width, src.crs, src.transform

    BS = args.block_size
    srcs = [rasterio.open(p) for p in valid_s2]
    manifest = {"block_size": BS, "grid_shape": info["grid_shape"], "crs": str(ref_crs),
                "n_channels": n_ch, "band_names": band_desc, "dates": dates,
                "blocks_per_split": info["blocks_per_split"], "blocks": []}
    t0 = time.time()
    try:
        for i, blk in enumerate(info["blocks"], 1):
            br, bc, split = blk["block_row"], blk["block_col"], blk["split"]
            r0, c0 = br * BS, bc * BS
            h, w = min(BS, H - r0), min(BS, W - c0)
            win = Window(c0, r0, w, h)
            wt = win_transform(win, ref_tf)
            bdir = out / split / f"block_r{br}_c{bc}"
            bdir.mkdir(parents=True, exist_ok=True)

            s2_prof = dict(driver="GTiff", dtype="float32", count=n_ch, height=h, width=w,
                           crs=ref_crs, transform=wt, nodata=S2_NODATA,
                           compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
            with rasterio.open(bdir / "s2.tif", "w", **s2_prof) as dst:
                ci = 1
                for s in srcs:
                    arr = s.read(window=win).astype(np.float32)
                    arr[(arr < 0) | ~np.isfinite(arr)] = S2_NODATA
                    for b in range(arr.shape[0]):
                        dst.write(arr[b], ci); dst.set_band_description(ci, band_desc[ci - 1]); ci += 1

            with rasterio.open(cdl_path) as csrc:
                cdl_crop = csrc.read(1, window=win)
            cdl_prof = dict(driver="GTiff", dtype="uint8", count=1, height=h, width=w,
                            crs=ref_crs, transform=wt, nodata=0, compress="lzw")
            with rasterio.open(bdir / "cdl.tif", "w", **cdl_prof) as dst:
                dst.write(cdl_crop, 1)

            manifest["blocks"].append({
                "block_row": br, "block_col": bc, "split": split, "n_patches": blk["n_patches"],
                "window": {"col_off": c0, "row_off": r0, "width": w, "height": h},
                "s2": str((bdir / "s2.tif").relative_to(out)),
                "cdl": str((bdir / "cdl.tif").relative_to(out))})
            log.info(f"  [{i}/{len(info['blocks'])}] {split:5s} block_r{br}_c{bc} ({w}x{h})")
    finally:
        for s in srcs:
            s.close()

    json.dump(manifest, open(out / "blocks_manifest.json", "w"), indent=2)
    ns = s2_dir / "norm_stats_percentile.npz"
    if ns.exists():
        shutil.copy2(ns, out / "norm_stats_percentile.npz")
    log.info(f"Done in {time.time()-t0:.0f}s → {out}  |  blocks: {info['blocks_per_split']}")


if __name__ == "__main__":
    main()
