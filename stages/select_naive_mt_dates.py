#!/usr/bin/env python3
"""Multi-temporal date selection — max-NDVI per calendar quarter.

For each quarter (Q1: Jan–Mar, Q2: Apr–Jun, Q3: Jul–Sep, Q4: Oct–Dec),
picks the acquisition date with the highest mean NDVI over crop pixels.
Caches result as phenol_dates.json alongside the S2 data.

Usage:
    python stages/select_naive_mt_dates.py
    python stages/select_naive_mt_dates.py --force          # ignore cache, recompute
    python stages/select_naive_mt_dates.py --s2-dir DIR --cdl PATH
"""

import argparse
import glob as _glob
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent          # → crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))         # parent on path for package import

from crop_mapping_pipeline.config import S2_TRAIN_DIR, CDL_TRAIN
from crop_mapping_pipeline.stages.experiments.base import build_local_band_map
from crop_mapping_pipeline.stages.experiments.exp_b import _select_phenol_dates

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description="Multi-temporal calendar date selection")
    ap.add_argument("--s2-dir", default=str(S2_TRAIN_DIR), help="dir of *_processed.tif S2 files")
    ap.add_argument("--cdl", default=str(CDL_TRAIN), help="CDL label raster path")
    ap.add_argument("--force", action="store_true", help="ignore phenol_dates.json cache, recompute")
    args = ap.parse_args()

    s2_dir = Path(args.s2_dir)
    s2_paths = sorted(
        p for p in (_glob.glob(str(s2_dir / "*_processed.tif")) + _glob.glob(str(s2_dir / "S2H_*.tif")))
        if not Path(p).name.startswith("._")
    )
    if not s2_paths:
        raise FileNotFoundError(f"No processed S2 files in {s2_dir}")
    log.info(f"S2 files: {len(s2_paths)} in {s2_dir}")
    log.info(f"CDL: {args.cdl}")

    if args.force:
        from crop_mapping_pipeline.config import PROCESSED_DIR
        cache = PROCESSED_DIR / "phenol_dates.json"
        if cache.exists():
            cache.unlink()
            log.info(f"Removed cache → {cache}")

    (_names, _b2i, local_date_to_idx, _m2d) = build_local_band_map(s2_paths)

    phenol_map = _select_phenol_dates(local_date_to_idx, s2_paths=s2_paths, cdl_path=args.cdl)

    print("\nCalendar-MT dates:")
    for q, d in phenol_map.items():
        print(f"  {q}: {d}")


if __name__ == "__main__":
    main()
