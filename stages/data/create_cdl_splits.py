"""Create per-area CDL files for train / test_a / test_b.

Each area's CDL is reprojected and clipped to match that area's S2 grid,
then filtered to KEEP_CLASSES (8 crops, v6.1 >=1M-px selection).

Outputs:
  <processed>/cdl/cdl_train.tif
  <processed>/cdl/cdl_test_a.tif
  <processed>/cdl/cdl_test_b.tif

Usage:
    python stages/create_cdl_splits.py --raw-cdl /path/to/2024_30m_cdls.tif
    python stages/create_cdl_splits.py \
        --raw-cdl /Volumes/T7/.../2024_30m_cdls.tif \
        --data-dir /Volumes/T7/.../data/processed \
        --overwrite
"""

import argparse
import logging
import sys
from glob import glob
from pathlib import Path

_ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT.parent))

from cropmap_pipeline.config import PROCESSED_DIR, SPATIAL_TEST_AREAS
from cropmap_pipeline.stages.process_data import process_cdl

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

AREAS = [
    {"name": "train",  "s2_subdir": "s2/2024"},
    {"name": "test_a", "s2_subdir": "s2/test_a"},
    {"name": "test_b", "s2_subdir": "s2/test_b"},
]


def _pick_s2_ref(s2_dir: Path) -> Path | None:
    tifs = sorted(f for f in s2_dir.glob("*.tif") if not f.name.startswith("._"))
    return tifs[0] if tifs else None


def main():
    parser = argparse.ArgumentParser(description="Create per-area CDL splits (train/test_a/test_b)")
    parser.add_argument("--raw-cdl",  required=True, help="Raw CDL .tif (EPSG:5070, 30m)")
    parser.add_argument("--data-dir", default=None,  help="Override processed data root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    processed = Path(args.data_dir) if args.data_dir else PROCESSED_DIR
    cdl_out_dir = processed / "cdl"
    cdl_out_dir.mkdir(parents=True, exist_ok=True)

    raw_cdl = Path(args.raw_cdl)
    if not raw_cdl.exists():
        log.error("Raw CDL not found: %s", raw_cdl)
        sys.exit(1)

    for area in AREAS:
        name    = area["name"]
        s2_dir  = processed / area["s2_subdir"]
        s2_ref  = _pick_s2_ref(s2_dir)

        if s2_ref is None:
            log.warning("[%s] No S2 files in %s — skipping", name, s2_dir)
            continue

        log.info("[%s] Using S2 reference: %s", name, s2_ref.name)

        out_reproj   = str(cdl_out_dir / f"cdl_{name}_reprojected.tif")
        out_filtered = str(cdl_out_dir / f"cdl_{name}.tif")

        process_cdl(
            cdl_raw_path    = str(raw_cdl),
            s2_ref_path     = str(s2_ref),
            out_reprojected = out_reproj,
            out_filtered    = out_filtered,
            overwrite       = args.overwrite,
        )
        log.info("[%s] Done → %s", name, out_filtered)

    log.info("All areas processed.")


if __name__ == "__main__":
    main()
