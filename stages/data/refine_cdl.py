"""Apply a majority filter to an already-processed CDL label raster.

Use this when CDL has already been preprocessed (reprojected + label-filtered)
and you want to smooth it without re-running the full pipeline.

Usage:
    python stages/refine_cdl.py                          # uses CDL_TRAIN from config
    python stages/refine_cdl.py --in path/to/cdl.tif     # custom input
    python stages/refine_cdl.py --kernel 5               # 5x5 majority filter
    python stages/refine_cdl.py --in-place               # overwrite input file
"""

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import CDL_TRAIN
from crop_mapping_pipeline.utils.label import majority_filter_labels


def main():
    parser = argparse.ArgumentParser(description="Majority-filter a CDL label raster")
    parser.add_argument(
        "--in", dest="in_path", type=str, default=None,
        help="Input CDL GeoTIFF (default: CDL_TRAIN from config)",
    )
    parser.add_argument(
        "--out", dest="out_path", type=str, default=None,
        help="Output path (default: <stem>_mf<k>.tif next to input)",
    )
    parser.add_argument(
        "--kernel", type=int, default=3,
        help="Majority filter kernel size (default: 3)",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Overwrite input file (backup saved as <stem>.bak.tif)",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path) if args.in_path else CDL_TRAIN
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        sys.exit(1)

    if args.in_place:
        backup = in_path.with_suffix(".bak.tif")
        shutil.copy2(in_path, backup)
        print(f"Backup → {backup}")
        out_path = in_path
    elif args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = in_path.with_stem(in_path.stem + f"_mf{args.kernel}")

    print(f"Input  : {in_path}")
    print(f"Output : {out_path}")
    print(f"Kernel : {args.kernel}×{args.kernel}")
    majority_filter_labels(str(in_path), str(out_path), kernel_size=args.kernel)
    print("Done.")


if __name__ == "__main__":
    main()
