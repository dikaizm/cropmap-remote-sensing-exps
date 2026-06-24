"""Apply boundary-erosion / connected-component cleanup / majority filter to an
already-processed CDL label raster.

Use this when CDL has already been preprocessed (reprojected + label-filtered)
and you want to clean it up without re-running the full pipeline.

Usage:
    python stages/refine_cdl.py                          # uses CDL_TRAIN from config, majority filter only
    python stages/refine_cdl.py --in path/to/cdl.tif    # custom input
    python stages/refine_cdl.py --kernel 5               # 5×5 majority filter
    python stages/refine_cdl.py --erode                  # CalCROP21-style: 1px boundary erosion + drop <4px components → unknown(255)
    python stages/refine_cdl.py --erode --no-majority    # erosion/cleanup only, skip majority filter
    python stages/refine_cdl.py --in-place               # overwrite input file
"""

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import CDL_TRAIN
from crop_mapping_pipeline.utils.label import majority_filter_labels, erode_and_clean_labels


def main():
    parser = argparse.ArgumentParser(description="Majority-filter CDL label raster")
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
    parser.add_argument(
        "--erode", action="store_true",
        help="Erode 1px class boundaries + drop small connected components → unknown (CalCROP21-style)",
    )
    parser.add_argument(
        "--erode-iter", type=int, default=1,
        help="Boundary erosion depth in pixels (default: 1)",
    )
    parser.add_argument(
        "--min-size", type=int, default=4,
        help="Connected components smaller than this (px) become unknown (default: 4)",
    )
    parser.add_argument(
        "--unknown-value", type=int, default=255,
        help="Sentinel label value for eroded/small-component pixels (default: 255)",
    )
    parser.add_argument(
        "--no-majority", action="store_true",
        help="Skip the majority filter step (useful with --erode for cleanup-only)",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path) if args.in_path else CDL_TRAIN
    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        sys.exit(1)

    if not args.erode and args.no_majority:
        print("ERROR: --no-majority with no --erode leaves nothing to do.")
        sys.exit(1)

    suffix = ("_erode" if args.erode else "") + ("" if args.no_majority else f"_mf{args.kernel}")

    if args.in_place:
        backup = in_path.with_suffix(".bak.tif")
        shutil.copy2(in_path, backup)
        print(f"Backup → {backup}")
        out_path = in_path
    elif args.out_path:
        out_path = Path(args.out_path)
    else:
        out_path = in_path.with_stem(in_path.stem + suffix)

    print(f"Input  : {in_path}")
    print(f"Options: erode={args.erode}, erode_iter={args.erode_iter}, min_size={args.min_size}, kernel={args.kernel}")
    print(f"Output : {out_path}")

    current = in_path
    tmp_path = None
    if args.erode:
        tmp_path = out_path if args.no_majority else in_path.with_stem(in_path.stem + "_erode_tmp")
        print(f"Erode  : {args.erode_iter}px boundary, drop <{args.min_size}px components, unknown={args.unknown_value}")
        erode_and_clean_labels(
            str(current), str(tmp_path),
            erosion_iter=args.erode_iter, min_size=args.min_size, unknown_value=args.unknown_value,
        )
        current = tmp_path

    if not args.no_majority:
        print(f"Kernel : {args.kernel}×{args.kernel}")
        majority_filter_labels(str(current), str(out_path), kernel_size=args.kernel)
        if tmp_path is not None and tmp_path != out_path:
            tmp_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
