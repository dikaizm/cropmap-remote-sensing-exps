"""Full tile scan — reads every 256×256 tile across all bands for each processed S2 file.

Usage:
    python stages/verify_tiles.py
    python stages/verify_tiles.py --data-dir /workspace/crop_mapping_pipeline/data/processed
    python stages/verify_tiles.py --years 2022
    python stages/verify_tiles.py --years 2022 2023 --tile-size 512
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
import rasterio.windows

TILE = 256
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Valid TIFF/BigTIFF magic byte sequences
_TIFF_MAGIC = {
    b"II\x2a\x00",  # little-endian TIFF
    b"MM\x00\x2a",  # big-endian TIFF
    b"II\x2b\x00",  # little-endian BigTIFF
    b"MM\x00\x2b",  # big-endian BigTIFF
}


def _check_magic(path: Path) -> str | None:
    """Return error string if TIFF magic bytes are invalid, else None."""
    with open(path, "rb") as f:
        header = f.read(4)
    if header not in _TIFF_MAGIC:
        return f"bad TIFF magic: {header.hex()} (expected II/MM + 2a/2b)"
    return None


def scan_file(path: Path, tile_size: int = TILE) -> tuple[Path, bool, str, int, float]:
    """Read every tile of every band. Return (path, ok, error_msg, tiles_read, elapsed_s)."""
    t0 = time.monotonic()
    tiles_read = 0

    magic_err = _check_magic(path)
    if magic_err:
        return path, False, magic_err, 0, time.monotonic() - t0

    try:
        with rasterio.open(path) as src:
            h, w, nb = src.height, src.width, src.count
            for band in range(1, nb + 1):
                for y in range(0, h, tile_size):
                    for x in range(0, w, tile_size):
                        ph = min(tile_size, h - y)
                        pw = min(tile_size, w - x)
                        src.read(band, window=rasterio.windows.Window(x, y, pw, ph))
                        tiles_read += 1
        return path, True, "", tiles_read, time.monotonic() - t0
    except Exception as e:
        return path, False, str(e), tiles_read, time.monotonic() - t0


def main():
    parser = argparse.ArgumentParser(description="Full tile scan of processed S2 files")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to processed/ directory (default: pipeline data/processed/)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2022", "2023"],
        help="Years to scan (default: 2022 2023)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=TILE,
        help=f"Tile size in pixels (default: {TILE})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel file workers (default: 4)",
    )
    args = parser.parse_args()

    s2_dir = args.data_dir / "s2"
    if not s2_dir.exists():
        print(f"ERROR: S2 directory not found: {s2_dir}", file=sys.stderr)
        sys.exit(1)

    files = []
    for yr in args.years:
        yr_dir = s2_dir / yr
        if not yr_dir.exists():
            print(f"WARN: year dir missing: {yr_dir}", file=sys.stderr)
            continue
        yr_files = sorted(yr_dir.glob("S2H_*_processed.tif"))
        if not yr_files:
            print(f"WARN: no processed TIFs in {yr_dir}", file=sys.stderr)
        files.extend((yr, f) for f in yr_files)

    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    total = len(files)
    print(f"Scanning {total} files  tile={args.tile_size}px  workers={args.workers}")
    print()

    failed = []
    ok_count = 0
    total_tiles = 0
    wall_t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(scan_file, f, args.tile_size): (yr, f)
            for yr, f in files
        }
        results = {}
        done = 0
        for future in as_completed(future_map):
            yr, f = future_map[future]
            path, ok, err, tiles, elapsed = future.result()
            results[(yr, f)] = (ok, err, tiles, elapsed)
            done += 1
            date = f.stem.replace("_processed", "")
            status = "OK  " if ok else "FAIL"
            print(
                f"[{done:>3}/{total}  {done/total*100:5.1f}%]  "
                f"{status}  {yr}/{date}  "
                f"{tiles:>6} tiles  {elapsed:5.1f}s"
                + (f"\n         {err}" if not ok else "")
            )

    wall_elapsed = time.monotonic() - wall_t0

    print()
    print("=" * 60)
    for yr, f in files:
        ok, err, tiles, _ = results[(yr, f)]
        if ok:
            ok_count += 1
            total_tiles += tiles
        else:
            failed.append((yr, f, err))

    print(f"Result : {ok_count}/{total} files OK")
    print(f"Tiles  : {total_tiles:,} total tiles read")
    print(f"Time   : {wall_elapsed:.1f}s wall  ({wall_elapsed/total:.1f}s avg/file)")

    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for yr, f, err in failed:
            print(f"  {yr}/{f.name}")
            print(f"    {err}")
        sys.exit(1)
    else:
        print("All files clean.")


if __name__ == "__main__":
    main()
