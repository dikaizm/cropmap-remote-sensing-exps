import os
import glob
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import argparse

def check_files(directory):
    print(f"Scanning for corrupt TIF files in: {directory}")
    tif_files = sorted(glob.glob(os.path.join(directory, "**/*.tif"), recursive=True))
    print(f"Found {len(tif_files)} TIF files")

    corrupt_files = []

    for tif in tqdm(tif_files):
        try:
            with rasterio.open(tif) as src:
                # Read the entire first band in blocks matching the file's tiling
                # to ensure every compressed tile is decompressed and validated.
                block_shapes = src.block_shapes
                tile_h, tile_w = block_shapes[0] if block_shapes else (256, 256)
                height, width = src.height, src.width
                for row_off in range(0, height, tile_h):
                    for col_off in range(0, width, tile_w):
                        win = Window(
                            col_off, row_off,
                            min(tile_w, width  - col_off),
                            min(tile_h, height - row_off),
                        )
                        src.read(1, window=win)
        except Exception as e:
            print(f"\n[CORRUPT] {tif}")
            print(f"  Error: {e}")
            corrupt_files.append(tif)

    if corrupt_files:
        print(f"\nFound {len(corrupt_files)} corrupt file(s):")
        for f in corrupt_files:
            print(f"  {f}")
        print("\nDelete these files and re-download / re-process them.")
    else:
        print("\nAll files OK — no corruption detected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for corrupt GeoTIFF files")
    parser.add_argument("dir", help="Directory to scan (e.g. data/processed)")
    args = parser.parse_args()
    
    check_files(args.dir)
