import rasterio
import numpy as np


def majority_filter_labels(in_path, out_path, kernel_size=3):
    """Apply majority (mode) filter to a categorical label raster.

    Each pixel is replaced by the most common class value in its
    kernel_size × kernel_size neighbourhood. Removes salt-and-pepper
    noise and smooths staircase artefacts caused by reprojecting 30 m
    CDL labels to ~10 m Sentinel-2 grid.

    Uses skimage.filters.rank.modal (C-accelerated, ~seconds on full tile).

    Args:
        in_path:     Path to input label GeoTIFF (uint8).
        out_path:    Path to write filtered label GeoTIFF (same profile).
        kernel_size: Square neighbourhood side length (default 3).
    """
    from skimage.filters.rank import modal
    try:
        from skimage.morphology import footprint_rectangle
        _fp = lambda k: footprint_rectangle((k, k))
    except ImportError:
        from skimage.morphology import square as _fp  # skimage < 0.25

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        data = src.read(1)  # (H, W) uint8

    filtered = modal(data, _fp(kernel_size))

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    print(f"Majority-filtered (k={kernel_size}) → {out_path}")


def label_filtering(in_path, out_path, keep_classes=[]):
    # Open raster
    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        data = src.read(1)  # read first band
        nodata_val = src.nodata if src.nodata is not None else 0

    # Ensure nodata is declared in the output profile
    profile.update(nodata=nodata_val)

    # Keep selected classes; everything else becomes nodata
    filtered = np.where(np.isin(data, keep_classes), data, nodata_val)

    # Save filtered raster
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(filtered, 1)

    print("Saved filtered raster:", out_path)