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

    print(f"✅ Majority-filtered (k={kernel_size}) → {out_path}")


def erode_and_clean_labels(in_path, out_path, erosion_iter=1, min_size=4, unknown_value=255):
    """Erode class boundaries and drop small connected components (CalCROP21-style).

    Mixed pixels at class boundaries (from reprojecting 30 m CDL onto the
    10 m S2 grid) and isolated speckle are replaced with `unknown_value`
    rather than forced into a neighbouring class, so a loss with
    ignore_index=unknown_value can exclude them from training.

    Per class: binary_erosion(iterations=erosion_iter) strips a boundary
    ring; pixels in that ring become unknown. Remaining per-class connected
    components smaller than min_size pixels also become unknown.

    Args:
        in_path:       Path to input label GeoTIFF (uint8).
        out_path:      Path to write cleaned label GeoTIFF (same profile).
        erosion_iter:  Erosion iterations (pixel depth) per class boundary.
        min_size:      Connected components smaller than this (px) → unknown.
        unknown_value: Sentinel value for excluded pixels (default 255).
    """
    from scipy.ndimage import binary_erosion, label as cc_label

    with rasterio.open(in_path) as src:
        profile = src.profile.copy()
        data = src.read(1)  # (H, W) uint8

    cleaned = data.copy()
    for cls in np.unique(data):
        mask = data == cls
        eroded = binary_erosion(mask, iterations=erosion_iter, border_value=0)
        boundary_ring = mask & ~eroded
        cleaned[boundary_ring] = unknown_value

        labeled, n_components = cc_label(eroded)
        if n_components:
            sizes = np.bincount(labeled.ravel())
            small_components = np.isin(labeled, np.nonzero(sizes < min_size)[0])
            small_components &= labeled > 0  # exclude background label 0 of cc_label
            cleaned[small_components] = unknown_value

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cleaned, 1)

    n_unknown = int((cleaned == unknown_value).sum())
    print(f"✅ Eroded boundaries + dropped components <{min_size}px "
          f"({n_unknown} px → unknown) → {out_path}")


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

    print("✅ Saved filtered raster:", out_path)