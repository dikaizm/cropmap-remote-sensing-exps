"""Tiled full-scene inference over an S2 stack + CDL ground-truth remap."""

import logging

import numpy as np
import torch
import rasterio
import rasterio.windows

from cropmap_pipeline.config import S2_NODATA, REMAP_LUT
from cropmap_pipeline.stages.training.normalization import _per_channel_percentiles
from cropmap_pipeline.stages.training.run_state import DEVICE

log = logging.getLogger(__name__)


def run_full_inference(model, s2_paths, band_indices, patch_size=256, stride=256,
                       channel_stats=None, band_percentiles=None, norm_mode="percentile"):
    """Tiled inference — reads one window at a time, never loads full rasters."""
    assert band_percentiles is not None, "band_percentiles required"
    with rasterio.open(s2_paths[0]) as src:
        H, W    = src.height, src.width
        profile = dict(src.profile)

    srcs     = [rasterio.open(p) for p in s2_paths]
    pred_map = np.zeros((H, W), dtype=np.uint8)
    n_rows   = (H + stride - 1) // stride
    n_cols   = (W + stride - 1) // stride
    total    = n_rows * n_cols
    K        = len(band_indices)

    lo_per_ch, hi_per_ch = _per_channel_percentiles(band_indices, *band_percentiles)
    denom_per_ch = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
    lo_per_ch    = lo_per_ch.astype(np.float32)

    model.eval()
    done = 0
    try:
        with torch.no_grad():
            for y in range(0, H, stride):
                for x in range(0, W, stride):
                    ph  = min(patch_size, H - y)
                    pw  = min(patch_size, W - x)
                    win = rasterio.windows.Window(x, y, pw, ph)

                    # Read only this window from each file
                    bands = []
                    for src in srcs:
                        try:
                            arr = src.read(window=win).astype(np.float32)
                        except Exception:
                            arr = np.zeros((src.count, ph, pw), dtype=np.float32)
                        arr[arr == S2_NODATA] = 0.0
                        arr[~np.isfinite(arr)] = 0.0
                        bands.append(arr)

                    patch = np.concatenate(bands, axis=0)[band_indices]  # (K, ph, pw)
                    patch = (patch - lo_per_ch[:, None, None]) / denom_per_ch[:, None, None]
                    if norm_mode != "zscore":
                        patch = np.clip(patch, 0.0, 1.0)

                    # Pad to patch_size if at border
                    if ph < patch_size or pw < patch_size:
                        padded = np.zeros((K, patch_size, patch_size), dtype=np.float32)
                        padded[:, :ph, :pw] = patch
                        patch = padded

                    t   = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                    out = model(t).argmax(dim=1).squeeze().cpu().numpy()
                    pred_map[y:y + ph, x:x + pw] = out[:ph, :pw]
                    done += 1
                    if done % 200 == 0 or done == total:
                        log.info(f"  {done}/{total} tiles")
    finally:
        for src in srcs:
            src.close()

    return pred_map, profile


def load_gt_remap(cdl_path):
    with rasterio.open(cdl_path) as src:
        cdl     = src.read(1).astype(np.int32)
        profile = dict(src.profile)
    gt = REMAP_LUT[np.clip(cdl, 0, 255)]
    return gt.astype(np.uint8), profile
