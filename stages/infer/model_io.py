"""Checkpoint loading + tiled inference for a single block."""

import numpy as np
import torch

from cropmap_pipeline.stages.training.train_segmentation import build_model


def load_model(ckpt_path, device, arch_default, num_classes_default):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    arch = ck.get("architecture", arch_default)
    inch = ck.get("in_channels")
    ncls = ck.get("num_classes", num_classes_default)
    m = build_model(arch, inch, ncls).to(device)
    m.load_state_dict(state)
    m.eval()
    return m, arch, ck


def infer_block(model, s2, chan_idx, lo, hi, patch, nodata, device):
    """Tiled inference over a full (C, H, W) block, returns (H, W) class-index map."""
    den = np.maximum(hi - lo, 1.0)
    n = len(chan_idx)
    Hh, Ww = s2.shape[1], s2.shape[2]
    pred = np.zeros((Hh, Ww), np.uint8)
    with torch.no_grad():
        for y in range(0, Hh, patch):
            for x in range(0, Ww, patch):
                ph, pw = min(patch, Hh - y), min(patch, Ww - x)
                tile = s2[np.ix_(chan_idx, range(y, y + ph), range(x, x + pw))].astype(np.float32)
                tile[tile == nodata] = 0.0
                tile[~np.isfinite(tile)] = 0.0
                tile = np.clip((tile - lo[:, None, None]) / den[:, None, None], 0, 1)
                pad = np.zeros((n, patch, patch), np.float32)
                pad[:, :ph, :pw] = tile
                out = (
                    model(torch.from_numpy(pad).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
                )
                pred[y : y + ph, x : x + pw] = out[:ph, :pw]
    return pred


def band_norm_stats(lo_b, hi_b, band_indices, n_bands_per_date):
    """Per-channel [lo, hi] for the given global band indices, tiled from per-band-position stats."""
    lo = np.array([lo_b[int(b) % n_bands_per_date] for b in band_indices], np.float32)
    hi = np.array([hi_b[int(b) % n_bands_per_date] for b in band_indices], np.float32)
    return lo, hi
