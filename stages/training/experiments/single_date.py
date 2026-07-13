"""Single-date experiments — peak NDVI date, all VEGE_BANDS or GSI-selected bands."""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

_ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT.parent))

from cropmap_pipeline.config import (
    S2_BAND_NAMES, N_BANDS_PER_DATE, VEGE_BANDS, KEEP_CLASSES,
)

import logging
log = logging.getLogger(__name__)

# B4=Red, B8=NIR (0-based in S2_BAND_NAMES)
_B4_IDX = S2_BAND_NAMES.index("B4") + 1   # rasterio 1-based
_B8_IDX = S2_BAND_NAMES.index("B8") + 1


def _mean_ndvi(tif_path, cdl_arr, valid_thresh=0.80):
    """Return (mean_ndvi, valid_frac) over crop pixels. Returns (None, 0) on failure."""
    try:
        with rasterio.open(tif_path) as src:
            nodata = src.nodata if src.nodata is not None else -9999.0
            b4 = src.read(_B4_IDX).astype(np.float32)
            b8 = src.read(_B8_IDX).astype(np.float32)
        valid = (cdl_arr > 0) & (b4 != nodata) & (b8 != nodata) & np.isfinite(b4) & np.isfinite(b8)
        valid_frac = valid.sum() / max(cdl_arr.sum(), 1)
        if valid_frac < valid_thresh:
            return None, valid_frac
        denom = np.where((b8[valid] + b4[valid]) == 0, 1e-6, b8[valid] + b4[valid])
        return float(np.mean((b8[valid] - b4[valid]) / denom)), valid_frac
    except Exception:
        return None, 0.0


def _find_peak_ndvi_date(local_date_to_idx, s2_paths=None, cdl_path=None):
    """Return peak-NDVI date string. Caches result alongside S2 data. Falls back to Jul heuristic."""
    available_dates = sorted(local_date_to_idx.keys())

    # Cache key: sorted date list (proxy for which S2 files are present)
    cache_path = (
        Path(s2_paths[0]).parent / "peak_ndvi_date.json"
        if s2_paths else None
    )
    if cache_path and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("dates_key") == available_dates and cached.get("date"):
                log.info(f"single_date: peak NDVI date cached → {cached['date']}")
                return cached["date"]
        except Exception:
            pass

    if s2_paths and cdl_path:
        try:
            with rasterio.open(cdl_path) as src:
                cdl_arr = np.isin(src.read(1), KEEP_CLASSES).astype(np.uint8)
            ndvi_scores = {}
            for d in available_dates:
                fi = local_date_to_idx[d]
                ndvi, _ = _mean_ndvi(s2_paths[fi], cdl_arr)
                if ndvi is not None:
                    ndvi_scores[d] = ndvi
            if ndvi_scores:
                best = max(ndvi_scores, key=ndvi_scores.get)
                log.info(f"single_date: NDVI-selected date={best} (NDVI={ndvi_scores[best]:.4f})")
                if cache_path:
                    with open(cache_path, "w") as f:
                        json.dump({"date": best, "dates_key": available_dates, "ndvi_scores": ndvi_scores}, f)
                return best
        except Exception as e:
            log.warning(f"single_date: NDVI selection failed ({e}), falling back to Jul heuristic")

    best = next(
        (k for k in available_dates if k[4:6] == "07" and k[6:8] in ("14", "29", "30")),
        available_dates[-1],
    )
    log.info(f"single_date: heuristic date={best}")
    return best


def build_single_date_indices(local_date_to_idx, local_band_to_idx,
                              s2_paths=None, cdl_path=None):
    """Single date (peak NDVI) × all S2 bands — conventional baseline (no band selection)."""
    best_date = _find_peak_ndvi_date(local_date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)
    off   = local_date_to_idx[best_date] * N_BANDS_PER_DATE
    idx   = [off + S2_BAND_NAMES.index(b) for b in S2_BAND_NAMES]
    names = [f"{b}_{best_date}" for b in S2_BAND_NAMES]
    log.info(f"single_date: {len(idx)} channels (all {len(S2_BAND_NAMES)} bands, no selection)")
    return idx, names, best_date


# backwards-compat aliases
build_exp_A_indices = build_single_date_indices
