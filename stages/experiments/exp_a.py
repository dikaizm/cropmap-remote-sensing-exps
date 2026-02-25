"""Single-date experiments — peak NDVI date, all VEGE_BANDS or GSI-selected bands."""

import json
import sys
from pathlib import Path

import numpy as np
import rasterio

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
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


def build_single_date_selected_indices(
    local_date_to_idx,
    local_band_to_idx,
    s2_paths=None,
    cdl_path=None,
    top_k: int | None = 5,
    candidates_json: Path | None = None,
    force: bool = False,
    best_date: str | None = None,
):
    """Single date (peak NDVI) × GSI or RF top-K band union.

    When candidates_json is None: runs scoped GSI on the peak date's single
    file and caches to gsi_single_date_candidates.json alongside the data.
    When candidates_json is provided (RF variant): loads that JSON directly.

    Parameters
    ----------
    top_k : int | None
        Bands per crop before union. None = use all ranked bands.
    candidates_json : Path | None
        Pre-computed candidates JSON (RF variant). None → compute GSI inline.
    force : bool
        Re-run scoped GSI even if cached JSON exists.
    best_date : str | None
        Pre-computed peak NDVI date (YYYYMMDD). Skips NDVI scan if provided.
    """
    if best_date is None:
        best_date = _find_peak_ndvi_date(local_date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)

    if candidates_json is not None:
        json_path = Path(candidates_json)
        if not json_path.exists():
            raise FileNotFoundError(f"Candidates JSON not found: {json_path}")
        with open(json_path) as f:
            band_candidates = json.load(f)["band_candidates_per_crop"]
    else:
        if s2_paths is None or cdl_path is None:
            raise ValueError("s2_paths and cdl_path required for scoped GSI scoring")
        from crop_mapping_pipeline.stages.selections.band_scoring.gsi.v3 import compute_band_candidates
        peak_file = s2_paths[local_date_to_idx[best_date]]
        out_json  = Path(s2_paths[0]).parent / "gsi_single_date_candidates.json"
        band_candidates = compute_band_candidates([peak_file], cdl_path, out_json=out_json, force=force)

    seen: set[str] = set()
    union_bands: list[str] = []
    for crop_id in KEEP_CLASSES:
        ranked = band_candidates.get(str(crop_id), [])
        k = top_k if top_k is not None else len(ranked)
        for band in ranked[:k]:
            if band not in seen and band in S2_BAND_NAMES:
                seen.add(band)
                union_bands.append(band)

    if not union_bands:
        raise ValueError("No valid bands found in band_candidates_per_crop")

    idx, names, skipped = [], [], 0
    for band in union_bands:
        local_name = f"{band}_{best_date}"
        i = local_band_to_idx.get(local_name)
        if i is not None:
            idx.append(i)
            names.append(local_name)
        else:
            skipped += 1

    if not idx:
        raise ValueError(
            f"single_date_selected: no bands matched date={best_date}. "
            "Check S2 files include this date."
        )
    if skipped:
        log.warning(f"single_date_selected: {skipped}/{len(union_bands)} band(s) not in local band map for date={best_date}")

    log.info(
        f"single_date_selected: date={best_date}, top_k={top_k} per crop "
        f"→ {len(union_bands)} union bands → {len(idx)} channels"
    )
    log.info(f"single_date_selected bands: {union_bands}")
    return idx, names, best_date


# backwards-compat aliases
build_exp_A_indices = build_single_date_indices
