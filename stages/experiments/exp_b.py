"""Multi-temporal baseline — 4 peak-NDVI dates per calendar quarter × bands."""

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import rasterio

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES, N_BANDS_PER_DATE, KEEP_CLASSES, PROCESSED_DIR,
)
from crop_mapping_pipeline.stages.experiments.exp_a import _mean_ndvi

import logging
log = logging.getLogger(__name__)

# Bump when date selection logic or JSON schema changes.
_PHENOL_CACHE_VERSION = "quarterly_ndvi_v1"

_QUARTERS = [
    (1,  1,  3, 31),
    (4,  1,  6, 30),
    (7,  1,  9, 30),
    (10, 1, 12, 31),
]


def _date_in_quarter(d: str, q_idx: int, year: int) -> bool:
    """Check if date string YYYYMMDD falls within the given quarter index (0–3)."""
    sm, sd, em, ed = _QUARTERS[q_idx]
    s = date(year, sm, sd)
    e = date(year, em, ed)
    dt = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    return s <= dt <= e


def _quarter_label(q_idx: int, year: int) -> str:
    """Return a human-readable quarter label e.g. 'Q1 (01/01–03/31)'."""
    sm, sd, em, ed = _QUARTERS[q_idx]
    return f"Q{q_idx+1} ({sm:02d}/{sd:02d}–{em:02d}/{ed:02d})"


def _select_phenol_dates(local_date_to_idx, s2_paths=None, cdl_path=None, phenol_json=None):
    """Return {Q1..Q4: date_str} for 4 dates — max-NDVI within each calendar quarter.

    Computes full-year NDVI (mean over crop pixels), then picks the single
    date with the highest NDVI per quarter. Falls back to quarter midpoints
    if NDVI cannot be computed (no s2_paths/cdl_path).

    If phenol_json is provided, reads/writes cache at that path instead of
    the default PROCESSED_DIR/phenol_dates.json.
    """
    available_dates = sorted(local_date_to_idx.keys())

    if not available_dates:
        raise ValueError("No available dates")

    ref_year = int(available_dates[0][:4])

    cache_path = Path(phenol_json) if phenol_json else (PROCESSED_DIR / "phenol_dates.json")
    log.info(f"Phenol cache path: {cache_path}")
    if cache_path and cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if (cached.get("version") == _PHENOL_CACHE_VERSION
                    and cached.get("dates_key") == available_dates
                    and cached.get("phenol_map")):
                log.info(f"quarterly-NDVI dates cached → {list(cached['phenol_map'].values())}")
                return cached["phenol_map"]
        except Exception:
            pass

    # ── full-year NDVI ──
    ndvi_year = {}
    if s2_paths and cdl_path:
        try:
            with rasterio.open(cdl_path) as src:
                cdl_arr = np.isin(src.read(1), KEEP_CLASSES).astype(np.uint8)
            for d in available_dates:
                fi = local_date_to_idx[d]
                ndvi, _ = _mean_ndvi(s2_paths[fi], cdl_arr)
                if ndvi is not None:
                    ndvi_year[d] = float(round(ndvi, 5))
            if ndvi_year:
                log.info(f"full-year NDVI: {len(ndvi_year)} dates")
        except Exception as e:
            log.warning(f"NDVI scan failed ({e}), falling back to midpoints")

    # ── per-quarter max-NDVI (with midpoints fallback) ──
    phenol_map = {}
    selection_method = "ndvi_max" if ndvi_year else "midpoint"

    for qi in range(4):
        quarter_dates = [d for d in available_dates if _date_in_quarter(d, qi, ref_year)]

        if selection_method == "ndvi_max" and quarter_dates:
            # filter to dates that have NDVI computed
            candidates = [(d, ndvi_year[d]) for d in quarter_dates if d in ndvi_year]
            if candidates:
                best = max(candidates, key=lambda x: x[1])[0]
                qlbl = f"Q{qi+1}"
                log.info(f"  {_quarter_label(qi, ref_year)}: {best} (NDVI={ndvi_year[best]:.4f})")
                phenol_map[qlbl] = best
                continue

        # fallback: nearest to quarter midpoint
        sm, sd, em, ed = _QUARTERS[qi]
        mid = (date(ref_year, sm, sd).toordinal() + date(ref_year, em, ed).toordinal()) // 2
        mid_mmdd = date.fromordinal(mid).strftime("%m%d")
        target = int(mid_mmdd)
        best = min(available_dates, key=lambda d: abs(int(d[4:]) - target))
        qlbl = f"Q{qi+1}"
        ndvi_str = f" (NDVI={ndvi_year.get(best, '?'):.4f})" if ndvi_year.get(best) else ""
        log.info(f"  {_quarter_label(qi, ref_year)} fallback → {best}{ndvi_str}")
        phenol_map[qlbl] = best

    log.info(f"selection_method={selection_method} → dates={list(phenol_map.values())}")

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump({"phenol_map": phenol_map, "dates_key": available_dates,
                       "ndvi_year": ndvi_year, "selection_method": selection_method,
                       "version": _PHENOL_CACHE_VERSION}, f)

    return phenol_map


def _band_union_from_candidates(band_candidates: dict, top_k: int | None = None) -> list[str]:
    """Return union of top-K bands per crop from band_candidates_per_crop dict."""
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
    return union_bands


def build_naive_multitemporal_indices(local_date_to_idx, local_band_to_idx,
                                      s2_paths=None, cdl_path=None,
                                      phenol_json=None):
    """4 calendar dates × all S2_BAND_NAMES = up to 40 channels.

    phenol_json: optional path to pre-computed phenol_dates.json.
    """
    phenol_map = _select_phenol_dates(local_date_to_idx, s2_paths=s2_paths,
                                      cdl_path=cdl_path, phenol_json=phenol_json)

    idx, names = [], []
    for _label, d in phenol_map.items():
        off    = local_date_to_idx[d] * N_BANDS_PER_DATE
        idx   += [off + S2_BAND_NAMES.index(b) for b in S2_BAND_NAMES]
        names += [f"{b}_{d}" for b in S2_BAND_NAMES]

    seen, dedup_idx, dedup_names = set(), [], []
    for i, name in zip(idx, names):
        if i not in seen:
            seen.add(i)
            dedup_idx.append(i)
            dedup_names.append(name)

    log.info(f"naive_multitemporal: {len(dedup_idx)} channels")
    return dedup_idx, dedup_names, phenol_map


def build_naive_multitemporal_selected_indices(
    local_date_to_idx,
    local_band_to_idx,
    s2_paths=None,
    cdl_path=None,
    top_k: int | None = 5,
    candidates_json: Path | None = None,
    force: bool = False,
    phenol_map: dict | None = None,
    phenol_json: str | None = None,
):
    """4 calendar dates × GSI or RF top-K band union.

    When candidates_json is None: runs scoped GSI on only the 4 calendar
    dates and caches to gsi_naive_mt_candidates.json alongside the data.
    When candidates_json is provided (RF variant): loads that JSON directly.

    Parameters
    ----------
    top_k : int | None
        Bands per crop before union. None = use all ranked bands.
    candidates_json : Path | None
        Pre-computed candidates JSON (RF variant). None → compute GSI inline.
    force : bool
        Re-run scoped GSI even if cached JSON exists.
    phenol_map : dict | None
        Pre-computed {Q1..Q4: date_str}. If provided, skips date selection.
    phenol_json : str | None
        Path to pre-computed phenol_dates.json cache.
    """
    if phenol_map is None:
        phenol_map = _select_phenol_dates(local_date_to_idx, s2_paths=s2_paths,
                                          cdl_path=cdl_path, phenol_json=phenol_json)

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
        phenol_files = [s2_paths[local_date_to_idx[d]] for d in phenol_map.values()]
        out_json     = Path(s2_paths[0]).parent / "gsi_naive_mt_candidates.json"
        band_candidates = compute_band_candidates(phenol_files, cdl_path, out_json=out_json, force=force)

    union_bands = _band_union_from_candidates(band_candidates, top_k=top_k)

    idx, names, skipped = [], [], 0
    for _label, d in phenol_map.items():
        for band in union_bands:
            local_name = f"{band}_{d}"
            i = local_band_to_idx.get(local_name)
            if i is not None:
                idx.append(i)
                names.append(local_name)
            else:
                skipped += 1

    seen, dedup_idx, dedup_names = set(), [], []
    for i, name in zip(idx, names):
        if i not in seen:
            seen.add(i)
            dedup_idx.append(i)
            dedup_names.append(name)

    if not dedup_idx:
        raise ValueError(
            "naive_multitemporal_selected: no bands matched. "
            "Check S2 files include the selected dates."
        )
    if skipped:
        log.warning(
            f"naive_multitemporal_selected: {skipped} (band, date) combos not in local band map"
        )

    log.info(
        f"naive_multitemporal_selected: top_k={top_k} per crop "
        f"→ {len(union_bands)} union bands × {len(phenol_map)} dates → {len(dedup_idx)} channels"
    )
    log.info(f"naive_multitemporal_selected bands: {union_bands}")
    return dedup_idx, dedup_names, phenol_map


# backwards-compat alias
build_exp_B_indices = build_naive_multitemporal_indices
