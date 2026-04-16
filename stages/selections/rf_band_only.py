"""RF band-only selection — ranks bands on fixed domain dates (no date selection).

Trains ONE multi-class RandomForestClassifier on the given domain channels,
decomposes per-class Gini importance (class-conditional MDI), collapses to
band-level ranking by averaging across dates, outputs top-K bands per crop.

Follows Wei et al. (2023, Remote Sensing 15:3212) — same multi-class RF +
per-class MDI approach as rf_direct.py. Unlike rf_direct.py, this does NOT
select dates — dates come from domain knowledge (phenological stages). Only
bands are ranked via RF.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, CDL_CLASS_NAMES, S2_BAND_NAMES,
)
from crop_mapping_pipeline.stages.selections._utils import build_channel_names, sample_pixels
from crop_mapping_pipeline.stages.selections.rf_direct import (
    _train_multiclass_rf, _per_class_importance,
)

log = logging.getLogger(__name__)


def run_rf_band_only(
    s2_paths: list[str],
    cdl_path: str,
    domain_channel_names: list[str],
    top_k: int = 9,
) -> dict[str, list[str]]:
    """One multi-class RF on domain-date channels → per-class MDI → band ranking.

    For single-date: each band appears once → importance is band ranking.
    For multi-date: per-class MDI averaged across dates per band.

    Follows Wei et al. (2023) — matches rf_direct.py approach.

    Args:
        s2_paths: reference-year S2 files.
        cdl_path: reference-year CDL raster path.
        domain_channel_names: channel names for fixed domain dates.
        top_k: bands per crop after ranking (default 9 = all VEGE_BANDS).

    Returns:
        {str(crop_id): [ranked_band_names, ...]} — ready for band_candidates_per_crop.
    """
    all_bandnames, _, _ = build_channel_names(s2_paths)

    valid_channels = [ch for ch in domain_channel_names if ch in all_bandnames]
    missing = set(domain_channel_names) - set(valid_channels)
    if missing:
        log.warning("rf_band_only: %d domain channels missing from S2 data", len(missing))
    if not valid_channels:
        raise ValueError("No valid domain channels found in S2 data")

    log.info("rf_band_only (multi-class): %d domain channels, top_k=%d",
             len(valid_channels), top_k)

    df = sample_pixels(s2_paths, cdl_path, all_bandnames)
    # Restrict df to valid domain channels only
    df_domain = df[["class_label"] + valid_channels].copy()

    # One multi-class RF — same approach as rf_direct.py
    rf = _train_multiclass_rf(df_domain, valid_channels, seed=42)
    imp_per_crop = _per_class_importance(rf, valid_channels, KEEP_CLASSES)

    band_candidates: dict[str, list[str]] = {}
    for crop_id in KEEP_CLASSES:
        importance = imp_per_crop[crop_id]  # pd.Series indexed by valid_channels

        # Collapse channel importance to band-level by averaging across dates
        band_imp: dict[str, list[float]] = {}
        for ch in valid_channels:
            band = ch.split("_")[0]
            if band in S2_BAND_NAMES:
                band_imp.setdefault(band, []).append(float(importance[ch]))

        band_avg = {b: float(np.mean(v)) for b, v in band_imp.items()}
        ranked   = sorted(band_avg, key=band_avg.get, reverse=True)[:top_k]
        band_candidates[str(crop_id)] = ranked

        log.info("  %-20s: top-3 bands = %s", CDL_CLASS_NAMES[crop_id], ranked[:3])

    return band_candidates


def save_rf_band_json(
    band_candidates: dict[str, list[str]],
    json_path: Path,
):
    """Save band_candidates_per_crop to JSON (same key as gsi_candidates.json)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selector": "rf_band_only",
        "band_candidates_per_crop": band_candidates,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved rf_band_only → %s", json_path)
