"""Experiment registry for Stage 3 training.

Registered experiments (all optional — only built when band indices are provided):
  single_date — peak NDVI date, all S2_BAND_NAMES (single-date baseline)
  mt_base     — 4 phenological dates, all VEGE_BANDS (multi-temporal naive baseline)
  gsi         — GSI-direct top-K channels per crop, union (spectral-temporal selection)
  rf          — RF-direct top-K channels per crop, union (multi-class MDI selection)

To add an experiment:
  1. Build its band indices in main() of train_segmentation.py
  2. Add an ExperimentConfig entry in build_registry() below
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from crop_mapping_pipeline.config import (
    MLFLOW_EXPERIMENT_TRAIN_V6_1_SAME_AREA,
)


@dataclass
class ExperimentConfig:
    key:               str
    description:       str
    band_indices:      Any           # list[int] or dict{yr: (list[int], list[str])}
    band_names:        list          # reference-year channel names
    default_loss:      str  = "wce"  # "wce" | "focal_tversky" | "dynamic_balanced"
    mlflow_experiment: str  = MLFLOW_EXPERIMENT_TRAIN_V6_1_SAME_AREA
    extra_kw:          dict = field(default_factory=dict)


def build_registry(
    single_date_idx = None,  single_date_names = None,  single_date_key = None,
    mt_base_idx     = None,  mt_base_names     = None,  phenol_map      = None,
    gsi_idx         = None,  gsi_names         = None,
    rf_idx          = None,  rf_names          = None,
) -> dict[str, ExperimentConfig]:
    """Build and return the experiment registry.

    Only experiments whose band indices are not None are registered.
    """
    reg: dict[str, ExperimentConfig] = {}

    if single_date_idx is not None:
        reg["single_date"] = ExperimentConfig(
            key         = "single_date",
            description = f"Single-date {single_date_key}, all bands (baseline) — {len(single_date_idx)}ch",
            band_indices= single_date_idx,
            band_names  = single_date_names,
        )

    if mt_base_idx is not None:
        reg["mt_base"] = ExperimentConfig(
            key         = "mt_base",
            description = f"4 calendar dates {list(phenol_map.values())}, all VEGE_BANDS (baseline) — {len(mt_base_idx)}ch",
            band_indices= mt_base_idx,
            band_names  = mt_base_names,
        )

    if gsi_idx is not None:
        reg["gsi"] = ExperimentConfig(
            key         = "gsi",
            description = f"GSI-direct top-K, {len(gsi_idx)}ch — spectral-temporal selection",
            band_indices= gsi_idx,
            band_names  = gsi_names,
        )

    if rf_idx is not None:
        reg["rf"] = ExperimentConfig(
            key         = "rf",
            description = f"RF-direct top-K, {len(rf_idx)}ch — RF importance selection",
            band_indices= rf_idx,
            band_names  = rf_names,
        )

    return reg


def expand_exp_keys(
    requested: list[str],
    registry:  dict[str, ExperimentConfig],
) -> list[str]:
    """Pass-through: all keys are concrete (no shorthands needed)."""
    return list(requested)
