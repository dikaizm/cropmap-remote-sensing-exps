"""Experiment channel builders for band selection comparison study."""

from crop_mapping_pipeline.stages.experiments.base import (
    parse_date,
    build_local_band_map,
)
from crop_mapping_pipeline.stages.experiments.exp_a import (
    build_single_date_indices,
    build_single_date_selected_indices,
    build_exp_A_indices,          # backwards-compat
)
from crop_mapping_pipeline.stages.experiments.exp_b import (
    build_naive_multitemporal_indices,
    build_naive_multitemporal_selected_indices,
    build_exp_B_indices,          # backwards-compat
)
from crop_mapping_pipeline.stages.experiments.registry import (
    ExperimentConfig,
    build_registry,
    expand_exp_keys,
)

__all__ = [
    "parse_date",
    "build_local_band_map",
    "build_single_date_indices",
    "build_single_date_selected_indices",
    "build_exp_A_indices",
    "build_naive_multitemporal_indices",
    "build_naive_multitemporal_selected_indices",
    "build_exp_B_indices",
    "ExperimentConfig",
    "build_registry",
    "expand_exp_keys",
]
