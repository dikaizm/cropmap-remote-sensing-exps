"""Experiment channel builders for band selection comparison study."""

from stages.training.experiments.base import (
    parse_date,
    build_local_band_map,
)
from stages.training.experiments.single_date import (
    build_single_date_indices,
    build_exp_A_indices,          # backwards-compat
)
from stages.training.experiments.mt_ndvi import (
    build_naive_multitemporal_indices,
    build_exp_B_indices,          # backwards-compat
)
from stages.training.experiments.full_stack import (
    build_full_stack_indices,
)
from stages.training.experiments.registry import (
    ExperimentConfig,
    build_registry,
    expand_exp_keys,
)

__all__ = [
    "parse_date",
    "build_local_band_map",
    "build_single_date_indices",
    "build_exp_A_indices",
    "build_naive_multitemporal_indices",
    "build_exp_B_indices",
    "build_full_stack_indices",
    "ExperimentConfig",
    "build_registry",
    "expand_exp_keys",
]
