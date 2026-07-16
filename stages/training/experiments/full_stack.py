"""Full-stack baseline — every date x every band, no selection."""

import logging

log = logging.getLogger(__name__)


def build_full_stack_indices(local_band_names):
    """All local channels (all dates x all S2_BAND_NAMES) = reference-year total.

    No band or date selection: upper-bound channel-count baseline against
    single_date / mt_ndvi / gsi / rf.
    """
    idx   = list(range(len(local_band_names)))
    names = list(local_band_names)
    log.info(f"full_stack: {len(idx)} channels")
    return idx, names
