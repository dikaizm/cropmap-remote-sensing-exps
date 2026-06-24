"""Shared utilities for all experiment index builders."""

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import S2_BAND_NAMES, N_BANDS_PER_DATE

import logging
log = logging.getLogger(__name__)


def parse_date(path):
    """Extract YYYYMMDD date string from a processed S2 filename.

    Expected pattern: S2H_YYYY_MM_DD[_processed].tif → 'YYYYMMDD'.
    Returns None if the filename does not match.
    """
    m = re.search(r"_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$", Path(path).name)
    if not m:
        return None
    return m.group(1).replace("_", "")


def build_local_band_map(s2_processed):
    """
    Return (local_band_names, local_band_to_idx, local_date_to_idx, mmdd_to_date)
    based on the year with the most processed files (used as reference).
    """
    by_year = {}
    for p in s2_processed:
        yr = Path(p).name.split("_")[1]
        by_year.setdefault(yr, []).append(p)

    ref_yr    = max(by_year, key=lambda y: len(by_year[y]))
    ref_files = sorted(by_year[ref_yr])

    local_band_names  = []
    local_date_to_idx = {}
    for i, p in enumerate(ref_files):
        d = parse_date(p)
        local_date_to_idx[d] = i
        local_band_names.extend([f"{b}_{d}" for b in S2_BAND_NAMES])

    local_band_to_idx = {n: i for i, n in enumerate(local_band_names)}
    available_dates   = sorted(local_date_to_idx.keys())
    mmdd_to_date      = {d[4:]: d for d in available_dates}

    log.info(
        f"Reference year={ref_yr}, {len(ref_files)} dates, "
        f"{len(local_band_names)} local channels"
    )
    return local_band_names, local_band_to_idx, local_date_to_idx, mmdd_to_date
