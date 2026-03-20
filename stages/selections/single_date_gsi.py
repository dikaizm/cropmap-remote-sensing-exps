"""GSI band selection scoped to the peak-NDVI date (single-date experiments)."""

import logging
from pathlib import Path

from crop_mapping_pipeline.stages.experiments.base import build_local_band_map
from crop_mapping_pipeline.stages.experiments.exp_a import _find_peak_ndvi_date
from crop_mapping_pipeline.stages.selections.band_scoring.gsi.v3 import compute_band_candidates

log = logging.getLogger(__name__)


def run_single_date_gsi(
    s2_paths: list[str],
    cdl_path: str,
    force: bool = False,
) -> tuple[dict[str, list[str]], str]:
    """Score bands on peak-NDVI date using scoped GSI.

    Saves gsi_single_date_candidates.json alongside the S2 data.
    Returns (band_candidates_per_crop, peak_date).
    """
    _, _, date_to_idx, _ = build_local_band_map(s2_paths)
    peak_date = _find_peak_ndvi_date(date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)
    peak_file = s2_paths[date_to_idx[peak_date]]
    out_json  = Path(s2_paths[0]).parent / "gsi_single_date_candidates.json"

    log.info(f"single_date_gsi: peak_date={peak_date}, scoring {Path(peak_file).name}")
    band_candidates = compute_band_candidates([peak_file], cdl_path, out_json=out_json, force=force)
    log.info(f"single_date_gsi: saved → {out_json}")
    return band_candidates, peak_date
