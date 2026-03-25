"""GSI band selection scoped to 4 calendar dates (multi-temporal experiments)."""

import logging
from pathlib import Path

from crop_mapping_pipeline.stages.experiments.base import build_local_band_map
from crop_mapping_pipeline.stages.experiments.exp_b import _select_phenol_dates
from crop_mapping_pipeline.stages.selections.band_scoring.gsi.v3 import compute_band_candidates

log = logging.getLogger(__name__)


def run_naive_mt_gsi(
    s2_paths: list[str],
    cdl_path: str,
    force: bool = False,
) -> tuple[dict[str, list[str]], dict]:
    """Score bands on 4 calendar dates using scoped GSI.

    Saves gsi_naive_mt_candidates.json alongside the S2 data.
    Returns (band_candidates_per_crop, phenol_map).
    """
    _, _, date_to_idx, _ = build_local_band_map(s2_paths)
    phenol_map   = _select_phenol_dates(date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)
    phenol_files = [s2_paths[date_to_idx[d]] for d in phenol_map.values()]
    out_json     = Path(s2_paths[0]).parent / "gsi_naive_mt_candidates.json"

    log.info(f"naive_mt_gsi: phenol_map={phenol_map}")
    band_candidates = compute_band_candidates(phenol_files, cdl_path, out_json=out_json, force=force)
    log.info(f"naive_mt_gsi: saved → {out_json}")
    return band_candidates, phenol_map
