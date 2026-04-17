"""RF band selection scoped to 4 calendar dates (multi-temporal experiments)."""

import json
import logging
from pathlib import Path

from crop_mapping_pipeline.config import VEGE_BANDS
from crop_mapping_pipeline.stages.experiments.base import build_local_band_map
from crop_mapping_pipeline.stages.experiments.exp_b import _select_phenol_dates
from crop_mapping_pipeline.stages.selections.rf_band_only import run_rf_band_only, save_rf_band_json

log = logging.getLogger(__name__)


def run_naive_mt_rf(
    s2_paths: list[str],
    cdl_path: str,
    data_dir: str | None = None,
    force: bool = False,
) -> tuple[dict[str, list[str]], dict]:
    """Score bands on 4 calendar dates using RF importance.

    Saves rf_band_naive_mt.json to data_dir (defaults to processed root).
    Returns (band_candidates_per_crop, phenol_map).
    """
    out_dir  = Path(data_dir) if data_dir else Path(s2_paths[0]).parent.parent.parent
    out_json = out_dir / "rf_band_naive_mt.json"

    _, _, date_to_idx, _ = build_local_band_map(s2_paths)
    phenol_map = _select_phenol_dates(date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)

    if out_json.exists() and not force:
        log.info(f"naive_mt_rf: cached → {out_json}")
        with open(out_json) as f:
            return json.load(f)["band_candidates_per_crop"], phenol_map

    phenol_files = [s2_paths[date_to_idx[d]] for d in phenol_map.values()]
    domain_names = [f"{b}_{d}" for d in phenol_map.values() for b in VEGE_BANDS]

    log.info(f"naive_mt_rf: phenol_map={phenol_map}")
    band_candidates = run_rf_band_only(phenol_files, cdl_path, domain_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rf_band_json(band_candidates, out_json)
    log.info(f"naive_mt_rf: saved → {out_json}")
    return band_candidates, phenol_map
