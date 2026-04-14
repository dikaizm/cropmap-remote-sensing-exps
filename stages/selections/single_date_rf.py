"""RF band selection scoped to the peak-NDVI date (single-date experiments)."""

import json
import logging
from pathlib import Path

from crop_mapping_pipeline.config import VEGE_BANDS
from crop_mapping_pipeline.stages.experiments.base import build_local_band_map
from crop_mapping_pipeline.stages.experiments.exp_a import _find_peak_ndvi_date
from crop_mapping_pipeline.stages.selections.rf_band_only import run_rf_band_only, save_rf_band_json

log = logging.getLogger(__name__)


def run_single_date_rf(
    s2_paths: list[str],
    cdl_path: str,
    data_dir: str | None = None,
    force: bool = False,
) -> tuple[dict[str, list[str]], str]:
    """Score bands on peak-NDVI date using RF importance.

    Saves rf_band_single_date.json to data_dir (defaults to processed root).
    Returns (band_candidates_per_crop, peak_date).
    """
    out_dir  = Path(data_dir) if data_dir else Path(s2_paths[0]).parent.parent.parent
    out_json = out_dir / "rf_band_single_date.json"

    _, _, date_to_idx, _ = build_local_band_map(s2_paths)
    peak_date = _find_peak_ndvi_date(date_to_idx, s2_paths=s2_paths, cdl_path=cdl_path)

    if out_json.exists() and not force:
        log.info(f"single_date_rf: cached → {out_json}")
        with open(out_json) as f:
            return json.load(f)["band_candidates_per_crop"], peak_date

    peak_file    = s2_paths[date_to_idx[peak_date]]
    domain_names = [f"{b}_{peak_date}" for b in VEGE_BANDS]

    log.info(f"single_date_rf: peak_date={peak_date}, scoring {Path(peak_file).name}")
    band_candidates = run_rf_band_only([peak_file], cdl_path, domain_names)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rf_band_json(band_candidates, out_json)
    log.info(f"single_date_rf: saved → {out_json}")
    return band_candidates, peak_date
