"""Band scoring — per-crop GSI and RF importance scoring for band selection comparison.

Produces:
  select_gsi_direct_k*.json — joint spectral-temporal top-K by GSI (used by gsi experiment)
  select_rf_direct_k*.json  — joint spectral-temporal top-K by RF importance (used by rf experiment)
"""

import argparse
import json
import logging
import os
import pathlib
from pathlib import Path
import sys
from datetime import datetime
from glob import glob

_ROOT = next(_p for _p in pathlib.Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT.parent))

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
import mlflow

from crop_mapping_pipeline.utils.mlflow_utils import patch_artifact_logging
patch_artifact_logging()

from crop_mapping_pipeline.config import (
    CDL_BY_YEAR as _CDL_BY_YEAR,
    CDL_TRAIN as _CDL_TRAIN,
    CDL_CLASS_NAMES,
    FIGURES_DIR as _FIGURES_DIR,
    GSI_CANDIDATES_JSON as _GSI_CANDIDATES_JSON,
    KEEP_CLASSES,
    LOGS_DIR as _LOGS_DIR,
    MAX_BANDS_PER_CROP,
    MAX_DATES_PER_CROP,
    MLFLOW_EXPERIMENT_FEATURE,
    MLFLOW_TRACKING_URI,
    PROCESSED_DIR as _PROCESSED_DIR,
    REMAP_LUT,
    RF_IMPORTANCE_THRESH,
    RF_MAX_PIXELS,
    RF_N_ESTIMATORS,
    S2_BAND_NAMES,
    S2_NODATA,
    S2_TRAIN_DIR as _S2_TRAIN_DIR,
    SAMPLE_FRACTION,
    SELECT_GSI_DIRECT_JSON,
    SELECT_GSI_DIRECT_BANDS,
    SELECT_RF_DIRECT_JSON,
    SELECT_RF_DIRECT_BANDS,
    SELECT_TOP_K_PER_CROP,
    TEST_YEAR,
    TOP_BANDS_PER_CROP,
    TOP_DATES_PER_CROP,
    TRAIN_YEARS,
    VEGE_BANDS,
)

log = logging.getLogger(__name__)

S2_TRAIN_DIR     = _S2_TRAIN_DIR
S2_PROCESSED_DIR = S2_TRAIN_DIR   # backwards-compat alias
CDL_TRAIN        = _CDL_TRAIN
CDL_BY_YEAR      = dict(_CDL_BY_YEAR)
PROCESSED_DIR    = _PROCESSED_DIR
FIGURES_DIR      = _FIGURES_DIR
LOGS_DIR         = _LOGS_DIR
GSI_CANDIDATES_JSON = _GSI_CANDIDATES_JSON


def configure_data_dir(data_dir: str | None) -> None:
    global S2_TRAIN_DIR, S2_PROCESSED_DIR, CDL_TRAIN, CDL_BY_YEAR, PROCESSED_DIR, FIGURES_DIR, GSI_CANDIDATES_JSON

    if not data_dir:
        return

    processed = pathlib.Path(data_dir)
    PROCESSED_DIR    = processed
    S2_TRAIN_DIR     = processed / "s2" / "2024"
    S2_PROCESSED_DIR = S2_TRAIN_DIR
    CDL_TRAIN        = processed / "cdl" / "cdl_2024_study_area_filtered.tif"
    CDL_BY_YEAR      = {"2024": CDL_TRAIN}
    GSI_CANDIDATES_JSON = processed / "s2" / "2024" / "gsi_candidates.json"
    log.info(f"Data dir overridden to {processed}")


def _glob_s2_train() -> list[str]:
    """Glob S2 files from flat train/ dir, then drop low-validity dates
    (same filter as the training pipeline → standalone selection stays consistent)."""
    from crop_mapping_pipeline.stages.data.valid_dates import filter_valid_s2_dates
    files = sorted(glob(str(S2_TRAIN_DIR / "*_processed.tif")) + glob(str(S2_TRAIN_DIR / "S2H_*.tif")))
    seen  = set()
    files = [p for p in files if not (p in seen or seen.add(p))]
    valid, _ = filter_valid_s2_dates(files, cache_dir=S2_TRAIN_DIR)
    return valid


def get_train_year_inputs() -> tuple[str, list[str], str]:
    """Training data from flat train/ dir."""
    s2_files = _glob_s2_train()
    assert s2_files, f"No S2 files in {S2_TRAIN_DIR}"
    cdl_path = str(CDL_TRAIN)
    assert os.path.exists(cdl_path), f"CDL not found: {cdl_path}"
    return TRAIN_YEARS[0], s2_files, cdl_path


def get_stage1_inputs() -> list[tuple[str, list[str], str]]:
    """Training data for GSI band scoring — flat train/ dir.
    Returns [(year, s2_files, cdl_path)].
    """
    s2_files = _glob_s2_train()
    cdl_path = str(CDL_TRAIN)
    if not s2_files:
        raise FileNotFoundError(f"No S2 files in {S2_TRAIN_DIR}")
    if not os.path.exists(cdl_path):
        raise FileNotFoundError(f"CDL not found: {cdl_path}")
    return [(TRAIN_YEARS[0], s2_files, cdl_path)]


_MLFLOW_EXPERIMENT_OVERRIDE: str | None = None


def main(force: bool = False, data_dir: str = None, output_dir: str = None,
         mode: str = "select", selector: str = "gsi_direct",
         top_k_values: list[int] | None = None) -> None:
    global _MLFLOW_EXPERIMENT_OVERRIDE, KEEP_CLASSES, CDL_CLASS_NAMES
    _MLFLOW_EXPERIMENT_OVERRIDE = None

    configure_data_dir(data_dir)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if mode == "select":
        from crop_mapping_pipeline.stages.selection.gsi_selection import run_gsi_direct
        from crop_mapping_pipeline.stages.selection.feature_importance_selection import run_rf_direct
        if selector not in _DIRECT_OUTPUT_MAP:
            raise ValueError(
                f"--selector must be 'gsi_direct' or 'rf_direct', got {selector!r}"
            )
        ks = top_k_values or [SELECT_TOP_K_PER_CROP]
        fn = run_gsi_direct if selector == "gsi_direct" else run_rf_direct
        years_data = get_stage1_inputs()
        out_base = Path(output_dir) if output_dir else (Path(data_dir) if data_dir else _DIRECT_OUTPUT_MAP[selector][0].parent)
        out_base.mkdir(parents=True, exist_ok=True)
        for k in ks:
            stem     = f"select_{selector}_k{k}"
            json_out = out_base / f"{stem}.json"
            if not force and json_out.exists():
                log.info(f"  k={k}: output exists ({json_out.name}) — skipping (--force to re-run)")
                continue
            log.info(f"  Running {selector} top_k={k} ...")
            fn(years_data, top_k=k, data_dir=str(out_base), out_stem=stem)
            log.info(f"  k={k} complete → {json_out}")
        log.info(f"Direct selection ({selector}) sweep complete: k={ks}")
        return

    log.info("Band scoring complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Band scoring: GSI and RF importance for band selection comparison")
    parser.add_argument(
        "--mode",
        choices=["gsi", "select"],
        default="gsi",
        help="'gsi' runs per-crop GSI scoring; 'select' runs joint spectral-temporal direct selection.",
    )
    parser.add_argument(
        "--selector",
        choices=["gsi_direct", "rf_direct"],
        default="gsi_direct",
        help="Direct selector for --mode select.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run even if outputs exist")
    parser.add_argument("--top-k", type=int, nargs="+", default=None, metavar="K",
                        help="Top-K per crop sweep (e.g. --top-k 5 10 15 20 30)")
    parser.add_argument("--data-dir", type=str, default=None, help="Override processed data directory (S2/CDL input)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for selection output JSONs (--mode select only)")
    return parser


def configure_logging() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_DIR / f"band_scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        ],
    )


def cli(argv=None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging()
    main(force=args.force, data_dir=args.data_dir, output_dir=args.output_dir,
         mode=args.mode, selector=args.selector, top_k_values=args.top_k)


if __name__ == "__main__":
    cli()
