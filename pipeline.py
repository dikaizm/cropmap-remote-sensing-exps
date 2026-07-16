"""
Pipeline orchestrator — runs all steps end-to-end.

Steps:
  fetch    — download processed S2 + CDL from Google Drive
  feature  — feature selection: GSI + RF, normalized score >= 0.5
  train    — train segmentation models for band selection comparison (train_segmentation.py)
  all      — run fetch + feature + train in order

Raw S2/CDL processing is a standalone step — run stages/process_data.py directly.

Usage:
    python pipeline.py --stages fetch feature train
    python pipeline.py --stages train
    python pipeline.py --stages all --shutdown
    python pipeline.py --force
    python pipeline.py --data-dir /mnt/data

Logs are written to logs/pipeline_YYYYMMDD_HHMMSS.log in addition to stdout.
"""

import sys
import os
import subprocess
import argparse
import logging
import time
import traceback
from datetime import datetime
from pathlib import Path

_ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(_ROOT / ".env")

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
import mlflow

from config import (
    LOGS_DIR, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_PIPELINE,
)

log = logging.getLogger(__name__)

VALID_STAGES = ["fetch-processed", "fetch", "feature", "train", "all"]


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_fetch(force=False, data_dir=None, years=None, **_):
    """fetch / fetch-processed — download processed S2 + CDL from Google Drive."""
    log.info("=" * 60)
    log.info("FETCH PROCESSED — Download processed data from Google Drive")
    log.info("=" * 60)
    import os
    from googleapiclient.http import MediaIoBaseDownload
    from stages.data.fetch_data import _build_drive_service, list_folder
    from config import (
        GDRIVE_PROCESSED_S2_FOLDER_IDS, GDRIVE_PROCESSED_CDL_FOLDER_ID,
        S2_PROCESSED_DIR, CDL_DIR,
    )

    service     = _build_drive_service()
    dl_years    = years or ["2022", "2023", "2024"]

    def _dl_folder(folder_id, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        name_to_id = list_folder(folder_id)
        for fname, fid in sorted(name_to_id.items()):
            out_path = os.path.join(out_dir, fname)
            if not force and os.path.exists(out_path):
                log.info("  Already exists — skip: %s", fname)
                continue
            log.info("  Downloading %s ...", fname)
            req  = service.files().get_media(fileId=fid)
            with open(out_path, "wb") as fh:
                dl = MediaIoBaseDownload(fh, req, chunksize=50 * 1024 * 1024)
                done = False
                while not done:
                    status, done = dl.next_chunk()
                    if status:
                        log.info("    %s: %d%%", fname, int(status.progress() * 100))

    for yr in dl_years:
        folder_id = GDRIVE_PROCESSED_S2_FOLDER_IDS.get(yr)
        if not folder_id:
            log.warning("No processed S2 GDrive folder for year %s — skipping", yr)
            continue
        log.info("Fetching processed S2 for %s ...", yr)
        _dl_folder(folder_id, str(S2_PROCESSED_DIR / yr))

    log.info("Fetching CDL ...")
    _dl_folder(GDRIVE_PROCESSED_CDL_FOLDER_ID, str(CDL_DIR))


def run_feature(force=False, data_dir=None):
    """Feature selection — GSI + RF, normalized-score threshold >= 0.5 (Wei 2023)."""
    log.info("=" * 60)
    log.info("FEATURE SELECTION — GSI + RF (normalized score >= 0.5)")
    log.info("=" * 60)
    from stages.selection.band_scoring import (
        configure_data_dir, get_stage1_inputs,
    )
    from stages.selection.gsi_selection import run_gsi_direct
    from stages.selection.feature_importance_selection import run_rf_direct
    from config import PROCESSED_DIR

    configure_data_dir(data_dir)
    years_data = get_stage1_inputs()
    out_base   = data_dir or str(PROCESSED_DIR)
    run_gsi_direct(years_data, score_threshold=0.5, data_dir=out_base, out_stem="select_gsi_direct_s0.5")
    run_rf_direct(years_data,  score_threshold=0.5, data_dir=out_base, out_stem="select_rf_direct_s0.5")


def run_train(force=False, data_dir=None):
    """Train segmentation models for band selection comparison."""
    log.info("=" * 60)
    log.info("TRAINING — Band selection comparison")
    log.info("=" * 60)
    from stages.training.train_segmentation import main as train_main
    train_main(force=force, data_dir=data_dir)


# ── Pipeline runner ───────────────────────────────────────────────────────────

STAGE_FNS = {
    "fetch-processed": run_fetch,
    "fetch":          run_fetch,      # alias for fetch-processed
    "feature":        run_feature,
    "train":          run_train,
}


def run_pipeline(stages, force=False, data_dir=None, years=None, log_file=None):
    """Execute each stage in order, recording timing and errors."""
    if "all" in stages:
        stages = ["fetch", "feature", "train"]

    results = {}
    pipeline_start = time.time()

    for stage in stages:
        fn = STAGE_FNS.get(stage)
        if fn is None:
            log.error(f"Unknown stage: {stage!r}  — skipping")
            continue

        t0 = time.time()
        log.info(f"\n{'─' * 60}")
        log.info(f"Starting stage: {stage}")
        log.info(f"{'─' * 60}")

        try:
            if stage in ("fetch", "fetch-processed"):
                fn(force=force, data_dir=data_dir, years=years)
            else:
                fn(force=force, data_dir=data_dir)
            elapsed        = time.time() - t0
            results[stage] = {"status": "ok", "elapsed_s": round(elapsed, 1)}
            log.info(f"Stage '{stage}' completed in {elapsed:.1f}s")
        except Exception:
            elapsed        = time.time() - t0
            results[stage] = {"status": "error", "elapsed_s": round(elapsed, 1)}
            log.error(f"Stage '{stage}' FAILED after {elapsed:.1f}s")
            log.error(traceback.format_exc())

    total = time.time() - pipeline_start

    log.info("\n" + "=" * 60)
    log.info("PIPELINE SUMMARY")
    log.info("=" * 60)
    any_error = False
    for stage, r in results.items():
        status = "OK" if r["status"] == "ok" else "ERROR"
        log.info(f"  {stage:10s}  {status}  ({r['elapsed_s']}s)")
        if r["status"] != "ok":
            any_error = True
    log.info(f"\nTotal wall time: {total / 60:.1f} min")

    if any_error:
        log.error("One or more stages failed — check logs above")
    else:
        log.info("All stages completed successfully")

    _upload_log(stages, results, total, log_file, any_error)

    if any_error:
        sys.exit(1)


def _schedule_shutdown(delay_min: int = 8) -> None:
    """
    Stop the pod/server after `delay_min` minutes.
    - RunPod: uses RunPod API (requires RUNPOD_API_KEY env var)
    - Other Linux VPS: falls back to `sudo shutdown -h`
    """
    import urllib.request, urllib.error, json

    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if pod_id and api_key:
        log.warning("=" * 60)
        log.warning(f"RunPod pod {pod_id} will stop in {delay_min} minutes.")
        log.warning("=" * 60)
        time.sleep(delay_min * 60)

        query = f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) {{ id desiredStatus }} }}"}}'
        req   = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data    = query.encode(),
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
                log.info(f"Pod stop response: {result}")
        except urllib.error.URLError as e:
            log.error(f"Failed to stop pod via RunPod API: {e}")
    else:
        log.warning("=" * 60)
        log.warning(f"SERVER SHUTDOWN in {delay_min} minutes.")
        log.warning("Cancel with:  sudo shutdown -c")
        log.warning("=" * 60)
        try:
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_min}"], check=True)
        except Exception as e:
            log.error(f"Failed to schedule shutdown: {e}")


def _upload_log(stages, results, total_s, log_file, any_error):
    """Upload the pipeline log file + summary metrics to MLflow."""
    try:
        for handler in logging.root.handlers:
            handler.flush()

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PIPELINE)

        ts           = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name     = f"pipeline_{'_'.join(stages)}_{ts}"
        final_status = "FAILED" if any_error else "FINISHED"

        with mlflow.start_run(run_name=run_name) as pipeline_run:
            mlflow.log_params({"stages": str(stages), "n_stages": len(stages)})
            mlflow.log_metric("total_wall_time_min", round(total_s / 60, 2))
            for stage, r in results.items():
                mlflow.log_metric(f"{stage}_elapsed_s", r["elapsed_s"])
                mlflow.set_tag(f"{stage}_status", r["status"])
            mlflow.set_tag("pipeline_status", "error" if any_error else "ok")

            if log_file and Path(log_file).exists():
                mlflow.log_artifact(str(log_file), artifact_path="logs")
                log.info(f"Log uploaded to MLflow run: {pipeline_run.info.run_id}")

        mlflow.end_run(status=final_status)

    except Exception:
        log.warning(f"MLflow log upload failed (non-fatal):\n{traceback.format_exc()}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Crop-mapping pipeline orchestrator")
    parser.add_argument(
        "--stages", nargs="+", default=["all"], choices=VALID_STAGES, metavar="STAGE",
        help=f"Stages to run: {VALID_STAGES}  (default: all)",
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=["2022", "2023", "2024"], metavar="YEAR",
        help="Years to fetch (default: all). Used by the fetch stage.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if outputs already exist",
    )
    parser.add_argument(
        "--data-dir", default=None, metavar="PATH",
        help="Override data/processed directory (absolute path)",
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Stop the RunPod pod 8 minutes after pipeline finishes",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )

    log.info(f"Pipeline log: {log_file}")
    log.info(f"Stages: {args.stages}  years={args.years}  force={args.force}  "
             f"data_dir={args.data_dir}  shutdown={args.shutdown}")

    run_pipeline(
        stages      = args.stages,
        force       = args.force,
        data_dir    = args.data_dir,
        years       = args.years,
        log_file    = log_file,
    )

    if args.shutdown:
        _schedule_shutdown(delay_min=8)


if __name__ == "__main__":
    main()
