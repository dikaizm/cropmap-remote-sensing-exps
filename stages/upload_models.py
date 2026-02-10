"""
Upload model checkpoints to Google Drive and tag the corresponding MLflow runs.

Steps:
  1. List all runs in a given MLflow experiment.
  2. Match each run name to a subdirectory under MODELS_DIR.
  3. Upload best_model.pth / last_model.pth to GDrive.
  4. Set gdrive_best_model.pth / gdrive_last_model.pth tags on the MLflow run.

Usage:
    python stages/upload_models.py
    python stages/upload_models.py --experiment cropmap_segmentation_s2_v3
    python stages/upload_models.py --dry-run
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

log = logging.getLogger(__name__)


# ── GDrive helpers (duplicated from train_segmentation to keep this script standalone) ──

def _build_drive_service():
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    from crop_mapping_pipeline.config import GDRIVE_OAUTH_TOKEN
    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally: python stages/process_data.py --auth"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(service, name, parent_id):
    query  = (f"name='{name}' and '{parent_id}' in parents "
              f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        return result["files"][0]["id"]
    meta   = {"name": name, "mimeType": "application/vnd.google-apps.folder",
               "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def _upload_file_gdrive(service, local_path, folder_id):
    from googleapiclient.http import MediaFileUpload

    fname  = os.path.basename(local_path)
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        fid = result["files"][0]["id"]
        log.info("    GDrive: already exists — %s (id=%s)", fname, fid)
        return fid

    size  = os.path.getsize(local_path)
    log.info("    GDrive: uploading %s  (%.0f MB)", fname, size / 1e6)
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)
    meta  = {"name": fname, "parents": [folder_id]}
    req   = service.files().create(body=meta, media_body=media, fields="id")
    resp  = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            log.info("      %d%%", int(status.progress() * 100))
    log.info("    GDrive: done — %s (id=%s)", fname, resp["id"])
    return resp["id"]


# ── Main logic ────────────────────────────────────────────────────────────────

def upload_and_tag(experiment_name: str, dry_run: bool = False) -> None:
    import os
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

    import mlflow
    from mlflow.tracking import MlflowClient
    from crop_mapping_pipeline.config import (
        MLFLOW_TRACKING_URI, MODELS_DIR, GDRIVE_MODELS_FOLDER_ID,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    # ── Step 1: list runs ──────────────────────────────────────────────────────
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        log.error("Experiment '%s' not found in MLflow.", experiment_name)
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        max_results=500,
    )
    log.info("Found %d run(s) in experiment '%s'", len(runs), experiment_name)

    if not runs:
        return

    # ── Step 2: match run names → model dirs ──────────────────────────────────
    # Build a lookup: run_name → list of run objects (there may be duplicates from re-runs)
    run_map: dict[str, list] = {}
    for r in runs:
        name = r.info.run_name or ""
        run_map.setdefault(name, []).append(r)

    if not MODELS_DIR.exists():
        log.error("MODELS_DIR does not exist: %s", MODELS_DIR)
        return

    model_dirs = {d.name: d for d in MODELS_DIR.iterdir() if d.is_dir()}
    log.info("Found %d model dir(s) under %s", len(model_dirs), MODELS_DIR)

    matched = []   # list of (run, model_dir)
    unmatched_runs = []
    unmatched_dirs = set(model_dirs.keys())

    for run_name, run_list in run_map.items():
        if run_name in model_dirs:
            # If multiple runs share the same name, tag the most recent one
            latest_run = max(run_list, key=lambda r: r.info.start_time)
            matched.append((latest_run, model_dirs[run_name]))
            unmatched_dirs.discard(run_name)
        else:
            for r in run_list:
                unmatched_runs.append(r.info.run_name)

    if unmatched_runs:
        log.warning("Runs with no matching model dir (%d): %s",
                    len(unmatched_runs), unmatched_runs)
    if unmatched_dirs:
        log.warning("Model dirs with no matching MLflow run (%d): %s",
                    len(unmatched_dirs), sorted(unmatched_dirs))

    log.info("Matched %d run(s) to model dirs", len(matched))
    if not matched:
        return

    # ── Steps 3+4: upload + tag ───────────────────────────────────────────────
    if not dry_run:
        try:
            service  = _build_drive_service()
            runs_fid = _get_or_create_folder(service, "runs", GDRIVE_MODELS_FOLDER_ID)
        except Exception as e:
            log.error("GDrive auth failed: %s", e)
            return

    for run, model_dir in matched:
        model_files = [
            f for f in [model_dir / "best_model.pth", model_dir / "last_model.pth"]
            if f.exists()
        ]
        if not model_files:
            log.warning("  %s — no checkpoint files, skipping", model_dir.name)
            continue

        log.info("  %s — %s", model_dir.name, [f.name for f in model_files])

        if dry_run:
            log.info("    [dry-run] would upload to GDrive and tag run %s", run.info.run_id)
            continue

        # Upload
        try:
            run_fid = _get_or_create_folder(service, model_dir.name, runs_fid)
            links = {}
            for path in model_files:
                file_id = _upload_file_gdrive(service, str(path), run_fid)
                links[path.name] = f"https://drive.google.com/file/d/{file_id}/view"
        except Exception as e:
            log.warning("    GDrive upload failed (%s) — skipping tags for this run", e)
            continue

        # Tag the MLflow run
        for fname, link in links.items():
            tag_key = f"gdrive_{fname}"
            client.set_tag(run.info.run_id, tag_key, link)
            log.info("    tagged %s = %s", tag_key, link)

    log.info("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="Upload model checkpoints to GDrive and tag MLflow runs."
    )
    parser.add_argument(
        "--experiment", "-e",
        default="cropmap_segmentation_s2_v2",
        help="MLflow experiment name (default: cropmap_segmentation_s2_v2).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without uploading or tagging.",
    )
    args = parser.parse_args()

    upload_and_tag(experiment_name=args.experiment, dry_run=args.dry_run)
