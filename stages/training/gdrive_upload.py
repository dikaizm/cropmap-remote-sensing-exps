"""Google Drive checkpoint upload + OAuth token check."""

import os
import logging

from config import GDRIVE_OAUTH_TOKEN, GDRIVE_MODELS_FOLDER_ID

log = logging.getLogger(__name__)


def _check_gdrive_token() -> None:
    """Attempt to refresh the OAuth token; log warning if unavailable or expired."""
    import pickle
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        log.warning(f"GDrive token not found at {GDRIVE_OAUTH_TOKEN} — artifact upload will fail")
        return
    try:
        with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
            creds = pickle.load(f)
        if creds.expired and creds.refresh_token:
            log.info("GDrive token expired — refreshing...")
            creds.refresh(Request())
            with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
                pickle.dump(creds, f)
            log.info("GDrive token refreshed and saved.")
        elif creds.expired:
            log.warning("GDrive token expired and no refresh_token — artifact upload will fail")
        else:
            log.info("GDrive token valid.")
    except Exception as e:
        log.warning(f"GDrive token check failed ({e}) — artifact upload will fail")


def _build_drive_service():
    """Authenticate GDrive API v3 using the OAuth token."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally with:\n"
            "  python stages/process_data.py --auth\n"
            "Then copy to the server via scp."
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _get_or_create_folder(service, name, parent_id):
    """Return GDrive folder ID for `name` under `parent_id`, creating it if needed."""
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
    """Upload a single file to a GDrive folder (resumable). Skips if already exists."""
    from googleapiclient.http import MediaFileUpload

    fname  = os.path.basename(local_path)
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id)").execute()
    if result.get("files"):
        log.info(f"  GDrive: already exists — {fname}")
        return result["files"][0]["id"]

    size  = os.path.getsize(local_path)
    log.info(f"  GDrive: uploading {fname}  ({size/1e6:.0f} MB)")
    media = MediaFileUpload(local_path, mimetype="application/octet-stream", resumable=True)
    meta  = {"name": fname, "parents": [folder_id]}
    req   = service.files().create(body=meta, media_body=media, fields="id")
    resp  = None
    while resp is None:
        status, resp = req.next_chunk()
        if status:
            log.info(f"    {int(status.progress() * 100)}%")
    log.info(f"  GDrive: uploaded {fname}  (id={resp['id']})")
    return resp["id"]


def upload_models_to_gdrive(run_name, model_files):
    """
    Upload model checkpoint files to GDrive under:
      <GDRIVE_MODELS_FOLDER_ID>/runs/<run_name>/

    Creates the `runs/` and `<run_name>/` folders if they don't exist.
    Returns dict {filename: gdrive_view_link} for MLflow tag logging.
    """
    try:
        service   = _build_drive_service()
        runs_id   = _get_or_create_folder(service, "runs", GDRIVE_MODELS_FOLDER_ID)
        run_id    = _get_or_create_folder(service, run_name, runs_id)
        links = {}
        for path in model_files:
            file_id = _upload_file_gdrive(service, str(path), run_id)
            links[os.path.basename(path)] = f"https://drive.google.com/file/d/{file_id}/view"
        log.info(f"  GDrive upload complete for {run_name}")
        return links
    except Exception as e:
        log.warning(f"  GDrive upload failed ({e}) — models kept locally only")
        return {}
