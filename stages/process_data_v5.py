"""
Stage 0.5b (v5) — Process single-file-per-date GEE S2 exports.

Unlike v2, GEE exports one file per date (no tile splitting), so no merge step.
Pipeline per date:
    raw S2H_{year}_{YYYY_MM_DD}.tif
        → assign NoData (-9999, float32)
        → DEFLATE+predictor=3 compression
        → overviews/pyramids
        → upload to processed_v3/s2/{year}/
        → delete raw

Checks processed_v3 on GDrive before downloading raw files — only downloads
and processes dates that are missing from processed_v3.

Usage:
    python stages/process_data_v5.py --years 2022 2023
    python stages/process_data_v5.py --years 2022 --skip-upload
    python stages/process_data_v5.py --years 2022 --overwrite
    python stages/process_data_v5.py --auth
"""

import os
import re
import sys
import logging
import argparse
import pathlib
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR,
    S2_NODATA, KEEP_CLASSES,
    GDRIVE_OAUTH_TOKEN,
)
from crop_mapping_pipeline.utils.constants import USDA_CDL_NAMES
from crop_mapping_pipeline.utils.label import label_filtering, majority_filter_labels

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]

_FILE_RE = re.compile(r"^(S2H_\d{4}_\d{4}_\d{2}_\d{2})\.tif$")

_TIFF_MAGIC = {
    b"II\x2a\x00", b"MM\x00\x2a",
    b"II\x2b\x00", b"MM\x00\x2b",
}


# ── Raw file listing ─────────────────────────────────────────────────────────────

def list_raw_files(raw_dir: Path, year: str) -> dict:
    """Return {date_key: path} for all raw S2 files in raw_dir."""
    files = {}
    for p in sorted(raw_dir.glob(f"S2H_{year}_*.tif")):
        m = _FILE_RE.match(p.name)
        if m:
            files[m.group(1)] = p
    if not files:
        log.warning("  No raw files found in %s for year=%s", raw_dir, year)
    else:
        log.info("  Found %d raw file(s) in %s", len(files), raw_dir)
    return files


# ── Valid-data check ─────────────────────────────────────────────────────────────

def _is_valid_tiff(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) in _TIFF_MAGIC
    except OSError:
        return False


def _has_valid_data(path: str, min_valid_frac: float = 0.01,
                    sample_size: int = 1024) -> bool:
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        row  = max(0, (h - sample_size) // 2)
        col  = max(0, (w - sample_size) // 2)
        ph   = min(sample_size, h)
        pw   = min(sample_size, w)
        data = src.read(window=rasterio.windows.Window(col, row, pw, ph)).astype(np.float32)
    valid = np.all((data > 0) & np.isfinite(data), axis=0)
    frac  = valid.sum() / valid.size
    log.info("  Valid pixel fraction (sample): %.2f%%", frac * 100)
    return frac >= min_valid_frac


# ── NoData assignment + compression + pyramids ──────────────────────────────────

def assign_nodata(in_path: str, out_path: str, overwrite: bool = False) -> str:
    """
    Assign NoData (negative/NaN/Inf → S2_NODATA), cast to float32,
    apply DEFLATE+predictor=3 compression, build overviews.
    Band-by-band to avoid loading full raster into RAM.
    """
    out = Path(out_path)
    if out.exists() and not overwrite:
        log.info("  Already processed: %s", out.name)
        return out_path

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.tif")

    try:
        with rasterio.open(in_path) as src:
            profile = src.profile.copy()
            profile.update(
                dtype      = "float32",
                nodata     = S2_NODATA,
                compress   = "deflate",
                predictor  = 3,
                tiled      = True,
                blockxsize = 256,
                blockysize = 256,
            )
            total_invalid = 0
            with rasterio.open(tmp, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    data              = src.read(band).astype(np.float32)
                    invalid           = (data < 0) | np.isnan(data) | np.isinf(data)
                    data[invalid]     = S2_NODATA
                    total_invalid    += int(invalid.sum())
                    dst.write(data, band)

        with rasterio.open(tmp, "r+") as dst:
            dst.build_overviews([4, 8, 16, 32], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

        tmp.rename(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    log.info("  Processed: %s  (invalid_px=%s)", out.name, f"{total_invalid:,}")
    return out_path


# ── CDL processing ───────────────────────────────────────────────────────────────

def process_cdl(cdl_raw_path: str, s2_ref_path: str,
                out_reprojected: str, out_filtered: str,
                overwrite: bool = False,
                majority_kernel: int = 3) -> None:
    if Path(out_reprojected).exists() and not overwrite:
        log.info("  CDL reprojected already exists: %s", Path(out_reprojected).name)
    else:
        log.info("  Reprojecting CDL → %s", Path(out_reprojected).name)
        with rasterio.open(s2_ref_path) as s2_ref:
            target_crs, target_transform = s2_ref.crs, s2_ref.transform
            target_width, target_height  = s2_ref.width, s2_ref.height

        with rasterio.open(cdl_raw_path) as cdl_src:
            dst_data = np.zeros((1, target_height, target_width), dtype=np.uint8)
            reproject(
                source        = rasterio.band(cdl_src, 1),
                destination   = dst_data,
                src_transform = cdl_src.transform,
                src_crs       = cdl_src.crs,
                dst_transform = target_transform,
                dst_crs       = target_crs,
                resampling    = Resampling.nearest,
            )

        profile = {
            "driver": "GTiff", "dtype": "uint8", "nodata": 0,
            "width": target_width, "height": target_height, "count": 1,
            "crs": target_crs, "transform": target_transform,
            "compress": "lzw",
        }
        Path(out_reprojected).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_reprojected, "w", **profile) as dst:
            dst.write(dst_data)

    if Path(out_filtered).exists() and not overwrite:
        log.info("  CDL filtered already exists: %s", Path(out_filtered).name)
    else:
        log.info("  Filtering CDL → %d classes", len(KEEP_CLASSES))
        _tmp_filtered = str(out_filtered) + ".tmp.tif"
        label_filtering(
            in_path      = out_reprojected,
            out_path     = _tmp_filtered,
            keep_classes = KEEP_CLASSES,
        )
        if majority_kernel and majority_kernel > 1:
            log.info("  Applying majority filter (k=%d) to CDL labels", majority_kernel)
            majority_filter_labels(_tmp_filtered, out_filtered, kernel_size=majority_kernel)
            Path(_tmp_filtered).unlink(missing_ok=True)
        else:
            Path(_tmp_filtered).rename(out_filtered)
        log.info("  CDL filtered: %s", Path(out_filtered).name)


# ── Google Drive ─────────────────────────────────────────────────────────────────

def _build_drive_service():
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally:  python process_data_v5.py --auth"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_subfolder(parent_id: str, name: str, service) -> str:
    query   = (f"name='{name}' and '{parent_id}' in parents "
               f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    result  = service.files().list(q=query, fields="files(id,name)").execute()
    folders = result.get("files", [])
    if folders:
        return folders[0]["id"]
    meta   = {"name": name, "mimeType": "application/vnd.google-apps.folder",
               "parents": [parent_id]}
    folder = service.files().create(body=meta, fields="id").execute()
    log.info("  Created GDrive subfolder: %s", name)
    return folder["id"]


def list_gdrive_processed(v3_folder_id: str, yr: str, service) -> set:
    """Return filenames already in processed_v3/s2/{yr}/."""
    try:
        s2_sub = get_or_create_subfolder(v3_folder_id, "s2", service)
        yr_sub = get_or_create_subfolder(s2_sub, yr, service)
        names, page_token = set(), None
        while True:
            kwargs = dict(
                q        = f"'{yr_sub}' in parents and trashed=false",
                fields   = "nextPageToken, files(name)",
                pageSize = 1000,
            )
            if page_token:
                kwargs["pageToken"] = page_token
            result     = service.files().list(**kwargs).execute()
            names     |= {f["name"] for f in result.get("files", [])}
            page_token = result.get("nextPageToken")
            if not page_token:
                break
        log.info("  GDrive processed_v3/s2/%s/ has %d file(s)", yr, len(names))
        return names
    except Exception as exc:
        log.warning("  Could not list processed_v3/s2/%s: %s", yr, exc)
        return set()


def upload_file(local_path: str, folder_id: str, service,
                overwrite: bool = False) -> str:
    from googleapiclient.http import MediaFileUpload

    fname   = Path(local_path).name
    query   = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result  = service.files().list(q=query, fields="files(id,name)").execute()
    existing = result.get("files", [])
    size    = os.path.getsize(local_path)
    media   = MediaFileUpload(local_path, mimetype="image/tiff", resumable=True)

    if existing and overwrite:
        log.info("  Replacing on GDrive: %s  (%.0f MB)", fname, size / 1e6)
        request = service.files().update(
            fileId=existing[0]["id"], media_body=media, fields="id"
        )
    elif existing:
        log.info("  Already on GDrive: %s", fname)
        return existing[0]["id"]
    else:
        log.info("  Uploading: %s  (%.0f MB)", fname, size / 1e6)
        meta    = {"name": fname, "parents": [folder_id]}
        request = service.files().create(body=meta, media_body=media, fields="id")

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            log.info("    %d%%", int(status.progress() * 100))
    log.info("  Uploaded: %s", fname)
    return response.get("id")


def delete_files(paths: list, label: str = "raw") -> None:
    freed = 0
    for p in paths:
        if Path(p).exists():
            freed += os.path.getsize(p)
            os.remove(p)
            log.info("  Deleted %s: %s", label, Path(p).name)
    log.info("  Freed: %.2f GB", freed / 1e9)


# ── Shutdown ─────────────────────────────────────────────────────────────────────

def _schedule_shutdown(delay_min: int = 8) -> None:
    import time, urllib.request, urllib.error, json

    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if pod_id and api_key:
        log.warning("RunPod pod %s will stop in %d minutes.", pod_id, delay_min)
        time.sleep(delay_min * 60)
        query = (f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) '
                 f'{{ id desiredStatus }} }}"}}')
        req   = urllib.request.Request(
            "https://api.runpod.io/graphql",
            data    = query.encode(),
            headers = {"Content-Type": "application/json",
                       "Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                log.info("Pod stop: %s", json.loads(resp.read()))
        except urllib.error.URLError as e:
            log.error("Failed to stop pod: %s", e)
    else:
        log.warning("VPS SHUTDOWN in %d minutes. Cancel: sudo shutdown -c", delay_min)
        try:
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_min}"], check=True)
        except Exception as e:
            log.error("Shutdown failed: %s", e)


# ── Pipeline ──────────────────────────────────────────────────────────────────────

_SENTINEL = object()


def _pipeline_year(
    raw_files      : dict,    # {date_key: raw_path}
    yr             : str,
    s2_out_dir     : Path,
    skip_upload    : bool = False,
    skip_delete    : bool = False,
    overwrite      : bool = False,
    process_workers: int  = 2,
    upload_workers : int  = 1,
    s2_folder_ids  : dict = None,
    cdl_folder_id  : str  = None,
) -> tuple:
    """
    2-stage concurrent pipeline per date:
      Stage 1 (process_workers threads): assign_nodata
      Stage 2 (upload_workers threads):  upload to GDrive → delete raw
    Returns (processed_paths, s2_ref_path).
    """
    s2_out_dir.mkdir(parents=True, exist_ok=True)

    upload_q: Queue         = Queue(maxsize=process_workers + 2)
    processed_paths: list   = []
    s2_ref_path: list       = [None]
    lock                    = threading.Lock()
    errors: list            = []

    # ── Stage 1: assign nodata ────────────────────────────────────────────────
    def _process_date(date_key: str, raw_path: Path) -> None:
        processed_path = s2_out_dir / f"{date_key}_processed.tif"
        try:
            if not _has_valid_data(str(raw_path)):
                log.warning("[%s] No valid data — skipped", date_key)
                return

            assign_nodata(str(raw_path), str(processed_path), overwrite=overwrite)

            with lock:
                processed_paths.append(str(processed_path))
                if s2_ref_path[0] is None:
                    s2_ref_path[0] = str(processed_path)

            upload_q.put((date_key, str(raw_path), str(processed_path)))

        except Exception as exc:
            log.error("[%s] Process error: %s", date_key, exc)
            errors.append(f"{date_key}: {exc}")

    # ── Stage 2: upload + delete ──────────────────────────────────────────────
    def _upload_worker() -> None:
        service = None
        if not skip_upload and s2_folder_ids:
            try:
                service = _build_drive_service()
            except Exception as exc:
                log.error("GDrive auth failed: %s", exc)

        parent_folder = (s2_folder_ids or {}).get(yr, "")
        s2_folder     = ""
        if service and parent_folder:
            try:
                s2_sub    = get_or_create_subfolder(parent_folder, "s2", service)
                s2_folder = get_or_create_subfolder(s2_sub, yr, service)
            except Exception as exc:
                log.error("Failed to get/create s2/%s subfolder: %s", yr, exc)

        while True:
            item = upload_q.get()
            if item is _SENTINEL:
                upload_q.task_done()
                break

            date_key, raw_path, processed_path = item
            try:
                if service and s2_folder:
                    upload_file(processed_path, s2_folder, service)
                elif not skip_upload:
                    log.warning("[%s] No GDrive folder — upload skipped", date_key)

                if not skip_delete:
                    delete_files([raw_path], label="raw")
            except Exception as exc:
                log.error("[%s] Upload/cleanup error: %s", date_key, exc)
                errors.append(f"{date_key} upload: {exc}")
            finally:
                upload_q.task_done()

    upload_threads = [
        threading.Thread(target=_upload_worker, daemon=True, name=f"upload-{i}")
        for i in range(upload_workers)
    ]
    for t in upload_threads:
        t.start()

    log.info("[%s] Pipeline start — %d dates, %d process workers, %d upload workers",
             yr, len(raw_files), process_workers, upload_workers)

    with ThreadPoolExecutor(max_workers=process_workers, thread_name_prefix="proc") as pool:
        futures = {
            pool.submit(_process_date, dk, path): dk
            for dk, path in raw_files.items()
        }
        for fut in as_completed(futures):
            dk  = futures[fut]
            exc = fut.exception()
            if exc:
                log.error("[%s] Uncaught exception: %s", dk, exc)

    for _ in upload_threads:
        upload_q.put(_SENTINEL)
    for t in upload_threads:
        t.join()

    if errors:
        log.warning("[%s] Pipeline finished with %d error(s):", yr, len(errors))
        for e in errors:
            log.warning("  %s", e)
    else:
        log.info("[%s] Pipeline done — %d date(s) processed", yr, len(processed_paths))

    return processed_paths, s2_ref_path[0]


# ── Main ──────────────────────────────────────────────────────────────────────────

def main(
    years           : list = None,
    raw_s2_dir      : str  = None,
    raw_cdl_dir     : str  = None,
    data_dir        : str  = None,
    s2_folder_ids   : dict = None,
    cdl_folder_id   : str  = None,
    skip_upload     : bool = False,
    skip_delete     : bool = False,
    shutdown        : bool = False,
    overwrite       : bool = False,
    process_workers : int  = 2,
    upload_workers  : int  = 1,
    download_workers: int  = 2,
    cdl_only        : bool = False,
) -> None:
    global S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR

    if data_dir:
        processed        = pathlib.Path(data_dir)
        PROCESSED_DIR    = processed
        S2_PROCESSED_DIR = processed / "s2"
        CDL_BY_YEAR      = {
            yr: processed / "cdl" / f"cdl_{yr}_study_area_filtered.tif"
            for yr in ALL_YEARS
        }

    years = years or ALL_YEARS

    from crop_mapping_pipeline.config import (
        GDRIVE_PROCESSED_V5_FOLDER_ID, GDRIVE_PROCESSED_CDL_FOLDER_ID,
        GDRIVE_RAW_S2_V5_FOLDER_ID as GDRIVE_RAW_S2_V2_FOLDER_ID,
    )

    for yr in years:
        log.info("=" * 60)
        log.info("Year: %s", yr)
        log.info("=" * 60)

        s2_raw_dir = (
            pathlib.Path(raw_s2_dir) / yr if raw_s2_dir
            else _ROOT / "data" / "raw" / "s2" / yr
        )

        _s2_ids = s2_folder_ids or {yr: GDRIVE_PROCESSED_V5_FOLDER_ID for yr in ALL_YEARS}
        _cdl_id = cdl_folder_id or GDRIVE_PROCESSED_CDL_FOLDER_ID
        _v3     = GDRIVE_PROCESSED_V5_FOLDER_ID

        # ── CDL-only mode: skip all S2 steps, use existing processed S2 as grid ref ──
        if cdl_only:
            existing = sorted((S2_PROCESSED_DIR / yr).glob("*_processed.tif"))
            if not existing:
                log.error("  --cdl-only: no processed S2 in %s — cannot determine grid", S2_PROCESSED_DIR / yr)
                continue
            s2_ref_path = str(existing[0])
            log.info("  --cdl-only: grid ref = %s", pathlib.Path(s2_ref_path).name)
        else:
            # ── Step 1: Check processed_v3 — what's already uploaded ─────────
            already_uploaded: set = set()
            if not overwrite and not skip_upload:
                try:
                    _svc = _build_drive_service()
                    already_uploaded = list_gdrive_processed(_v3, yr, _svc)
                except Exception as exc:
                    log.warning("  GDrive pre-check failed (%s) — processing all local dates", exc)

            # ── Step 2: Scan local raw files ──────────────────────────────────
            local_files = list_raw_files(s2_raw_dir, yr)

            # ── Step 3: Filter to dates still needed ─────────────────────────
            needed_local = {
                dk: path for dk, path in local_files.items()
                if f"{dk}_processed.tif" not in already_uploaded
            }
            n_skipped = len(local_files) - len(needed_local)
            if n_skipped:
                log.info("  Skipping %d local date(s) already in processed_v3/s2/%s/",
                         n_skipped, yr)

            # ── Step 4: Download missing dates from GDrive raw ────────────────
            try:
                from crop_mapping_pipeline.stages.fetch_data_v5 import (
                    list_dates_by_year, download_date_keys,
                )
                gdrive_date_keys = set(
                    list_dates_by_year(GDRIVE_RAW_S2_V2_FOLDER_ID, years=[yr]).get(yr, [])
                )
                needed_gdrive = {
                    dk for dk in gdrive_date_keys
                    if f"{dk}_processed.tif" not in already_uploaded
                }
                to_download = needed_gdrive - set(local_files.keys())
                if to_download:
                    log.info("  Downloading %d missing date(s) from GDrive raw...", len(to_download))
                    download_date_keys(
                        folder_id  = GDRIVE_RAW_S2_V2_FOLDER_ID,
                        output_dir = str(s2_raw_dir.parent),
                        date_keys  = list(to_download),
                        workers    = download_workers,
                    )
                    local_files  = list_raw_files(s2_raw_dir, yr)
                    needed_local = {
                        dk: path for dk, path in local_files.items()
                        if f"{dk}_processed.tif" not in already_uploaded
                    }
            except Exception as exc:
                log.warning("  GDrive raw listing/download failed (%s) — using local files only", exc)

            if not needed_local:
                log.info("  All dates for year %s already in processed_v3 — skipping", yr)
                continue

            s2_out_dir = S2_PROCESSED_DIR / yr

            all_processed, s2_ref_path = _pipeline_year(
                raw_files       = needed_local,
                yr              = yr,
                s2_out_dir      = s2_out_dir,
                skip_upload     = skip_upload,
                skip_delete     = skip_delete,
                overwrite       = overwrite,
                process_workers = process_workers,
                upload_workers  = upload_workers,
                s2_folder_ids   = _s2_ids,
                cdl_folder_id   = _cdl_id,
            )
            log.info("  Processed %d date(s) for year %s", len(all_processed), yr)

        # ── CDL processing ────────────────────────────────────────────────────
        from glob import glob as _glob
        from crop_mapping_pipeline.config import GDRIVE_RAW_CDL_FOLDER_ID
        cdl_dir = (pathlib.Path(raw_cdl_dir) if raw_cdl_dir
                   else _ROOT / "data" / "raw" / "cdl")
        cdl_subdir = cdl_dir / f"{yr}_30m_cdls"
        cdl_raw = next((_glob(str(cdl_subdir / "*.tif")).__iter__()), None)

        # Auto-download CDL from USDA NASS — stream-extract to avoid storing zip on disk
        if not cdl_raw:
            from crop_mapping_pipeline.config import CDL_DOWNLOAD_URLS
            url = CDL_DOWNLOAD_URLS.get(yr)
            if url:
                log.info("  Raw CDL for %s not found — stream-downloading from USDA NASS...", yr)
                try:
                    import urllib.request, zipfile, io
                    cdl_subdir.mkdir(parents=True, exist_ok=True)
                    tif_dest = cdl_subdir / f"{yr}_30m_cdls.tif"
                    if not tif_dest.exists():
                        log.info("  Streaming %s (no temp zip — direct extract)...", url)
                        with urllib.request.urlopen(url) as resp:
                            total = int(resp.headers.get("Content-Length", 0))
                            buf = io.BytesIO()
                            downloaded = 0
                            chunk = 8 * 1024 * 1024  # 8 MB chunks
                            while True:
                                data = resp.read(chunk)
                                if not data:
                                    break
                                buf.write(data)
                                downloaded += len(data)
                                if total:
                                    log.info("    CDL buffer: %d%%",
                                             downloaded * 100 // total)
                        log.info("  Extracting from buffer (%.0f MB)...",
                                 buf.tell() / 1e6)
                        buf.seek(0)
                        with zipfile.ZipFile(buf) as zf:
                            tif_members = [m for m in zf.namelist()
                                           if m.endswith(".tif")]
                            if not tif_members:
                                raise RuntimeError("No TIF in ZIP")
                            for member in tif_members:
                                log.info("  Extracting: %s", member)
                                src = zf.open(member)
                                with open(tif_dest, "wb") as dst:
                                    import shutil
                                    shutil.copyfileobj(src, dst)
                        log.info("  CDL extracted: %.0f MB",
                                 tif_dest.stat().st_size / 1e6)
                    else:
                        log.info("  CDL TIF already present: %s", tif_dest.name)
                    cdl_raw = str(tif_dest) if tif_dest.exists() else None
                except Exception as exc:
                    log.error("  CDL download/extract failed: %s", exc)
            else:
                log.warning("  No download URL configured for CDL year %s", yr)

        cdl_filtered = None
        cdl_reprojected = None
        if not cdl_raw:
            log.warning("  Raw CDL for %s not found — skipping CDL processing", yr)
        elif s2_ref_path is None:
            log.warning("  No processed S2 reference — skipping CDL processing")
        else:
            cdl_out_dir     = S2_PROCESSED_DIR.parent / "cdl"
            cdl_reprojected = str(cdl_out_dir / f"cdl_{yr}_study_area.tif")
            cdl_filtered    = str(cdl_out_dir / f"cdl_{yr}_study_area_filtered.tif")
            process_cdl(cdl_raw, s2_ref_path, cdl_reprojected, cdl_filtered,
                        overwrite=overwrite)
            # Delete raw CDL after processing — full CONUS TIF is large (~8 GB)
            if not skip_delete and pathlib.Path(cdl_raw).exists():
                freed = pathlib.Path(cdl_raw).stat().st_size
                pathlib.Path(cdl_raw).unlink()
                log.info("  Deleted raw CDL TIF (freed %.1f GB)", freed / 1e9)

        if not skip_upload and _s2_ids:
            service    = _build_drive_service()
            v3_parent  = next(iter(_s2_ids.values()))
            cdl_folder = get_or_create_subfolder(v3_parent, "cdl", service)
            if cdl_reprojected and pathlib.Path(cdl_reprojected).exists():
                upload_file(cdl_reprojected, cdl_folder, service, overwrite=overwrite)
            if cdl_filtered and pathlib.Path(cdl_filtered).exists():
                upload_file(cdl_filtered, cdl_folder, service, overwrite=overwrite)

        log.info("Year %s done.\n", yr)

    if shutdown:
        _schedule_shutdown(delay_min=8)


def generate_oauth_token():
    import pickle
    from google_auth_oauthlib.flow import InstalledAppFlow
    from crop_mapping_pipeline.config import GDRIVE_OAUTH_SECRET

    flow  = InstalledAppFlow.from_client_secrets_file(
        str(GDRIVE_OAUTH_SECRET),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds = flow.run_local_server(port=0)
    with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
        pickle.dump(creds, f)
    print(f"Token saved: {GDRIVE_OAUTH_TOKEN}")


# ── CLI ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process single-file-per-date GEE S2 exports: NoData → upload."
    )
    parser.add_argument("--years", nargs="+", default=None, choices=ALL_YEARS)
    parser.add_argument("--raw-s2-dir", default=None)
    parser.add_argument("--raw-cdl-dir", default=None)
    parser.add_argument("--data-dir", default=None,
                        help="Override processed output directory.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--skip-delete", action="store_true")
    parser.add_argument("--cdl-only", action="store_true",
                        help="Skip S2 processing; reproject+filter CDL only, using existing processed S2 as grid ref")
    parser.add_argument("--shutdown", action="store_true")
    parser.add_argument("--process-workers", type=int, default=2)
    parser.add_argument("--upload-workers", type=int, default=1)
    parser.add_argument("--download-workers", type=int, default=2)
    parser.add_argument("--auth", action="store_true")
    args = parser.parse_args()

    if args.auth:
        generate_oauth_token()
        sys.exit(0)

    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s %(levelname)s %(message)s",
        handlers= [logging.StreamHandler()],
    )

    main(
        years            = args.years,
        raw_s2_dir       = args.raw_s2_dir,
        raw_cdl_dir      = args.raw_cdl_dir,
        data_dir         = args.data_dir,
        skip_upload      = args.skip_upload,
        skip_delete      = args.skip_delete,
        cdl_only         = args.cdl_only,
        shutdown         = args.shutdown,
        overwrite        = args.overwrite,
        process_workers  = args.process_workers,
        upload_workers   = args.upload_workers,
        download_workers = args.download_workers,
    )
