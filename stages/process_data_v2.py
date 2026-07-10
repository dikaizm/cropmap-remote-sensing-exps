"""
Stage 0.5b — Process multi-tile GEE exports for the new (larger) study area.

For large study areas GEE splits exports into tiles named:
    S2H_{year}_{YYYY_MM_DD}-{row_offset:010d}-{col_offset:010d}.tif

This script adds a merge step before the standard NoData assignment:

  For each year:
    1. Group raw tiles by date key
    2. Merge tiles → single mosaic per date  (rasterio.merge)
    3. Assign NoData (-9999, float32)         (same as process_data.py)
    4. Process CDL — reproject + clip to merged S2 grid + filter classes
    5. Upload processed files to Google Drive
    6. Delete raw tiles + temp merged files

Usage:
    python process_data_v2.py --years 2022                      # one year
    python process_data_v2.py --years 2022 2023 2024            # all years
    python process_data_v2.py --years 2022 --skip-upload        # process only
    python process_data_v2.py --years 2022 --skip-delete        # keep raw
    python process_data_v2.py --raw-s2-dir /path/to/tiles       # custom raw dir
    python process_data_v2.py --auth                            # generate OAuth token
    python process_data_v2.py --test-merge --years 2022         # merge + inspect only

Google Drive folder IDs (raw tiles):
    Set GDRIVE_RAW_S2_V2_FOLDER_IDS in config.py before downloading raw tiles.
    Processed outputs go to GDRIVE_PROCESSED_S2_V2_FOLDER_IDS.
"""

import os
import re
import sys
import logging
import argparse
import pathlib
import subprocess
import tempfile
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from queue import Queue

import numpy as np
import rasterio
import rasterio.windows
from rasterio.warp import reproject, Resampling
from dotenv import load_dotenv

_ROOT = pathlib.Path(__file__).parent.parent   # crop_mapping_pipeline/
load_dotenv(_ROOT / ".env")
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_PROCESSED_DIR, CDL_BY_YEAR, PROCESSED_DIR,
    S2_NODATA, KEEP_CLASSES,
    GDRIVE_OAUTH_TOKEN,
)
from crop_mapping_pipeline.utils.constants import USDA_CDL_NAMES
from crop_mapping_pipeline.utils.label import label_filtering

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]

# GEE tile filename pattern: S2H_{year}_{YYYY_MM_DD}-{10digits}-{10digits}.tif
_TILE_RE = re.compile(
    r"^(S2H_\d{4}_\d{4}_\d{2}_\d{2})-(\d{10})-(\d{10})\.tif$"
)


# ── Tile grouping ───────────────────────────────────────────────────────────────

def group_tiles_by_date(raw_dir: str, year: str) -> dict:
    """
    Scan raw_dir for GEE multi-tile files that match:
        S2H_{year}_{YYYY_MM_DD}-{row:010d}-{col:010d}.tif

    Returns OrderedDict  {date_key → [(row, col, path), ...]},
    each list sorted by (row_offset, col_offset) for deterministic merge order.
    date_key example: "S2H_2022_2022_01_01"
    """
    pattern = str(Path(raw_dir) / f"S2H_{year}_*.tif")
    all_files = sorted(glob(pattern))
    if not all_files:
        log.warning("  No tiles found in %s matching year=%s", raw_dir, year)
        return {}

    groups = defaultdict(list)
    unmatched = []
    for fpath in all_files:
        fname = Path(fpath).name
        m = _TILE_RE.match(fname)
        if m:
            date_key  = m.group(1)             # e.g. "S2H_2022_2022_01_01"
            row_off   = int(m.group(2))
            col_off   = int(m.group(3))
            groups[date_key].append((row_off, col_off, fpath))
        else:
            unmatched.append(fname)

    if unmatched:
        log.warning("  %d file(s) did not match tile pattern — skipped: %s",
                    len(unmatched), unmatched[:5])

    # Sort tiles within each date by (row, col)
    sorted_groups = {}
    for dk in sorted(groups):
        sorted_groups[dk] = sorted(groups[dk], key=lambda x: (x[0], x[1]))

    log.info("  Found %d date(s) with tiles (year=%s)", len(sorted_groups), year)
    return sorted_groups


# ── Merge ───────────────────────────────────────────────────────────────────────

_TIFF_MAGIC = {
    b"II\x2a\x00",  # little-endian TIFF
    b"MM\x00\x2a",  # big-endian TIFF
    b"II\x2b\x00",  # little-endian BigTIFF
    b"MM\x00\x2b",  # big-endian BigTIFF
}


def _is_valid_tiff(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) in _TIFF_MAGIC
    except OSError:
        return False


def merge_tiles(tile_paths: list, out_path: str) -> None:
    """
    Mosaic tiles via gdal.BuildVRT + gdal.Translate — streams block-by-block,
    never loads full mosaic into RAM. Produces clean libtiff-compatible GeoTIFF.

    Explicitly computes the union of all tile bounds and passes outputBounds +
    xRes/yRes to BuildVRT so GDAL never truncates the mosaic due to tile
    alignment differences (replicates rasterio.merge extent behaviour).
    """
    from osgeo import gdal
    gdal.UseExceptions()

    out = Path(out_path)
    if out.exists():
        if _is_valid_tiff(out):
            log.info("  Merged already exists: %s", out.name)
            return
        log.warning("  Corrupt merged file, re-merging: %s", out.name)
        out.unlink()

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp      = out.with_suffix(".tmp.tif")
    vrt_path = str(out.with_suffix(".vrt"))

    # Compute union of all tile bounds (read metadata only — no pixel data loaded)
    left = right = top = bottom = None
    xres = yres = None
    for p in tile_paths:
        with rasterio.open(p) as src:
            b = src.bounds
            left   = b.left   if left   is None else min(left,   b.left)
            bottom = b.bottom if bottom is None else min(bottom, b.bottom)
            right  = b.right  if right  is None else max(right,  b.right)
            top    = b.top    if top    is None else max(top,    b.top)
            if xres is None:
                xres, yres = src.res

    log.info("  Tile union bounds: (%.6f, %.6f, %.6f, %.6f)  res=(%.8f, %.8f)",
             left, bottom, right, top, xres, yres)

    try:
        vrt = gdal.BuildVRT(
            vrt_path,
            [str(p) for p in tile_paths],
            outputBounds = (left, bottom, right, top),
            xRes         = xres,
            yRes         = yres,
            resampleAlg  = "nearest",
        )
        if vrt is None:
            raise RuntimeError(f"gdal.BuildVRT failed for {out.name}")
        vrt.FlushCache()
        vrt = None

        ds = gdal.Translate(
            str(tmp),
            vrt_path,
            format          = "GTiff",
            creationOptions = [
                "COMPRESS=LZW",
                "TILED=YES",
                "BLOCKXSIZE=256",
                "BLOCKYSIZE=256",
                "BIGTIFF=IF_SAFER",
            ],
        )
        if ds is None:
            raise RuntimeError(f"gdal.Translate failed for {out.name}")
        ds.FlushCache()
        ds = None

        tmp.rename(out)
    except Exception:
        tmp.unlink(missing_ok=True)
        Path(vrt_path).unlink(missing_ok=True)
        raise
    finally:
        Path(vrt_path).unlink(missing_ok=True)

    log.info("  Merged → %s  (%d tiles)", out.name, len(tile_paths))


# ── Valid-data check ────────────────────────────────────────────────────────────

def _has_valid_data(path: str, min_valid_frac: float = 0.01,
                    sample_size: int = 1024) -> bool:
    """
    Sample a centre window to check valid pixel fraction — avoids loading full raster.
    Valid = positive, finite, non-zero across all bands.
    """
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


# ── NoData assignment ───────────────────────────────────────────────────────────

def assign_nodata(in_path: str, out_path: str, overwrite: bool = False) -> str:
    """
    Assign NoData (negative / NaN / Inf → S2_NODATA) and cast to float32.
    Processes band-by-band to avoid loading full raster into RAM.
    Skips if out_path already exists (unless overwrite=True).
    Returns out_path.
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
            profile.update(dtype="float32", nodata=S2_NODATA,
                           compress="deflate", predictor=3,
                           tiled=True, blockxsize=256, blockysize=256)
            total_invalid = 0
            with rasterio.open(tmp, "w", **profile) as dst:
                for band in range(1, src.count + 1):
                    data                     = src.read(band).astype(np.float32)
                    invalid                  = (data < 0) | np.isnan(data) | np.isinf(data)
                    data[invalid]            = S2_NODATA
                    total_invalid           += int(invalid.sum())
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


# ── CDL processing ──────────────────────────────────────────────────────────────

def download_cdl_usda(year: str, output_dir: Path) -> Path | None:
    """Download California CDL (FIPS=06) for `year` from USDA NASS CropScape.

    Saves the extracted GeoTIFF to:
        output_dir / {year}_30m_cdls / CDL_{year}_06.tif

    Returns the path to the .tif, or None on failure.
    """
    import urllib.request
    import zipfile

    url      = f"https://nassgeodata.gmu.edu/nass_data_cache/byfips/CDL_{year}_06.zip"
    dest_dir = output_dir / f"{year}_30m_cdls"
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = dest_dir / f"CDL_{year}_06.zip"
    tif_path = dest_dir / f"CDL_{year}_06.tif"

    if tif_path.exists():
        log.info("  CDL %s already downloaded: %s", year, tif_path.name)
        return tif_path

    log.info("  Downloading CDL %s from USDA NASS ...", year)
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        log.error("  CDL download failed for %s: %s", year, e)
        return None

    log.info("  Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path, "r") as zf:
        tif_names = [n for n in zf.namelist() if n.lower().endswith(".tif")]
        if not tif_names:
            log.error("  No .tif found inside %s", zip_path.name)
            return None
        extracted = dest_dir / Path(tif_names[0]).name
        zf.extract(tif_names[0], dest_dir)
        if extracted != tif_path:
            extracted.rename(tif_path)

    zip_path.unlink(missing_ok=True)
    log.info("  CDL %s saved: %s", year, tif_path)
    return tif_path


def process_cdl(cdl_raw_path: str, s2_ref_path: str,
                out_reprojected: str, out_filtered: str) -> None:
    """Reproject CDL to S2 grid, then filter to hardcoded KEEP_CLASSES.

    Uses the S2 processed file as the spatial reference (CRS, transform, size),
    so CDL is clipped to exactly the S2 study area.
    KEEP_CLASSES is fixed in config.py: [3, 6, 24, 36, 37, 54, 69, 75, 76, 210].
    """
    if Path(out_reprojected).exists():
        log.info("  CDL reprojected already exists: %s", Path(out_reprojected).name)
    else:
        log.info("  Reprojecting CDL → %s", Path(out_reprojected).name)
        with rasterio.open(s2_ref_path) as s2_ref:
            target_crs       = s2_ref.crs
            target_transform = s2_ref.transform
            target_width     = s2_ref.width
            target_height    = s2_ref.height

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
        log.info("  CDL reprojected: %s", Path(out_reprojected).name)

    if Path(out_filtered).exists():
        log.info("  CDL filtered already exists: %s", Path(out_filtered).name)
    else:
        log.info("  Filtering CDL → %d classes: %s",
                 len(KEEP_CLASSES),
                 [USDA_CDL_NAMES.get(c, c) for c in KEEP_CLASSES])
        label_filtering(
            in_path      = out_reprojected,
            out_path     = out_filtered,
            keep_classes = KEEP_CLASSES,
        )
        log.info("  CDL filtered: %s", Path(out_filtered).name)


# ── Google Drive upload ─────────────────────────────────────────────────────────

def _build_drive_service():
    """Build an authenticated Google Drive API v3 service (OAuth token)."""
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Generate it locally:  python process_data_v2.py --auth\n"
            f"Then copy to server:  scp {GDRIVE_OAUTH_TOKEN} user@server:{GDRIVE_OAUTH_TOKEN}"
        )

    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)

    if creds.expired and creds.refresh_token:
        log.info("Refreshing expired OAuth token...")
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_or_create_subfolder(parent_id: str, name: str, service) -> str:
    """Return GDrive folder ID for `name` inside `parent_id`, creating it if missing."""
    query  = (f"name='{name}' and '{parent_id}' in parents "
              f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
    result = service.files().list(q=query, fields="files(id,name)").execute()
    folders = result.get("files", [])
    if folders:
        return folders[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=meta, fields="id").execute()
    log.info("  Created GDrive subfolder: %s (id=%s)", name, folder["id"])
    return folder["id"]


def upload_file(local_path: str, folder_id: str, service=None,
                overwrite: bool = False) -> str:
    """Upload a single file. Replaces existing file if overwrite=True, else skips. Returns GDrive file ID."""
    from googleapiclient.http import MediaFileUpload

    if service is None:
        service = _build_drive_service()

    fname  = Path(local_path).name
    query  = f"name='{fname}' and '{folder_id}' in parents and trashed=false"
    result = service.files().list(q=query, fields="files(id,name)").execute()
    existing = result.get("files", [])

    size  = os.path.getsize(local_path)
    media = MediaFileUpload(local_path, mimetype="image/tiff", resumable=True)

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
            log.info("    %d%% uploaded", int(status.progress() * 100))

    file_id = response.get("id")
    log.info("  Uploaded: %s  (id=%s)", fname, file_id)
    return file_id


def list_gdrive_processed(v3_folder_id: str, yr: str, service) -> set:
    """Return set of filenames already uploaded to processed_v3/s2/{yr}/."""
    try:
        s2_sub = get_or_create_subfolder(v3_folder_id, "s2", service)
        yr_sub = get_or_create_subfolder(s2_sub, yr, service)
        names  = set()
        page_token = None
        while True:
            kwargs = dict(
                q      = f"'{yr_sub}' in parents and trashed=false",
                fields = "nextPageToken, files(name)",
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
        log.warning("  Could not list GDrive processed_v3/s2/%s: %s", yr, exc)
        return set()


def upload_year(s2_processed_paths: list, cdl_filtered_path: str,
                year: str, s2_folder_ids: dict, cdl_folder_id: str,
                overwrite: bool = False) -> None:
    """Upload all processed S2 + CDL files for one year."""
    s2_folder = s2_folder_ids.get(year, "")
    if not s2_folder or not cdl_folder_id:
        raise ValueError(
            f"s2_folder_ids['{year}'] and cdl_folder_id must be set before uploading. "
            "Pass --skip-upload to skip."
        )

    service = _build_drive_service()
    log.info("  Uploading %d S2 files (year=%s)...", len(s2_processed_paths), year)
    for path in s2_processed_paths:
        upload_file(path, s2_folder, service, overwrite=overwrite)

    if cdl_filtered_path and Path(cdl_filtered_path).exists():
        log.info("  Uploading CDL filtered file...")
        upload_file(cdl_filtered_path, cdl_folder_id, service, overwrite=overwrite)


# ── Cleanup ─────────────────────────────────────────────────────────────────────

def delete_files(paths: list, label: str = "raw") -> None:
    freed = 0
    for p in paths:
        if Path(p).exists():
            size = os.path.getsize(p)
            os.remove(p)
            freed += size
            log.info("  Deleted %s: %s", label, Path(p).name)
    log.info("  Freed: %.2f GB", freed / 1e9)


# ── Shutdown ────────────────────────────────────────────────────────────────────

def _schedule_shutdown(delay_min: int = 8) -> None:
    import time, urllib.request, urllib.error, json

    pod_id  = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")

    if pod_id and api_key:
        log.warning("=" * 60)
        log.warning("RunPod pod %s will stop in %d minutes.", pod_id, delay_min)
        log.warning("=" * 60)
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
                log.info("Pod stop response: %s", json.loads(resp.read()))
        except urllib.error.URLError as e:
            log.error("Failed to stop pod: %s", e)
    else:
        log.warning("=" * 60)
        log.warning("VPS SHUTDOWN in %d minutes.  Cancel: sudo shutdown -c", delay_min)
        log.warning("=" * 60)
        try:
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_min}"], check=True)
        except Exception as e:
            log.error("Shutdown failed: %s", e)


# ── Test helper ─────────────────────────────────────────────────────────────────

def process_date_batch(
    date_groups  : dict,
    yr           : str,
    s2_out_dir,
    merge_tmp_dir,
    keep_merged  : bool = False,
    overwrite    : bool = False,
) -> tuple:
    """
    Merge + assign NoData for a subset of dates already downloaded locally.

    Parameters
    ----------
    date_groups   : {date_key: [(row, col, path), ...]} — output of group_tiles_by_date(),
                    filtered to the desired batch.
    yr            : year string (used for logging).
    s2_out_dir    : Path — directory to write *_processed.tif files.
    merge_tmp_dir : Path — temporary directory for intermediate merged TIFs.
    keep_merged   : bool — keep intermediate merged files.

    Returns
    -------
    (raw_tile_paths, processed_paths, s2_ref_path)
        raw_tile_paths : list[str] — all raw tile paths consumed (for deletion).
        processed_paths: list[str] — all *_processed.tif paths produced.
        s2_ref_path    : str | None — first processed file (CDL grid reference).
    """
    Path(s2_out_dir).mkdir(parents=True, exist_ok=True)
    Path(merge_tmp_dir).mkdir(parents=True, exist_ok=True)

    raw_tile_paths  = []
    processed_paths = []
    s2_ref_path     = None

    for date_key, tiles in date_groups.items():
        tile_paths = [t[2] for t in tiles]
        raw_tile_paths.extend(tile_paths)

        merged_path    = str(Path(merge_tmp_dir) / f"{date_key}_merged.tif")
        processed_path = str(Path(s2_out_dir)   / f"{date_key}_processed.tif")

        merge_tiles(tile_paths, merged_path)

        if not _has_valid_data(merged_path):
            log.warning("  Skipping %s — no valid data (all fill/NoData)", date_key)
            if Path(merged_path).exists():
                os.remove(merged_path)
            continue

        assign_nodata(merged_path, processed_path, overwrite=overwrite)
        processed_paths.append(processed_path)

        if s2_ref_path is None:
            s2_ref_path = processed_path

        if not keep_merged and Path(merged_path).exists():
            os.remove(merged_path)

    log.info("  Batch: processed %d date(s) for year %s", len(processed_paths), yr)
    return raw_tile_paths, processed_paths, s2_ref_path


_SENTINEL = object()  # end-of-queue signal


def _pipeline_year(
    date_groups   : dict,
    yr            : str,
    s2_out_dir    : Path,
    merge_tmp_dir : Path,
    skip_upload   : bool = False,
    skip_delete   : bool = False,
    overwrite     : bool = False,
    process_workers: int = 2,
    upload_workers : int = 1,
    s2_folder_ids  : dict = None,
    cdl_folder_id  : str  = None,
) -> tuple:
    """
    3-stage concurrent pipeline per date:
      Stage 1 (process_workers threads): merge_tiles + assign_nodata
      Stage 2 (upload_workers threads):  upload to GDrive
      Stage 3 (in upload thread):        delete raw tiles

    Returns (processed_paths, s2_ref_path).
    """
    s2_out_dir.mkdir(parents=True, exist_ok=True)
    merge_tmp_dir.mkdir(parents=True, exist_ok=True)

    upload_q: Queue = Queue(maxsize=process_workers + 2)

    processed_paths: list[str] = []
    s2_ref_path: list[str | None] = [None]
    lock = threading.Lock()
    errors: list[str] = []

    # ── Stage 1: process one date (runs in thread pool) ──────────────────────
    def _process_date(date_key: str, tiles: list) -> None:
        tile_paths     = [t[2] for t in tiles]
        merged_path    = merge_tmp_dir / f"{date_key}_merged.tif"
        processed_path = s2_out_dir   / f"{date_key}_processed.tif"

        try:
            merge_tiles(tile_paths, str(merged_path))

            if not _has_valid_data(str(merged_path)):
                log.warning("[%s] No valid data — skipped", date_key)
                merged_path.unlink(missing_ok=True)
                return

            assign_nodata(str(merged_path), str(processed_path), overwrite=overwrite)
            merged_path.unlink(missing_ok=True)

            with lock:
                processed_paths.append(str(processed_path))
                if s2_ref_path[0] is None:
                    s2_ref_path[0] = str(processed_path)

            upload_q.put((date_key, tile_paths, str(processed_path)))

        except Exception as exc:
            log.error("[%s] Process error: %s", date_key, exc)
            errors.append(f"{date_key}: {exc}")
            merged_path.unlink(missing_ok=True)

    # ── Stage 2+3: upload then delete (runs in upload thread) ────────────────
    def _upload_worker() -> None:
        service = None
        if not skip_upload and s2_folder_ids:
            try:
                service = _build_drive_service()
            except Exception as exc:
                log.error("GDrive auth failed: %s", exc)

        # Resolve processed_v3/s2/{yr}/ — create intermediate folders if needed
        parent_folder = (s2_folder_ids or {}).get(yr, "")
        s2_folder = ""
        if service and parent_folder:
            try:
                s2_subfolder = get_or_create_subfolder(parent_folder, "s2", service)
                s2_folder    = get_or_create_subfolder(s2_subfolder,  yr,  service)
            except Exception as exc:
                log.error("Failed to get/create s2/%s subfolder: %s", yr, exc)

        while True:
            item = upload_q.get()
            if item is _SENTINEL:
                upload_q.task_done()
                break

            date_key, tile_paths, processed_path = item
            try:
                if service and s2_folder:
                    upload_file(processed_path, s2_folder, service)
                elif not skip_upload:
                    log.warning("[%s] No GDrive folder ID — upload skipped", date_key)

                if not skip_delete:
                    delete_files(tile_paths, label="tile")
            except Exception as exc:
                log.error("[%s] Upload/cleanup error: %s", date_key, exc)
                errors.append(f"{date_key} upload: {exc}")
            finally:
                upload_q.task_done()

    # ── Launch upload thread(s) ───────────────────────────────────────────────
    upload_threads = [
        threading.Thread(target=_upload_worker, daemon=True, name=f"upload-{i}")
        for i in range(upload_workers)
    ]
    for t in upload_threads:
        t.start()

    # ── Launch process pool ───────────────────────────────────────────────────
    log.info("[%s] Pipeline start — %d dates, %d process workers, %d upload workers",
             yr, len(date_groups), process_workers, upload_workers)

    with ThreadPoolExecutor(max_workers=process_workers, thread_name_prefix="proc") as pool:
        futures = {
            pool.submit(_process_date, dk, tiles): dk
            for dk, tiles in date_groups.items()
        }
        for fut in as_completed(futures):
            dk = futures[fut]
            exc = fut.exception()
            if exc:
                log.error("[%s] Uncaught exception: %s", dk, exc)

    # ── Signal upload threads to stop ─────────────────────────────────────────
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


def test_merge(raw_dir: str, year: str, out_dir: str) -> None:
    """
    Merge + inspect only — no CDL, no upload, no delete.
    Useful for verifying bounds and band count before full processing.
    """
    groups = group_tiles_by_date(raw_dir, year)
    if not groups:
        log.error("No tile groups found.")
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for date_key, tiles in groups.items():
        tile_paths  = [t[2] for t in tiles]
        merged_path = str(Path(out_dir) / f"{date_key}_merged.tif")

        log.info("─" * 60)
        log.info("Date key : %s", date_key)
        log.info("Tiles    : %d", len(tiles))
        for row, col, p in tiles:
            log.info("  offset=(%d,%d)  %s", row, col, Path(p).name)

        merge_tiles(tile_paths, merged_path)

        with rasterio.open(merged_path) as src:
            b = src.bounds
            log.info("Merged   : %dx%d  bands=%d  dtype=%s",
                     src.width, src.height, src.count, src.dtypes[0])
            log.info("SW       : (%.5f, %.5f)", b.left, b.bottom)
            log.info("NE       : (%.5f, %.5f)", b.right, b.top)
            log.info("CRS      : %s", src.crs)


# ── Main ────────────────────────────────────────────────────────────────────────

def main(
    years              : list = None,
    raw_s2_dir         : str  = None,
    raw_cdl_dir        : str  = None,
    data_dir           : str  = None,
    s2_folder_ids      : dict = None,   # GDrive folder IDs for processed S2 per year
    cdl_folder_id      : str  = None,   # GDrive folder ID for processed CDL
    skip_upload        : bool = False,
    skip_delete        : bool = False,
    keep_merged        : bool = False,
    flat_dir           : bool = False,  # raw_s2_dir has no year subdir
    shutdown           : bool = False,
    overwrite          : bool = False,
    process_workers    : int  = 2,
    upload_workers     : int  = 1,
    download_workers   : int  = 2,
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

    for yr in years:
        log.info("=" * 60)
        log.info("Year: %s", yr)
        log.info("=" * 60)

        # ── Locate raw S2 tile directory ───────────────────────────────────────
        if flat_dir and raw_s2_dir:
            s2_raw_dir = pathlib.Path(raw_s2_dir)
        elif raw_s2_dir:
            s2_raw_dir = pathlib.Path(raw_s2_dir) / yr
        else:
            s2_raw_dir = _ROOT / "data" / "raw" / "s2" / yr

        from crop_mapping_pipeline.config import (
            GDRIVE_PROCESSED_V3_FOLDER_ID, GDRIVE_PROCESSED_CDL_FOLDER_ID,
            GDRIVE_RAW_S2_V2_FOLDER_ID,
        )
        _s2_ids = s2_folder_ids or {yr: GDRIVE_PROCESSED_V3_FOLDER_ID for yr in ALL_YEARS}
        _cdl_id = cdl_folder_id or GDRIVE_PROCESSED_CDL_FOLDER_ID
        _v3     = GDRIVE_PROCESSED_V3_FOLDER_ID

        # ── Step 1: Check processed_v3 first — what's already uploaded ────────
        already_uploaded: set = set()
        if not overwrite and not skip_upload:
            try:
                _svc = _build_drive_service()
                already_uploaded = list_gdrive_processed(_v3, yr, _svc)
            except Exception as exc:
                log.warning("  GDrive pre-check failed (%s) — will process all local dates", exc)

        # ── Step 2: Scan local raw tiles ──────────────────────────────────────
        local_groups = group_tiles_by_date(str(s2_raw_dir), yr)

        # ── Step 3: Filter local to dates that still need processing ──────────
        needed_local = {
            dk: tiles for dk, tiles in local_groups.items()
            if f"{dk}_processed.tif" not in already_uploaded
        }
        n_skipped_local = len(local_groups) - len(needed_local)
        if n_skipped_local:
            log.info("  Skipping %d local date(s) already in processed_v3/s2/%s/",
                     n_skipped_local, yr)

        # ── Step 4: Discover + download missing/incomplete raw dates ─────────
        try:
            from crop_mapping_pipeline.stages.fetch_data_v2 import (
                list_tile_counts_by_date, download_dates,
            )
            gdrive_counts = list_tile_counts_by_date(GDRIVE_RAW_S2_V2_FOLDER_ID, years=[yr])

            # Dates on GDrive not yet uploaded to processed_v3
            needed_gdrive_keys = {
                dk for dk in gdrive_counts
                if f"{dk}_processed.tif" not in already_uploaded
            }

            # Dates missing entirely from local
            to_download = needed_gdrive_keys - set(local_groups.keys())

            # Dates present locally but with fewer tiles than GDrive (partial download)
            incomplete = {
                dk for dk, tiles in local_groups.items()
                if dk in gdrive_counts and len(tiles) < gdrive_counts[dk]
                and f"{dk}_processed.tif" not in already_uploaded
            }
            if incomplete:
                log.warning(
                    "  %d date(s) have incomplete local tiles (local < GDrive count) — re-downloading: %s",
                    len(incomplete), sorted(incomplete),
                )
                to_download |= incomplete

            if to_download:
                log.info("  Downloading %d date(s) from GDrive (missing=%d, incomplete=%d)...",
                         len(to_download),
                         len(to_download - incomplete),
                         len(to_download & incomplete))
                download_dates(
                    folder_id  = GDRIVE_RAW_S2_V2_FOLDER_ID,
                    output_dir = str(s2_raw_dir.parent),
                    date_keys  = list(to_download),
                    overwrite  = True,   # replace partial tiles
                    workers    = download_workers,
                )
                local_groups = group_tiles_by_date(str(s2_raw_dir), yr)
                needed_local = {
                    dk: tiles for dk, tiles in local_groups.items()
                    if f"{dk}_processed.tif" not in already_uploaded
                }
        except Exception as exc:
            log.warning("  GDrive raw listing/download failed (%s) — using local tiles only", exc)

        groups = needed_local

        if not groups:
            log.info("  All dates for year %s already in processed_v3 or no raw tiles — skipping", yr)
            continue

        merge_tmp_dir = _ROOT / "data" / "raw" / "s2" / yr / "_merged"
        s2_out_dir    = S2_PROCESSED_DIR / yr

        all_processed, s2_ref_path = _pipeline_year(
            date_groups    = groups,
            yr             = yr,
            s2_out_dir     = s2_out_dir,
            merge_tmp_dir  = merge_tmp_dir,
            skip_upload    = skip_upload,
            skip_delete    = skip_delete,
            overwrite      = overwrite,
            process_workers= process_workers,
            upload_workers = upload_workers,
            s2_folder_ids  = _s2_ids,
            cdl_folder_id  = _cdl_id,
        )
        log.info("  Processed %d date(s) for year %s", len(all_processed), yr)

        # ── CDL processing (after pipeline — needs s2_ref_path) ───────────────
        cdl_dir  = (pathlib.Path(raw_cdl_dir) if raw_cdl_dir
                    else _ROOT / "data" / "raw" / "cdl")
        cdl_raw  = next(
            (p for p in glob(str(cdl_dir / f"{yr}_30m_cdls" / "*.tif"))),
            None,
        )
        cdl_filtered = None
        if not cdl_raw:
            log.warning("  Raw CDL for %s not found — skipping CDL processing", yr)
        elif s2_ref_path is None:
            log.warning("  No processed S2 reference — skipping CDL processing")
        else:
            cdl_out_dir     = S2_PROCESSED_DIR.parent / "cdl"
            cdl_reprojected = str(cdl_out_dir / f"cdl_{yr}_study_area.tif")
            cdl_filtered    = str(cdl_out_dir / f"cdl_{yr}_study_area_filtered.tif")
            process_cdl(cdl_raw, s2_ref_path, cdl_reprojected, cdl_filtered)

        # ── Upload CDL into processed_v3/cdl/ ────────────────────────────────
        if not skip_upload and cdl_filtered and _s2_ids:
            service    = _build_drive_service()
            v3_parent  = next(iter(_s2_ids.values()))
            cdl_folder = get_or_create_subfolder(v3_parent, "cdl", service)
            upload_file(cdl_filtered, cdl_folder, service)

        log.info("Year %s done.\n", yr)

    if shutdown:
        _schedule_shutdown(delay_min=8)


def generate_oauth_token():
    """Run OAuth flow in browser (run locally once, then copy token to server)."""
    import pickle
    from google_auth_oauthlib.flow import InstalledAppFlow
    from crop_mapping_pipeline.config import GDRIVE_OAUTH_SECRET, GDRIVE_OAUTH_TOKEN

    if not GDRIVE_OAUTH_SECRET.exists():
        raise FileNotFoundError(
            f"OAuth client secret not found: {GDRIVE_OAUTH_SECRET}\n"
            "Download from Google Cloud Console → APIs & Services → Credentials."
        )

    flow  = InstalledAppFlow.from_client_secrets_file(
        str(GDRIVE_OAUTH_SECRET),
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds = flow.run_local_server(port=0)

    with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
        pickle.dump(creds, f)

    print(f"Token saved: {GDRIVE_OAUTH_TOKEN}")
    print(f"Copy to server:\n  scp {GDRIVE_OAUTH_TOKEN} user@server:{GDRIVE_OAUTH_TOKEN}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process multi-tile GEE S2 exports: merge → NoData → CDL → upload."
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=ALL_YEARS, metavar="YEAR",
        help=f"Years to process (default: all — {ALL_YEARS})",
    )
    parser.add_argument(
        "--raw-s2-dir", default=None,
        help=(
            "Directory containing raw S2 tile dirs. "
            "By default the script appends /{year}/ to this path. "
            "Use --flat-dir if tiles are directly in this directory (no year subdir)."
        ),
    )
    parser.add_argument(
        "--flat-dir", action="store_true",
        help=(
            "Treat --raw-s2-dir as the direct tile directory (no year subdir). "
            "Useful when pointing at a GDrive-mounted export folder."
        ),
    )
    parser.add_argument(
        "--raw-cdl-dir", default=None,
        help="Directory containing raw CDL folders (YYYY_30m_cdls/).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override processed output directory (default: data/processed/).",
    )
    parser.add_argument(
        "--s2-folder-ids", nargs="+", metavar="YEAR:FOLDER_ID",
        default=None,
        help=(
            "GDrive folder IDs for processed S2 output, one per year. "
            "Format: 2022:FOLDER_ID 2023:FOLDER_ID 2024:FOLDER_ID"
        ),
    )
    parser.add_argument(
        "--cdl-folder-id", default=None,
        help="GDrive folder ID for processed CDL output.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-process dates that already have a *_processed.tif output.",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Process files but do not upload to Google Drive.",
    )
    parser.add_argument(
        "--skip-delete", action="store_true",
        help="Keep raw tile files after processing.",
    )
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="Keep intermediate merged (pre-NoData) TIFs for inspection.",
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Stop the VPS 8 minutes after processing (Linux / RunPod).",
    )
    parser.add_argument(
        "--process-workers", type=int, default=2,
        help="Parallel merge+nodata workers (default: 2; limited by RAM — each holds ~1 GB mosaic)",
    )
    parser.add_argument(
        "--upload-workers", type=int, default=1,
        help="Parallel GDrive upload workers (default: 1; GDrive throttles concurrent uploads)",
    )
    parser.add_argument(
        "--download-workers", type=int, default=2,
        help="Parallel GDrive download workers for auto-download (default: 2)",
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Generate OAuth token via browser (run locally once).",
    )
    parser.add_argument(
        "--test-merge", action="store_true",
        help=(
            "Merge tiles and print bounds/shape only — no CDL, upload, or delete. "
            "Output goes to data/raw/s2/{year}/_merged/. "
            "Requires --years."
        ),
    )
    args = parser.parse_args()

    if args.auth:
        generate_oauth_token()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Parse --s2-folder-ids YEAR:ID pairs
    s2_folder_ids = None
    if args.s2_folder_ids:
        s2_folder_ids = {}
        for item in args.s2_folder_ids:
            yr, fid = item.split(":", 1)
            s2_folder_ids[yr] = fid

    if args.test_merge:
        if not args.years:
            parser.error("--test-merge requires --years")
        for yr in (args.years or ALL_YEARS):
            if args.flat_dir and args.raw_s2_dir:
                s2_raw_dir = pathlib.Path(args.raw_s2_dir)
            elif args.raw_s2_dir:
                s2_raw_dir = pathlib.Path(args.raw_s2_dir) / yr
            else:
                s2_raw_dir = _ROOT / "data" / "raw" / "s2" / yr
            out_dir = _ROOT / "data" / "raw" / "s2" / yr / "_merged"
            test_merge(str(s2_raw_dir), yr, str(out_dir))
        sys.exit(0)

    main(
        years           = args.years,
        raw_s2_dir      = args.raw_s2_dir,
        raw_cdl_dir     = args.raw_cdl_dir,
        data_dir        = args.data_dir,
        s2_folder_ids   = s2_folder_ids,
        cdl_folder_id   = args.cdl_folder_id,
        skip_upload     = args.skip_upload,
        skip_delete     = args.skip_delete,
        keep_merged     = args.keep_merged,
        flat_dir        = args.flat_dir,
        shutdown        = args.shutdown,
        overwrite       = args.overwrite,
        process_workers  = args.process_workers,
        upload_workers   = args.upload_workers,
        download_workers = args.download_workers,
    )
