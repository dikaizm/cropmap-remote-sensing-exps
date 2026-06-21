"""
Stage 0b (v5) — Download raw S2 files from Google Drive.

Unlike v2, GEE exports one file per date (no tile splitting):
    S2H_{year}_{YYYY_MM_DD}.tif

Files are sorted into year subdirectories:
    {output_dir}/2022/S2H_2022_*.tif
    {output_dir}/2023/S2H_2023_*.tif
    {output_dir}/2024/S2H_2024_*.tif

Usage:
    python fetch_data_v6.py --folder-id FOLDER_ID
    python fetch_data_v6.py --folder-id FOLDER_ID --years 2022
    python fetch_data_v6.py --folder-id FOLDER_ID --years 2022 --overwrite
    python fetch_data_v6.py --folder-id FOLDER_ID --list-files
    python fetch_data_v6.py --auth
"""

import os
import re
import sys
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import GDRIVE_OAUTH_TOKEN

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]

# S2H_{year}_{YYYY_MM_DD}.tif  (raw) or S2H_{year}_{YYYY_MM_DD}_processed.tif
_FILE_RE = re.compile(r"^S2H_(\d{4})_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$")


# ── Auth ────────────────────────────────────────────────────────────────────────

def _build_drive_service():
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Run:  python stages/process_data_v6.py --auth"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


_thread_local = threading.local()


def _get_thread_service():
    if not hasattr(_thread_local, "service"):
        _thread_local.service = _build_drive_service()
    return _thread_local.service


# ── Filename helpers ────────────────────────────────────────────────────────────

def _year_from_filename(fname: str) -> str:
    m = _FILE_RE.match(fname)
    return m.group(1) if m else ""


def _date_key_from_filename(fname: str) -> str:
    """Return date key, e.g. 'S2H_2022_2022_01_16'."""
    m = _FILE_RE.match(fname)
    if not m:
        return ""
    return f"S2H_{m.group(1)}_{m.group(2)}"


# ── Folder listing ──────────────────────────────────────────────────────────────

def _list_children(service, folder_id: str):
    """Return (tifs, subfolders) as {name: id} dicts for direct children."""
    tifs, subfolders = {}, {}
    page_token = None
    while True:
        resp = service.files().list(
            q         = f"'{folder_id}' in parents and trashed = false",
            fields    = "nextPageToken, files(id, name, mimeType)",
            pageSize  = 1000,
            pageToken = page_token,
        ).execute()
        for f in resp.get("files", []):
            if f["mimeType"] == "application/vnd.google-apps.folder":
                subfolders[f["name"]] = f["id"]
            else:
                tifs[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return tifs, subfolders


def _find_subfolder(service, folder_id: str, name: str):
    """Return folder ID of a named subfolder, or None."""
    _, subs = _list_children(service, folder_id)
    return subs.get(name)


def list_folder(folder_id: str, years: list = None) -> dict:
    """Return {filename: file_id} for all S2 TIFs reachable from folder_id.

    Handles three layouts automatically:
      - Flat:            folder/S2H_*.tif
      - Year-subdir:     folder/2022/S2H_*.tif
      - s2/year-subdir:  folder/s2/2022/S2H_*.tif  (GDrive processed layout v6)
    """
    service    = _build_drive_service()
    name_to_id = {}

    def _collect_s2(fid):
        tifs, subs = _list_children(service, fid)
        for name, tid in tifs.items():
            if _FILE_RE.match(name):
                name_to_id[name] = tid
        for sub_name, sub_id in subs.items():
            _collect_s2(sub_id)   # recurse into any subfolders (year dirs, etc.)

    # check if there's an s2/ subfolder — if so, only recurse into that
    s2_sub = _find_subfolder(service, folder_id, "s2")
    _collect_s2(s2_sub if s2_sub else folder_id)

    log.info("  %d S2 file(s) found in folder", len(name_to_id))

    if years:
        years_set  = set(years)
        name_to_id = {n: fid for n, fid in name_to_id.items()
                      if _year_from_filename(n) in years_set}
        log.info("  %d file(s) after year filter=%s", len(name_to_id), years)

    return name_to_id


def list_dates_by_year(folder_id: str, years: list = None) -> dict:
    """Return {year: sorted [date_keys]} from GDrive folder."""
    name_to_id    = list_folder(folder_id, years=years)
    dates_by_year: dict = {}
    for fname in name_to_id:
        yr  = _year_from_filename(fname)
        key = _date_key_from_filename(fname)
        if yr and key:
            dates_by_year.setdefault(yr, set()).add(key)
    return {yr: sorted(keys) for yr, keys in sorted(dates_by_year.items())}


# ── Download ────────────────────────────────────────────────────────────────────

def _download_one(fname: str, file_id: str, output_dir: str,
                  overwrite: bool = False, flat: bool = False) -> tuple[str, str]:
    """Download one file into {output_dir}/{year}/ (or flat {output_dir}/). Returns (path, status)."""
    from googleapiclient.http import MediaIoBaseDownload

    if flat:
        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_path = dest_dir / fname
    else:
        yr = _year_from_filename(fname)
        if not yr:
            log.warning("  Cannot parse year from '%s' — skipping", fname)
            return "", "error"
        yr_dir   = Path(output_dir) / yr
        yr_dir.mkdir(parents=True, exist_ok=True)
        out_path = yr_dir / fname

    if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
        log.info("  Skip (exists): %s/%s", yr, fname)
        return str(out_path), "skip"

    service = _get_thread_service()
    request = service.files().get_media(fileId=file_id)
    tmp     = out_path.with_suffix(".tmp.tif")
    try:
        with open(tmp, "wb") as fh:
            dl   = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
            done = False
            while not done:
                status, done = dl.next_chunk()
                if status:
                    log.info("  %s: %d%%", fname, int(status.progress() * 100))
        tmp.rename(out_path)
        log.info("  Done: %s  (%.0f MB)", fname, out_path.stat().st_size / 1e6)
        return str(out_path), "new"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        log.error("  Failed: %s (%s)", fname, exc)
        return "", "error"


def _download_many(name_to_id: dict, output_dir: str,
                   overwrite: bool = False, workers: int = 2,
                   flat: bool = False) -> list:
    """Parallel download — thread-local Drive service per worker."""
    total     = len(name_to_id)
    results   = []
    new_count = skipped = errors = 0
    lock      = threading.Lock()

    log.info("  Downloading %d file(s) with %d worker(s)...", total, workers)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl") as pool:
        futures = {
            pool.submit(_download_one, fname, fid, output_dir, overwrite, flat): fname
            for fname, fid in name_to_id.items()
        }
        done_n = 0
        for fut in as_completed(futures):
            fname   = futures[fut]
            done_n += 1
            try:
                path, status = fut.result()
            except Exception as exc:
                log.error("  [%d/%d] Error %s: %s", done_n, total, fname, exc)
                with lock:
                    errors += 1
                continue
            with lock:
                if status == "new":
                    results.append(path)
                    new_count += 1
                elif status == "skip":
                    results.append(path)
                    skipped += 1
                else:
                    errors += 1

    log.info("  Done: %d new, %d skipped, %d errors", new_count, skipped, errors)
    return results


def download_folder_by_year(folder_id: str, output_dir: str,
                            years: list = None, overwrite: bool = False,
                            workers: int = 2) -> list:
    """Download all S2 files from folder, sorted into year subdirs."""
    name_to_id = list_folder(folder_id, years=years)
    if not name_to_id:
        log.warning("  No files to download.")
        return []
    return _download_many(name_to_id, output_dir, overwrite, workers)


def download_cdl(folder_id: str, output_dir: str,
                 overwrite: bool = False, workers: int = 2) -> list:
    """Download CDL TIFs from folder/cdl/ into {output_dir}/cdl/."""
    service = _build_drive_service()
    cdl_fid = _find_subfolder(service, folder_id, "cdl")
    if not cdl_fid:
        log.warning("  No 'cdl' subfolder found in folder %s", folder_id)
        return []
    tifs, _ = _list_children(service, cdl_fid)
    if not tifs:
        log.warning("  No CDL files found in cdl/ subfolder")
        return []

    from googleapiclient.http import MediaIoBaseDownload
    cdl_dir = Path(output_dir) / "cdl"
    cdl_dir.mkdir(parents=True, exist_ok=True)
    results, new_count, skipped, errors = [], 0, 0, 0

    log.info("  Downloading %d CDL file(s) → %s", len(tifs), cdl_dir)
    for fname, fid in sorted(tifs.items()):
        out_path = cdl_dir / fname
        if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            log.info("  Skip (exists): cdl/%s", fname)
            skipped += 1
            results.append(str(out_path))
            continue
        tmp = out_path.with_suffix(".tmp.tif")
        try:
            svc     = _build_drive_service()
            request = svc.files().get_media(fileId=fid)
            with open(tmp, "wb") as fh:
                dl   = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
                done = False
                while not done:
                    status, done = dl.next_chunk()
            tmp.rename(out_path)
            log.info("  Done: cdl/%s  (%.0f MB)", fname, out_path.stat().st_size / 1e6)
            new_count += 1
            results.append(str(out_path))
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            log.error("  Failed: cdl/%s (%s)", fname, exc)
            errors += 1

    log.info("  CDL done: %d new, %d skipped, %d errors", new_count, skipped, errors)
    return results


def fetch_preload_cache(folder_id: str, output_dir: str,
                        overwrite: bool = False) -> list:
    """Download a cloud-built portable preload cache flat into output_dir.

    Grabs every `preload_*.npy` + `preload_*_masks.pt` reachable from folder_id
    (also descends into an optional `preload_cache/` subfolder). Filenames are
    content-hash keyed by PreloadedDataset, so a matching file lands as a cache
    hit at train time — no local rebuild. Pairs with `--build-cache-only`, which
    builds the same files locally for upload.
    """
    from googleapiclient.http import MediaIoBaseDownload

    service = _build_drive_service()
    sub     = _find_subfolder(service, folder_id, "preload_cache")
    files, _ = _list_children(service, sub if sub else folder_id)
    cache = {n: fid for n, fid in files.items()
             if n.startswith("preload_") and (n.endswith(".npy") or n.endswith(".pt"))}
    if not cache:
        log.warning("  No preload cache files (preload_*.npy / *_masks.pt) in folder %s", folder_id)
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, new_count, skipped, errors = [], 0, 0, 0
    log.info("  Downloading %d preload cache file(s) → %s", len(cache), out_dir)

    for fname, fid in sorted(cache.items()):
        out_path = out_dir / fname
        if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            log.info("  Skip (exists): %s", fname)
            skipped += 1
            results.append(str(out_path))
            continue
        tmp = out_path.with_name(out_path.name + ".tmp")
        try:
            request = service.files().get_media(fileId=fid)
            with open(tmp, "wb") as fh:
                dl   = MediaIoBaseDownload(fh, request, chunksize=50 * 1024 * 1024)
                done = False
                while not done:
                    status, done = dl.next_chunk()
                    if status:
                        log.info("  %s: %d%%", fname, int(status.progress() * 100))
            tmp.rename(out_path)
            log.info("  Done: %s  (%.0f MB)", fname, out_path.stat().st_size / 1e6)
            new_count += 1
            results.append(str(out_path))
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            log.error("  Failed: %s (%s)", fname, exc)
            errors += 1

    log.info("  Preload cache: %d new, %d skipped, %d errors", new_count, skipped, errors)
    return results


def upload_preload_cache(folder_id: str, cache_dir: str,
                         overwrite: bool = False) -> list:
    """Upload locally-built portable preload cache files to a GDrive folder.

    Pushes every `preload_*.npy` + `preload_*_masks.pt` under cache_dir. Skips
    files already present in the folder unless overwrite=True (then replaces
    in place via files().update). Pairs with `--build-cache-only` so a cache
    built on one machine is reusable by `--preload-cache-gdrive` on another.
    """
    from googleapiclient.http import MediaFileUpload

    cache_dir = Path(cache_dir)
    local = sorted(
        [p for p in cache_dir.glob("preload_*.npy") if p.stat().st_size > 0] +
        [p for p in cache_dir.glob("preload_*_masks.pt") if p.stat().st_size > 0]
    )
    if not local:
        log.warning("  No preload cache files to upload in %s", cache_dir)
        return []

    service       = _build_drive_service()
    existing, _   = _list_children(service, folder_id)   # {name: id}
    results, new_count, replaced, skipped, errors = [], 0, 0, 0, 0
    log.info("  Uploading %d preload cache file(s) → GDrive folder %s", len(local), folder_id)

    for path in local:
        fname  = path.name
        exists = existing.get(fname)
        if exists and not overwrite:
            log.info("  Skip (exists): %s", fname)
            skipped += 1
            results.append(fname)
            continue
        media = MediaFileUpload(str(path), resumable=True, chunksize=50 * 1024 * 1024)
        try:
            if exists:
                req = service.files().update(fileId=exists, media_body=media)
            else:
                req = service.files().create(
                    body={"name": fname, "parents": [folder_id]},
                    media_body=media, fields="id",
                )
            resp = None
            while resp is None:
                status, resp = req.next_chunk()
                if status:
                    log.info("  %s: %d%%", fname, int(status.progress() * 100))
            log.info("  Done: %s  (%.0f MB)%s", fname, path.stat().st_size / 1e6,
                     " [replaced]" if exists else "")
            replaced += 1 if exists else 0
            new_count += 0 if exists else 1
            results.append(fname)
        except Exception as exc:
            log.error("  Failed: %s (%s)", fname, exc)
            errors += 1

    log.info("  Preload cache upload: %d new, %d replaced, %d skipped, %d errors",
             new_count, replaced, skipped, errors)
    return results


def download_date_keys(folder_id: str, output_dir: str,
                       date_keys: list, overwrite: bool = False,
                       workers: int = 2) -> list:
    """Download only files matching the given date keys."""
    date_keys_set = set(date_keys)
    years         = {_year_from_filename(dk + ".tif") for dk in date_keys}
    years         = {y for y in years if y}

    name_to_id = list_folder(folder_id, years=list(years) if years else None)
    name_to_id = {
        n: fid for n, fid in name_to_id.items()
        if _date_key_from_filename(n) in date_keys_set
    }
    if not name_to_id:
        log.warning("  No files found for date_keys=%s", date_keys)
        return []
    return _download_many(name_to_id, output_dir, overwrite, workers)


# ── Verify ──────────────────────────────────────────────────────────────────────

def verify(output_dir: str, years: list = None) -> bool:
    years  = years or ALL_YEARS
    all_ok = True
    print(f"\nS2 files under {output_dir}/{{year}}/:")
    for yr in sorted(years):
        yr_dir = Path(output_dir) / yr
        files  = sorted(yr_dir.glob(f"S2H_{yr}_*.tif")) if yr_dir.exists() else []
        files  = [f for f in files if _FILE_RE.match(f.name)]
        status = "OK" if files else "MISSING"
        print(f"  {yr}: {status}  {len(files)} date(s)")
        for f in files:
            print(f"    {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
        if not files:
            all_ok = False
    return all_ok


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


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download single-file-per-date S2 exports from GDrive."
    )
    parser.add_argument("--folder-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--years", nargs="+", default=None, choices=ALL_YEARS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--list-files", action="store_true")
    parser.add_argument("--include-cdl", action="store_true",
                        help="Also download CDL files (v6.1 processed CDL folder) into cdl/")
    parser.add_argument("--cdl-only", action="store_true",
                        help="Download only CDL (skip S2 + test areas). Implies --include-cdl.")
    parser.add_argument("--raw", action="store_true",
                        help="Download raw S2 files (no _processed suffix) from raw GDrive folders")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--test-areas", action="store_true",
                        help="Download test_a and test_b S2 files to s2/test_a/ and s2/test_b/")
    parser.add_argument("--auth", action="store_true")
    args = parser.parse_args()

    if args.cdl_only:
        args.include_cdl = True

    if args.auth:
        generate_oauth_token()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    from crop_mapping_pipeline.config import (
        GDRIVE_PROCESSED_S2_V6_FOLDER_IDS,
        GDRIVE_RAW_S2_V5_FOLDER_IDS,
        GDRIVE_PROCESSED_CDL_FOLDER_ID_V6,
    )

    if args.raw:
        output_dir    = args.output_dir or str(_ROOT / "data" / "raw")
        s2_output_dir = str(Path(output_dir) / "s2")
        folder_ids    = GDRIVE_RAW_S2_V5_FOLDER_IDS
    else:
        output_dir    = args.output_dir or str(_ROOT / "data" / "processed")
        s2_output_dir = str(Path(output_dir) / "s2")
        folder_ids    = GDRIVE_PROCESSED_S2_V6_FOLDER_IDS  # v6.1 processed S2 (2024)

    years = args.years or ALL_YEARS

    if args.verify_only:
        ok = verify(s2_output_dir, years=years)
        sys.exit(0 if ok else 1)

    # S2 — download each year from its own folder → {s2_output_dir}/{year}/
    for yr in ([] if args.cdl_only else years):
        fid = args.folder_id or folder_ids.get(yr)
        if not fid:
            log.warning("  No folder ID for year %s — skipping", yr)
            continue
        log.info("  Fetching S2 year=%s from folder %s", yr, fid)
        name_to_id = list_folder(fid, years=[yr])
        if args.list_files:
            for name in sorted(name_to_id):
                print(f"  {name}")
            continue
        _download_many(name_to_id, s2_output_dir,
                       overwrite=args.overwrite, workers=args.workers)

    if args.raw:
        globals()["_FILE_RE"] = _orig_file_re

    if args.list_files:
        sys.exit(0)

    # Spatial test areas — flat folders → {s2_output_dir}/test_a/ and test_b/
    if args.test_areas and not args.cdl_only:
        from crop_mapping_pipeline.config import (
            GDRIVE_S2_TEST_A_FOLDER_ID,
            GDRIVE_S2_TEST_B_FOLDER_ID,
        )
        for area_name, fid in [("test_a", GDRIVE_S2_TEST_A_FOLDER_ID),
                                ("test_b", GDRIVE_S2_TEST_B_FOLDER_ID)]:
            area_out = str(Path(s2_output_dir) / area_name)
            log.info("  Fetching S2 %s from folder %s → %s", area_name, fid, area_out)
            name_to_id = list_folder(fid)
            if args.list_files:
                for name in sorted(name_to_id):
                    print(f"  [{area_name}] {name}")
                continue
            _download_many(name_to_id, area_out,
                           overwrite=args.overwrite, workers=args.workers, flat=True)

    # CDL — flat folder → {output_dir}/cdl/
    if args.include_cdl:
        cdl_fid = GDRIVE_PROCESSED_CDL_FOLDER_ID_V6
        log.info("  Fetching CDL from folder %s", cdl_fid)
        service = _build_drive_service()
        tifs, _ = _list_children(service, cdl_fid)
        if tifs:
            cdl_dir = Path(output_dir) / "cdl"
            cdl_dir.mkdir(parents=True, exist_ok=True)
            from googleapiclient.http import MediaIoBaseDownload
            for fname, fid in sorted(tifs.items()):
                out_path = cdl_dir / fname
                if not args.overwrite and out_path.exists() and out_path.stat().st_size > 0:
                    log.info("  Skip (exists): cdl/%s", fname)
                    continue
                tmp = out_path.with_suffix(".tmp.tif")
                try:
                    req = service.files().get_media(fileId=fid)
                    with open(tmp, "wb") as fh:
                        dl = MediaIoBaseDownload(fh, req, chunksize=50*1024*1024)
                        done = False
                        while not done:
                            _, done = dl.next_chunk()
                    tmp.rename(out_path)
                    log.info("  Done: cdl/%s  (%.0f MB)", fname, out_path.stat().st_size/1e6)
                except Exception as exc:
                    tmp.unlink(missing_ok=True)
                    log.error("  Failed: cdl/%s (%s)", fname, exc)
        else:
            log.warning("  No CDL files found in folder %s", cdl_fid)

    if not args.cdl_only:
        verify(s2_output_dir, years=years)
