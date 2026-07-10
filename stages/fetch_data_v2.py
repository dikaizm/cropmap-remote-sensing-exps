"""
Stage 0b — Download raw S2 tile files from a single flat Google Drive folder.

For the new (larger) study area, GEE exports all years into one flat folder:
    S2H_{year}_{YYYY_MM_DD}-{row:010d}-{col:010d}.tif

Unlike fetch_data.py (which expects one GDrive folder per year), this script
downloads from a single GDrive folder and automatically sorts files into
year subdirectories:
    {output_dir}/2022/S2H_2022_*.tif
    {output_dir}/2023/S2H_2023_*.tif
    {output_dir}/2024/S2H_2024_*.tif

process_data_v2.py then reads these year subdirs and merges tiles per date.

Usage:
    python fetch_data_v2.py --folder-id FOLDER_ID
    python fetch_data_v2.py --folder-id FOLDER_ID --output-dir /data/raw/s2
    python fetch_data_v2.py --folder-id FOLDER_ID --years 2022
    python fetch_data_v2.py --folder-id FOLDER_ID --years 2022 2023 --overwrite
    python fetch_data_v2.py --folder-id FOLDER_ID --verify-only
    python fetch_data_v2.py --folder-id FOLDER_ID --list-files
    python fetch_data_v2.py --folder-id FOLDER_ID --delete
    python fetch_data_v2.py --auth                    # generate OAuth token once

Folder ID is taken from:
  1. --folder-id CLI argument  (highest priority)
  2. GDRIVE_RAW_S2_V2_FOLDER_ID in config.py
"""

import os
import sys
import re
import argparse
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import GDRIVE_OAUTH_TOKEN

log = logging.getLogger(__name__)

ALL_YEARS = ["2022", "2023", "2024"]

# Matches raw tiles:      S2H_{year}_{YYYY_MM_DD}-{10d}-{10d}.tif
_TILE_RE = re.compile(r"^S2H_(\d{4})_\d{4}_\d{2}_\d{2}-\d{10}-\d{10}\.tif$")
# Matches processed files: S2H_{year}_{YYYY_MM_DD}_processed.tif[.*]
_PROC_RE = re.compile(r"^S2H_(\d{4})_\d{4}_\d{2}_\d{2}_processed\.tif")


# ── Auth ────────────────────────────────────────────────────────────────────────

def _build_drive_service():
    import pickle
    from googleapiclient.discovery import build
    from google.auth.transport.requests import Request

    if not GDRIVE_OAUTH_TOKEN.exists():
        raise FileNotFoundError(
            f"OAuth token not found: {GDRIVE_OAUTH_TOKEN}\n"
            "Run:  python stages/process_data.py --auth\n"
            f"Then: scp {GDRIVE_OAUTH_TOKEN} user@server:{GDRIVE_OAUTH_TOKEN}"
        )
    with open(GDRIVE_OAUTH_TOKEN, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        log.info("Refreshing OAuth token...")
        creds.refresh(Request())
        with open(GDRIVE_OAUTH_TOKEN, "wb") as f:
            pickle.dump(creds, f)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# One Drive service per thread (httplib2 is not thread-safe)
_thread_local = threading.local()


def _get_thread_service():
    if not hasattr(_thread_local, "service"):
        _thread_local.service = _build_drive_service()
    return _thread_local.service


def _download_one(fname: str, file_id: str, output_dir: str,
                  overwrite: bool = False) -> tuple[str, str]:
    """Download a single tile file. Returns (out_path, status) where status is 'new'|'skip'|'error'."""
    from googleapiclient.http import MediaIoBaseDownload

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
        log.info("  Done: %s/%s  (%.0f MB)", yr, fname, out_path.stat().st_size / 1e6)
        return str(out_path), "new"
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        log.error("  Download failed — deleted partial: %s (%s)", fname, exc)
        return "", "error"


# ── Folder listing ──────────────────────────────────────────────────────────────

def list_folder(folder_id: str, years: list = None, dates: list = None) -> dict:
    """
    Return {filename: file_id} for all files in the flat GDrive folder.
    Optionally filter by years and/or dates (YYYY-MM-DD or YYYY_MM_DD).
    """
    service    = _build_drive_service()
    name_to_id = {}
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        for f in resp.get("files", []):
            name_to_id[f["name"]] = f["id"]
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    log.info("  %d file(s) found in folder", len(name_to_id))

    if years:
        years_set = set(years)
        name_to_id = {
            name: fid for name, fid in name_to_id.items()
            if _year_from_filename(name) in years_set
        }
        log.info("  %d file(s) match years=%s", len(name_to_id), years)

    if dates:
        dates_set = {_normalize_date(d) for d in dates}
        name_to_id = {
            name: fid for name, fid in name_to_id.items()
            if _date_from_filename(name) in dates_set
        }
        log.info("  %d file(s) match dates=%s", len(name_to_id), dates)

    return name_to_id


def _year_from_filename(fname: str) -> str:
    """Extract year string from filename (raw tile or processed)."""
    m = _TILE_RE.match(fname) or _PROC_RE.match(fname)
    return m.group(1) if m else ""


def _date_from_filename(fname: str) -> str:
    """Extract YYYY_MM_DD from raw tile or processed filename."""
    m = re.search(r"_(\d{4}_\d{2}_\d{2})[-_]", fname)
    return m.group(1) if m else ""


def _normalize_date(d: str) -> str:
    """Normalize user date input to YYYY_MM_DD (accepts YYYY-MM-DD or YYYY_MM_DD)."""
    return d.replace("-", "_")


def _date_key_from_filename(fname: str) -> str:
    """Extract date key from tile filename, e.g. 'S2H_2022_2022_01_01-...' → 'S2H_2022_2022_01_01'."""
    m = _TILE_RE.match(fname)
    if not m:
        return ""
    # Reconstruct date key: everything before the first tile offset
    # fname = S2H_{year}_{YYYY_MM_DD}-{row}-{col}.tif → stem without offsets
    stem = fname[: fname.index("-")]
    return stem


def list_dates_by_year(folder_id: str, years: list = None) -> dict:
    """
    List all unique date keys per year from GDrive without downloading.
    Returns {year: sorted list of date keys}, e.g.:
        {'2022': ['S2H_2022_2022_01_01', 'S2H_2022_2022_01_16', ...], ...}
    """
    name_to_id = list_folder(folder_id, years=years)
    dates_by_year: dict = {}
    for fname in name_to_id:
        yr  = _year_from_filename(fname)
        key = _date_key_from_filename(fname)
        if yr and key:
            dates_by_year.setdefault(yr, set()).add(key)
    return {yr: sorted(keys) for yr, keys in sorted(dates_by_year.items())}


def list_tile_counts_by_date(folder_id: str, years: list = None) -> dict:
    """
    Return {date_key: tile_count} for all raw tiles in GDrive folder.
    Used to detect incomplete local downloads (local tile count < GDrive count).
    """
    name_to_id = list_folder(folder_id, years=years)
    counts: dict = {}
    for fname in name_to_id:
        key = _date_key_from_filename(fname)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return counts


# ── Download ────────────────────────────────────────────────────────────────────

def _download_many(name_to_id: dict, output_dir: str,
                   overwrite: bool = False, workers: int = 2) -> list:
    """Parallel download core — one Drive service per thread via thread-local storage."""
    total     = len(name_to_id)
    results   = []
    new_count = skipped = errors = 0
    lock      = threading.Lock()

    log.info("  Downloading %d file(s) with %d worker(s)...", total, workers)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dl") as pool:
        futures = {
            pool.submit(_download_one, fname, fid, output_dir, overwrite): fname
            for fname, fid in name_to_id.items()
        }
        done_n = 0
        for fut in as_completed(futures):
            fname  = futures[fut]
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
                            years: list = None, dates: list = None,
                            overwrite: bool = False, workers: int = 2) -> list:
    """
    Download all tile files from a flat GDrive folder, sorted into year subdirs.
    Returns list of downloaded file paths.
    """
    name_to_id = list_folder(folder_id, years=years, dates=dates)
    if not name_to_id:
        log.warning("  No files to download.")
        return []
    return _download_many(name_to_id, output_dir, overwrite, workers)


def download_dates(folder_id: str, output_dir: str,
                   date_keys: list, overwrite: bool = False,
                   workers: int = 2) -> list:
    """
    Download only tiles whose date key matches one of `date_keys`.
    Returns list of downloaded file paths.
    """
    date_keys_set = set(date_keys)
    years         = {_year_from_filename(dk + "-0000000000-0000000000.tif") for dk in date_keys}
    years         = {y for y in years if y}

    name_to_id = list_folder(folder_id, years=list(years) if years else None)
    name_to_id = {
        name: fid for name, fid in name_to_id.items()
        if _date_key_from_filename(name) in date_keys_set
    }

    if not name_to_id:
        log.warning("  No tiles found for date_keys=%s", date_keys)
        return []

    return _download_many(name_to_id, output_dir, overwrite, workers)


# ── Flat download (CDL / non-tiled files) ───────────────────────────────────────

def download_flat(folder_id: str, output_dir: str, overwrite: bool = False) -> list:
    """Download all files from a flat GDrive folder into output_dir (no year subdir)."""
    from googleapiclient.http import MediaIoBaseDownload

    name_to_id = list_folder(folder_id)
    if not name_to_id:
        log.warning("  No files found in folder.")
        return []

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    service    = _build_drive_service()
    downloaded = []
    skipped    = 0

    for fname in sorted(name_to_id):
        out_path = Path(output_dir) / fname
        if not overwrite and out_path.exists() and out_path.stat().st_size > 0:
            log.info("  Skip (exists): %s", fname)
            skipped += 1
            continue
        log.info("  Downloading → %s  (id=%s)", fname, name_to_id[fname])
        req = service.files().get_media(fileId=name_to_id[fname])
        with open(out_path, "wb") as fh:
            dl = MediaIoBaseDownload(fh, req, chunksize=32 * 1024 * 1024)
            done = False
            while not done:
                _, done = dl.next_chunk()
        downloaded.append(str(out_path))
        log.info("  Done: %s", fname)

    log.info("Flat download complete: %d new, %d skipped", len(downloaded), skipped)
    return downloaded


# ── Verify ──────────────────────────────────────────────────────────────────────

def verify(output_dir: str, years: list = None) -> bool:
    """
    Check which tile files are present locally under {output_dir}/{year}/.
    Returns True if at least one file exists per requested year.
    """
    years = years or ALL_YEARS
    all_ok = True
    print(f"\nTile files under {output_dir}/{{year}}/:")
    for yr in sorted(years):
        yr_dir   = Path(output_dir) / yr
        by_date: dict = {}
        for p in sorted(yr_dir.glob("S2H_*.tif")) if yr_dir.exists() else []:
            if not _TILE_RE.match(p.name):
                continue
            date_key = re.sub(r"-\d{10}-\d{10}$", "", p.stem)
            by_date.setdefault(date_key, []).append(p)

        n_dates = len(by_date)
        n_tiles = sum(len(v) for v in by_date.values())
        status  = "✅" if n_tiles > 0 else "❌ MISSING"
        print(f"  {yr}: {status}  {n_dates} date(s), {n_tiles} tile(s)")
        for dk in sorted(by_date):
            print(f"      {dk}: {len(by_date[dk])} tile(s)")
        if n_tiles == 0:
            all_ok = False

    print(f"\nStatus: {'✅ All present' if all_ok else '⚠️  Some files missing'}")
    return all_ok


# ── Delete ──────────────────────────────────────────────────────────────────────

def delete_tiles(output_dir: str, years: list = None) -> None:
    """Delete downloaded tile files under {output_dir}/{year}/ for the given years."""
    freed   = 0
    deleted = 0
    for yr in (years or ALL_YEARS):
        yr_dir = Path(output_dir) / yr
        for p in sorted(yr_dir.glob("S2H_*.tif")) if yr_dir.exists() else []:
            freed   += p.stat().st_size
            p.unlink()
            deleted += 1
            log.info("  Deleted: %s/%s", yr, p.name)
    log.info("Deleted %d tile(s), freed %.2f GB", deleted, freed / 1e9)


# ── Main ────────────────────────────────────────────────────────────────────────

def main(
    folder_id   : str,
    output_dir  : str,
    years       : list = None,
    dates       : list = None,
    overwrite   : bool = False,
    verify_only : bool = False,
    list_files  : bool = False,
    delete      : bool = False,
    workers     : int  = 2,
) -> None:

    if list_files:
        name_to_id = list_folder(folder_id, years=years, dates=dates)
        print(f"\n{len(name_to_id)} file(s) in folder {folder_id}:")
        for name in sorted(name_to_id):
            print(f"  {name}")
        return

    if verify_only:
        ok = verify(output_dir, years=years)
        sys.exit(0 if ok else 1)

    download_folder_by_year(folder_id, output_dir,
                            years=years, dates=dates, overwrite=overwrite,
                            workers=workers)
    verify(output_dir, years=years)

    if delete:
        log.info("Deleting downloaded tiles to free disk space...")
        delete_tiles(output_dir, years=years)


def generate_oauth_token():
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
        description="Download raw S2 tiles from a single flat GDrive folder."
    )
    parser.add_argument(
        "--folder-id", default=None,
        help=(
            "GDrive folder ID containing the raw tile files. "
            "Falls back to GDRIVE_RAW_S2_V2_FOLDER_ID in config.py if not set."
        ),
    )
    parser.add_argument(
        "--output-dir", default=None,
        help=(
            "Local directory to write tiles into (flat, no year subdirs). "
            "Default: data/raw/s2/  relative to project root."
        ),
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=ALL_YEARS, metavar="YEAR",
        help="Only download tiles for these years (default: all years in the folder).",
    )
    parser.add_argument(
        "--dates", nargs="+", default=None, metavar="DATE",
        help=(
            "Only download files for these dates (YYYY-MM-DD or YYYY_MM_DD). "
            "Works with both raw and --processed modes. "
            "Example: --dates 2022-01-16 2023-01-01"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download files that already exist locally.",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Report what is present locally without downloading.",
    )
    parser.add_argument(
        "--list-files", action="store_true",
        help="List all files in the GDrive folder without downloading.",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete downloaded tiles after verification (frees disk space).",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Parallel download workers (default: 2; GDrive throttles above 2-3)",
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Generate OAuth token via browser (run locally once, then copy to server).",
    )
    parser.add_argument(
        "--processed", action="store_true",
        help=(
            "Download processed S2 files (GDRIVE_PROCESSED_S2_FOLDER_IDS per year) "
            "into data/processed/s2/{year}/."
        ),
    )
    parser.add_argument(
        "--cdl", action="store_true",
        help=(
            "Download CDL files (GDRIVE_PROCESSED_CDL_FOLDER_ID) "
            "into data/processed/cdl/."
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

    # --cdl: download CDL files flat into data/processed/cdl/
    if args.cdl:
        from crop_mapping_pipeline.config import GDRIVE_PROCESSED_CDL_FOLDER_ID
        cdl_out = args.output_dir or str(_ROOT / "data" / "processed" / "cdl")
        log.info(f"=== Downloading CDL files → {cdl_out} ===")
        download_flat(
            folder_id  = GDRIVE_PROCESSED_CDL_FOLDER_ID,
            output_dir = cdl_out,
            overwrite  = args.overwrite,
        )
        sys.exit(0)

    # --processed: iterate per-year folders from GDRIVE_PROCESSED_S2_FOLDER_IDS
    if args.processed:
        from crop_mapping_pipeline.config import GDRIVE_PROCESSED_S2_FOLDER_IDS
        years_to_dl = args.years if args.years else list(GDRIVE_PROCESSED_S2_FOLDER_IDS.keys())
        base_out = args.output_dir or str(_ROOT / "data" / "processed" / "s2")
        for year in sorted(years_to_dl):
            if year not in GDRIVE_PROCESSED_S2_FOLDER_IDS:
                log.warning(f"No processed folder ID for year {year} — skipping")
                continue
            log.info(f"=== Downloading processed S2 for {year} → {base_out}/{year}/ ===")
            main(
                folder_id   = GDRIVE_PROCESSED_S2_FOLDER_IDS[year],
                output_dir  = base_out,
                years       = [year],
                dates       = args.dates,
                overwrite   = args.overwrite,
                verify_only = args.verify_only,
                list_files  = args.list_files,
                delete      = args.delete,
                workers     = args.workers,
            )
        sys.exit(0)

    # Resolve folder ID (raw mode)
    folder_id = args.folder_id
    if not folder_id:
        try:
            from crop_mapping_pipeline.config import GDRIVE_RAW_S2_V2_FOLDER_ID
            folder_id = GDRIVE_RAW_S2_V2_FOLDER_ID
        except ImportError:
            pass
    if not folder_id:
        parser.error(
            "--folder-id is required (or set GDRIVE_RAW_S2_V2_FOLDER_ID in config.py)"
        )

    output_dir = args.output_dir or str(_ROOT / "data" / "raw" / "s2")

    main(
        folder_id   = folder_id,
        output_dir  = output_dir,
        years       = args.years,
        dates       = args.dates,
        overwrite   = args.overwrite,
        verify_only = args.verify_only,
        list_files  = args.list_files,
        delete      = args.delete,
        workers     = args.workers,
    )
