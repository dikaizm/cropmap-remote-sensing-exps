"""
Batch download + process for the v2 study area dataset (~194 GB total).

Downloads N dates at a time from a single flat GDrive folder, processes them
(merge tiles → assign NoData), optionally uploads processed files, then deletes
the raw tiles before moving to the next batch. CDL is processed once per year
using the first processed S2 file as the grid reference.

Usage:
    python stages/batch_process_v2.py --folder-id FOLDER_ID
    python stages/batch_process_v2.py --folder-id FOLDER_ID --batch-size 10
    python stages/batch_process_v2.py --folder-id FOLDER_ID --years 2022 2023
    python stages/batch_process_v2.py --folder-id FOLDER_ID --skip-upload --shutdown
    python stages/batch_process_v2.py --auth
"""

import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent   # crop_mapping_pipeline/
sys.path.insert(0, str(_ROOT.parent))

log = logging.getLogger(__name__)

ALL_YEARS    = ["2022", "2023", "2024"]
DEFAULT_BATCH = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _chunks(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    folder_id    : str,
    output_dir   : str,
    years        : list  = None,
    batch_size   : int   = DEFAULT_BATCH,
    data_dir     : str   = None,
    raw_cdl_dir  : str   = None,
    s2_folder_ids: dict  = None,
    cdl_folder_id: str   = None,
    skip_upload  : bool  = False,
    skip_delete  : bool  = False,
    keep_merged  : bool  = False,
    overwrite    : bool  = False,
    download_cdl : bool  = False,
    cdl_only     : bool  = False,
    shutdown     : bool  = False,
) -> None:
    from crop_mapping_pipeline.config import (
        GDRIVE_RAW_S2_V2_FOLDER_ID,
        GDRIVE_PROCESSED_V2_FOLDER_ID,
    )
    from crop_mapping_pipeline.stages.fetch_data_v2 import (
        download_dates,
        list_dates_by_year,
    )
    from crop_mapping_pipeline.stages.process_data_v2 import (
        _schedule_shutdown,
        delete_files,
        download_cdl_usda,
        group_tiles_by_date,
        process_cdl,
        process_date_batch,
        upload_file,
        upload_year,
    )
    import crop_mapping_pipeline.stages.process_data_v2 as pdv2

    # Allow data_dir override (mirrors process_data_v2 behaviour)
    if data_dir:
        processed        = pathlib.Path(data_dir)
        pdv2.PROCESSED_DIR    = processed
        pdv2.S2_PROCESSED_DIR = processed / "s2"
        pdv2.CDL_BY_YEAR      = {
            yr: processed / "cdl" / f"cdl_{yr}_study_area_filtered.tif"
            for yr in ALL_YEARS
        }

    years = years or ALL_YEARS

    # ── CDL download (optional) ───────────────────────────────────────────────
    cdl_dir = (pathlib.Path(raw_cdl_dir) if raw_cdl_dir
               else _ROOT / "data" / "raw" / "cdl")
    if download_cdl:
        log.info("Downloading CDL from USDA NASS for years: %s", years)
        for yr in years:
            download_cdl_usda(yr, cdl_dir)

    # ── Resolve upload folder IDs (needed by both cdl-only and full mode) ─────
    cdl_folder_id_resolved = cdl_folder_id
    if not skip_upload and not s2_folder_ids:
        try:
            from crop_mapping_pipeline.stages.process_data_v2 import _build_drive_service

            def _get_or_create_folder(svc, name, parent_id):
                q = (f"name='{name}' and '{parent_id}' in parents "
                     f"and mimeType='application/vnd.google-apps.folder' and trashed=false")
                res = svc.files().list(q=q, fields="files(id)").execute()
                if res.get("files"):
                    return res["files"][0]["id"]
                meta   = {"name": name, "mimeType": "application/vnd.google-apps.folder",
                          "parents": [parent_id]}
                folder = svc.files().create(body=meta, fields="id").execute()
                return folder["id"]

            _svc = _build_drive_service()
            s2_parent  = _get_or_create_folder(_svc, "s2",  GDRIVE_PROCESSED_V2_FOLDER_ID)
            cdl_parent = _get_or_create_folder(_svc, "cdl", GDRIVE_PROCESSED_V2_FOLDER_ID)
            s2_folder_ids = {
                yr: _get_or_create_folder(_svc, yr, s2_parent)
                for yr in years
            }
            cdl_folder_id_resolved = cdl_parent
            log.info("Upload folders resolved under v2 parent:")
            for yr, fid in s2_folder_ids.items():
                log.info("  s2/%s → %s", yr, fid)
            log.info("  cdl   → %s", cdl_folder_id_resolved)
        except Exception as e:
            log.warning("Could not resolve upload folders (%s) — upload will be skipped", e)
            s2_folder_ids = None

    # ── CDL-only mode: reprocess CDL without touching S2 ─────────────────────
    if cdl_only:
        from glob import glob as _glob
        for yr in years:
            log.info("CDL-only: processing year %s ...", yr)
            # Use any existing processed S2 file as grid reference
            s2_refs = sorted(_glob(str(pdv2.S2_PROCESSED_DIR / yr / "*_processed.tif")))
            if not s2_refs:
                log.warning("  No processed S2 files found for %s — skipping", yr)
                continue
            s2_ref_path  = s2_refs[0]
            cdl_out_dir  = pdv2.S2_PROCESSED_DIR.parent / "cdl"
            cdl_raw = next(
                (p for p in _glob(str(cdl_dir / f"{yr}_30m_cdls" / "*.tif"))), None
            )
            if not cdl_raw:
                log.warning("  Raw CDL for %s not found in %s — skipping", yr, cdl_dir)
                continue
            cdl_reprojected = str(cdl_out_dir / f"cdl_{yr}_study_area.tif")
            cdl_filtered    = str(cdl_out_dir / f"cdl_{yr}_study_area_filtered.tif")
            # Remove existing outputs so process_cdl rewrites them
            for p in (cdl_reprojected, cdl_filtered):
                if pathlib.Path(p).exists():
                    pathlib.Path(p).unlink()
                    log.info("  Removed existing: %s", pathlib.Path(p).name)
            process_cdl(cdl_raw, s2_ref_path, cdl_reprojected, cdl_filtered)
            log.info("  CDL %s done.", yr)
            if not skip_upload and cdl_folder_id_resolved:
                log.info("  Uploading CDL files → GDrive ...")
                upload_file(cdl_reprojected, cdl_folder_id_resolved)
                upload_file(cdl_filtered, cdl_folder_id_resolved)
        if shutdown:
            _schedule_shutdown(delay_min=8)
        return

    log.info("Listing dates from GDrive folder %s ...", folder_id)
    dates_by_year = list_dates_by_year(folder_id, years=years)
    for yr, dates in dates_by_year.items():
        log.info("  %s: %d date(s) found", yr, len(dates))

    # ── Filter already-processed dates ───────────────────────────────────────
    for yr in list(dates_by_year.keys()):
        s2_out_dir = pdv2.S2_PROCESSED_DIR / yr
        all_dates  = dates_by_year[yr]
        remaining  = [
            dk for dk in all_dates
            if not (s2_out_dir / f"{dk}_processed.tif").exists()
        ]
        skipped = len(all_dates) - len(remaining)
        if skipped:
            log.info("  %s: %d already processed, %d remaining", yr, skipped, len(remaining))
        dates_by_year[yr] = remaining


    for yr in years:
        all_dates = dates_by_year.get(yr, [])
        if not all_dates:
            log.info("Year %s — all dates already processed, skipping", yr)
            continue

        log.info("=" * 60)
        log.info("Year %s — %d date(s), batch size %d", yr, len(all_dates), batch_size)
        log.info("=" * 60)

        s2_out_dir    = pdv2.S2_PROCESSED_DIR / yr
        merge_tmp_dir = Path(output_dir) / yr / "_merged"
        s2_out_dir.mkdir(parents=True, exist_ok=True)
        merge_tmp_dir.mkdir(parents=True, exist_ok=True)

        cdl_processed = False
        s2_ref_path   = None     # set from first processed batch

        for batch_num, date_batch in enumerate(_chunks(all_dates, batch_size), start=1):
            log.info("─── Batch %d/%d  (%d dates: %s … %s)",
                     batch_num,
                     -(-len(all_dates) // batch_size),   # ceil div
                     len(date_batch),
                     date_batch[0], date_batch[-1])

            # ── Step 1: Download tiles for this batch ──────────────────────
            log.info("  [1/4] Downloading %d date(s)...", len(date_batch))
            download_dates(
                folder_id  = folder_id,
                output_dir = output_dir,
                date_keys  = date_batch,
                overwrite  = overwrite,
            )

            # ── Step 2: Group tiles by date (only for this batch) ──────────
            raw_yr_dir   = Path(output_dir) / yr
            all_groups   = group_tiles_by_date(str(raw_yr_dir), yr)
            batch_groups = {dk: all_groups[dk] for dk in date_batch if dk in all_groups}

            if not batch_groups:
                log.warning("  No tile groups found for batch — skipping")
                continue

            # ── Step 3: Merge + NoData ─────────────────────────────────────
            log.info("  [2/4] Processing %d date(s)...", len(batch_groups))
            raw_tiles, processed_paths, batch_ref = process_date_batch(
                date_groups   = batch_groups,
                yr            = yr,
                s2_out_dir    = s2_out_dir,
                merge_tmp_dir = merge_tmp_dir,
                keep_merged   = keep_merged,
            )

            if s2_ref_path is None and batch_ref:
                s2_ref_path = batch_ref

            # ── Step 4: CDL (once per year, after first batch) ────────────
            if not cdl_processed and s2_ref_path:
                log.info("  [3/4] Processing CDL for year %s...", yr)
                from glob import glob as _glob
                cdl_raw = next(
                    (p for p in _glob(str(cdl_dir / f"{yr}_30m_cdls" / "*.tif"))),
                    None,
                )
                if cdl_raw:
                    cdl_out_dir     = pdv2.S2_PROCESSED_DIR.parent / "cdl"
                    cdl_reprojected = str(cdl_out_dir / f"cdl_{yr}_study_area.tif")
                    cdl_filtered    = str(cdl_out_dir / f"cdl_{yr}_study_area_filtered.tif")
                    process_cdl(cdl_raw, s2_ref_path, cdl_reprojected, cdl_filtered)
                    cdl_processed = True
                else:
                    log.warning("  Raw CDL for %s not found — skipping CDL step", yr)
            else:
                log.info("  [3/4] CDL already processed — skipped")

            # ── Step 5: Upload ─────────────────────────────────────────────
            if not skip_upload:
                if s2_folder_ids and cdl_folder_id_resolved:
                    log.info("  [4/4] Uploading %d processed file(s)...",
                             len(processed_paths))
                    cdl_filtered_path = str(
                        pdv2.S2_PROCESSED_DIR.parent / "cdl" /
                        f"cdl_{yr}_study_area_filtered.tif"
                    )
                    upload_year(
                        s2_processed_paths = processed_paths,
                        cdl_filtered_path  = cdl_filtered_path if cdl_processed else "",
                        year               = yr,
                        s2_folder_ids      = s2_folder_ids,
                        cdl_folder_id      = cdl_folder_id_resolved,
                    )
                else:
                    log.warning("  [4/4] No GDrive folder IDs — skipping upload. "
                                "Use --skip-upload or provide --s2-folder-ids / --cdl-folder-id")
            else:
                log.info("  [4/4] Upload skipped (--skip-upload)")

            # ── Step 6: Delete raw tiles for this batch ───────────────────
            if not skip_delete:
                log.info("  Deleting %d raw tile(s)...", len(raw_tiles))
                delete_files(raw_tiles, label="tile")
            else:
                log.info("  Raw tiles kept (--skip-delete)")

            log.info("  Batch %d done.\n", batch_num)

        log.info("Year %s complete.\n", yr)

    if shutdown:
        _schedule_shutdown(delay_min=8)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch download + process v2 study area tiles (~194 GB)."
    )
    parser.add_argument(
        "--folder-id", default=None,
        help="GDrive folder ID (flat, all years). Falls back to GDRIVE_RAW_S2_V2_FOLDER_ID.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Local directory for raw tiles (year subdirs created automatically). "
             "Default: data/raw/s2/ relative to project root.",
    )
    parser.add_argument(
        "--years", nargs="+", default=None, choices=ALL_YEARS, metavar="YEAR",
        help="Only process these years (default: all).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help=f"Number of dates per batch (default: {DEFAULT_BATCH}).",
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Override processed data directory (default: data/processed/).",
    )
    parser.add_argument(
        "--raw-cdl-dir", default=None,
        help="Directory containing raw CDL files (default: data/raw/cdl/).",
    )
    parser.add_argument(
        "--s2-folder-ids", nargs="+", default=None, metavar="YEAR:ID",
        help="GDrive folder IDs for processed S2 upload, e.g. 2022:1ABC 2023:1DEF.",
    )
    parser.add_argument(
        "--cdl-folder-id", default=None,
        help="GDrive folder ID for processed CDL upload.",
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Do not upload processed files to GDrive.",
    )
    parser.add_argument(
        "--skip-delete", action="store_true",
        help="Keep raw tiles after processing (useful for debugging).",
    )
    parser.add_argument(
        "--keep-merged", action="store_true",
        help="Keep intermediate merged TIFs (pre-NoData assignment).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-download tiles that already exist locally.",
    )
    parser.add_argument(
        "--download-cdl", action="store_true",
        help=(
            "Download California CDL (FIPS=06) from USDA NASS CropScape for each year "
            "before processing.  Saves to --raw-cdl-dir (default: data/raw/cdl/)."
        ),
    )
    parser.add_argument(
        "--cdl-only", action="store_true",
        help=(
            "Reprocess CDL only — skip S2 download and processing entirely. "
            "Uses the first existing processed S2 file as the grid reference. "
            "Combine with --download-cdl to also re-download raw CDL first."
        ),
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Stop the RunPod pod after all batches complete.",
    )
    parser.add_argument(
        "--auth", action="store_true",
        help="Generate OAuth token via browser (run locally once, then copy to server).",
    )
    args = parser.parse_args()

    if args.auth:
        from crop_mapping_pipeline.stages.fetch_data_v2 import generate_oauth_token
        generate_oauth_token()
        sys.exit(0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Resolve folder ID
    folder_id = args.folder_id
    if not folder_id:
        try:
            from crop_mapping_pipeline.config import GDRIVE_RAW_S2_V2_FOLDER_ID
            folder_id = GDRIVE_RAW_S2_V2_FOLDER_ID
        except ImportError:
            pass
    if not folder_id:
        parser.error("--folder-id is required (or set GDRIVE_RAW_S2_V2_FOLDER_ID in config.py)")

    output_dir = args.output_dir or str(_ROOT / "data" / "raw" / "s2")

    s2_folder_ids = None
    if args.s2_folder_ids:
        s2_folder_ids = {}
        for item in args.s2_folder_ids:
            yr, fid = item.split(":", 1)
            s2_folder_ids[yr] = fid

    main(
        folder_id     = folder_id,
        output_dir    = output_dir,
        years         = args.years,
        batch_size    = args.batch_size,
        data_dir      = args.data_dir,
        raw_cdl_dir   = args.raw_cdl_dir,
        s2_folder_ids = s2_folder_ids,
        cdl_folder_id = args.cdl_folder_id,
        skip_upload   = args.skip_upload,
        skip_delete   = args.skip_delete,
        keep_merged   = args.keep_merged,
        overwrite     = args.overwrite,
        download_cdl  = args.download_cdl,
        cdl_only      = args.cdl_only,
        shutdown      = args.shutdown,
    )
