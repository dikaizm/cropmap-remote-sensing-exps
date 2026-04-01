import logging
import os
import pathlib
import re
import tempfile
from datetime import datetime
import json

import numpy as np
import pandas as pd
import rasterio

import mlflow

import crop_mapping_pipeline.stages.feature_analysis_v2 as fa2

log = logging.getLogger(__name__)


def _sample_year(s2_paths: list[str], cdl_path: str) -> tuple[list[str], list[str], pd.DataFrame]:
    """Load one year's S2 + CDL, return (bandnames, dates, sampled pixel DataFrame).
    Reads one S2 file at a time to avoid stacking all 25 files (~28 GB) into RAM.
    """
    all_bandnames = []
    all_dates_set = []
    for s2_path in s2_paths:
        fname = os.path.basename(s2_path)
        match = re.search(r"_(\d{4})_(\d{2})_(\d{2})(_processed)?\.tif$", fname)
        date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}" if match else fname[:8]
        if date_str not in all_dates_set:
            all_dates_set.append(date_str)
        all_bandnames.extend([f"{band}_{date_str}" for band in fa2.S2_BAND_NAMES])

    all_dates = sorted(all_dates_set)
    n_channels = len(all_bandnames)
    log.info(f"  {len(s2_paths)} files | {n_channels} channels | {len(all_dates)} dates")

    # Read CDL once to get valid pixel indices
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1).astype(np.int32)
        height, width = cdl.shape

    lbl_1d = cdl.flatten()
    del cdl

    valid_mask = np.isin(lbl_1d, fa2.KEEP_CLASSES)
    valid_indices = np.where(valid_mask)[0]
    lbl_valid = lbl_1d[valid_mask]
    del lbl_1d

    rng = np.random.default_rng(42)
    n = min(len(valid_indices), max(1000, int(len(valid_indices) * fa2.SAMPLE_FRACTION)))
    chosen = rng.choice(len(valid_indices), n, replace=False)
    sample_flat_idx = valid_indices[chosen]
    lbl_sample = lbl_valid[chosen]
    del valid_indices, lbl_valid

    log.info(f"  Sampled {n:,} pixels — reading {len(s2_paths)} files one at a time...")

    # Fill data column-by-column, one S2 file at a time
    data = np.full((n, n_channels), np.nan, dtype=np.float32)
    col = 0
    for s2_path in s2_paths:
        with rasterio.open(s2_path) as src:
            n_file_bands = src.count
            arr = src.read().astype(np.float32)
        arr[arr == fa2.S2_NODATA] = np.nan
        arr_2d = arr.reshape(n_file_bands, -1).T
        del arr
        data[:, col:col + n_file_bands] = arr_2d[sample_flat_idx]
        del arr_2d
        col += n_file_bands

    df = pd.DataFrame(data, columns=all_bandnames)
    df.insert(0, "class_label", lbl_sample.astype(int))
    log.info(f"  DataFrame shape: {df.shape}")

    return all_bandnames, all_dates, df


def _compute_gsi(df: pd.DataFrame, bandnames: list[str]) -> dict[int, pd.Series]:
    """Per-crop binary SI_global (one-vs-all). Returns {crop_id: pd.Series(index=bandnames)}."""
    x_all = df[bandnames].values.astype(np.float32)
    y_all = df["class_label"].values

    gsi_dict = {}
    for crop_id in fa2.KEEP_CLASSES:
        crop_mask = y_all == crop_id
        rest_mask = np.isin(y_all, fa2.KEEP_CLASSES) & ~crop_mask
        if crop_mask.sum() < 10:
            log.warning(
                f"  Crop {crop_id} ({fa2.CDL_CLASS_NAMES[crop_id]}) has only "
                f"{crop_mask.sum()} samples — using zeros"
            )
            gsi_dict[crop_id] = pd.Series(0.0, index=bandnames)
            continue
        x_crop = x_all[crop_mask]
        x_rest = x_all[rest_mask]
        mean_crop = np.nanmean(x_crop, axis=0)
        std_crop  = np.nanstd(x_crop, axis=0)
        mean_rest = np.nanmean(x_rest, axis=0)
        std_rest  = np.nanstd(x_rest, axis=0)
        # SI(j,k) = |mean_s - mean_o| / (1.96 * (std_s + std_o))  — Li et al. 2023 (rs15040875),
        # adapted from Somers & Asner 2013 (RSE 136:14-27).
        si = np.abs(mean_crop - mean_rest) / (1.96 * (std_crop + std_rest) + 1e-9)
        gsi_dict[crop_id] = pd.Series(si.astype(np.float32), index=bandnames)

    return gsi_dict


def _band_level_gsi(gsi_dict: dict[int, pd.Series], bandnames: list[str]) -> dict[int, pd.Series]:
    """Collapse date dimension — return {crop_id: Series(index=S2_BAND_NAMES)} with mean SI per band."""
    result = {}
    for crop_id, si_series in gsi_dict.items():
        band_si = {}
        for band in fa2.S2_BAND_NAMES:
            keys = [k for k in bandnames if k.startswith(f"{band}_")]
            band_si[band] = float(si_series[keys].mean()) if keys else 0.0
        result[crop_id] = pd.Series(band_si)
    return result


def run_stage1v3(
    years_data: list[tuple[str, list[str], str]],
    data_dir: str | None = None,
):
    """Stage 1v3: per-crop GSI over all training years.

    years_data: [(year, s2_paths, cdl_path), ...]
      - Date candidates come from the primary year (first entry, 2022) only,
        so they remain compatible with Stage 2 which operates on 2022 files.
      - Band ranking averages band-level GSI across all years for robustness.
    """
    log.info("Stage 1v3: computing per-crop GSI across training years...")
    log.info(f"  Years: {[yr for yr, _, _ in years_data]}")

    fa2.mlflow_setup()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage1_run = mlflow.start_run(run_name=f"stage1v3_{ts}")

    # ── Per-year GSI ──────────────────────────────────────────────────────────
    year_gsi: dict[str, dict[int, pd.Series]] = {}       # year → {crop → channel SI}
    year_band_gsi: dict[str, dict[int, pd.Series]] = {}  # year → {crop → band SI}
    primary_bandnames: list[str] = []
    primary_dates: list[str] = []
    primary_band_name_to_idx: dict[str, int] = {}
    total_files = 0

    for i, (year, s2_paths, cdl_path) in enumerate(years_data):
        log.info(f"Loading year {year} ({len(s2_paths)} files)...")
        bandnames, dates, df = _sample_year(s2_paths, cdl_path)
        gsi_dict = _compute_gsi(df, bandnames)
        year_gsi[year] = gsi_dict
        year_band_gsi[year] = _band_level_gsi(gsi_dict, bandnames)
        total_files += len(s2_paths)

        if i == 0:
            primary_bandnames = bandnames
            primary_dates = dates
            primary_band_name_to_idx = {name: idx for idx, name in enumerate(bandnames)}
            primary_df = df  # keep for logging stats

    n_channels = len(primary_bandnames)
    log.info(f"Primary year channels: {n_channels}  |  dates: {len(primary_dates)}")

    # ── Date ranking — primary year only ─────────────────────────────────────
    primary_gsi = year_gsi[years_data[0][0]]
    primary_gsi_df = pd.DataFrame(primary_gsi)
    gsi_mean_global = primary_gsi_df.mean(axis=1).sort_values(ascending=False)

    values = primary_gsi_df.values
    log.info(
        f"Primary GSI  SI range:  min={np.nanmin(values):.4f}  "
        f"median={np.nanmedian(values):.4f}  max={np.nanmax(values):.4f}"
    )
    log.info(
        f"Top-K selection: TOP_DATES_PER_CROP={fa2.TOP_DATES_PER_CROP}  "
        f"TOP_BANDS_PER_CROP={fa2.TOP_BANDS_PER_CROP}"
    )

    date_candidates_per_crop: dict[str, list[str]] = {}

    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        si_crop = primary_gsi_df[crop_id] if crop_id in primary_gsi_df.columns else gsi_mean_global

        date_si = {}
        for date in primary_dates:
            band_keys = [f"{band}_{date}" for band in fa2.S2_BAND_NAMES if f"{band}_{date}" in si_crop.index]
            date_si[date] = float(si_crop[band_keys].mean()) if band_keys else 0.0
        sorted_dates = sorted(date_si.items(), key=lambda item: item[1], reverse=True)
        date_candidates_per_crop[crop_key] = [d for d, _ in sorted_dates[: fa2.TOP_DATES_PER_CROP]]

    # ── Band ranking — averaged across all years ──────────────────────────────
    band_candidates_per_crop: dict[str, list[str]] = {}
    n_years = len(year_band_gsi)

    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        avg_band_si: dict[str, float] = {}
        for band in fa2.S2_BAND_NAMES:
            scores = []
            for yr_band_gsi in year_band_gsi.values():
                if crop_id in yr_band_gsi and band in yr_band_gsi[crop_id].index:
                    scores.append(float(yr_band_gsi[crop_id][band]))
            avg_band_si[band] = float(np.mean(scores)) if scores else 0.0

        sorted_bands = sorted(avg_band_si.items(), key=lambda item: item[1], reverse=True)
        band_candidates_per_crop[crop_key] = [b for b, _ in sorted_bands[: fa2.TOP_BANDS_PER_CROP]]

        log.info(
            f"  {fa2.CDL_CLASS_NAMES[crop_id]:20s}: "
            f"top dates={date_candidates_per_crop[crop_key][:3]}...  "
            f"top bands={band_candidates_per_crop[crop_key][:3]}..."
            f"  (band GSI averaged over {n_years} year(s))"
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    stage1_path = fa2.STAGE1V3_CANDIDATES_JSON
    if data_dir:
        stage1_path = pathlib.Path(data_dir) / "s2" / "2022" / "stage1v3_candidates.json"
    os.makedirs(os.path.dirname(stage1_path), exist_ok=True)

    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "run_ts": run_ts,
        "years": [yr for yr, _, _ in years_data],
        "primary_year": years_data[0][0],
        "all_dates": primary_dates,
        "date_candidates_per_crop": date_candidates_per_crop,
        "band_candidates_per_crop": band_candidates_per_crop,
    }
    with open(stage1_path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(f"Stage 1v3 candidates saved: {stage1_path}")

    fa2.save_exp_d_bands(date_candidates_per_crop, band_candidates_per_crop,
                         primary_band_name_to_idx, data_dir=data_dir)

    # ── MLflow logging ────────────────────────────────────────────────────────
    nan_pixels = np.isnan(primary_df[primary_bandnames].values).any(axis=1).sum()
    mlflow.log_params({
        "stage":             "1v3_date_band_ranking",
        "version":           "v3",
        "years":             str([yr for yr, _, _ in years_data]),
        "primary_year":      years_data[0][0],
        "n_train_years":     n_years,
        "n_images_total":    total_files,
        "n_dates_primary":   len(primary_dates),
        "total_channels":    n_channels,
        "sample_fraction":   fa2.SAMPLE_FRACTION,
        "n_sampled_primary": len(primary_df),
        "nan_pixels":        int(nan_pixels),
        "top_dates_per_crop": fa2.TOP_DATES_PER_CROP,
        "top_bands_per_crop": fa2.TOP_BANDS_PER_CROP,
        "keep_classes":      str(fa2.KEEP_CLASSES),
    })

    rows = []
    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        for rank, date in enumerate(date_candidates_per_crop[crop_key], start=1):
            rows.append({"crop_id": crop_id, "crop_name": fa2.CDL_CLASS_NAMES[crop_id],
                         "type": "date", "rank": rank, "value": date})
        for rank, band in enumerate(band_candidates_per_crop[crop_key], start=1):
            rows.append({"crop_id": crop_id, "crop_name": fa2.CDL_CLASS_NAMES[crop_id],
                         "type": "band", "rank": rank, "value": band})

    with tempfile.TemporaryDirectory() as tmp:
        artifact_path = pathlib.Path(tmp) / "stage1v3_per_crop_candidates.csv"
        artifact_path.write_text(pd.DataFrame(rows).to_csv(index=False))
        mlflow.log_artifact(str(artifact_path))
    mlflow.log_artifact(str(stage1_path))
    if fa2.STAGE3_EXP_D_JSON.exists():
        mlflow.log_artifact(str(fa2.STAGE3_EXP_D_JSON))
    if fa2.STAGE3_EXP_D_BANDS.exists():
        mlflow.log_artifact(str(fa2.STAGE3_EXP_D_BANDS))

    heatmap_dir = fa2.FIGURES_DIR / "stage1v3_gsi"
    for heatmap_path in fa2.plot_gsi_heatmaps(primary_gsi_df, primary_dates, heatmap_dir):
        mlflow.log_artifact(str(heatmap_path))

    mlflow.end_run(status="FINISHED")
    log.info(f"Stage 1v3 MLflow run_id: {stage1_run.info.run_id}")

    return date_candidates_per_crop, band_candidates_per_crop, primary_band_name_to_idx, primary_dates
