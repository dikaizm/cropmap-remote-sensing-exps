"""
Stage 2v3 — Incremental Top-K Channel Enumeration

Continuation of Stage 1v3 (date + band SI ranking).
No training — purely enumerates channel combinations for Stage 3.

Phase A — Band sweep on best date:
  Fix to rank-1 date (highest SI). For k = 1…len(band_cands):
    take top-k bands × top-1 date → channel set.
  Produces band_sweep_k1, band_sweep_k2, … per crop.

Phase B — Temporal sweep with all bands:
  Use all band candidates (full ranked list from Stage 1v3).
  For k = 1…len(date_cands):
    take all_bands × top-k dates → channel set.
  Produces date_sweep_k1, date_sweep_k2, … per crop.

For each k the union across all crops is saved as a Stage 3 input:
  stage3_exp_c_v3_band_sweep_k{k}_bands.txt
  stage3_exp_c_v3_date_sweep_k{k}_bands.txt

A full structured JSON is also saved:
  stage2v3_sweep_per_crop_results.json
"""

import json
import logging
import os
import pathlib
import tempfile
from datetime import datetime

import pandas as pd
import mlflow

import crop_mapping_pipeline.stages.feature_analysis_v2 as fa2

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _channels_for(dates: list[str], bands: list[str], band_name_to_idx: dict) -> list[str]:
    """Return ordered channel names for (dates × bands) that exist in the index."""
    return [
        f"{band}_{date}"
        for date in dates
        for band in bands
        if f"{band}_{date}" in band_name_to_idx
    ]


def _union_channels(per_crop: dict[int, list[str]]) -> list[str]:
    """Ordered union of channel lists across all crops (insertion order, no dupes)."""
    seen: set[str] = set()
    union: list[str] = []
    for channels in per_crop.values():
        for ch in channels:
            if ch not in seen:
                seen.add(ch)
                union.append(ch)
    return union


def _save_channel_file(channels: list[str], path: pathlib.Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    path.write_text("\n".join(channels))
    log.info(f"Saved {len(channels):>3} channels → {path.name}")


# ── main entry ────────────────────────────────────────────────────────────────

def run_stage2v3(
    date_candidates_per_crop: dict,
    band_candidates_per_crop: dict,
    band_name_to_idx: dict,
    data_dir: str | None = None,
) -> dict:
    """
    Generate all Phase A and Phase B channel combinations from Stage 1v3 output.

    Parameters
    ----------
    date_candidates_per_crop : {crop_key: [date, ...]} from Stage 1v3 (SI-ranked)
    band_candidates_per_crop : {crop_key: [band, ...]} from Stage 1v3 (SI-ranked)
    band_name_to_idx         : {"{band}_{date}": channel_index}
    data_dir                 : optional override for output paths

    Returns
    -------
    dict with keys:
        "band_sweep"  → {k: {"per_crop": {crop_id: [channels]}, "union": [channels]}}
        "date_sweep"  → {k: {"per_crop": {crop_id: [channels]}, "union": [channels]}}
        "per_crop"    → {crop_id: {crop_name, band_sweep_channels, date_sweep_channels}}
    """
    if data_dir:
        fa2.configure_data_dir(data_dir)

    os.makedirs(fa2.PROCESSED_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")

    fa2.mlflow_setup()
    mlflow_run = mlflow.start_run(run_name=f"stage2v3_sweep_{run_ts}")
    mlflow.log_params(
        {
            "stage": "2v3_topk_enumeration",
            "version": "v3",
            "n_crops": len(fa2.KEEP_CLASSES),
            "top_dates_per_crop": fa2.TOP_DATES_PER_CROP,
            "top_bands_per_crop": fa2.TOP_BANDS_PER_CROP,
        }
    )

    band_sweep: dict[int, dict] = {}   # k → {per_crop, union}
    date_sweep: dict[int, dict] = {}   # k → {per_crop, union}
    per_crop_summary: dict = {}

    # max sweep sizes (across all crops)
    max_k_bands = max(len(v) for v in band_candidates_per_crop.values())
    max_k_dates = max(len(v) for v in date_candidates_per_crop.values())

    log.info(f"Stage 2v3: enumerating band sweep k=1..{max_k_bands}, date sweep k=1..{max_k_dates}")
    log.info(f"Crops: {[fa2.CDL_CLASS_NAMES[c] for c in fa2.KEEP_CLASSES]}")

    # ── Phase A: band sweep on top-1 date ────────────────────────────────────
    log.info("\n=== Phase A: Band sweep (top-1 date per crop) ===")
    for k in range(1, max_k_bands + 1):
        per_crop_k: dict[int, list[str]] = {}
        for crop_id in fa2.KEEP_CLASSES:
            crop_key = str(crop_id)
            date_cands = date_candidates_per_crop.get(crop_key, [])
            band_cands = band_candidates_per_crop.get(crop_key, [])
            if not date_cands or not band_cands:
                per_crop_k[crop_id] = []
                continue
            top1_date = [date_cands[0]]
            top_k_bands = band_cands[:k]
            per_crop_k[crop_id] = _channels_for(top1_date, top_k_bands, band_name_to_idx)

        union = _union_channels(per_crop_k)
        band_sweep[k] = {"per_crop": {str(c): v for c, v in per_crop_k.items()}, "union": union}

        out_path = fa2.PROCESSED_DIR / f"stage3_exp_c_v3_band_sweep_k{k:02d}_bands.txt"
        _save_channel_file(union, out_path)
        mlflow.log_metric("band_sweep_union_channels", len(union), step=k)
        try:
            mlflow.log_artifact(str(out_path))
        except Exception:
            pass

    # ── Phase B: date sweep with all bands ───────────────────────────────────
    log.info("\n=== Phase B: Date sweep (all bands per crop) ===")
    for k in range(1, max_k_dates + 1):
        per_crop_k: dict[int, list[str]] = {}
        for crop_id in fa2.KEEP_CLASSES:
            crop_key = str(crop_id)
            date_cands = date_candidates_per_crop.get(crop_key, [])
            band_cands = band_candidates_per_crop.get(crop_key, [])
            if not date_cands or not band_cands:
                per_crop_k[crop_id] = []
                continue
            top_k_dates = date_cands[:k]
            per_crop_k[crop_id] = _channels_for(top_k_dates, band_cands, band_name_to_idx)

        union = _union_channels(per_crop_k)
        date_sweep[k] = {"per_crop": {str(c): v for c, v in per_crop_k.items()}, "union": union}

        out_path = fa2.PROCESSED_DIR / f"stage3_exp_c_v3_date_sweep_k{k:02d}_bands.txt"
        _save_channel_file(union, out_path)
        mlflow.log_metric("date_sweep_union_channels", len(union), step=k)
        try:
            mlflow.log_artifact(str(out_path))
        except Exception:
            pass

    # ── Per-crop summary ──────────────────────────────────────────────────────
    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        date_cands = date_candidates_per_crop.get(crop_key, [])
        band_cands = band_candidates_per_crop.get(crop_key, [])
        per_crop_summary[crop_key] = {
            "crop_name": fa2.CDL_CLASS_NAMES[crop_id],
            "top1_date": date_cands[0] if date_cands else None,
            "band_sweep_channels": {
                str(k): band_sweep[k]["per_crop"].get(crop_key, [])
                for k in band_sweep
            },
            "date_sweep_channels": {
                str(k): date_sweep[k]["per_crop"].get(crop_key, [])
                for k in date_sweep
            },
        }

    result = {
        "run_ts": run_ts,
        "band_sweep": band_sweep,
        "date_sweep": date_sweep,
        "per_crop": per_crop_summary,
    }

    # ── Save structured JSON ──────────────────────────────────────────────────
    json_path = fa2.STAGE2V3_SWEEP_PER_CROP_JSON
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved: {json_path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    rows = []
    for crop_id in fa2.KEEP_CLASSES:
        crop_key = str(crop_id)
        crop_name = fa2.CDL_CLASS_NAMES[crop_id]
        date_cands = date_candidates_per_crop.get(crop_key, [])
        band_cands = band_candidates_per_crop.get(crop_key, [])
        for k, sweep_data in band_sweep.items():
            ch = sweep_data["per_crop"].get(crop_key, [])
            rows.append({
                "phase": "A_band_sweep",
                "crop_id": crop_id,
                "crop_name": crop_name,
                "k": k,
                "top1_date": date_cands[0] if date_cands else "",
                "bands_used": str(band_cands[:k]),
                "n_channels": len(ch),
                "channels": str(ch),
            })
        for k, sweep_data in date_sweep.items():
            ch = sweep_data["per_crop"].get(crop_key, [])
            rows.append({
                "phase": "B_date_sweep",
                "crop_id": crop_id,
                "crop_name": crop_name,
                "k": k,
                "top1_date": date_cands[0] if date_cands else "",
                "bands_used": str(band_cands),
                "n_channels": len(ch),
                "channels": str(ch),
            })

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = pathlib.Path(tmp) / "stage2v3_sweep_summary.csv"
        csv_path.write_text(pd.DataFrame(rows).to_csv(index=False))
        mlflow.log_artifact(str(csv_path))

    mlflow.log_artifact(str(json_path))
    mlflow.log_metrics({
        "n_band_sweep_variants": max_k_bands,
        "n_date_sweep_variants": max_k_dates,
    })
    mlflow.end_run(status="FINISHED")
    log.info(f"Stage 2v3 run_id: {mlflow_run.info.run_id}")
    log.info(f"Phase A: {max_k_bands} band-sweep variants generated")
    log.info(f"Phase B: {max_k_dates} date-sweep variants generated")

    return result
