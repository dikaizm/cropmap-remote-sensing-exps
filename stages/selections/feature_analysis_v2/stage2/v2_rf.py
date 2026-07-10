import logging
import os
from datetime import datetime

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import mlflow

import crop_mapping_pipeline.stages.feature_analysis_v2 as fa2

log = logging.getLogger(__name__)


def run_stage2v2_rf(s2_paths, cdl_path, date_candidates_per_crop, band_candidates_per_crop, band_name_to_idx, all_dates, data_dir=None):
    os.makedirs(fa2.PROCESSED_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = fa2.PROCESSED_DIR / f"stage2v2_rf_run_{run_ts}.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    fa2.mlflow_setup()
    parent_run = mlflow.start_run(run_name=f"stage2v3_rf_{run_ts}")
    mlflow.log_params(
        {
            "stage": "2v2_rf_date_band_selection",
            "selector": "rf",
            "rf_n_estimators": fa2.RF_N_ESTIMATORS,
            "rf_max_pixels": fa2.RF_MAX_PIXELS,
            "rf_importance_thresh": fa2.RF_IMPORTANCE_THRESH,
            "top_dates_per_crop": fa2.TOP_DATES_PER_CROP,
            "top_bands_per_crop": fa2.TOP_BANDS_PER_CROP,
            "max_bands_per_crop": fa2.MAX_BANDS_PER_CROP,
            "n_crops": len(fa2.KEEP_CLASSES),
        }
    )

    log.info("Pre-loading all patches into RAM (all bands)...")
    full_dataset = fa2.RasterPatchDataset(
        s2_paths=s2_paths,
        cdl_path=cdl_path,
        patch_size=fa2.S2_PATCH_SIZE,
        stride=fa2.S2_STRIDE,
        min_valid_frac=fa2.S2_MIN_VALID,
        band_indices=None,
        remap_lut=None,
        target_class_id=None,
    )
    preloaded_all = fa2.preload_patches(full_dataset)
    imgs_all, masks_all = preloaded_all.tensors
    _, channels, height, width = imgs_all.shape
    log.info(f"Pre-loaded: imgs={tuple(imgs_all.shape)}  masks={tuple(masks_all.shape)}")

    pixels_x = imgs_all.permute(0, 2, 3, 1).reshape(-1, channels).numpy().astype(np.float32)
    pixels_y = masks_all.reshape(-1).numpy().astype(np.int32)
    log.info(f"Flattened pixel matrix: {pixels_x.shape}  labels: {pixels_y.shape}")

    results_per_crop = {}
    n_crops = len(fa2.KEEP_CLASSES)

    log.info("\nStage 2v2-RF — per-crop binary RF feature selection (date + band phases)")
    log.info(
        f"  n_estimators={fa2.RF_N_ESTIMATORS}  max_pixels={fa2.RF_MAX_PIXELS}  "
        f"importance_thresh={fa2.RF_IMPORTANCE_THRESH}"
    )

    try:
        for crop_idx, crop_id in enumerate(fa2.KEEP_CLASSES, start=1):
            crop_name = fa2.CDL_CLASS_NAMES[crop_id]
            crop_key = str(crop_id)
            date_cands = date_candidates_per_crop[crop_key]
            band_cands = band_candidates_per_crop[crop_key]

            log.info(f"\n{'=' * 60}")
            log.info(
                f"[{crop_idx}/{n_crops}] Crop: {crop_name} (CDL id={crop_id}) "
                f"— {len(date_cands)} date candidates, {len(band_cands)} band candidates"
            )

            crop_px_mask = pixels_y == crop_id
            rest_px_mask = np.isin(pixels_y, fa2.KEEP_CLASSES) & ~crop_px_mask
            n_crop = crop_px_mask.sum()
            n_rest = rest_px_mask.sum()
            log.info(f"  Crop pixels: {n_crop:,}  rest pixels: {n_rest:,}")

            if n_crop < 50:
                log.warning(f"  Too few crop pixels ({n_crop}) — skipping")
                results_per_crop[crop_id] = {
                    "dates": date_cands[:1] if date_cands else [],
                    "bands": band_cands[:1] if band_cands else [],
                    "k_dates": 1,
                    "k_bands": 1,
                    "best_iou_after_dates": 0.0,
                    "best_iou_after_bands": 0.0,
                    "fallback_dates": True,
                    "fallback_bands": True,
                }
                continue

            rng = np.random.default_rng(42)
            n_sample_each = min(fa2.RF_MAX_PIXELS // 2, n_crop, n_rest)
            crop_idx_arr = rng.choice(np.where(crop_px_mask)[0], n_sample_each, replace=False)
            rest_idx_arr = rng.choice(np.where(rest_px_mask)[0], n_sample_each, replace=False)
            sample_idx = np.concatenate([crop_idx_arr, rest_idx_arr])
            x_sample = pixels_x[sample_idx]
            y_binary = np.concatenate(
                [
                    np.ones(n_sample_each, dtype=np.int32),
                    np.zeros(n_sample_each, dtype=np.int32),
                ]
            )
            log.info(f"  RF sample: {len(x_sample):,} pixels ({n_sample_each:,} crop + {n_sample_each:,} rest)")

            with mlflow.start_run(run_name=f"stage2v3_rf_{crop_name.replace('/', '-')}_{run_ts}", nested=True) as crop_run:
                mlflow.log_params(
                    {
                        "crop_id": crop_id,
                        "crop_name": crop_name,
                        "n_date_candidates": len(date_cands),
                        "n_band_candidates": len(band_cands),
                        "n_crop_pixels": int(n_crop),
                        "n_sample_pixels": len(x_sample),
                    }
                )

                log.info(f"\n  === Phase A: Date selection for {crop_name} ===")
                selected_dates = []
                fallback_dates = False

                if date_cands:
                    phase_a_cols, phase_a_feat_names = [], []
                    for date in date_cands:
                        for band in fa2.VEGE_BANDS:
                            key = f"{band}_{date}"
                            if key in band_name_to_idx:
                                phase_a_cols.append(band_name_to_idx[key])
                                phase_a_feat_names.append(key)

                    if phase_a_cols:
                        rf_a = RandomForestClassifier(
                            n_estimators=fa2.RF_N_ESTIMATORS,
                            n_jobs=-1,
                            random_state=42,
                            class_weight="balanced",
                        )
                        rf_a.fit(x_sample[:, phase_a_cols], y_binary)
                        date_importance = {}
                        for feat_idx, feat_name in enumerate(phase_a_feat_names):
                            date = feat_name.split("_", 1)[1]
                            date_importance.setdefault(date, []).append(rf_a.feature_importances_[feat_idx])
                        date_mean_imp = {date: float(np.mean(values)) for date, values in date_importance.items()}
                        max_imp = max(date_mean_imp.values()) if date_mean_imp else 1e-9
                        thresh = fa2.RF_IMPORTANCE_THRESH * max_imp
                        sorted_dates_imp = sorted(date_mean_imp.items(), key=lambda item: item[1], reverse=True)
                        selected_dates = [date for date, imp in sorted_dates_imp if imp >= thresh][: fa2.MAX_DATES_PER_CROP]
                        log.info("  Date importances: " + "  ".join(f"{date}={value:.4f}" for date, value in sorted_dates_imp[:5]))
                        mlflow.log_metrics({f"phaseA_imp_{date}": value for date, value in date_mean_imp.items()})
                    else:
                        log.warning(f"  Phase A: no valid features for {crop_name}")

                if not selected_dates and date_cands:
                    selected_dates = [date_cands[0]]
                    fallback_dates = True
                    log.warning(f"  K_dates=0 for {crop_name} — fallback to top-1: {date_cands[0]}")
                    mlflow.set_tag("phaseA_fallback_date", date_cands[0])

                log.info(f"\n  Phase A done: K_dates={len(selected_dates)}  dates={selected_dates}")
                mlflow.log_metric("phaseA_k_dates", len(selected_dates))
                mlflow.set_tag("phaseA_selected_dates", str(selected_dates))

                log.info(f"\n  === Phase B: Band selection for {crop_name} (fixed dates={selected_dates}) ===")
                selected_bands = []
                fallback_bands = False

                if band_cands and selected_dates:
                    phase_b_cols, phase_b_feat_names = [], []
                    for band in band_cands:
                        for date in selected_dates:
                            key = f"{band}_{date}"
                            if key in band_name_to_idx:
                                phase_b_cols.append(band_name_to_idx[key])
                                phase_b_feat_names.append(key)

                    if phase_b_cols:
                        rf_b = RandomForestClassifier(
                            n_estimators=fa2.RF_N_ESTIMATORS,
                            n_jobs=-1,
                            random_state=42,
                            class_weight="balanced",
                        )
                        rf_b.fit(x_sample[:, phase_b_cols], y_binary)
                        band_importance = {}
                        for feat_idx, feat_name in enumerate(phase_b_feat_names):
                            band = feat_name.split("_")[0]
                            band_importance.setdefault(band, []).append(rf_b.feature_importances_[feat_idx])
                        band_mean_imp = {band: float(np.mean(values)) for band, values in band_importance.items()}
                        max_imp = max(band_mean_imp.values()) if band_mean_imp else 1e-9
                        thresh = fa2.RF_IMPORTANCE_THRESH * max_imp
                        sorted_bands_imp = sorted(band_mean_imp.items(), key=lambda item: item[1], reverse=True)
                        selected_bands = [band for band, imp in sorted_bands_imp if imp >= thresh][: fa2.MAX_BANDS_PER_CROP]
                        log.info("  Band importances: " + "  ".join(f"{band}={value:.4f}" for band, value in sorted_bands_imp))
                        mlflow.log_metrics({f"phaseB_imp_{band}": value for band, value in band_mean_imp.items()})
                    else:
                        log.warning(f"  Phase B: no valid features for {crop_name}")

                if not selected_bands and band_cands:
                    selected_bands = [band_cands[0]]
                    fallback_bands = True
                    log.warning(f"  K_bands=0 for {crop_name} — fallback to top-1: {band_cands[0]}")
                    mlflow.set_tag("phaseB_fallback_band", band_cands[0])

                log.info(f"\n  Phase B done: K_bands={len(selected_bands)}  bands={selected_bands}")
                mlflow.log_metric("phaseB_k_bands", len(selected_bands))
                mlflow.set_tag("phaseB_selected_bands", str(selected_bands))

                results_per_crop[crop_id] = {
                    "dates": selected_dates,
                    "bands": selected_bands,
                    "k_dates": len(selected_dates),
                    "k_bands": len(selected_bands),
                    "best_iou_after_dates": 0.0,
                    "best_iou_after_bands": 0.0,
                    "fallback_dates": fallback_dates,
                    "fallback_bands": fallback_bands,
                    "mlflow_run_id": crop_run.info.run_id,
                }

            log.info(f"\n  -> [{crop_idx}/{n_crops}] {crop_name}: K_dates={len(selected_dates)}  K_bands={len(selected_bands)}")
            mlflow.log_metrics({f"crop_{crop_id}_k_dates": len(selected_dates), f"crop_{crop_id}_k_bands": len(selected_bands)})

        fa2.save_results_v2(
            results_per_crop,
            band_name_to_idx,
            per_crop_json=fa2.STAGE2V3_RF_PER_CROP_JSON,
            exp_json=fa2.STAGE3_EXP_C_V2_RF_JSON,
            exp_bands=fa2.STAGE3_EXP_C_V2_RF_BANDS,
        )
        mlflow.log_artifact(str(fa2.STAGE2V3_RF_PER_CROP_JSON))
        mlflow.log_artifact(str(fa2.STAGE3_EXP_C_V2_RF_JSON))
        mlflow.log_artifact(str(fa2.STAGE3_EXP_C_V2_RF_BANDS))
        log.info(f"Stage 2v2-RF parent run_id: {parent_run.info.run_id}")
        logging.getLogger().removeHandler(file_handler)
        file_handler.flush()
        file_handler.close()
        mlflow.log_artifact(str(log_path))
        mlflow.end_run(status="FINISHED")
    except Exception as exc:
        logging.getLogger().removeHandler(file_handler)
        file_handler.flush()
        file_handler.close()
        mlflow.log_artifact(str(log_path))
        mlflow.end_run(status="FAILED")
        raise exc

    return results_per_crop
