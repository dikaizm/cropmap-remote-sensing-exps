import logging
import os
import time
from datetime import datetime

import numpy as np

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import crop_mapping_pipeline.stages.feature_analysis_v2 as fa2

log = logging.getLogger(__name__)


def _train_trial_unet(crop_imgs, crop_masks, trial_idx):
    trial_ds = TensorDataset(crop_imgs[:, trial_idx, :, :], crop_masks)
    train_ds, val_ds = fa2.split_tensor_dataset(trial_ds)
    train_dl, val_dl, n_workers, use_pin = fa2.build_dataloaders(train_ds, val_ds)
    model = fa2.build_unet(len(trial_idx))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_epoch_iou, no_improve_ep = 0.0, 0
    for _epoch in range(fa2.S2_EPOCHS):
        model.train()
        for imgs_batch, masks_batch in train_dl:
            imgs_batch = imgs_batch.to(fa2.DEVICE)
            masks_batch = masks_batch.to(fa2.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs_batch), masks_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs_batch, masks_batch in val_dl:
                preds = model(imgs_batch.to(fa2.DEVICE)).argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(masks_batch)
        iou = fa2.compute_iou_class1(torch.cat(all_preds), torch.cat(all_labels))
        if iou > best_epoch_iou + 1e-4:
            best_epoch_iou = iou
            no_improve_ep = 0
        else:
            no_improve_ep += 1
            if no_improve_ep >= fa2.S2_PATIENCE:
                break

    return best_epoch_iou, len(train_ds), len(val_ds), n_workers, use_pin


def run_stage2v2(s2_paths, cdl_path, date_candidates_per_crop, band_candidates_per_crop, band_name_to_idx, all_dates, data_dir=None):
    os.makedirs(fa2.PROCESSED_DIR, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = fa2.PROCESSED_DIR / f"stage2v2_run_{run_ts}.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(file_handler)

    fa2.mlflow_setup()
    parent_run = mlflow.start_run(run_name=f"stage2v3_{run_ts}")
    mlflow.log_params(
        {
            "stage": "2v2_date_band_fwd_selection",
            "version": "v2",
            "encoder": fa2.S2_ENCODER,
            "patch_size": fa2.S2_PATCH_SIZE,
            "stride": fa2.S2_STRIDE,
            "min_valid": fa2.S2_MIN_VALID,
            "epochs": fa2.S2_EPOCHS,
            "patience": fa2.S2_PATIENCE,
            "date_delta": fa2.S2_DATE_DELTA,
            "date_no_improve": fa2.S2_DATE_NO_IMPROVE,
            "max_dates": fa2.S2_MAX_DATES,
            "band_delta": fa2.S2_BAND_DELTA,
            "band_no_improve": fa2.S2_BAND_NO_IMPROVE,
            "max_bands": fa2.S2_MAX_BANDS_V2,
            "top_dates_per_crop": fa2.TOP_DATES_PER_CROP,
            "top_bands_per_crop": fa2.TOP_BANDS_PER_CROP,
            "max_bands_per_crop": fa2.MAX_BANDS_PER_CROP,
            "device": fa2.DEVICE,
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
    log.info(f"Pre-loaded: imgs={tuple(imgs_all.shape)}  masks={tuple(masks_all.shape)}")

    results_per_crop = {}
    n_crops = len(fa2.KEEP_CLASSES)

    log.info("\nStage 2v2 — per-crop binary CNN forward selection (date + band phases)")
    log.info(
        f"  Crops: {n_crops}  |  δ_date={fa2.S2_DATE_DELTA}  δ_band={fa2.S2_BAND_DELTA}  "
        f"max_dates={fa2.S2_MAX_DATES}  max_bands={fa2.S2_MAX_BANDS_V2}"
    )
    log.info(
        f"  Epochs={fa2.S2_EPOCHS}  patience={fa2.S2_PATIENCE}  "
        f"batch={fa2.S2_BATCH_SIZE}  patch={fa2.S2_PATCH_SIZE}px  stride={fa2.S2_STRIDE}px"
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

            binary_masks = (masks_all == crop_id).long()
            crop_has_class = binary_masks.sum(dim=(1, 2)) > 0
            crop_imgs = imgs_all[crop_has_class]
            crop_masks = binary_masks[crop_has_class]
            crop_tensor_ds = TensorDataset(crop_imgs, crop_masks)

            log.info(f"  Crop patches (has class {crop_id}): {len(crop_tensor_ds)}")
            if len(crop_tensor_ds) < 4:
                log.warning(f"  Too few patches for crop {crop_id} — skipping")
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

            with mlflow.start_run(run_name=f"stage2v3_{crop_name.replace('/', '-')}_{run_ts}", nested=True) as crop_run:
                mlflow.log_params(
                    {
                        "crop_id": crop_id,
                        "crop_name": crop_name,
                        "n_date_candidates": len(date_cands),
                        "n_band_candidates": len(band_cands),
                    }
                )

                log.info(f"\n  === Phase A: Date selection for {crop_name} ===")
                selected_dates, best_iou_dates, no_improve_dates = [], 0.0, 0

                for step, date in enumerate(date_cands):
                    if len(selected_dates) >= fa2.S2_MAX_DATES:
                        log.info(f"  max_dates={fa2.S2_MAX_DATES} reached — stopping Phase A")
                        break
                    if no_improve_dates >= fa2.S2_DATE_NO_IMPROVE:
                        log.info(f"  {fa2.S2_DATE_NO_IMPROVE} consecutive date rejections — stopping Phase A")
                        break

                    trial_dates = selected_dates + [date]
                    trial_idx = fa2.dates_to_band_indices(trial_dates, band_name_to_idx)
                    if not trial_idx:
                        log.warning(f"  date={date}: no valid band indices — skipping")
                        no_improve_dates += 1
                        continue

                    log.info(
                        f"\n  --- Phase A step {step + 1}/{len(date_cands)}  "
                        f"trial date: {date}  (selected: {len(selected_dates)}) ---"
                    )
                    t0 = time.time()
                    iou, n_train, n_val, n_workers, use_pin = _train_trial_unet(crop_imgs, crop_masks, trial_idx)
                    if step == 0:
                        log.info(
                            f"    U-Net  in_ch={len(trial_idx)}  train={n_train} patches  val={n_val} patches  "
                            f"max_epochs={fa2.S2_EPOCHS}  patience={fa2.S2_PATIENCE}  "
                            f"workers={n_workers}  pin_memory={use_pin}"
                        )
                    elapsed = time.time() - t0
                    gain = iou - best_iou_dates
                    accepted = gain >= fa2.S2_DATE_DELTA

                    if accepted:
                        selected_dates = selected_dates + [date]
                        best_iou_dates = iou
                        no_improve_dates = 0
                    else:
                        no_improve_dates += 1

                    mlflow.log_metrics(
                        {
                            "phaseA_iou": iou,
                            "phaseA_gain": gain,
                            "phaseA_accepted": int(accepted),
                            "phaseA_n_dates": len(selected_dates),
                        },
                        step=step,
                    )
                    log.info(f"  [{'OK' if accepted else '--'}] +{date}  IoU={iou:.4f} gain={gain:+.4f} ({elapsed:.0f}s)")

                fallback_dates = False
                if not selected_dates and date_cands:
                    selected_dates = [date_cands[0]]
                    fallback_dates = True
                    log.warning(f"  K_dates=0 for {crop_name} — fallback to top-1 date: {date_cands[0]}")
                    mlflow.set_tag("phaseA_fallback_date", date_cands[0])

                log.info(
                    f"\n  Phase A done: K_dates={len(selected_dates)}  "
                    f"best_iou={best_iou_dates:.4f}  dates={selected_dates}"
                )
                mlflow.log_metrics({"phaseA_final_iou": best_iou_dates, "phaseA_k_dates": len(selected_dates)})
                mlflow.set_tag("phaseA_selected_dates", str(selected_dates))

                log.info(f"\n  === Phase B: Band selection for {crop_name} (fixed dates={selected_dates}) ===")
                selected_bands, best_iou_bands, no_improve_bands = [], 0.0, 0

                for step, band in enumerate(band_cands):
                    if len(selected_bands) >= fa2.S2_MAX_BANDS_V2:
                        log.info(f"  max_bands={fa2.S2_MAX_BANDS_V2} reached — stopping Phase B")
                        break
                    if no_improve_bands >= fa2.S2_BAND_NO_IMPROVE:
                        log.info(f"  {fa2.S2_BAND_NO_IMPROVE} consecutive band rejections — stopping Phase B")
                        break

                    trial_bands = selected_bands + [band]
                    trial_idx = fa2.dates_bands_to_indices(selected_dates, trial_bands, band_name_to_idx)
                    if not trial_idx:
                        log.warning(f"  band={band}: no valid indices — skipping")
                        no_improve_bands += 1
                        continue

                    log.info(
                        f"\n  --- Phase B step {step + 1}/{len(band_cands)}  "
                        f"trial band: {band}  (selected: {len(selected_bands)}) ---"
                    )
                    t0 = time.time()
                    iou, _n_train, _n_val, _n_workers, _use_pin = _train_trial_unet(crop_imgs, crop_masks, trial_idx)
                    elapsed = time.time() - t0
                    gain = iou - best_iou_bands
                    accepted = gain >= fa2.S2_BAND_DELTA

                    if accepted:
                        selected_bands = selected_bands + [band]
                        best_iou_bands = iou
                        no_improve_bands = 0
                    else:
                        no_improve_bands += 1

                    mlflow.log_metrics(
                        {
                            "phaseB_iou": iou,
                            "phaseB_gain": gain,
                            "phaseB_accepted": int(accepted),
                            "phaseB_n_bands": len(selected_bands),
                        },
                        step=step,
                    )
                    log.info(f"  [{'OK' if accepted else '--'}] +{band}  IoU={iou:.4f} gain={gain:+.4f} ({elapsed:.0f}s)")

                fallback_bands = False
                if not selected_bands and band_cands:
                    selected_bands = [band_cands[0]]
                    fallback_bands = True
                    log.warning(f"  K_bands=0 for {crop_name} — fallback to top-1 band: {band_cands[0]}")
                    mlflow.set_tag("phaseB_fallback_band", band_cands[0])

                log.info(
                    f"\n  Phase B done: K_bands={len(selected_bands)}  "
                    f"best_iou={best_iou_bands:.4f}  bands={selected_bands}"
                )
                mlflow.log_metrics({"phaseB_final_iou": best_iou_bands, "phaseB_k_bands": len(selected_bands)})
                mlflow.set_tag("phaseB_selected_bands", str(selected_bands))

                result = {
                    "dates": selected_dates,
                    "bands": selected_bands,
                    "k_dates": len(selected_dates),
                    "k_bands": len(selected_bands),
                    "best_iou_after_dates": round(best_iou_dates, 4),
                    "best_iou_after_bands": round(best_iou_bands, 4),
                    "fallback_dates": fallback_dates,
                    "fallback_bands": fallback_bands,
                    "mlflow_run_id": crop_run.info.run_id,
                }
                results_per_crop[crop_id] = result

            log.info(
                f"\n  -> [{crop_idx}/{n_crops}] {crop_name}: "
                f"K_dates={len(selected_dates)}  K_bands={len(selected_bands)}  "
                f"IoU_dates={best_iou_dates:.4f}  IoU_bands={best_iou_bands:.4f}"
            )
            mlflow.log_metrics(
                {
                    f"crop_{crop_id}_k_dates": len(selected_dates),
                    f"crop_{crop_id}_k_bands": len(selected_bands),
                    f"crop_{crop_id}_iou_dates": best_iou_dates,
                    f"crop_{crop_id}_iou_bands": best_iou_bands,
                }
            )

        fa2.save_results_v2(
            results_per_crop,
            band_name_to_idx,
            per_crop_json=fa2.STAGE2V3_PER_CROP_JSON,
            exp_json=fa2.STAGE3_EXP_C_V2_JSON,
            exp_bands=fa2.STAGE3_EXP_C_V2_BANDS,
        )
        mlflow.log_artifact(str(fa2.STAGE2V3_PER_CROP_JSON))
        mlflow.log_artifact(str(fa2.STAGE3_EXP_C_V2_JSON))
        mlflow.log_artifact(str(fa2.STAGE3_EXP_C_V2_BANDS))
        log.info(f"Stage 2v2 parent run_id: {parent_run.info.run_id}")
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
