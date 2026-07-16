"""Per-block and full-test-set evaluation metrics."""

import os

import numpy as np
import torch
import rasterio


def iou(g, p, k):
    i = ((g == k) & (p == k)).sum()
    u = ((g == k) | (p == k)).sum()
    return (i / u) if u else np.nan


def block_metrics(gt, preds, cdl_class_names, keep_classes):
    """preds: {label: pred_map} -> {label: {mIoU, OA, per_crop}}"""
    metrics = {}
    for label, pred in preds.items():
        per = {cdl_class_names[c]: iou(gt, pred, k + 1) for k, c in enumerate(keep_classes)}
        metrics[label] = {
            "mIoU": float(np.nanmean([v for v in per.values() if not np.isnan(v)])),
            "OA": float((gt == pred).mean()),
            "per_crop": per,
        }
    return metrics


def evaluate_full_test_set(
    scenario_models, split_dir, all_blocks, lo_b, hi_b, C, patch, device, arch_default,
    load_model_fn, infer_band_norm_fn,
):
    """Aggregate confusion matrix over every valid patch in every test block.

    scenario_models: {label: ckpt_path}
    load_model_fn: model_io.load_model
    infer_band_norm_fn: model_io.band_norm_stats
    """
    keep = set(C.KEEP_CLASSES)
    min_frac = C.MIN_VALID_FRAC
    n_bands = C.N_BANDS_PER_DATE

    full_results = {}
    for label, ckpt in scenario_models.items():
        if not os.path.exists(ckpt):
            print(f"! missing {label}")
            continue
        model, arch, ck = load_model_fn(ckpt, device, arch_default, C.NUM_CLASSES)
        bi = np.array(ck["band_indices"])
        ci = list(bi)
        lo, hi = infer_band_norm_fn(lo_b, hi_b, bi, n_bands)
        den = np.maximum(hi - lo, 1.0)

        conf = np.zeros((C.NUM_CLASSES, C.NUM_CLASSES), np.int64)
        n_patches = 0

        for blk in all_blocks:
            with rasterio.open(split_dir / blk["s2"]) as src:
                s2_blk = src.read().astype(np.float32)
            with rasterio.open(split_dir / blk["cdl"]) as src:
                cdl_raw = src.read(1).astype(np.int32)
            gt_blk = C.REMAP_LUT[np.clip(cdl_raw, 0, 255)].astype(np.int64)
            H, W = s2_blk.shape[1], s2_blk.shape[2]

            s2_sel = s2_blk[ci].copy()
            s2_sel[s2_sel == C.S2_NODATA] = 0.0
            s2_sel[~np.isfinite(s2_sel)] = 0.0
            s2_sel = np.clip((s2_sel - lo[:, None, None]) / den[:, None, None], 0, 1).astype(np.float32)

            valid_patches = [
                (r, c)
                for r in range(0, H - patch + 1, patch)
                for c in range(0, W - patch + 1, patch)
                if np.isin(cdl_raw[r : r + patch, c : c + patch], list(keep)).mean() >= min_frac
            ]
            n_patches += len(valid_patches)

            with torch.no_grad():
                for r, c in valid_patches:
                    t = s2_sel[:, r : r + patch, c : c + patch].copy()
                    out = model(torch.from_numpy(t).unsqueeze(0).to(device)).argmax(1)[0].cpu().numpy()
                    gt_p = gt_blk[r : r + patch, c : c + patch]
                    for truth in range(C.NUM_CLASSES):
                        mask = gt_p == truth
                        if mask.any():
                            for pred_cls in np.unique(out[mask]):
                                conf[truth, pred_cls] += int(mask[out == pred_cls].sum())

        per_iou = {}
        for k, nm in zip(range(1, C.NUM_CLASSES), C.CDL_CLASS_NAMES.values()):
            tp = conf[k, k]
            fp = conf[:, k].sum() - tp
            fn = conf[k, :].sum() - tp
            per_iou[nm] = tp / max(tp + fp + fn, 1)
        miou = float(np.mean(list(per_iou.values())))
        full_results[label] = {"mIoU": miou, "per_iou": per_iou, "n_patches": n_patches}

        print(f"\n{label} ({n_patches} patches)")
        print(f"  {'Crop':14s} {'IoU':>7}")
        for nm, iou_v in per_iou.items():
            print(f"  {nm:14s} {iou_v:.4f}")
        print(f"  {'mIoU':14s} {miou:.4f}")

    return full_results
