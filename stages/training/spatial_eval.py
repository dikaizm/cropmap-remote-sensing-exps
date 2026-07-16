"""Held-out spatial-area evaluation (a separate test region, not the in-area split).

Currently unused by the training pipeline (kept for ad-hoc cross-area testing).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
from torch.utils.data import DataLoader

from config import (
    PATCH_SIZE, STRIDE, KEEP_CLASSES, REMAP_LUT, MIN_VALID_FRAC,
    NUM_CLASSES, CDL_CLASS_NAMES,
)
from stages.training import run_state
from stages.training.run_state import DEVICE
from stages.training.helpers import _filter_s2_by_band_indices
from stages.training.datasets import NormalizedDataset, PreloadedDataset
from stages.training.metrics import evaluate_test_set
from stages.training.viz import _plot_confusion_matrix, save_segmentation_map
from stages.training.full_scene_inference import load_gt_remap, run_full_inference
from stages.training.experiments import build_local_band_map
from geoai.geoai.train import RasterPatchDataset

log = logging.getLogger(__name__)


def _evaluate_spatial_area(
    model,
    area: dict,
    band_names: list,
    exp_name: str,
    exp_dir: Path,
    skip_viz: bool = False,
    channel_stats: "tuple | None" = None,  # kept for API compat, unused
    band_percentiles: "tuple | None" = None,
    no_preload: bool = False,
    norm_mode: str = "percentile",
) -> "dict | None":
    """Evaluate model on one held-out spatial test area.

    area: {"name": str, "s2_dir": Path, "cdl": Path}
    band_names: channel names from experiment (e.g. ["B4_20240730", ...]).
    Returns evaluate_test_set result dict, or None if area data missing.
    """
    import glob as _glob

    area_name = area["name"]
    s2_dir    = Path(area["s2_dir"])
    cdl_path  = Path(area["cdl"])

    area_s2 = sorted(f for f in _glob.glob(str(s2_dir / "*.tif")) if not Path(f).name.startswith("._"))
    if not area_s2:
        log.warning(f"  Spatial test {area_name}: no S2 files in {s2_dir} — skipping")
        return None
    if not cdl_path.exists():
        log.warning(f"  Spatial test {area_name}: CDL not found at {cdl_path} — skipping")
        return None

    log.info(f"  Spatial test [{area_name}]: {len(area_s2)} S2 files, CDL={cdl_path.name}")

    _, area_band_to_idx, _, _ = build_local_band_map(area_s2)

    area_global_indices = []
    skipped_bands = []
    for bname in band_names:
        idx = area_band_to_idx.get(bname)
        if idx is not None:
            area_global_indices.append(idx)
        else:
            skipped_bands.append(bname)

    if skipped_bands:
        log.warning(f"  Spatial test {area_name}: {len(skipped_bands)} band(s) not found in area files (date mismatch?): {skipped_bands[:3]}...")
    if not area_global_indices:
        log.error(f"  Spatial test {area_name}: no matching bands — skipping")
        return None

    area_s2_filtered, area_idx_local = _filter_s2_by_band_indices(area_s2, area_global_indices)

    area_ds = RasterPatchDataset(
        s2_paths=area_s2_filtered, cdl_path=str(cdl_path),
        patch_size=PATCH_SIZE, stride=STRIDE,
        keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
        min_valid_frac=MIN_VALID_FRAC, band_indices=area_idx_local,
    )
    if no_preload:
        area_norm = NormalizedDataset(area_ds, band_percentiles=band_percentiles,
                                      norm_mode=norm_mode)
    else:
        area_norm = PreloadedDataset(area_ds, desc=area_name, cache_dir=run_state.PRELOAD_CACHE_DIR,
                                     band_percentiles=band_percentiles, norm_mode=norm_mode)
    area_dl = DataLoader(area_norm, batch_size=run_state.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    area_r = evaluate_test_set(model, area_dl, NUM_CLASSES, DEVICE)
    log.info(f"  [{area_name}] mIoU={area_r['miou']:.4f}  OA={area_r['oa']:.4f}")
    log.info(f"  {'Class':<20} {'IoU':>7}")
    for cls_id, iou in area_r["per_class_iou"].items():
        cdl_id = KEEP_CLASSES[cls_id - 1]
        name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
        log.info(f"  {name:<20} {iou:.4f}" if not np.isnan(iou) else f"  {name:<20}     nan")

    # MLflow metrics prefixed with area name
    mlflow.log_metrics({
        f"{area_name}_miou": area_r["miou"],
        f"{area_name}_mf1":  area_r["mf1"],
        f"{area_name}_oa":   area_r["oa"],
    })
    for cls_id, iou in area_r["per_class_iou"].items():
        if not np.isnan(iou):
            cdl_id = KEEP_CLASSES[cls_id - 1]
            cname  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
            mlflow.log_metric(
                f"{area_name}_iou_{cname.lower().replace('/', '_').replace(' ', '_')}",
                iou,
            )
    for cls_id, f1v in area_r["per_class_f1"].items():
        if not np.isnan(f1v):
            cdl_id = KEEP_CLASSES[cls_id - 1]
            cname  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
            mlflow.log_metric(
                f"{area_name}_f1_{cname.lower().replace('/', '_').replace(' ', '_')}",
                f1v,
            )

    # Per-class metrics CSV (IoU + F1)
    iou_rows = [
        {
            "class_id":   cls_id,
            "cdl_id":     KEEP_CLASSES[cls_id - 1],
            "class_name": CDL_CLASS_NAMES.get(KEEP_CLASSES[cls_id - 1], f"cls{cls_id}"),
            "iou":        round(iou, 4) if not np.isnan(iou) else float("nan"),
            "f1":         round(area_r["per_class_f1"].get(cls_id, float("nan")), 4)
                          if not np.isnan(area_r["per_class_f1"].get(cls_id, float("nan")))
                          else float("nan"),
        }
        for cls_id, iou in area_r["per_class_iou"].items()
    ]
    iou_csv = exp_dir / f"{area_name}_per_class_iou.csv"
    pd.DataFrame(iou_rows).to_csv(iou_csv, index=False)
    mlflow.log_artifact(str(iou_csv))

    # Confusion matrix
    cm_path = exp_dir / f"{area_name}_confusion_matrix.png"
    _plot_confusion_matrix(area_r["preds"], area_r["labels"], str(cm_path))
    mlflow.log_artifact(str(cm_path))

    # Segmentation map
    if not skip_viz:
        gt_map, _   = load_gt_remap(str(cdl_path))
        pred_map, _ = run_full_inference(
            model, area_s2_filtered, area_idx_local,
            patch_size=PATCH_SIZE, stride=PATCH_SIZE,
            channel_stats=None, band_percentiles=band_percentiles,
        )
        seg_path = exp_dir / f"{area_name}_segmentation_map.png"
        save_segmentation_map(
            pred_map, gt_map,
            title=f"{exp_name} — {area_name}",
            save_path=str(seg_path),
        )
        mlflow.log_artifact(str(seg_path))
        del pred_map, gt_map

    return area_r
