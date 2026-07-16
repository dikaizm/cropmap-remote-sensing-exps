"""Segmentation metrics: mIoU / per-class IoU-F1-OA, validation, test eval, latency.

Pure module — every function takes what it needs as arguments (num_classes,
device, class maps), so nothing here depends on runtime/config globals.
"""

import os
import time
import logging

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import numpy as np
import torch
import mlflow
from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)


def compute_miou(logits, labels, num_classes):
    preds  = logits.argmax(dim=1).view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    ious   = []
    for cls in range(1, num_classes):
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum()
        union = (p | l).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_per_class_iou(logits, labels, num_classes):
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    ious   = {}
    for cls in range(1, num_classes):
        p = (preds == cls)
        l = (labels == cls)
        inter = (p & l).sum()
        union = (p | l).sum()
        ious[cls] = float(inter / union) if union > 0 else float("nan")
    return ious


def compute_per_class_f1(logits, labels, num_classes):
    """Per-class F1 via precision × recall. Excludes background (class 0)."""
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    f1s = {}
    for cls in range(1, num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fp = int(((preds == cls) & (labels != cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        prec   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        if not (np.isnan(prec) or np.isnan(recall)) and (prec + recall) > 0:
            f1s[cls] = float(2 * prec * recall / (prec + recall))
        else:
            f1s[cls] = float("nan")
    return f1s


def compute_per_class_oa(logits, labels, num_classes):
    """Per-class OA = recall = TP / (TP + FN). Excludes background (class 0)."""
    preds  = logits.argmax(dim=1).view(-1).numpy()
    labels = labels.view(-1).numpy()
    oas = {}
    for cls in range(1, num_classes):
        tp = int(((preds == cls) & (labels == cls)).sum())
        fn = int(((preds != cls) & (labels == cls)).sum())
        oas[cls] = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    return oas


def mean_f1(f1_dict):
    vals = [v for v in f1_dict.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else 0.0


def per_class_metric_dict(values, prefix, keep_classes, cdl_class_names):
    """Map {model_cls_id: value} → {f'{prefix}_{crop_slug}': value}, skipping NaN.

    Consolidates the per-class metric-flattening loop used for epoch logging,
    best-val logging, and test logging (prefixes like 'val_iou', 'best_val_f1',
    'test_iou'). model_cls_id 1..N-1 maps to keep_classes[id-1] → CDL name → slug.
    """
    out = {}
    for cls_id, v in values.items():
        if np.isnan(v):
            continue
        cdl_id = keep_classes[cls_id - 1]
        name   = cdl_class_names.get(cdl_id, f"cls{cls_id}")
        slug   = name.lower().replace('/', '_').replace(' ', '_')
        out[f"{prefix}_{slug}"] = v
    return out


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        imgs        = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
        logits      = model(imgs)
        loss        = criterion(logits, masks)
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(masks.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=1)
    oa              = (preds == all_labels).float().mean().item()
    miou            = compute_miou(all_logits, all_labels, num_classes)
    per_class_iou   = compute_per_class_iou(all_logits, all_labels, num_classes)
    per_class_f1    = compute_per_class_f1(all_logits, all_labels, num_classes)
    per_class_oa    = compute_per_class_oa(all_logits, all_labels, num_classes)
    mf1             = mean_f1(per_class_f1)
    return {
        "loss": total_loss / len(loader), "miou": miou, "oa": oa,
        "mf1": mf1, "per_class_iou": per_class_iou, "per_class_f1": per_class_f1,
        "per_class_oa": per_class_oa,
    }


@torch.no_grad()
def _get_hardware_info() -> dict:
    """CPU/GPU/RAM identity for mlflow params — static per-machine, not a metric."""
    import platform

    cpu_name = platform.processor()
    if not cpu_name and platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass

    info = {
        "cpu_name":  cpu_name or "unknown",
        "cpu_cores": os.cpu_count(),
    }

    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        info["ram_total_gb"] = None

    if torch.cuda.is_available():
        info["gpu_name"]      = torch.cuda.get_device_name(0)
        info["gpu_count"]     = torch.cuda.device_count()
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
    else:
        info["gpu_name"]      = "none"
        info["gpu_count"]     = 0
        info["gpu_memory_gb"] = None

    return info


def evaluate_test_set(model, loader, num_classes, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=5.0, neginf=-5.0)
            logits = model(imgs.to(device))
            all_logits.append(logits.cpu())
            all_labels.append(masks.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    preds      = all_logits.argmax(dim=1)
    per_class_f1 = compute_per_class_f1(all_logits, all_labels, num_classes)
    return {
        "miou":          compute_miou(all_logits, all_labels, num_classes),
        "oa":            (preds == all_labels).float().mean().item(),
        "mf1":           mean_f1(per_class_f1),
        "per_class_iou": compute_per_class_iou(all_logits, all_labels, num_classes),
        "per_class_f1":  per_class_f1,
        "preds":         preds,
        "labels":        all_labels,
    }


def benchmark_inference_latency(model, loader, device, run_id):
    """Time inference one patch at a time (excludes data loading/metric compute).

    Logs per-patch latency (ms) to mlflow via log_batch (chunked, avoids one
    HTTP call per patch) plus avg/std/min/max summary metrics.
    """
    from mlflow.entities import Metric

    model.eval()
    client  = MlflowClient()
    metrics = []
    latencies_ms = []
    is_cuda = torch.cuda.is_available() and str(device) != "cpu"
    idx = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = torch.nan_to_num(imgs, nan=0.0, posinf=5.0, neginf=-5.0)
            for b in range(imgs.shape[0]):
                patch = imgs[b:b + 1].to(device)
                if is_cuda:
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(patch)
                if is_cuda:
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)
                metrics.append(Metric(
                    key="inference_time_ms_patch", value=elapsed_ms,
                    timestamp=int(time.time() * 1000), step=idx,
                ))
                idx += 1

    for i in range(0, len(metrics), 1000):
        client.log_batch(run_id, metrics=metrics[i:i + 1000])

    lat = np.array(latencies_ms)
    summary = {
        "inference_time_ms_avg":       float(lat.mean()),
        "inference_time_ms_std":       float(lat.std()),
        "inference_time_ms_min":       float(lat.min()),
        "inference_time_ms_max":       float(lat.max()),
        "inference_patches_benchmarked": len(lat),
    }
    mlflow.log_metrics(summary)
    log.info(
        f"  Inference latency: avg={summary['inference_time_ms_avg']:.2f}ms "
        f"std={summary['inference_time_ms_std']:.2f}ms "
        f"(min={summary['inference_time_ms_min']:.2f} max={summary['inference_time_ms_max']:.2f}) "
        f"over {len(lat)} patches"
    )
    return summary
