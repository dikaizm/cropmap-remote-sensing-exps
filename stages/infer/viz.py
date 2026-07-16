"""Plotting helpers for block RGB, ground truth, predictions, and per-crop IoU."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def stretch(a, nodata):
    a = a.copy()
    a[a == nodata] = np.nan
    lo, hi = np.nanpercentile(a, [2, 98])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def compute_rgb(s2, band_names, nodata, date="20240714"):
    """RGB composite (B4,B3,B2) for a single date, percentile-stretched. date format: YYYYMMDD."""
    bidx = {n: i for i, n in enumerate(band_names)}
    rgb = np.dstack([stretch(s2[bidx[f"{b}_{date}"]], nodata) for b in ("B4", "B3", "B2")])
    return rgb


def class_cmap_norm(cdl_class_names, num_classes):
    pal = plt.cm.tab10(np.linspace(0, 1, len(cdl_class_names)))
    cmap = ListedColormap([(0.9, 0.9, 0.9, 1)] + [tuple(c) for c in pal])
    norm = BoundaryNorm(np.arange(-0.5, num_classes + 0.5), num_classes)
    return pal, cmap, norm


def plot_gt_rgb(rgb, gt, date, block_name, cdl_class_names, num_classes):
    pal, cmap, norm = class_cmap_norm(cdl_class_names, num_classes)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(rgb)
    ax[0].set_title(f"{block_name} RGB ({date})")
    ax[0].axis("off")
    ax[1].imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
    ax[1].set_title("CDL ground truth")
    ax[1].axis("off")
    ax[1].legend(
        handles=[Patch(facecolor="0.9", label="bg")]
        + [Patch(facecolor=pal[i], label=n) for i, n in enumerate(cdl_class_names.values())],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )
    plt.tight_layout()
    plt.show()


def plot_pred_maps(gt, preds, rgb, date, metrics, cdl_class_names, num_classes, block_name, scenario, out_fig):
    pal, cmap, norm = class_cmap_norm(cdl_class_names, num_classes)
    labels = list(preds)
    ncol = 1 + len(labels)
    fig, ax = plt.subplots(2, ncol, figsize=(6 * ncol, 12), squeeze=False)
    ax[0, 0].imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
    ax[0, 0].set_title("Ground Truth (CDL)")
    ax[1, 0].imshow(rgb)
    ax[1, 0].set_title(f"RGB ({date})")
    for j, label in enumerate(labels, 1):
        ax[0, j].imshow(preds[label], cmap=cmap, norm=norm, interpolation="nearest")
        ax[0, j].set_title(
            f"{label}\nmIoU {metrics[label]['mIoU']:.3f} · OA {metrics[label]['OA']:.3f}"
        )
        ax[1, j].imshow((gt != preds[label]).astype(int), cmap="Reds", interpolation="nearest")
        ax[1, j].set_title(f"{label} — errors")
    for a in ax.flat:
        a.axis("off")
    ax[0, 0].legend(
        handles=[Patch(facecolor="0.9", label="bg")]
        + [Patch(facecolor=pal[i], label=nm) for i, nm in enumerate(cdl_class_names.values())],
        bbox_to_anchor=(0, 0),
        loc="upper right",
        fontsize=7,
    )
    fig.suptitle(f"Test block {block_name} — {scenario}", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_fig / f"pred_maps_{block_name}_{scenario}.png", dpi=130, bbox_inches="tight")
    plt.show()


def plot_per_crop_iou_bar(metrics, cdl_class_names, block_name, scenario, out_fig):
    labels = list(metrics)
    crops = list(cdl_class_names.values())
    x = np.arange(len(crops))
    w = 0.8 / max(1, len(labels))
    fig, axb = plt.subplots(figsize=(12, 5))
    for k, label in enumerate(labels):
        axb.bar(x + k * w, [metrics[label]["per_crop"][c] for c in crops], w, label=label)
    axb.set_xticks(x + w * (len(labels) - 1) / 2)
    axb.set_xticklabels(crops, rotation=30, ha="right")
    axb.set_ylabel("IoU")
    axb.set_ylim(0, 1)
    axb.legend()
    axb.set_title(f"Per-crop IoU — {scenario} @ {block_name}")
    fig.tight_layout()
    fig.savefig(out_fig / f"per_crop_iou_{block_name}_{scenario}.png", dpi=130, bbox_inches="tight")
    plt.show()


def to_rgb_display(s2_full_patch, raw_bands, lo_full, hi_full, stretch_factor=2.5):
    rgb = s2_full_patch[raw_bands].astype(np.float32)
    lo = lo_full[raw_bands][:, None, None]
    hi = hi_full[raw_bands][:, None, None]
    rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1.0), 0, 1)
    rgb = np.clip(rgb.transpose(1, 2, 0) * stretch_factor, 0, 1)
    return rgb


def plot_patch_grid(s2_blk, gt_blk, preds_by_patch, valid_patches, patch, raw_bands, lo_full, hi_full,
                     class_colors, cmap, norm, class_names, block_name, scenario, label):
    """preds_by_patch: {(r,c): pred_map} for one model label."""
    n = len(valid_patches)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    last_out = None
    for i, (r, c) in enumerate(valid_patches):
        out = preds_by_patch[(r, c)]
        last_out = out
        gt_p = gt_blk[r : r + patch, c : c + patch]
        rgb = to_rgb_display(s2_blk[:, r : r + patch, c : c + patch], raw_bands, lo_full, hi_full)

        axes[i, 0].imshow(rgb)
        axes[i, 1].imshow(gt_p, cmap=cmap, norm=norm)
        axes[i, 2].imshow(out, cmap=cmap, norm=norm)
        for ax in axes[i]:
            ax.axis("off")
        if i == 0:
            axes[i, 0].set_title("RGB")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 2].set_title("Prediction")

    present = sorted(set(np.unique(gt_blk)) | set(np.unique(last_out)))
    legend_handles = [Patch(color=class_colors[k], label=class_names.get(k, str(k))) for k in present]
    fig.legend(handles=legend_handles, loc="lower center", ncol=min(len(present), 6),
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{scenario}/{label} — {block_name}", y=1.0)
    fig.tight_layout()
    plt.show()
