"""NDVI-based resolution of CDL-vs-prediction disagreement.

Follows Ghosh et al. (2021), CalCROP21 (IEEE Big Data), Sec. 5.2 + 8.1: where the
ground truth (CDL) and a model's prediction disagree on a pixel, CDL is not
automatically correct — it carries its own noise (mixed 30 m pixels, stale
year-over-year labels, decision-tree misclassification). Instead, build a
per-class "characteristic NDVI series" from pixels where CDL and the prediction
already agree, then check which class's characteristic series each disagreement
pixel's own NDVI series actually tracks (lowest NMSE wins).

Inputs: test_pred_map.npy / test_gt_map.npy (written by train_segmentation.py
alongside test_segmentation_map.png) + the per-date S2 *_processed.tif stack
covering the same raster.

Usage:
    python stages/ndvi_disagreement_analysis.py --exp-dir ml_models/<run>/<exp_name>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, CDL_CLASS_NAMES, S2_BAND_NAMES, S2_TRAIN_DIR, S2_NODATA,
)

log = logging.getLogger(__name__)

B4_IDX = S2_BAND_NAMES.index("B4")
B8_IDX = S2_BAND_NAMES.index("B8")

# CalCROP21 Sec 5.2/8.1: minimum agreement pixels to trust a class's characteristic series.
MIN_AGREEMENT_PIXELS = 100


def _class_name(cid: int) -> str:
    if cid == 0:
        return "Background"
    cdl_id = KEEP_CLASSES[cid - 1]
    return CDL_CLASS_NAMES.get(cdl_id, f"cls{cid}")


def _load_ndvi_stack(s2_paths: list[str]) -> np.ndarray:
    """(n_dates, H, W) float32 NDVI; NaN where either band is nodata/invalid."""
    series = []
    for p in s2_paths:
        with rasterio.open(p) as src:
            b4 = src.read(B4_IDX + 1).astype(np.float32)
            b8 = src.read(B8_IDX + 1).astype(np.float32)
        invalid = (b4 == S2_NODATA) | (b8 == S2_NODATA) | ~np.isfinite(b4) | ~np.isfinite(b8)
        denom = b4 + b8
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (b8 - b4) / denom
        ndvi[invalid | (denom == 0)] = np.nan
        series.append(ndvi)
    return np.stack(series, axis=0)


def _characteristic_series(ndvi_stack: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Per-class median NDVI series over agreement pixels. None if support too low."""
    if mask.sum() < MIN_AGREEMENT_PIXELS:
        return None
    return np.nanmedian(ndvi_stack[:, mask], axis=1)


def build_characteristic_series(
    ndvi_stack: np.ndarray, gt_map: np.ndarray, pred_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class characteristic NDVI series, built from agreement (CDL==pred) pixels.

    Returns (char_matrix, char_var): char_matrix[cid] is the (n_dates,) median
    series for class cid (NaN row if support < MIN_AGREEMENT_PIXELS), char_var[cid]
    its variance (used to normalise MSE → NMSE downstream).
    """
    n_dates = ndvi_stack.shape[0]
    agreement_mask = pred_map == gt_map
    class_ids = sorted(set(np.unique(gt_map).tolist()) | set(np.unique(pred_map).tolist()))
    max_cid   = max(class_ids)
    char_matrix = np.full((max_cid + 1, n_dates), np.nan, dtype=np.float32)
    char_var    = np.full((max_cid + 1,), np.nan, dtype=np.float32)
    for cid in class_ids:
        m  = agreement_mask & (gt_map == cid)
        cs = _characteristic_series(ndvi_stack, m)
        if cs is not None:
            char_matrix[cid] = cs
            char_var[cid]    = np.nanvar(cs)
            log.info(f"  class {cid} ({_class_name(cid)}): {m.sum():,} agreement px → characteristic series")
        else:
            log.warning(f"  class {cid} ({_class_name(cid)}): < {MIN_AGREEMENT_PIXELS} agreement px — skipped")
    return char_matrix, char_var


def score_patch_verdict(
    ndvi_patch: np.ndarray,    # (n_dates, ph, pw)
    gt_patch: np.ndarray,      # (ph, pw)
    pred_patch: np.ndarray,    # (ph, pw)
    char_matrix: np.ndarray,
    char_var: np.ndarray,
) -> np.ndarray:
    """Per-pixel verdict map for one patch, against precomputed characteristic series.

    0 = agreement/background, 1 = prediction wins (lower NMSE), 2 = CDL ground
    truth wins, 3 = disagreement but unscored (no characteristic series for one
    of the two candidate classes).
    """
    verdict = np.zeros(gt_patch.shape, dtype=np.uint8)
    dis_mask = gt_patch != pred_patch
    if not dis_mask.any():
        return verdict

    ys, xs = np.where(dis_mask)
    max_cid = char_matrix.shape[0] - 1
    c_gt, c_pred = gt_patch[ys, xs].astype(np.int64), pred_patch[ys, xs].astype(np.int64)
    in_range = (c_gt <= max_cid) & (c_pred <= max_cid) & (c_gt >= 0) & (c_pred >= 0)
    verdict[ys[~in_range], xs[~in_range]] = 3
    ys, xs, c_gt, c_pred = ys[in_range], xs[in_range], c_gt[in_range], c_pred[in_range]

    has_ref = np.isfinite(char_var[c_gt]) & np.isfinite(char_var[c_pred])
    verdict[ys[~has_ref], xs[~has_ref]] = 3
    ys, xs, c_gt, c_pred = ys[has_ref], xs[has_ref], c_gt[has_ref], c_pred[has_ref]
    if len(ys) == 0:
        return verdict

    px_series = ndvi_patch[:, ys, xs]      # (n_dates, N)
    ref_gt    = char_matrix[c_gt].T        # (n_dates, N)
    ref_pred  = char_matrix[c_pred].T      # (n_dates, N)
    with np.errstate(invalid="ignore"):
        mse_gt   = np.nanmean((px_series - ref_gt) ** 2, axis=0)
        mse_pred = np.nanmean((px_series - ref_pred) ** 2, axis=0)
    nmse_gt   = mse_gt / char_var[c_gt]
    nmse_pred = mse_pred / char_var[c_pred]

    unscored  = ~(np.isfinite(nmse_gt) & np.isfinite(nmse_pred))
    pred_wins = (~unscored) & (nmse_pred < nmse_gt)
    gt_wins   = (~unscored) & ~pred_wins
    verdict[ys[unscored],  xs[unscored]]  = 3
    verdict[ys[pred_wins], xs[pred_wins]] = 1
    verdict[ys[gt_wins],   xs[gt_wins]]   = 2
    return verdict


def run_ndvi_disagreement(
    pred_map: np.ndarray,
    gt_map: np.ndarray,
    s2_paths: list[str],
    out_dir: Path,
    chunk_rows: int = 500,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Returns (overall_summary, char_matrix, char_var) — the latter two reusable
    for patch-scale visualisation without rebuilding characteristic series."""
    log.info(f"Loading NDVI stack from {len(s2_paths)} S2 date files...")
    ndvi_stack = _load_ndvi_stack(s2_paths)
    n_dates, H, W = ndvi_stack.shape
    log.info(f"  NDVI stack: {ndvi_stack.shape}")

    agreement_mask = pred_map == gt_map
    n_agree    = int(agreement_mask.sum())
    n_disagree = int((~agreement_mask).sum())
    class_ids  = sorted(set(np.unique(gt_map).tolist()) | set(np.unique(pred_map).tolist()))

    char_matrix, char_var = build_characteristic_series(ndvi_stack, gt_map, pred_map)

    # ── score disagreement pixels, chunked over rows to bound memory ──────────
    rows_y, rows_x, rows_cgt, rows_cpred, rows_nmse_gt, rows_nmse_pred = [], [], [], [], [], []
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        gt_chunk   = gt_map[r0:r1]
        pred_chunk = pred_map[r0:r1]
        dis_local  = gt_chunk != pred_chunk
        ys_local, xs = np.where(dis_local)
        if len(ys_local) == 0:
            continue

        c_gt, c_pred = gt_chunk[ys_local, xs], pred_chunk[ys_local, xs]
        has_ref = np.isfinite(char_var[c_gt]) & np.isfinite(char_var[c_pred])
        if not has_ref.any():
            continue
        ys_local, xs, c_gt, c_pred = ys_local[has_ref], xs[has_ref], c_gt[has_ref], c_pred[has_ref]
        ys = ys_local + r0

        px_series = ndvi_stack[:, ys, xs]      # (n_dates, N)
        ref_gt    = char_matrix[c_gt].T        # (n_dates, N)
        ref_pred  = char_matrix[c_pred].T       # (n_dates, N)

        with np.errstate(invalid="ignore"):
            mse_gt   = np.nanmean((px_series - ref_gt) ** 2, axis=0)
            mse_pred = np.nanmean((px_series - ref_pred) ** 2, axis=0)
        nmse_gt   = mse_gt / char_var[c_gt]
        nmse_pred = mse_pred / char_var[c_pred]

        keep = np.isfinite(nmse_gt) & np.isfinite(nmse_pred)
        rows_y.extend(ys[keep].tolist());           rows_x.extend(xs[keep].tolist())
        rows_cgt.extend(c_gt[keep].tolist());        rows_cpred.extend(c_pred[keep].tolist())
        rows_nmse_gt.extend(nmse_gt[keep].tolist());  rows_nmse_pred.extend(nmse_pred[keep].tolist())

        log.info(f"  rows {r0}-{r1}/{H}: {keep.sum():,} disagreement px scored")

    df = pd.DataFrame({
        "y": rows_y, "x": rows_x,
        "gt_class": rows_cgt, "pred_class": rows_cpred,
        "nmse_gt": rows_nmse_gt, "nmse_pred": rows_nmse_pred,
    })
    df["verdict"] = np.where(df["nmse_pred"] < df["nmse_gt"], "pred", "gt")

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "ndvi_disagreement_pixels.csv", index=False)

    # ── per-class summary (CalCROP21 Table 3 style: win rate + mean NMSE) ──────
    summary_rows = []
    for cid in sorted(c for c in class_ids if np.isfinite(char_var[c])):
        sub = df[(df["gt_class"] == cid) | (df["pred_class"] == cid)]
        if sub.empty:
            continue
        pred_wins = int((sub["verdict"] == "pred").sum())
        summary_rows.append({
            "class_id":        cid,
            "class_name":      _class_name(cid),
            "n_disagreement":  len(sub),
            "gt_cdl_wins":     len(sub) - pred_wins,
            "pred_wins":       pred_wins,
            "pred_win_rate":   round(pred_wins / len(sub), 4),
            "mean_nmse_gt":    round(float(sub["nmse_gt"].mean()), 4),
            "mean_nmse_pred":  round(float(sub["nmse_pred"].mean()), 4),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "ndvi_disagreement_summary.csv", index=False)

    overall = {
        "n_dates":               n_dates,
        "n_agreement_px":        n_agree,
        "n_disagreement_px":     n_disagree,
        "n_scored":              len(df),
        "pred_win_rate_overall": round(float((df["verdict"] == "pred").mean()), 4) if len(df) else None,
        "reference": "Ghosh et al. 2021, CalCROP21 (IEEE BigData) Sec 5.2/8.1 "
                      "— NDVI characteristic-series GT-vs-pred resolution",
    }
    with open(out_dir / "ndvi_disagreement_overall.json", "w") as f:
        json.dump(overall, f, indent=2)

    if overall["pred_win_rate_overall"] is not None:
        log.info(f"Overall: {overall['n_scored']:,} disagreement px scored, "
                  f"prediction wins {overall['pred_win_rate_overall']:.1%}")
    log.info(f"Saved → {out_dir}")
    return overall, char_matrix, char_var


def _scatter_plot(out_dir: Path, max_points: int = 50_000) -> None:
    """NMSE scatter (CalCROP21 Fig 11 style) — pred vs gt, capped sample for plotting."""
    import matplotlib.pyplot as plt

    df = pd.read_csv(out_dir / "ndvi_disagreement_pixels.csv")
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df["nmse_pred"], df["nmse_gt"], s=2, alpha=0.3)
    lim = float(np.nanpercentile(np.concatenate([df["nmse_pred"], df["nmse_gt"]]), 99))
    ax.plot([0, lim], [0, lim], color="red", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("NMSE (prediction)"); ax.set_ylabel("NMSE (CDL ground truth)")
    ax.set_title("NDVI disagreement resolution — points below the line favor the prediction")
    plt.tight_layout()
    plt.savefig(out_dir / "ndvi_disagreement_scatter.png", dpi=150)
    plt.close()
    log.info(f"  Saved scatter → {out_dir / 'ndvi_disagreement_scatter.png'}")


def main():
    parser = argparse.ArgumentParser(
        description="NDVI-based resolution of CDL-vs-prediction disagreement (CalCROP21 method)"
    )
    parser.add_argument("--exp-dir", required=True, type=str,
                        help="Run dir containing test_pred_map.npy + test_gt_map.npy "
                             "(produced by train_segmentation.py)")
    parser.add_argument("--s2-dir", type=str, default=None,
                        help=f"Dir of per-date S2 *_processed.tif (default: {S2_TRAIN_DIR})")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output dir for CSV/JSON (default: <exp-dir>/ndvi_disagreement)")
    parser.add_argument("--plot", action="store_true", help="Also save NMSE scatter plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    exp_dir = Path(args.exp_dir)
    pred_map = np.load(exp_dir / "test_pred_map.npy")
    gt_map   = np.load(exp_dir / "test_gt_map.npy")

    s2_dir = Path(args.s2_dir) if args.s2_dir else S2_TRAIN_DIR
    s2_paths = sorted(str(p) for p in s2_dir.glob("*_processed.tif"))
    if not s2_paths:
        raise FileNotFoundError(f"No *_processed.tif files found in {s2_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "ndvi_disagreement"
    run_ndvi_disagreement(pred_map, gt_map, s2_paths, out_dir)

    if args.plot:
        _scatter_plot(out_dir)


if __name__ == "__main__":
    main()
