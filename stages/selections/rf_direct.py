"""RF-direct selector — single multi-class RF, per-crop importance decomposition.

Follows Wei et al. (2023, Remote Sensing 15:3212):
  - ONE multi-class RandomForestClassifier trained on all crop classes simultaneously.
  - Per-crop importance derived by decomposing each node's Gini decrease, weighted
    by that class's representation at the node (class-conditional MDI).
  - Features ranked separately per crop; union selected for Stage 3.

Pixel samples pooled from all training years for robust importance estimates.
"""

import logging
import time
from datetime import datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from crop_mapping_pipeline.config import (
    KEEP_CLASSES, CDL_CLASS_NAMES,
    SELECT_TOP_K_PER_CROP, SELECT_RF_DIRECT_JSON, SELECT_RF_DIRECT_BANDS,
    RF_N_ESTIMATORS, RF_MAX_PIXELS,
)
from crop_mapping_pipeline.stages.selections._utils import (
    build_channel_names, sample_pixels, save_selection, log_selection_run, save_per_class_table,
)

log = logging.getLogger(__name__)


def _per_class_importance(
    rf: RandomForestClassifier,
    bandnames: list[str],
    keep_classes: list[int],
) -> dict[int, pd.Series]:
    """Decompose multi-class RF Gini importance per crop class.

    For class c, importance_c[j] = mean over trees of:
        Σ_nodes_using_j  (n_c_node / n_c_root) × (n_node / N) × ΔGini_node

    This class-conditional MDI reflects how much feature j contributes to
    separating class c from all other classes within the joint multi-class tree,
    matching the per-crop importance decomposition in Wei et al. (2023) Fig. 3.
    """
    classes_ = list(rf.classes_)
    class_to_ci = {c: i for i, c in enumerate(classes_)}
    n_features = len(bandnames)
    n_rf_classes = len(classes_)

    importances = np.zeros((n_rf_classes, n_features), dtype=np.float64)

    for tree in rf.estimators_:
        t = tree.tree_
        feature   = t.feature           # (n_nodes,)  -2 for leaves
        n_samp    = t.n_node_samples    # (n_nodes,)
        value     = t.value[:, 0, :]   # (n_nodes, n_rf_classes) — raw counts
        impurity  = t.impurity          # (n_nodes,)
        ch_left   = t.children_left
        ch_right  = t.children_right

        root_counts = value[0]          # class sample counts at root node
        N_root = n_samp[0]

        for nid in range(t.node_count):
            if ch_left[nid] == -1:      # leaf — no split
                continue
            fj = feature[nid]
            if fj < 0:
                continue

            lid = ch_left[nid]
            rid = ch_right[nid]
            n_p = n_samp[nid]
            n_l = n_samp[lid]
            n_r = n_samp[rid]

            delta_gini = (
                impurity[nid]
                - (n_l / n_p) * impurity[lid]
                - (n_r / n_p) * impurity[rid]
            )
            if delta_gini <= 0:
                continue

            node_weight = (n_p / N_root) * delta_gini

            for ci in range(n_rf_classes):
                if root_counts[ci] == 0:
                    continue
                # Class share at this node relative to total class samples
                class_share = value[nid, ci] / root_counts[ci]
                importances[ci, fj] += class_share * node_weight

    importances /= len(rf.estimators_)

    # Normalise each class so importances sum to 1 (standard MDI convention)
    for ci in range(n_rf_classes):
        s = importances[ci].sum()
        if s > 0:
            importances[ci] /= s

    result: dict[int, pd.Series] = {}
    for crop_id in keep_classes:
        if crop_id not in class_to_ci:
            result[crop_id] = pd.Series(0.0, index=bandnames)
            continue
        ci = class_to_ci[crop_id]
        result[crop_id] = pd.Series(importances[ci].astype(np.float32), index=bandnames)
    return result


def _train_multiclass_rf(df: pd.DataFrame, bandnames: list[str],
                         seed: int = 42) -> RandomForestClassifier:
    """Train one multi-class RF on df (class_label column + bandname columns)."""
    x = df[bandnames].values.astype(np.float32)
    y = df["class_label"].values.astype(int)

    # Cap total pixels to avoid OOM (None = no cap, use all sampled pixels)
    if RF_MAX_PIXELS is not None and len(y) > RF_MAX_PIXELS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), min(RF_MAX_PIXELS, len(y)), replace=False)
        x, y = x[idx], y[idx]
        log.info(f"  Pixel cap applied: {len(y)} / {RF_MAX_PIXELS} pixels used")

    # Impute NaN with column median
    col_medians = np.nanmedian(x, axis=0)
    x = np.where(np.isnan(x), col_medians, x)

    rf = RandomForestClassifier(
        n_estimators  = RF_N_ESTIMATORS,
        class_weight  = "balanced",
        oob_score     = True,
        n_jobs        = -1,
        random_state  = seed,
    )
    rf.fit(x, y)
    log.info(f"  Multi-class RF trained — OOB accuracy: {rf.oob_score_:.3f}  "
             f"classes: {list(rf.classes_)}  n_samples: {len(y)}")
    return rf


def run_rf_direct(
    years_data: list[tuple[str, list[str], str]],
    top_k: int = SELECT_TOP_K_PER_CROP,
    data_dir: str | None = None,
    out_stem: str | None = None,
    percentile: float | None = None,
    score_threshold: float | None = None,
) -> list[str]:
    """
    years_data: [(year, s2_paths, cdl_path), ...]
      Primary year (first) supplies channel names for output.
      All years contribute pixel samples for MMDD-level importance averaging.
    Returns union channel list.

    Follows Wei et al. (2023): single multi-class RF, per-crop importance
    decomposed via class-conditional MDI from the joint tree structure.
    """
    t_start = time.time()
    log.info("RF-direct (multi-class): scoring all channels, no prefilter")
    _mode_str = (f"score_threshold={score_threshold:g}" if score_threshold is not None
                 else f"percentile={percentile:g}" if percentile is not None
                 else f"top_k={top_k}")
    log.info(f"  years={[yr for yr, _, _ in years_data]}  mode={_mode_str}  n_trees={RF_N_ESTIMATORS}")

    primary_year, primary_s2, primary_cdl = years_data[0]
    primary_bandnames, _, _ = build_channel_names(primary_s2)
    n_channels = len(primary_bandnames)
    log.info(f"  Primary year {primary_year}: {n_channels} channels")

    # ── Sample + train on primary year ────────────────────────────────────────
    log.info(f"  Sampling {primary_year}...")
    df_primary = sample_pixels(primary_s2, primary_cdl, primary_bandnames)
    rf_primary = _train_multiclass_rf(df_primary, primary_bandnames, seed=42)
    imp_primary = _per_class_importance(rf_primary, primary_bandnames, KEEP_CLASSES)

    # ── MMDD helpers ──────────────────────────────────────────────────────────
    def _doy(mmdd: str) -> int:
        return _dt.strptime(f"2000{mmdd}", "%Y%m%d").timetuple().tm_yday

    def _mmdd_level(imp: pd.Series, bandnames: list[str]) -> dict[str, float]:
        """Collapse channel importance to {band_MMDD: mean_importance}."""
        result: dict[str, list[float]] = {}
        for ch in bandnames:
            parts = ch.rsplit("_", 1)
            if len(parts) != 2:
                continue
            band, date8 = parts[0], parts[1]
            mmdd = date8[4:]
            key  = f"{band}_{mmdd}"
            result.setdefault(key, []).append(float(imp[ch]))
        return {k: float(np.mean(v)) for k, v in result.items()}

    # ── Extra years: sample + train separate multi-class RF, MMDD-level imp ───
    extra_imps: list[tuple[str, list[str], dict[int, dict[str, float]]]] = []
    for year, s2_paths, cdl_path in years_data[1:]:
        log.info(f"  Sampling extra year {year}...")
        bandnames_yr, _, _ = build_channel_names(s2_paths)
        df_yr = sample_pixels(s2_paths, cdl_path, bandnames_yr)
        rf_yr = _train_multiclass_rf(df_yr, bandnames_yr, seed=42)
        imp_yr = _per_class_importance(rf_yr, bandnames_yr, KEEP_CLASSES)
        # Collapse to MMDD level per crop
        mmdd_yr = {
            crop_id: _mmdd_level(imp_yr[crop_id], bandnames_yr)
            for crop_id in KEEP_CLASSES
        }
        extra_imps.append((year, bandnames_yr, mmdd_yr))

    # ── Adjust primary-year importances via multi-year MMDD averaging ─────────
    adjusted_per_crop: dict[int, pd.Series] = {}

    for crop_id in KEEP_CLASSES:
        si_primary = imp_primary[crop_id]

        if extra_imps:
            primary_mmdd = _mmdd_level(si_primary, primary_bandnames)
            adjusted = si_primary.copy()
            for ch in primary_bandnames:
                parts = ch.rsplit("_", 1)
                if len(parts) != 2:
                    continue
                band, date8 = parts[0], parts[1]
                mmdd_p = date8[4:]
                doy_p  = _doy(mmdd_p)

                imps = [primary_mmdd.get(f"{band}_{mmdd_p}", float(si_primary[ch]))]
                for _yr, _bnames, mmdd_yr in extra_imps:
                    yr_mmdd = mmdd_yr[crop_id]
                    cands = [k for k in yr_mmdd if k.startswith(f"{band}_")]
                    if not cands:
                        continue
                    nearest = min(cands, key=lambda k: abs(_doy(k.split("_")[1]) - doy_p))
                    imps.append(yr_mmdd[nearest])
                adjusted[ch] = float(np.mean(imps))
        else:
            adjusted = si_primary

        adjusted_per_crop[crop_id] = adjusted.fillna(0.0)

    # ── Selection: score_threshold (Wei et al. 2023) / pooled-percentile / top-K ─
    per_crop: dict[int, list[str]] = {}
    thr: float | None = None
    if score_threshold is not None:
        # Per-crop min-max normalize importances to [0,1]; retain channels >= score_threshold.
        # Follows Wei et al. (2023): "features >= 0.5 have yielded quite results."
        log.info(f"  RF per-crop normalized score threshold = {score_threshold}")
        for crop_id in KEEP_CLASSES:
            s = adjusted_per_crop[crop_id]
            s_min, s_max = float(s.min()), float(s.max())
            if s_max > s_min:
                s_norm = (s - s_min) / (s_max - s_min)
            else:
                s_norm = pd.Series(0.0, index=s.index)
            sel = s_norm[s_norm >= score_threshold].sort_values(ascending=False).index.tolist()
            per_crop[crop_id] = sel
            log.info(f"  {CDL_CLASS_NAMES[crop_id]:20s}: {len(sel)} ch (norm≥{score_threshold}, top-3 {sel[:3]})")
    elif percentile is not None:
        pooled = np.concatenate([s.values for s in adjusted_per_crop.values()])
        thr    = float(np.percentile(pooled, percentile))
        log.info(f"  RF pooled P{percentile:g} threshold = {thr:.6f}")
        for crop_id in KEEP_CLASSES:
            s   = adjusted_per_crop[crop_id]
            sel = s[s >= thr].sort_values(ascending=False).index.tolist()
            per_crop[crop_id] = sel
            log.info(f"  {CDL_CLASS_NAMES[crop_id]:20s}: {len(sel)} ch (top-3 {sel[:3]})")
    else:
        for crop_id in KEEP_CLASSES:
            top_channels = adjusted_per_crop[crop_id].nlargest(top_k).index.tolist()
            per_crop[crop_id] = top_channels
            log.info(f"  {CDL_CLASS_NAMES[crop_id]:20s}: top-3 = {top_channels[:3]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    stem = out_stem or (
        f"select_rf_direct_s{score_threshold:g}" if score_threshold is not None
        else f"select_rf_direct_p{percentile:g}" if percentile is not None
        else f"select_rf_direct_k{top_k}"
    )
    base_dir  = Path(data_dir) if data_dir else SELECT_RF_DIRECT_JSON.parent
    json_path = base_dir / f"{stem}.json"
    txt_path  = base_dir / f"{stem}_bands.txt"

    union = save_selection(
        per_crop, json_path, txt_path,
        selector="rf_direct_multiclass", top_k=top_k, percentile=percentile,
        score_threshold=score_threshold,
        meta={
            "years":             [yr for yr, _, _ in years_data],
            "primary_year":      primary_year,
            "n_primary_channels": n_channels,
            "rf_n_estimators":   RF_N_ESTIMATORS,
            "rf_oob_score":      float(rf_primary.oob_score_),
            "method":            "multiclass_rf_per_class_mdi",
            "reference":         "Wei et al. 2023 Remote Sensing 15:3212",
        },
    )
    log.info(f"RF-direct (multi-class): {len(union)} union channels → {json_path}")

    table_paths = save_per_class_table(
        per_crop={int(k): v for k, v in per_crop.items()},
        save_dir=base_dir,
        stem=stem,
        score_label="RF_Importance",
        adjusted_per_crop=adjusted_per_crop,
    )
    log.info(f"RF-direct: tables saved ({len(table_paths)} files)")

    # ── MLflow ────────────────────────────────────────────────────────────────
    duration_s = time.time() - t_start
    log.info(f"RF-direct completed in {duration_s:.1f}s")
    if score_threshold is not None:
        sel_mode = "score_threshold"
    elif percentile is not None:
        sel_mode = "percentile"
    else:
        sel_mode = "top_k"
    log_selection_run(
        selector="rf_direct_multiclass",
        run_name_prefix="rf_direct",
        per_crop=per_crop,
        union=union,
        json_path=json_path,
        extra_artifacts=table_paths,
        params={
            "selector":         "rf_direct_multiclass",
            "selection_mode":   sel_mode,
            "top_k":            top_k,
            "percentile":       percentile,
            "score_threshold":  score_threshold,
            "years":            str([yr for yr, _, _ in years_data]),
            "primary_year":     primary_year,
            "n_channels":       n_channels,
            "n_union":          len(union),
            "n_crops":          len(KEEP_CLASSES),
            "rf_n_estimators":  RF_N_ESTIMATORS,
            "method":           "multiclass_rf_per_class_mdi",
        },
        duration_s=duration_s,
        threshold=thr,
        extra_metrics={"rf_oob_score": float(rf_primary.oob_score_)},
    )

    return union
