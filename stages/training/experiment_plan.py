"""Resolve per-experiment channel sets and build the (exp × arch) run plan.

Extracted from main(): turns the validated S2 file list + CLI selectors into
the concrete list of runs to execute (single_date / mt_ndvi / gsi / rf channel
indices → registry → flat plan), delegating to stages/training/experiments/.
"""

import logging
from pathlib import Path

from config import (
    ARCH_CFG, TRAIN_YEARS, PROCESSED_DIR, SELECT_GSI_DIRECT_JSON,
)
from stages.training.experiments import (
    build_local_band_map,
    build_single_date_indices,
    build_naive_multitemporal_indices,
    build_full_stack_indices,
    build_registry,
    expand_exp_keys,
)
from stages.training.experiments.feature_selection import build_direct_indices
from stages.training.helpers import _s2_for_year
from stages.training import run_state

log = logging.getLogger(__name__)


def build_experiment_plan(s2_processed, exps, archs, phenol_dates, score_threshold, data_dir):
    """Return (plan, registry).

    plan = list of (exp_key, arch, band_indices, band_names, description, extra_kw).
    registry maps exp_key → config entry (used by main for MLflow grouping).
    """
    # ── Build local band map (reference year) ──────────────────────────────
    (local_band_names, local_band_to_idx,
     local_date_to_idx, mmdd_to_date) = build_local_band_map(s2_processed)

    # ── Build experiment channel sets ─────────────────────────────────────
    _ref_year_s2  = _s2_for_year(s2_processed, TRAIN_YEARS[0])
    _ref_year_cdl = run_state.CDL_TRAIN

    # ── Base domain channels (all 9 VEGE_BANDS, no band selection) ─────────
    needs_sd  = not exps or "single_date" in exps
    needs_nmt = not exps or "mt_ndvi" in exps

    sd_base_idx = sd_base_names = sd_date_key = None
    nmt_base_idx = nmt_base_names = phenol_map_base = None

    if needs_sd:
        sd_base = build_single_date_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
        )
        sd_base_idx, sd_base_names, sd_date_key = sd_base

    if needs_nmt:
        nmt_base = build_naive_multitemporal_indices(
            local_date_to_idx, local_band_to_idx,
            s2_paths=_ref_year_s2, cdl_path=str(_ref_year_cdl),
            phenol_json=phenol_dates,
        )
        nmt_base_idx, nmt_base_names, phenol_map_base = nmt_base

    # ── single_date (peak NDVI date × ALL bands — conventional baseline) ──
    # No band selection: isolates the temporal variable against the multi-temporal
    # configurations. GSI/RF are not applied here.
    single_date_idx = single_date_names = single_date_key = None
    if not exps or "single_date" in exps:
        single_date_idx, single_date_names, single_date_key = sd_base_idx, sd_base_names, sd_date_key

    # ── mt_ndvi (4 calendar dates × ALL VEGE_BANDS — no selection) ──
    mt_base_idx = mt_base_names = phenol_map = None
    if not exps or "mt_ndvi" in exps:
        mt_base_idx, mt_base_names, phenol_map = nmt_base_idx, nmt_base_names, phenol_map_base

    def _find_direct_json(selector: str) -> Path:
        """Return the selection JSON for a direct selector (per-crop normalized score >= T)."""
        base = Path(data_dir) if data_dir else SELECT_GSI_DIRECT_JSON.parent
        return base / f"select_{selector}_s{score_threshold:g}.json"

    gsi_idx = gsi_names = None
    if not exps or "gsi" in exps:
        gsi_json = _find_direct_json("gsi_direct")
        gsi_idx, gsi_names = build_direct_indices(
            gsi_json, mmdd_to_date, local_band_to_idx,
            selector_name="gsi", subset_k=None,
        )
        log.info(f"gsi (s={score_threshold:g}): {len(gsi_idx)} channels")

    rf_idx = rf_names = None
    if not exps or "rf" in exps:
        rf_json = _find_direct_json("rf_direct")
        rf_idx, rf_names = build_direct_indices(
            rf_json, mmdd_to_date, local_band_to_idx,
            selector_name="rf", subset_k=None,
        )
        log.info(f"rf (s={score_threshold:g}): {len(rf_idx)} channels")

    # ── full (full multi-temporal stack — all dates x all bands, no selection) ──
    full_idx = full_names = None
    if not exps or "full" in exps:
        full_idx, full_names = build_full_stack_indices(local_band_names)

    # ── Build experiment registry & plan ───────────────────────────────────
    all_archs = list(ARCH_CFG.keys())
    run_exps  = exps  or ["single_date", "mt_ndvi", "gsi", "rf", "full"]
    run_archs = archs or all_archs

    registry = build_registry(
        single_date_idx=single_date_idx, single_date_names=single_date_names, single_date_key=sd_date_key,
        mt_base_idx=mt_base_idx,         mt_base_names=mt_base_names,         phenol_map=phenol_map,
        gsi_idx=gsi_idx,                 gsi_names=gsi_names,
        rf_idx=rf_idx,                   rf_names=rf_names,
        full_idx=full_idx,               full_names=full_names,
    )

    expanded_exps = expand_exp_keys(run_exps, registry)
    log.info(f"Selected experiments: {expanded_exps}")

    plan = []
    for exp_key in expanded_exps:
        cfg = registry.get(exp_key)
        if cfg is None:
            log.warning(f"Experiment '{exp_key}' not in registry — skipping")
            continue
        if cfg.band_indices is None:
            raise RuntimeError(
                f"Exp {exp_key}: band indices are None — required feature-selection output is missing."
            )
        for arch in run_archs:
            plan.append((exp_key, arch, cfg.band_indices, cfg.band_names,
                         f"{cfg.description}, {arch}", cfg.extra_kw))

    log.info(f"Planned {len(plan)} run(s): {[(e, a) for e, a, *_ in plan]}")
    return plan, registry
