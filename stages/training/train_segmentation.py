"""
Full Model Validation — CLI entry point + session orchestrator.

Five experiment configurations × 2 architectures = up to 10 training runs.

| Config             | Dates               | Band selection | Purpose                      |
|--------------------|---------------------|----------------|------------------------------|
| single_date        | peak NDVI           | none (all bands)| Baseline (isolates temporal) |
| mt_ndvi            | 4 calendar dates     | none           | Multi-temporal baseline   |
| gsi                | GSI          | GSI     | GSI spectral-temporal        |
| rf                 | RF           | RF      | RF spectral-temporal         |
| full               | all dates            | none (all bands)| Full-stack upper-bound baseline |

The heavy lifting is split into focused modules in this package:
  run_state           runtime overrides + HP-grid / optimizer / scheduler / session helpers
  helpers             s2/band index helpers, class weights, S2 file validation
  experiment_plan     resolve channel sets -> registry -> (exp × arch) plan
  runner              run_experiment — one training run end to end
  metrics / datasets / model_builder / viz / gdrive_upload / full_scene_inference

This file keeps the CLI, `main()` session loop, and backward-compatible
re-exports of the names external code imports from here.

Usage:
    python stages/training/train_segmentation.py                       # run all experiments
    python stages/training/train_segmentation.py --exp single_date     # only single-date baseline
    python stages/training/train_segmentation.py --exp gsi --arch segformer
    python stages/training/train_segmentation.py --force               # re-run even if ckpt exists
    python stages/training/train_segmentation.py --data-dir /mnt/data
    python stages/training/train_segmentation.py --exp gsi --random-channel-order  # channel-order sanity check
"""

import os
import sys
import logging
import argparse
from glob import glob
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")  # must precede `import mlflow`

import torch
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

_ROOT = next(_p for _p in Path(__file__).resolve().parents if (_p / "config.py").exists())
sys.path.insert(0, str(_ROOT))

from config import (
    S2_MIN_VALID_FRAC, MLFLOW_TRACKING_URI,
    TRAIN_YEARS, TEST_YEAR, VAL_FRAC, TEST_FRAC,
    LOGS_DIR, ARCH_CFG, GDRIVE_PRELOAD_CACHE_FOLDER_ID,
)
from stages.training import run_state
from stages.training.run_state import (
    _device_label, _load_hp_grid, _hp_tag, _flush_deferred_logs,
)
from stages.training.normalization import NORM_MODES
from stages.training.helpers import (
    compute_class_weights, validate_s2_files, _filter_s2_by_band_indices,
)
from stages.training.metrics import evaluate_test_set, _get_hardware_info
from stages.training.datasets import NormalizedDataset
from stages.training.model_builder import build_model
from stages.training.gdrive_upload import (
    _check_gdrive_token, upload_models_to_gdrive,
)
from stages.training.experiment_plan import build_experiment_plan
from stages.training.runner import run_experiment

# Backward-compat re-exports: external code imports these names from this module.
#   pipeline.py                 → main
#   stages/infer/model_io.py    → build_model
#   notebooks/05_*.ipynb        → build_model, compute_class_weights,
#                                 evaluate_test_set, NormalizedDataset,
#                                 _filter_s2_by_band_indices
__all__ = [
    "main", "run_experiment", "build_model", "compute_class_weights",
    "evaluate_test_set", "NormalizedDataset", "_filter_s2_by_band_indices",
]

log = logging.getLogger(__name__)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    exps=None,
    archs=None,
    loss="wce",
    force=False,
    data_dir=None,
    phenol_dates=None,
    skip_viz=False,
    score_threshold=0.5,
    batch_size=None,
    epochs=None,
    no_preload=False,
    cache_only=False,
    norm_mode="percentile",
    no_aug=False,
    hp=None,
    seed=None,
    random_channel_order=False,
    channel_shuffle_seed=None,
):
    # Runtime overrides live on run_state so every training module sees them.
    if batch_size:
        run_state.BATCH_SIZE = batch_size
        log.info(f"Batch size overridden: {run_state.BATCH_SIZE}")
    if epochs:
        run_state.MAX_EPOCHS = epochs
        log.info(f"Max epochs overridden: {run_state.MAX_EPOCHS}")

    if seed is not None:
        run_state.SEED = seed
        run_state.SEED_TAG = f"seed{seed}"
        log.info(f"Seed overridden: {run_state.SEED}")
    else:
        run_state.SEED_TAG = ""

    run_state.HP_OVERRIDE = hp or None
    run_state.HP_TAG = _hp_tag(hp) if hp else ""
    if run_state.HP_OVERRIDE:
        log.info(f"HP grid combo: {run_state.HP_OVERRIDE}  (tag={run_state.HP_TAG})")

    _check_gdrive_token()

    # Override data directories on run_state so all module-level functions pick
    # up the new paths at call time.
    if data_dir:
        data_dir = Path(data_dir)
        run_state.S2_TRAIN_DIR      = data_dir / "s2" / "2024"
        run_state.S2_PROCESSED_DIR  = run_state.S2_TRAIN_DIR
        run_state.CDL_TRAIN         = data_dir / "cdl" / "cdl_2024_study_area_filtered.tif"
        run_state.CDL_BY_YEAR       = {"2024": run_state.CDL_TRAIN}
        run_state.MODELS_DIR        = data_dir / "models"
        run_state.FIGURES_DIR       = data_dir / "figures"
        run_state.PRELOAD_CACHE_DIR = data_dir / "preload_cache"   # cache lives under the data dir
        run_state.PRELOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        log.info(f"Data dir overridden to {data_dir}  (preload_cache={run_state.PRELOAD_CACHE_DIR})")

    # ── Cloud preload cache — download a prebuilt cache instead of rebuilding ──
    # Filenames are content-hash keyed by PreloadedDataset, so a matching file is a
    # cache hit at train time. Skipped under --no-preload (no cache is consulted).
    _pc_gdrive = (GDRIVE_PRELOAD_CACHE_FOLDER_ID or None) if (
        "args" in globals() and getattr(args, "use_cloud_preload", False)) else None
    if _pc_gdrive and not getattr(args, "no_preload", False):
        run_state.PRELOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        from stages.data.fetch_data import fetch_preload_cache
        log.info(f"Fetching cloud preload cache from GDrive folder {_pc_gdrive} → {run_state.PRELOAD_CACHE_DIR}")
        got = fetch_preload_cache(_pc_gdrive, str(run_state.PRELOAD_CACHE_DIR), overwrite=False)
        log.info(f"Cloud preload cache: {len(got)} file(s) ready in {run_state.PRELOAD_CACHE_DIR}")

    s2_processed = sorted(
        glob(str(run_state.S2_TRAIN_DIR / "*_processed.tif")) +
        glob(str(run_state.S2_TRAIN_DIR / "S2H_*.tif"))
    )
    seen = set()
    s2_processed = [p for p in s2_processed if not (p in seen or seen.add(p))
                    and not Path(p).name.startswith("._")]
    if not s2_processed:
        raise FileNotFoundError(f"No processed S2 files in {run_state.S2_TRAIN_DIR}")

    # Drop corrupt / empty / low-validity dates (cached; see helpers.validate_s2_files).
    s2_processed = validate_s2_files(s2_processed, run_state.S2_TRAIN_DIR, S2_MIN_VALID_FRAC)

    run_state.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_state.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve channel sets → registry → (exp × arch) plan.
    plan, registry = build_experiment_plan(
        s2_processed, exps, archs, phenol_dates, score_threshold, data_dir,
    )

    # ── Sanity-check ablation: shuffle channel order per run ────────────────
    # Verifies the model learns from channel *content*, not its position in the
    # input stack — union-selected channel sets have no inherent phenological
    # order, so shuffling should not materially change accuracy.
    # channel_shuffle_seed is intentionally decoupled from run_state.SEED (the
    # training/split seed) so repeated trials vary ONLY the channel order —
    # split and model init stay fixed, isolating the one variable under test.
    effective_shuffle_seed = channel_shuffle_seed if channel_shuffle_seed is not None else run_state.SEED
    if random_channel_order:
        import random as _random
        _rng = _random.Random(effective_shuffle_seed)
        _shuffled_plan = []
        for exp_key, arch, band_idx, band_names, description, extra_kw in plan:
            if band_idx:
                paired = list(zip(band_idx, band_names))
                _rng.shuffle(paired)
                band_idx = [p[0] for p in paired]
                band_names = [p[1] for p in paired]
            _shuffled_plan.append((exp_key, arch, band_idx, band_names, description, extra_kw))
        plan = _shuffled_plan
        log.info(f"Channel order randomized (shuffle_seed={effective_shuffle_seed}) for {len(plan)} run(s)")

    # ── Class weights ──────────────────────────────────────────────────────
    cw_tensor, cw_counts = compute_class_weights(return_counts=True)
    log.info("Class weights computed")

    # ── MLflow setup ────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # ── Run experiments — one top-level run per exp_key, nested run per arch ─
    all_results = []
    exp_groups: dict = {}
    for exp_key, arch, band_idx, band_names, description, extra_kw in plan:
        exp_groups.setdefault(exp_key, []).append((arch, band_idx, band_names, description, extra_kw))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for exp_key, arch_runs in exp_groups.items():
        cfg_entry  = registry[exp_key]
        experiment = mlflow.set_experiment(cfg_entry.mlflow_experiment)
        MlflowClient().set_experiment_tag(
            experiment.experiment_id, "mlflow.note.content",
            "Segmentation training — 8-crop CalCROP21-style class selection "
            "(>=1M px threshold), block spatial split "
            f"({int(round((1-VAL_FRAC-TEST_FRAC)*100))}/{int(round(VAL_FRAC*100))}/{int(round(TEST_FRAC*100))}; "
            "block split groups whole grid cells per split to avoid patch-adjacency "
            "leakage). Compares band-selection "
            "experiments (single-date / multi-temporal NDVI / GSI / RF / full-stack) "
            f"across architectures. train_years={TRAIN_YEARS}, test_year={TEST_YEAR}.",
        )
        n_ch = len(arch_runs[0][1]) if arch_runs[0][1] else 0
        # score_threshold only governs gsi/rf direct band selection — for
        # single_date/mt_ndvi/full (no threshold-based selection involved) the
        # suffix is meaningless noise (e.g. exp_full_s0.5 despite score_threshold
        # playing no role), so only include it where it's actually load-bearing.
        _sel_sfx = f"_s{score_threshold:g}" if exp_key in ("gsi", "rf") else ""
        if run_state.HP_TAG:
            _sel_sfx += f"_{run_state.HP_TAG}"
        if run_state.SEED_TAG:
            _sel_sfx += f"_{run_state.SEED_TAG}"
        if random_channel_order:
            _sel_sfx += f"_randord{effective_shuffle_seed}"
        parent_run_name = f"exp_{exp_key}{_sel_sfx}_{timestamp}"
        if run_state.EVAL_ONLY_CKPT is not None:
            parent_run_name = f"eval_{parent_run_name}"
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            mlflow.log_params({
                "experiment":   f"exp_{exp_key}",
                "n_channels":   n_ch,
                "train_years":  str(TRAIN_YEARS),
                "test_year":    TEST_YEAR,
                "description":  cfg_entry.description,
                "loss":         loss,
                "seed":         run_state.SEED,
                "score_threshold": score_threshold,
                "random_channel_order": random_channel_order,
                "channel_shuffle_seed": effective_shuffle_seed if random_channel_order else None,
                **({f"hp_{k}": v for k, v in run_state.HP_OVERRIDE.items()} if run_state.HP_OVERRIDE else {}),
                **_get_hardware_info(),
            })
            mlflow.set_tag(
                "mlflow.note.content",
                f"Parent run grouping all architectures for experiment '{exp_key}': "
                f"{cfg_entry.description}. {n_ch} input channels, trained on "
                f"{TRAIN_YEARS} and tested on {TEST_YEAR}.",
            )
            log.info(f"Parent MLflow run: {parent_run_name}  (id={parent_run.info.run_id})")
            for arch, band_idx, band_names, description, extra_kw in arch_runs:
                exp_name = f"exp_{exp_key}{_sel_sfx}_{arch}"
                result = run_experiment(
                    exp_name=exp_name,
                    arch=arch,
                    band_indices=band_idx,
                    band_names_list=band_names,
                    description=description,
                    s2_processed=s2_processed,
                    class_weights_tensor=cw_tensor,
                    class_counts=cw_counts,
                    loss=loss,
                    force=force,
                    skip_viz=skip_viz,
                    no_preload=no_preload,
                    cache_only=cache_only,
                    norm_mode=norm_mode,
                    no_aug=no_aug,
                    **extra_kw,
                )
                if result is not None:
                    all_results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    if all_results:
        summary_df  = pd.DataFrame(all_results)
        sort_col = next(
            (c for c in ("test_miou", "best_val_miou") if c in summary_df.columns),
            None,
        )
        if sort_col:
            summary_df = summary_df.sort_values(sort_col, ascending=False)
        summary_csv = run_state.MODELS_DIR / "experiment_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        log.info("\n=== Experiment Summary ===")
        cols = [c for c in [
            "exp_name", "arch", "in_channels",
            "best_val_miou", "test_miou", "test_mf1", "test_oa",
            "total_epochs",
        ] if c in summary_df.columns]
        log.info("\n" + summary_df[cols].to_string(index=False))
        log.info(f"Saved: {summary_csv}")

    log.info("All experiments done — segmentation maps, confusion matrices, and IoU CSVs logged to MLflow.")


def _upload_existing_models(filter_exps=None, filter_archs=None):
    """Upload best_model.pth + last_model.pth for all existing run dirs.

    Scans MODELS_DIR for subdirectories that contain at least one of the two
    checkpoint files and uploads them to GDrive under runs/<run_dir_name>/.

    filter_exps  — optional list of exp shorthand keys (e.g. ["C_v3", "A_v2"]).
                   Run dir must contain any of the keys as a substring.
    filter_archs — optional list of arch names to further filter.
    """
    import re as _re

    def _matches(run_dir_name):
        if filter_exps:
            if not any(
                _re.search(r"(?i)" + _re.escape(e.lower()), run_dir_name.lower())
                for e in filter_exps
            ):
                return False
        if filter_archs:
            if not any(arch.lower() in run_dir_name.lower() for arch in filter_archs):
                return False
        return True

    models_dir = run_state.MODELS_DIR
    candidates = sorted(models_dir.iterdir()) if models_dir.exists() else []
    run_dirs = [
        d for d in candidates
        if d.is_dir() and _matches(d.name)
        and (
            (d / "best_model.pth").exists()
            or (d / "last_model.pth").exists()
        )
    ]

    if not run_dirs:
        log.warning("No matching run dirs with model checkpoints found under %s", models_dir)
        return

    log.info("Uploading models for %d run(s)…", len(run_dirs))
    for run_dir in run_dirs:
        model_files = [
            f for f in [run_dir / "best_model.pth", run_dir / "last_model.pth"]
            if f.exists()
        ]
        log.info("  %s: %s", run_dir.name, [f.name for f in model_files])
        links = upload_models_to_gdrive(run_name=run_dir.name, model_files=model_files)
        if links:
            for fname, link in links.items():
                log.info("    %s → %s", fname, link)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation models for band selection comparison")
    parser.add_argument(
        "--exp", nargs="+",
        choices=["single_date", "mt_ndvi", "gsi", "rf", "full"],
        default=["single_date", "mt_ndvi", "gsi", "rf", "full"],
        help=(
            "Experiments to run (default: all five). "
            "single_date=peak NDVI date + ALL bands (single-date baseline), "
            "mt_ndvi=4 calendar dates + ALL S2_BAND_NAMES (multi-temporal baseline, no selection), "
            "gsi=GSI-direct top-K, rf=RF-direct top-K (multi-class MDI), "
            "full=all dates + ALL S2_BAND_NAMES, no selection (full-stack upper-bound baseline)."
        ),
    )
    parser.add_argument(
        "--arch", nargs="+", choices=list(ARCH_CFG.keys()),
        default=None,
        help="Which architectures to run (default: all)",
    )
    parser.add_argument(
        "--loss",
        choices=["wce", "focal_tversky", "dynamic_balanced"],
        default="wce",
        help=(
            "Loss function: wce (default, WeightedCrossEntropy), "
            "focal_tversky (Focal Tversky, median-freq weighted class-mean), "
            "dynamic_balanced (per-batch Cui+2019 weights; thesis primary, DECB-CE)"
        ),
    )
    parser.add_argument("--force",      action="store_true", help="Re-run even if checkpoint exists")
    parser.add_argument("--skip-viz",   action="store_true", help="Skip full-image visualization")
    parser.add_argument("--no-preload", action="store_true",
                        help="Skip disk preload cache; use on-the-fly normalization. "
                             "Slower per epoch but avoids large disk/RAM allocation — useful for high channel counts.")
    parser.add_argument("--norm", default="percentile", choices=list(NORM_MODES),
                        help="Input normalization strategy for ablation. "
                             "percentile: clip [P2,P98]→[0,1] (default). "
                             "minmax: clip [min,max]→[0,1]. "
                             "zscore: (x-mean)/std, no clip.")
    parser.add_argument("--build-cache-only", action="store_true",
                        help="Build PreloadedDataset cache for all selected experiments then exit without training. "
                             "Transfer the cache dir to another machine and training will use it as a cache hit.")
    parser.add_argument(
        "--use-cloud-preload", action="store_true",
        help="Download the cloud-built portable preload cache (preload_*.npy + *_masks.pt) from "
             "config.GDRIVE_PRELOAD_CACHE_FOLDER_ID into the preload_cache dir before training, "
             "instead of rebuilding locally. Ignored with --no-preload.")
    parser.add_argument(
        "--upload-cache-gdrive", nargs="?", const=GDRIVE_PRELOAD_CACHE_FOLDER_ID or None,
        default=None, metavar="FOLDER_ID",
        help="After --build-cache-only, upload the built preload cache to this GDrive folder. "
             "Bare flag uses config.GDRIVE_PRELOAD_CACHE_FOLDER_ID. With --build-cache-only and a "
             "configured folder id, upload runs automatically.")
    parser.add_argument("--no-upload-cache", action="store_true",
                        help="Disable the automatic preload-cache upload after --build-cache-only.")
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable train-time augmentation (geometric + spectral). "
                             "Useful for ablation or fast debug runs.")
    parser.add_argument("--random-channel-order", action="store_true",
                        help="Shuffle each experiment's channel order before training. "
                             "Sanity-check ablation: confirms the model learns from channel "
                             "content, not its position in the input stack. Run names get a "
                             "'_randord{shuffle_seed}' suffix in MLflow. Combine with "
                             "--random-channel-order-trials for repeated independent shuffles.")
    parser.add_argument(
        "--random-channel-order-trials", type=int, default=1, metavar="N",
        help="With --random-channel-order: repeat the --exp/--arch matrix N times, each "
             "with a different, reproducible channel permutation (shuffle seeds 1..N). "
             "The channel-shuffle seed is decoupled from --seed/config.SEED, so the "
             "spatial split and model init stay fixed across trials — only channel order "
             "varies. E.g. --exp gsi rf --random-channel-order --random-channel-order-trials 4 "
             "gives 4 runs each for gsi-deeplab, gsi-segformer, rf-deeplab, rf-segformer (16 total).",
    )
    parser.add_argument("--data-dir", default=None, help="Override data/processed directory")
    parser.add_argument("--phenol-dates", default=None, help="Path to pre-computed phenol_dates.json for Exp B multi-temporal baseline")
    parser.add_argument("--shutdown", action="store_true", help="Stop the RunPod pod after training")
    parser.add_argument(
        "--upload-existing", action="store_true",
        help=(
            "Upload best_model.pth and last_model.pth for all existing run dirs under "
            "MODELS_DIR to Google Drive without re-training. "
            "Optionally filter with --exp / --arch."
        ),
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5, metavar="T",
        help="Per-crop normalized-score threshold for gsi/rf direct selection "
             "(loads select_gsi/rf_direct_s{T}.json). Wei et al. 2023: normalize per "
             "crop to [0,1], retain >= T. Default 0.5.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, metavar="N",
        help=f"Override BATCH_SIZE from config (default: {run_state.BATCH_SIZE}).",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, metavar="N",
        help=f"Override MAX_EPOCHS from config (default: {run_state.MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--eval-only", metavar="CKPT_PATH",
        help="Skip training — load checkpoint and run spatial test evaluation only.",
    )
    parser.add_argument(
        "--hp-grid", metavar="JSON_PATH", default=None,
        help="Hyperparameter-grid JSON. Per-arch (recommended): top-level keys = "
             "arch names, each {\"grid\":{...}} or {\"combos\":[...]} — separate search "
             "space per architecture. Shared: top-level {\"grid\":{\"lr\":[...], "
             "\"weight_decay\":[...], \"warmup_epochs\":[...], \"sched_power\":[...]}} or "
             "{\"combos\":[...]} applied to every --arch. Tunable keys: lr, weight_decay, "
             "warmup_epochs, sched_power, scheduler(polynomial|cosine), optimizer(adamw|"
             "adam|sgd), momentum, grad_clip(0=off), batch_size. Each combo overrides "
             "ARCH_CFG/config defaults, runs the --exp/--arch matrix, and logs to MLflow "
             "tagged with the combo. Combos run outermost. See configs/hp_grid_example.json.",
    )
    parser.add_argument(
        "--seed-grid", type=int, nargs="+", default=None, metavar="SEED",
        help="Run the full experiment matrix once per seed for stability testing. "
             "Each seed overrides config.SEED, tags run names with _seed{N}, and logs "
             "'seed' as an MLflow param. The spatial block split is re-seeded each run "
             "so splits differ across seeds. E.g. --seed-grid 42 123 456 789",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    # Suppress GDAL tile-decode noise (LZW/ZIP errors on legacy files).
    # Filters on a Logger only apply at that logger — not on propagation —
    # so we must filter on each Handler after basicConfig creates them.
    class _SuppressGDALFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            return "GDAL signalled an error" not in msg and "IReadBlock failed" not in msg

    _gdal_filter = _SuppressGDALFilter()
    # Also silence rasterio._err directly (covers worker processes via fork)
    logging.getLogger("rasterio._err").setLevel(logging.ERROR)

    run_state.SESSION_LOG_PATH = str(LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_state.SESSION_LOG_PATH),
        ],
    )
    for _h in logging.root.handlers:
        _h.addFilter(_gdal_filter)

    log.info(f"Device: {_device_label()}  PyTorch: {torch.__version__}")

    if args.upload_existing:
        _upload_existing_models(filter_exps=args.exp, filter_archs=args.arch)
        sys.exit(0)

    if args.eval_only:
        # Route through the normal --exp/--arch path so the deterministic
        # same-area split + correct band selection are rebuilt; run_experiment then
        # skips training, loads this checkpoint, and runs test eval + per-patch viz.
        ckpt_path = Path(args.eval_only)
        if not ckpt_path.exists():
            log.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        run_state.EVAL_ONLY_CKPT = str(ckpt_path)
        # Eval runs log to the tracking server (run names prefixed "eval_"), with
        # the full segmentation map, per-patch PNGs, and metrics CSV as artifacts.
        log.info(f"--eval-only: evaluating {ckpt_path} (logged to MLflow as eval_* runs)")

    # HP-grid combos run outermost; [(None, None)] = no grid (single default pass).
    # Each entry is (arch_or_None, combo): arch=None → use the --arch matrix;
    # an arch string pins the combo to that single architecture (per-arch grid).
    hp_combos = _load_hp_grid(args.hp_grid) if args.hp_grid else [(None, None)]
    if args.hp_grid:
        log.info(f"HP grid: {len(hp_combos)} combo(s) from {args.hp_grid}")
        for i, (a, c) in enumerate(hp_combos):
            log.info(f"  [{i+1}/{len(hp_combos)}] arch={a or 'ALL'}  {c}")

    seed_list = args.seed_grid if args.seed_grid else [None]
    if args.seed_grid:
        log.info(f"Seed grid: {seed_list} ({len(seed_list)} seed(s))")

    for seed_val in seed_list:
        if seed_val is not None:
            log.info(f"{'*'*65}")
            log.info(f"  Seed: {seed_val}")
            log.info(f"{'*'*65}")
        for hp_arch, hp in hp_combos:
            # Per-arch grid pinned to an arch excluded by --arch → skip.
            if hp_arch is not None and args.arch and hp_arch not in args.arch:
                log.info(f"Skip HP combo (arch {hp_arch} not in --arch {args.arch})")
                continue
            run_archs = [hp_arch] if hp_arch is not None else args.arch
            if hp is not None:
                log.info(f"{'#'*65}")
                log.info(f"  HP combo: arch={hp_arch or 'ALL'}  {hp}")
                log.info(f"{'#'*65}")

            # Channel-shuffle trials — innermost loop. Each trial reuses the same
            # seed/hp/split but draws a different, reproducible channel permutation
            # (shuffle seeds 1..N), so only channel order varies across trials.
            n_trials = args.random_channel_order_trials if args.random_channel_order else 1
            shuffle_seeds = list(range(1, n_trials + 1)) if n_trials > 1 else [None]
            if n_trials > 1:
                log.info(f"Channel-order trials: {n_trials} (shuffle seeds {shuffle_seeds})")

            for shuffle_seed in shuffle_seeds:
                if shuffle_seed is not None:
                    log.info(f"  -- channel-shuffle trial: seed={shuffle_seed} --")
                main(
                    exps=args.exp,
                    archs=run_archs,
                    loss=args.loss,
                    force=args.force,
                    data_dir=args.data_dir,
                    phenol_dates=args.phenol_dates,
                    skip_viz=args.skip_viz,
                    score_threshold=args.score_threshold,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    no_preload=args.no_preload,
                    cache_only=args.build_cache_only,
                    norm_mode=args.norm,
                    no_aug=args.no_aug,
                    hp=hp,
                    seed=seed_val,
                    random_channel_order=args.random_channel_order,
                    channel_shuffle_seed=shuffle_seed,
                )

    # ── Upload all logs once, after the whole session finished ────────────────
    _flush_deferred_logs()

    # ── Auto-upload preload cache after --build-cache-only ────────────────────
    if args.build_cache_only and not args.no_upload_cache:
        _up_folder = args.upload_cache_gdrive or GDRIVE_PRELOAD_CACHE_FOLDER_ID or None
        if _up_folder:
            from stages.data.fetch_data import upload_preload_cache
            log.info(f"Uploading built preload cache from {run_state.PRELOAD_CACHE_DIR} → GDrive {_up_folder}")
            up = upload_preload_cache(_up_folder, str(run_state.PRELOAD_CACHE_DIR), overwrite=False)
            log.info(f"Preload cache upload complete: {len(up)} file(s)")
        else:
            log.info("No upload folder set (config.GDRIVE_PRELOAD_CACHE_FOLDER_ID empty / no --upload-cache-gdrive) — skipping upload.")

    if args.shutdown:
        import urllib.request, urllib.error, json as _json, time as _time
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).parent.parent / ".env")
        pod_id  = os.environ.get("RUNPOD_POD_ID")
        api_key = os.environ.get("RUNPOD_API_KEY")
        delay   = 5   # minutes
        if pod_id and api_key:
            log.warning(f"RunPod pod {pod_id} will stop in {delay} minutes.")
            _time.sleep(delay * 60)
            query = f'{{"query": "mutation {{ podStop(input: {{podId: \\"{pod_id}\\"}}) {{ id desiredStatus }} }}"}}'
            req   = urllib.request.Request(
                "https://api.runpod.io/graphql",
                data    = query.encode(),
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    log.info(f"Pod stop response: {_json.loads(resp.read())}")
            except urllib.error.URLError as e:
                log.error(f"Failed to stop pod: {e}")
        else:
            log.warning(f"RUNPOD_POD_ID/RUNPOD_API_KEY not set — falling back to sudo shutdown in {delay} min")
            import subprocess
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay}"], check=False)
