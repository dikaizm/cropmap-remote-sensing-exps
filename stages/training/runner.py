"""Per-run orchestration: run one (experiment × architecture) training run.

setup → data split → model/optimiser/loss → MLflow child run → train loop →
test evaluation → artifacts → GDrive upload. Mutable CLI/config overrides are
read from ``run_state`` (as attributes) so main()'s reassignments are seen here.
"""

import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
import mlflow

from config import (
    CDL_CLASS_NAMES, KEEP_CLASSES, REMAP_LUT, NUM_CLASSES,
    PATCH_SIZE, STRIDE, MIN_VALID_FRAC,
    TRAIN_YEARS, TEST_YEAR, VAL_FRAC, TEST_FRAC,
    BLOCK_SIZE, MIN_CLASS_FRAC,
    EARLY_STOP, EARLY_STOP_DELTA, ARCH_CFG,
)
from stages.training import run_state
from stages.training.run_state import (
    DEVICE, _resolve_hp, _build_optimizer, _build_scheduler, _combo_done,
)
from stages.training.model_builder import build_model
from stages.training.metrics import (
    validate_one_epoch, evaluate_test_set, benchmark_inference_latency,
    _get_hardware_info, per_class_metric_dict,
)
from stages.training.datasets import (
    NormalizedDataset, PreloadedDataset, AugmentedSubset, _patch_weights,
)
from stages.training.viz import (
    _plot_confusion_matrix, save_segmentation_map, _load_rgb_for_viz,
    save_test_patch_visualizations, save_training_curve,
)
from stages.training.gdrive_upload import upload_models_to_gdrive
from stages.training.full_scene_inference import (
    run_full_inference, load_gt_remap,
)
from stages.training.helpers import (
    _s2_for_year, _valid_global_indices, _filter_s2_by_band_indices,
)
from stages.training.normalization import load_or_compute_norm_stats
from stages.training.losses import (
    build_wce, build_focal_tversky, build_dynamic_balanced,
)
from stages.data.spatial_split import (
    _block_spatial_split, _save_block_split_artifacts,
)
from geoai.geoai.train import RasterPatchDataset

log = logging.getLogger(__name__)


def run_experiment(
    exp_name,
    arch,
    band_indices,           # list[int]  OR  dict{yr: (list[int], list[str])}
    band_names_list,        # list[str]  (reference year; used for logging/metadata)
    description,
    s2_processed,
    class_weights_tensor,
    class_counts=None,      # required for focal_tversky effective-number weights
    loss="wce",             # "wce" | "focal_tversky" | "dynamic_balanced"
    force=False,
    skip_viz=False,
    no_preload=False,       # skip disk preload cache; use on-the-fly normalisation
    cache_only=False,       # build PreloadedDataset cache then exit without training
    norm_mode="percentile", # "percentile" | "minmax" | "zscore"
    no_aug=False,           # disable train-time geometric + spectral augmentation
):
    """band_indices: list[int] same for all years, or dict{yr: (idx, names)} per-year."""
    cfg           = ARCH_CFG[arch]
    hp            = _resolve_hp(cfg)
    bs            = hp["batch_size"] or run_state.BATCH_SIZE   # per-combo batch size override
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_only     = run_state.EVAL_ONLY_CKPT is not None
    if eval_only:
        # Write outputs (patch PNGs + test_patch_metrics.csv) to a local dir;
        # load weights from the provided checkpoint path.
        exp_dir   = run_state.MODELS_DIR / f"{exp_name}_evalonly_{run_timestamp}"
        best_ckpt = Path(run_state.EVAL_ONLY_CKPT)
        last_ckpt = best_ckpt
        exp_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Resume support: skip combos already finished (have a `.done` marker)
        # unless --force. Checked before creating a new dir so skips leave no litter.
        if not force and _combo_done(exp_name):
            log.info(f"already done — skipping {exp_name}  (use --force to re-run)")
            return None
        exp_dir   = run_state.MODELS_DIR / f"{exp_name}_{run_timestamp}"
        best_ckpt = exp_dir / "best_model.pth"
        last_ckpt = exp_dir / "last_model.pth"
        exp_dir.mkdir(parents=True, exist_ok=True)

    # Per-run log file — captured from start of training; uploaded as MLflow artifact at end
    run_log_path    = exp_dir / f"{exp_name}_train.log"
    run_log_handler = logging.FileHandler(run_log_path, mode="w")
    run_log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(run_log_handler)

    per_year = isinstance(band_indices, dict)

    def _yr_idx(yr):
        """Return (idx_list, names_list) for a given year."""
        if per_year:
            if yr in band_indices:
                return band_indices[yr]
            # fallback: use the first available year's indices
            fallback_yr = next(iter(band_indices))
            log.warning(
                f"Exp C projected: year {yr} not in projected map — "
                f"falling back to {fallback_yr} indices"
            )
            return band_indices[fallback_yr]
        return band_indices, band_names_list

    # Pre-pass: find globally consistent band indices available in ALL years.
    # Prevents channel-count mismatch when some years lack a file (e.g. excluded empty date).
    if not per_year:
        base_idx = band_indices  # same list applied to every year
        all_years = list(TRAIN_YEARS) + [TEST_YEAR]
        valid_sets = []
        for yr in all_years:
            yr_s2_all = _s2_for_year(s2_processed, yr)
            valid_sets.append(_valid_global_indices(yr_s2_all, base_idx))
        consistent = sorted(set.intersection(*valid_sets))
        dropped = len(base_idx) - len(consistent)
        if dropped:
            log.warning(
                f"  Dropping {dropped} channel(s) not available in all years "
                f"({', '.join(all_years)}) — keeping {len(consistent)} consistent channels"
            )
        consistent_set  = set(consistent)
        band_names_list = [name for gi, name in zip(base_idx, band_names_list) if gi in consistent_set]
        band_indices    = consistent

    in_channels = len(_yr_idx(TRAIN_YEARS[0])[0])
    log.info(f"\n{'='*65}")
    log.info(f" {exp_name}")
    log.info(f"  arch={arch}  in_channels={in_channels}  per_year_indices={per_year}")
    log.info(f"  {description}")
    log.info(f"{'='*65}\n")

    # ── Per-band normalisation stats (computed once from all training files) ────
    _stats_cache_dir = Path(s2_processed[0]).parent
    _all_train_s2 = []
    for yr in TRAIN_YEARS:
        _all_train_s2.extend(_s2_for_year(s2_processed, yr))
    band_percentiles = load_or_compute_norm_stats(norm_mode, _all_train_s2, _stats_cache_dir)
    log.info(f"  norm_mode={norm_mode}")

    # ── Year-based dataset split ──────────────────────────────────────────────
    train_year_datasets_raw = []   # RasterPatchDataset — for _patch_weights (needs _cdl etc.)
    train_year_datasets     = []   # PreloadedDataset  — for DataLoader
    primary_s2_filtered = None     # S2 paths for primary year (used for segmentation map)
    primary_idx_local   = None     # band indices for primary year
    for yr in TRAIN_YEARS:
        yr_s2  = _s2_for_year(s2_processed, yr)
        yr_cdl = run_state.CDL_TRAIN
        if not yr_s2 or not yr_cdl.exists():
            log.warning(f"Skipping train year {yr}: {'no S2' if not yr_s2 else 'CDL missing'}")
            continue
        yr_idx, _ = _yr_idx(yr)
        yr_s2_filtered, yr_idx_local = _filter_s2_by_band_indices(yr_s2, yr_idx)
        if primary_s2_filtered is None:
            primary_s2_filtered = yr_s2_filtered
            primary_idx_local   = yr_idx_local
        ds_raw = RasterPatchDataset(
            s2_paths=yr_s2_filtered, cdl_path=str(yr_cdl),
            patch_size=PATCH_SIZE, stride=STRIDE,
            keep_classes=KEEP_CLASSES, remap_lut=REMAP_LUT,
            min_valid_frac=MIN_VALID_FRAC, band_indices=yr_idx_local,
        )
        log.info(f"  [{yr}] {len(ds_raw):,} patches  ({len(yr_idx)} channels, {len(yr_s2_filtered)}/{len(yr_s2)} files)")
        train_year_datasets_raw.append(ds_raw)
        if no_preload:
            train_year_datasets.append(NormalizedDataset(ds_raw, band_percentiles=band_percentiles,
                                                          norm_mode=norm_mode))
        else:
            preloaded = PreloadedDataset(ds_raw, desc=yr, cache_dir=run_state.PRELOAD_CACHE_DIR,
                                         band_percentiles=band_percentiles, norm_mode=norm_mode)
            train_year_datasets.append(preloaded)

    assert train_year_datasets, "No training data for any TRAIN_YEAR"

    if cache_only:
        log.info(f"  [--build-cache-only] Cache built for {exp_name} — skipping training")
        log.removeHandler(run_log_handler)
        run_log_handler.close()
        return None

    train_val_ds = ConcatDataset(train_year_datasets)

    # Split: train / val / test — spatial block (grid) split; whole blocks per
    # split, prevents patch-adjacency spatial leakage. When TEST_FRAC=0: test
    # evaluation is skipped.
    n_total   = len(train_val_ds)
    tr_idx, va_idx, te_idx, split_info = _block_spatial_split(
        train_year_datasets_raw, BLOCK_SIZE, VAL_FRAC, TEST_FRAC,
        NUM_CLASSES, run_state.SEED, min_class_frac=MIN_CLASS_FRAC, log=log,
    )
    train_ds = torch.utils.data.Subset(train_val_ds, tr_idx)
    val_ds   = torch.utils.data.Subset(train_val_ds, va_idx)
    test_ds  = torch.utils.data.Subset(train_val_ds, te_idx) if te_idx else None
    n_train, n_val, n_test = len(tr_idx), len(va_idx), len(te_idx)
    split_label = f"block_spatial_{int(round((1-VAL_FRAC-TEST_FRAC)*100))}_{int(round(VAL_FRAC*100))}_{int(round(TEST_FRAC*100))}"
    split_artifacts = _save_block_split_artifacts(
        split_info, exp_dir, exp_name,
        class_names=[CDL_CLASS_NAMES[c] for c in KEEP_CLASSES], log=log,
    )
    test_s2_filtered = None
    test_idx_local   = None

    # Class-weighted sampler: rare-class patches sampled more frequently
    log.info("  Computing patch weights for class-balanced sampling...")
    all_weights = _patch_weights(train_year_datasets_raw)
    train_weights = all_weights[train_ds.indices]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(train_weights).double(),
        num_samples=n_train,
        replacement=True,
    )
    # Band indices threaded through to enable per-band (vs per-channel) spectral
    # augmentation. For per-year dict, use the primary year's indices.
    _aug_bi = primary_idx_local if isinstance(band_indices, dict) else band_indices
    aug_train_ds = train_ds if no_aug else AugmentedSubset(train_ds, band_indices=_aug_bi)
    # In --eval-only with on-the-fly (no-preload) datasets, workers can't pickle open
    # rasterio handles under macOS spawn; use 0 workers (single test pass, speed is fine).
    _nw = 0 if eval_only else 4
    train_dl = DataLoader(aug_train_ds, batch_size=bs, sampler=sampler, num_workers=_nw, pin_memory=True, drop_last=True)
    val_dl   = DataLoader(val_ds,       batch_size=bs, shuffle=False,   num_workers=_nw, pin_memory=True)
    test_dl  = DataLoader(test_ds,      batch_size=bs, shuffle=False,   num_workers=_nw, pin_memory=True) if test_ds is not None else None
    if n_test > 0:
        log.info(f"  Patches: {n_train:,} train / {n_val:,} val / {n_test:,} test ({split_label})")
    else:
        log.info(f"  Patches: {n_train:,} train / {n_val:,} val  (no test split — TEST_FRAC=0)")

    # ── Model + optimiser + scheduler + loss ──────────────────────────────────
    model     = build_model(arch, in_channels, NUM_CLASSES)
    grad_clip = hp["grad_clip"]
    optimizer = _build_optimizer(
        hp["optimizer"], model.parameters(),
        lr=hp["lr"], weight_decay=hp["weight_decay"], momentum=hp["momentum"],
    )
    scheduler = _build_scheduler(
        optimizer, run_state.MAX_EPOCHS, power=hp["sched_power"],
        warmup_epochs=hp["warmup_epochs"], kind=hp["scheduler"],
    )
    _decay_label = (f"CosineAnnealingLR" if hp["scheduler"] == "cosine"
                    else f"PolynomialLR(power={hp['sched_power']:g})")
    _sched_label = _decay_label + (f"+LinearWarmup({hp['warmup_epochs']}ep)" if hp["warmup_epochs"] else "")
    _opt_label   = {"adamw": "AdamW", "adam": "Adam", "sgd": f"SGD(m={hp['momentum']:g})"}[hp["optimizer"]]
    if run_state.HP_OVERRIDE:
        log.info(
            f"  HP override: opt={_opt_label} lr={hp['lr']:.2e} wd={hp['weight_decay']:.2e} "
            f"bs={bs} sched={_sched_label} grad_clip={grad_clip or 'off'}"
        )

    # ── Loss function (named) ──────────────────────────────────────────────
    if loss == "focal_tversky":
        criterion = build_focal_tversky(
            class_counts=class_counts,
            tv_alpha=0.7, tv_beta=0.3, tv_gamma=0.75,
        ).to(DEVICE)
        log.info("  Loss=focal_tversky — Focal Tversky (median-freq weighted class-mean)")
    elif loss == "dynamic_balanced":
        criterion = build_dynamic_balanced(
            num_classes=NUM_CLASSES, beta=0.9999, fallback_weight=2.0,
        ).to(DEVICE)
        log.info("  Loss=dynamic_balanced — Dynamic Effective Class Balanced (per-batch, β=0.9999)")
    else:
        criterion = build_wce(class_weights_tensor.to(DEVICE))
        log.info("  Loss=wce — WeightedCrossEntropy")

    # ── MLflow run (child — nested under parent created in main()) ────────────

    _child_run_name = f"eval_{exp_name}" if eval_only else exp_name
    with mlflow.start_run(run_name=_child_run_name, nested=True, log_system_metrics=True) as run:
        mlflow.log_params({
            "experiment":     exp_name,
            "architecture":   arch,
            "encoder":        cfg["encoder"],
            "in_channels":    in_channels,
            "num_classes":    NUM_CLASSES,
            "patch_size":     PATCH_SIZE,
            "stride":         STRIDE,
            "batch_size":     bs,
            "max_epochs":     run_state.MAX_EPOCHS,
            "early_stopping": EARLY_STOP,
            "learning_rate":  hp["lr"],
            "weight_decay":   hp["weight_decay"],
            "warmup_epochs":  hp["warmup_epochs"],
            "sched_power":    hp["sched_power"],
            "grad_clip":      grad_clip,
            "optimizer":      _opt_label,
            "lr_scheduler":   _sched_label,
            "loss":           loss,
            "norm_mode":      norm_mode,
            "augmentation":   not no_aug,
            "train_years":    str(TRAIN_YEARS),
            "test_year":      TEST_YEAR,
            "train_patches":  n_train,
            "val_patches":    n_val,
            "test_patches":   n_test,
            "split":          split_label,
            "block_size":     BLOCK_SIZE,
            "n_blocks":       (split_info or {}).get("n_blocks"),
            "description":    description,
            "keep_classes":   str(KEEP_CLASSES),
            "model_params":   getattr(model, "_n_params", None),
            **_get_hardware_info(),
        })
        mlflow.set_tag("band_names", str(band_names_list))
        mlflow.set_tag("n_bands",    str(in_channels))
        mlflow.set_tag(
            "mlflow.note.content",
            f"{description}. Arch={arch} ({cfg['encoder']}), {in_channels} input "
            f"channels, loss={loss}. Trained on {TRAIN_YEARS}, tested on {TEST_YEAR} "
            f"({split_label}: {n_train} train / {n_val} val / {n_test} test patches).",
        )


        # ── Training loop ─────────────────────────────────────────────────────
        best_miou              = 0.0
        best_val_mf1           = 0.0
        best_val_oa            = 0.0
        best_val_per_class_iou = {}
        best_val_per_class_f1  = {}
        no_improve             = 0
        history                = []
        t_start    = time.time()

        if eval_only:
            log.info(f"  [--eval-only] Skipping training — evaluating checkpoint {best_ckpt}")

        for epoch in ([] if eval_only else range(run_state.MAX_EPOCHS)):
            t_ep = time.time()

            model.train()
            train_loss_acc, n_batches = 0.0, 0
            _logged_vram = epoch > 0   # log VRAM once on first batch of epoch 0
            for imgs, masks in train_dl:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                imgs        = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = criterion(logits, masks)

                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_loss_acc += loss.item()
                n_batches += 1

                if not _logged_vram and torch.cuda.is_available():
                    alloc  = torch.cuda.memory_allocated()  / 1024**3
                    reserv = torch.cuda.memory_reserved()   / 1024**3
                    log.info(f"  [VRAM] allocated={alloc:.2f} GB  reserved={reserv:.2f} GB")
                    _logged_vram = True

            train_loss = train_loss_acc / n_batches
            val_m = validate_one_epoch(model, val_dl, criterion, DEVICE, NUM_CLASSES)
            scheduler.step()

            ep_t = time.time() - t_ep
            per_cls_metrics = {
                **per_class_metric_dict(val_m["per_class_iou"], "val_iou", KEEP_CLASSES, CDL_CLASS_NAMES),
                **per_class_metric_dict(val_m["per_class_f1"],  "val_f1",  KEEP_CLASSES, CDL_CLASS_NAMES),
                **per_class_metric_dict(val_m["per_class_oa"],  "val_oa",  KEEP_CLASSES, CDL_CLASS_NAMES),
            }
            mlflow.log_metrics({
                "train_loss":   train_loss,
                "val_loss":     val_m["loss"],
                "val_miou":     val_m["miou"],
                "val_mf1":      val_m["mf1"],
                "val_oa":       val_m["oa"],
                "lr":           scheduler.get_last_lr()[0],
                "epoch_time_s": ep_t,
                **per_cls_metrics,
            }, step=epoch)

            history.append({
                "epoch":      epoch + 1,
                "train_loss": round(train_loss,       4),
                "val_loss":   round(val_m["loss"],    4),
                "val_miou":   round(val_m["miou"],    4),
                "val_mf1":    round(val_m["mf1"],     4),
                "val_oa":     round(val_m["oa"],      4),
                "epoch_t_s":  round(ep_t,              1),
            })

            if val_m["miou"] > best_miou + EARLY_STOP_DELTA:
                best_miou              = val_m["miou"]
                best_val_mf1           = val_m["mf1"]
                best_val_oa            = val_m["oa"]
                best_val_per_class_iou = val_m["per_class_iou"]
                best_val_per_class_f1  = val_m["per_class_f1"]
                no_improve = 0
                torch.save({
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "best_miou":        best_miou,
                    "band_indices":     band_indices,
                    "band_names":       band_names_list,
                    "in_channels":      in_channels,
                    "num_classes":      NUM_CLASSES,
                    "architecture":     arch,
                }, best_ckpt)
            else:
                no_improve += 1

            total_min = (time.time() - t_start) / 60
            log.info(
                f"  Ep {epoch+1:3d}/{run_state.MAX_EPOCHS} "
                f"loss={train_loss:.4f} val={val_m['loss']:.4f} "
                f"mIoU={val_m['miou']:.4f} mF1={val_m['mf1']:.4f} OA={val_m['oa']:.4f} "
                f"best={best_miou:.4f} patience={no_improve}/{EARLY_STOP} "
                f"{ep_t:.0f}s  {total_min:.1f}min"
            )
            _iou_parts, _f1_parts, _oa_parts = [], [], []
            for cls_id, iou in val_m["per_class_iou"].items():
                cdl_id = KEEP_CLASSES[cls_id - 1]
                short  = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}").replace(" ", "")
                _iou_parts.append(f"{short}={iou:.3f}" if not np.isnan(iou) else f"{short}=  nan")
                f1v = val_m["per_class_f1"].get(cls_id, float("nan"))
                _f1_parts.append(f"{short}={f1v:.3f}" if not np.isnan(f1v) else f"{short}=  nan")
                oav = val_m["per_class_oa"].get(cls_id, float("nan"))
                _oa_parts.append(f"{short}={oav:.3f}" if not np.isnan(oav) else f"{short}=  nan")
            log.info("    IoU: " + "  ".join(_iou_parts))
            log.info("     F1: " + "  ".join(_f1_parts))
            log.info("     OA: " + "  ".join(_oa_parts))

            # Save last checkpoint every epoch (overwrites previous)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_miou":         val_m["miou"],
                "band_indices":     band_indices,
                "band_names":       band_names_list,
                "in_channels":      in_channels,
                "num_classes":      NUM_CLASSES,
                "architecture":     arch,
            }, last_ckpt)

            if no_improve >= EARLY_STOP:
                log.info(f"  Early stopping at epoch {epoch + 1}")
                break

        # Training time only — measured up to here, excludes test/inference below.
        train_time_total_s = time.time() - t_start
        mlflow.log_metrics({
            "train_time_total_s":   train_time_total_s,
            "train_time_total_min": train_time_total_s / 60,
        })
        log.info(f"  Training time: {train_time_total_s:.1f}s ({train_time_total_s / 60:.1f}min)")

        # ── Test evaluation (held-out same-area split, only when TEST_FRAC > 0) ─
        ckpt = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if eval_only:
            best_miou = ckpt.get("best_miou", best_miou)
            log.info(f"  [--eval-only] Loaded checkpoint (reported best_val_miou={best_miou:.4f})")

        if test_dl is not None:
            log.info("  Evaluating on held-out test set (same area, random split)...")
            test_r = evaluate_test_set(model, test_dl, NUM_CLASSES, DEVICE)
            log.info("  Benchmarking per-patch inference latency...")
            benchmark_inference_latency(model, test_dl, DEVICE, run.info.run_id)
        else:
            log.info("  No same-area test split — TEST_FRAC=0; skipping test evaluation")
            test_r = None

        _base_metrics = {
            "best_val_miou": best_miou,
            "best_val_mf1":  best_val_mf1,
            "best_val_oa":   best_val_oa,
            "total_epochs":  len(history),
        }
        if test_r is not None:
            _base_metrics.update({
                "test_miou": test_r["miou"],
                "test_mf1":  test_r["mf1"],
                "test_oa":   test_r["oa"],
            })
        mlflow.log_metrics(_base_metrics)

        for k, v in per_class_metric_dict(best_val_per_class_iou, "best_val_iou", KEEP_CLASSES, CDL_CLASS_NAMES).items():
            mlflow.log_metric(k, v)
        for k, v in per_class_metric_dict(best_val_per_class_f1, "best_val_f1", KEEP_CLASSES, CDL_CLASS_NAMES).items():
            mlflow.log_metric(k, v)

        if test_r is not None:
            for k, v in per_class_metric_dict(test_r["per_class_iou"], "test_iou", KEEP_CLASSES, CDL_CLASS_NAMES).items():
                mlflow.log_metric(k, v)
            for k, v in per_class_metric_dict(test_r["per_class_f1"], "test_f1", KEEP_CLASSES, CDL_CLASS_NAMES).items():
                mlflow.log_metric(k, v)

            # ── Log per-class IoU table to console ───────────────────────────
            log.info(f"  Test results  mIoU={test_r['miou']:.4f}  mF1={test_r['mf1']:.4f}  OA={test_r['oa']:.4f}")
            log.info(f"  {'Class':<20} {'CDL ID':>6}  {'IoU':>7}")
            log.info(f"  {'-'*38}")
            for cls_id, iou in test_r["per_class_iou"].items():
                cdl_id = KEEP_CLASSES[cls_id - 1]
                name   = CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}")
                iou_s  = f"{iou:.4f}" if not np.isnan(iou) else "    nan"
                log.info(f"  {name:<20} {cdl_id:>6}  {iou_s:>7}")
            log.info(f"  {'-'*38}")
            log.info(f"  {'mIoU':<20} {'':>6}  {test_r['miou']:>7.4f}")

        # ── eval-only: write full segmentation map + per-patch viz + metrics CSV ─
        # (skips training-only artifacts: history/curve/gdrive upload)
        if eval_only:
            # Full-scene segmentation map (same renderer as training finalize path)
            if not skip_viz and primary_s2_filtered is not None:
                log.info(f"  [--eval-only] Running full-image inference for {exp_name}...")
                gt_map, _   = load_gt_remap(str(run_state.CDL_TRAIN))
                pred_map, _ = run_full_inference(
                    model, primary_s2_filtered, primary_idx_local,
                    patch_size=PATCH_SIZE, stride=PATCH_SIZE,
                    channel_stats=None, band_percentiles=band_percentiles,
                    norm_mode=norm_mode,
                )
                seg_path = exp_dir / "test_segmentation_map.png"
                rgb_img  = _load_rgb_for_viz(primary_s2_filtered, band_percentiles, downsample=4)
                save_segmentation_map(
                    pred_map, gt_map,
                    title=f"Segmentation Map ({TEST_YEAR})",
                    save_path=str(seg_path),
                    rgb_img=rgb_img,
                )
                mlflow.log_artifact(str(seg_path))
                del pred_map, gt_map
            if test_r is not None and test_dl is not None:
                log.info(f"  [--eval-only] Saving per-patch test visualizations + metrics CSV for {exp_name}...")
                patch_dir = save_test_patch_visualizations(
                    test_dl, test_r["preds"], test_r["labels"],
                    s2_processed, test_ds, train_year_datasets_raw,
                    band_percentiles, exp_dir, exp_name,
                )
                mlflow.log_artifacts(str(patch_dir), artifact_path="test_patches")
                _metrics_csv = exp_dir / "test_patch_metrics.csv"
                if _metrics_csv.exists():
                    mlflow.log_artifact(str(_metrics_csv))
                log.info(f"  [--eval-only] Outputs written to {exp_dir} + logged to MLflow")
            else:
                log.warning("  [--eval-only] No test split available — nothing to write")
            _eval_run_id = run.info.run_id
            run_log_handler.flush()
            log.removeHandler(run_log_handler)
            run_log_handler.close()
            run_state._DEFERRED_LOG_RUNS.append((_eval_run_id, str(run_log_path)))
            return None

        # ── Artifacts ─────────────────────────────────────────────────────────

        # Training history CSV
        hist_df  = pd.DataFrame(history)
        hist_csv = exp_dir / "training_history.csv"
        hist_df.to_csv(hist_csv, index=False)

        # Training curve PNG
        curve_path = exp_dir / "training_curve.png"
        save_training_curve(hist_df, best_miou, exp_name, curve_path)

        # Per-class metrics CSV + confusion matrix (only when same-area test exists)
        iou_csv = exp_dir / "test_per_class_iou.csv"
        cm_path = exp_dir / "confusion_matrix.png"
        if test_r is not None:
            iou_rows = []
            for cls_id, iou in test_r["per_class_iou"].items():
                cdl_id = KEEP_CLASSES[cls_id - 1]
                f1v    = test_r["per_class_f1"].get(cls_id, float("nan"))
                iou_rows.append({
                    "class_id":   cls_id,
                    "cdl_id":     cdl_id,
                    "class_name": CDL_CLASS_NAMES.get(cdl_id, f"cls{cls_id}"),
                    "iou":        round(iou, 4) if not np.isnan(iou) else float("nan"),
                    "f1":         round(f1v, 4) if not np.isnan(f1v) else float("nan"),
                })
            pd.DataFrame(iou_rows).to_csv(iou_csv, index=False)
            if "preds" in test_r and "labels" in test_r:
                _plot_confusion_matrix(test_r["preds"], test_r["labels"], str(cm_path))

        # Segmentation map PNG (full-tile inference)
        seg_path = None
        if not skip_viz and test_s2_filtered is not None:
            log.info(f"  Running full-image inference for {exp_name}...")
            gt_map, _    = load_gt_remap(str(test_cdl))
            pred_map, _  = run_full_inference(
                model, test_s2_filtered, test_idx_local, patch_size=PATCH_SIZE, stride=PATCH_SIZE,
                channel_stats=None, band_percentiles=band_percentiles,
                norm_mode=norm_mode,
            )
            seg_path = exp_dir / "test_segmentation_map.png"
            rgb_img = _load_rgb_for_viz(test_s2_filtered, band_percentiles, downsample=4)
            save_segmentation_map(
                pred_map, gt_map,
                title=f"{exp_name} — Test Segmentation ({TEST_YEAR})",
                save_path=str(seg_path),
                rgb_img=rgb_img,
            )
            np.save(exp_dir / "test_pred_map.npy", pred_map)
            np.save(exp_dir / "test_gt_map.npy", gt_map)
            del pred_map, gt_map
        elif not skip_viz and primary_s2_filtered is not None:
            log.info(f"  Running full-image inference on training area for {exp_name}...")
            gt_map, _   = load_gt_remap(str(run_state.CDL_TRAIN))
            pred_map, _ = run_full_inference(
                model, primary_s2_filtered, primary_idx_local,
                patch_size=PATCH_SIZE, stride=PATCH_SIZE,
                channel_stats=None, band_percentiles=band_percentiles,
                norm_mode=norm_mode,
            )
            seg_path = exp_dir / "test_segmentation_map.png"
            rgb_img = _load_rgb_for_viz(primary_s2_filtered, band_percentiles, downsample=4)
            save_segmentation_map(
                pred_map, gt_map,
                title=f"{exp_name} — Segmentation Map ({TRAIN_YEARS[0]})",
                save_path=str(seg_path),
                rgb_img=rgb_img,
            )
            np.save(exp_dir / "test_pred_map.npy", pred_map)
            np.save(exp_dir / "test_gt_map.npy", gt_map)
            del pred_map, gt_map

        # Per-patch test visualizations
        if not skip_viz and test_r is not None and test_dl is not None:
            log.info(f"  Saving per-patch test visualizations for {exp_name}...")
            patch_dir = save_test_patch_visualizations(
                test_dl, test_r["preds"], test_r["labels"],
                s2_processed, test_ds, train_year_datasets_raw,
                band_percentiles, exp_dir, exp_name,
            )
            mlflow.log_artifacts(str(patch_dir), artifact_path="test_patches")

        gdrive_links = upload_models_to_gdrive(
            run_name=f"{exp_name}_{run_timestamp}",
            model_files=[best_ckpt, last_ckpt],
        )
        for fname, link in gdrive_links.items():
            mlflow.set_tag(f"gdrive_{fname}", link)
        mlflow.log_artifact(str(hist_csv))
        mlflow.log_artifact(str(curve_path))
        if split_artifacts is not None:
            for p in split_artifacts.values():
                if Path(p).exists():
                    mlflow.log_artifact(str(p), artifact_path="split")
        if iou_csv.exists():
            mlflow.log_artifact(str(iou_csv))
        if cm_path.exists():
            mlflow.log_artifact(str(cm_path))
        if seg_path is not None:
            mlflow.log_artifact(str(seg_path))

        run_id = run.info.run_id

    # Logs uploaded after the whole session ends (see _flush_deferred_logs).
    run_log_handler.flush()
    run_log_handler.close()
    log.removeHandler(run_log_handler)
    run_state._DEFERRED_LOG_RUNS.append((run_id, str(run_log_path)))

    summary = {
        "exp_name":      exp_name,
        "arch":          arch,
        "in_channels":   in_channels,
        "best_val_miou": round(best_miou, 4),
        "total_epochs":  len(history),
        "run_id":        run_id,
        "ckpt":          str(best_ckpt),
    }
    if test_r is not None:
        summary["test_miou"] = round(test_r["miou"], 4) if not np.isnan(test_r["miou"]) else float("nan")
        summary["test_mf1"]  = round(test_r["mf1"],  4) if not np.isnan(test_r["mf1"])  else float("nan")
        summary["test_oa"]   = round(test_r["oa"],   4) if not np.isnan(test_r["oa"])   else float("nan")
    if test_r is not None:
        spatial_str = f"test_mIoU={test_r['miou']:.4f}"
    else:
        spatial_str = "(no test set)"
    log.info(f"\n{exp_name}  val_mIoU={best_miou:.4f}  {spatial_str}  run={run_id}")

    # Resume marker — written last, so a crashed run is NOT marked done and reruns.
    try:
        (exp_dir / ".done").write_text(f"{run_timestamp}\trun_id={run_id}\tval_miou={best_miou:.4f}\n")
    except Exception as e:
        log.warning(f"  Could not write .done marker: {e}")
    return summary
