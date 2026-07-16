"""S2/band index helpers, class weights, and S2 file validation."""

import json
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import rasterio
import rasterio.windows

from config import (
    N_BANDS_PER_DATE, KEEP_CLASSES, NUM_CLASSES, CLASS_REMAP, S2_NODATA,
)
from stages.training import run_state

log = logging.getLogger(__name__)


def _s2_for_year(s2_processed, yr):
    # Flat train/ dir — all files belong to the single training year
    return sorted(s2_processed)


def _valid_global_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return the subset of band_indices that are in range for s2_paths."""
    if band_indices is None:
        return set()
    needed = sorted({gi // n_bands_per_file for gi in band_indices
                     if gi // n_bands_per_file < len(s2_paths)})
    new_idx_map = set()
    for fi in needed:
        for local in range(n_bands_per_file):
            new_idx_map.add(fi * n_bands_per_file + local)
    return set(gi for gi in band_indices if gi in new_idx_map)


def _filter_s2_by_band_indices(s2_paths, band_indices, n_bands_per_file=N_BANDS_PER_DATE):
    """Return (filtered_paths, remapped_indices) keeping only TIF files that
    contribute at least one channel in band_indices, with indices remapped to
    their positions in the reduced stack.

    Example: 25 files × 11 bands = 275 channels.  single_date selects bands [157..165]
    (file 14 only) → returns [s2_paths[14]], remapped to [0..8].
    """
    if band_indices is None:
        return s2_paths, None
    # Which file indices (0-based) are needed?
    needed_file_idxs = sorted({gi // n_bands_per_file for gi in band_indices
                                if gi // n_bands_per_file < len(s2_paths)})
    filtered_paths = [s2_paths[i] for i in needed_file_idxs]
    # Build global-index → new-stacked-index map for every band in kept files
    new_idx_map = {}
    stacked = 0
    for fi in needed_file_idxs:
        for local in range(n_bands_per_file):
            new_idx_map[fi * n_bands_per_file + local] = stacked
            stacked += 1
    skipped = [gi for gi in band_indices if gi not in new_idx_map]
    if skipped:
        log.warning("  Dropping %d channel(s) from excluded/empty S2 files: %s",
                    len(skipped), skipped)
    remapped = [new_idx_map[gi] for gi in band_indices if gi in new_idx_map]
    return filtered_paths, remapped


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(cdl_path=None, return_counts=False):
    """Inverse-frequency weights from CDL (train area). Caches result alongside CDL.

    If return_counts=True, returns (weights_tensor, counts_array).
    """
    ref_cdl   = Path(cdl_path) if cdl_path else run_state.CDL_TRAIN
    cache_key = {"cdl": str(ref_cdl), "keep_classes": KEEP_CLASSES, "num_classes": NUM_CLASSES}
    cache_h   = hashlib.sha256(json.dumps(cache_key, sort_keys=True).encode()).hexdigest()[:12]
    cache_path = ref_cdl.parent / f"class_weights_{cache_h}.json"

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                d = json.load(f)
            w = d["weights"]
            c = d.get("class_counts")
            log.info(f"Class weights cache hit → {cache_path.name}")
            wt = torch.tensor(w, dtype=torch.float32)
            if return_counts and c is not None:
                return wt, np.asarray(c, dtype=np.float64)
            if not return_counts:
                return wt
        except Exception:
            pass

    with rasterio.open(ref_cdl) as src:
        cdl_arr = src.read(1).astype(np.int32)

    class_counts      = np.zeros(NUM_CLASSES, dtype=np.float64)
    class_counts[0]   = (cdl_arr == 0).sum()
    for cdl_id, model_id in CLASS_REMAP.items():
        class_counts[model_id] += (cdl_arr == cdl_id).sum()

    freq    = class_counts / (class_counts.sum() + 1e-9)
    weights = 1.0 / (freq + 1e-9)
    weights /= weights.sum()

    with open(cache_path, "w") as f:
        json.dump({"weights": weights.tolist(), "class_counts": class_counts.tolist()}, f)
    log.info(f"Class weights cached → {cache_path.name}")

    wt = torch.tensor(weights, dtype=torch.float32)
    if return_counts:
        return wt, class_counts
    return wt


# ── S2 file validation ────────────────────────────────────────────────────────

def validate_s2_files(s2_processed, s2_train_dir, min_valid_frac):
    """Drop corrupt / empty / low-validity S2 dates; return the valid subset.

    Samples a 3×3 grid of windows across a few bands per file, measures the
    valid-pixel fraction, and excludes acquisitions below `min_valid_frac`
    (e.g. high-cloud / partial-capture dates). Result is cached in
    `s2_train_dir/s2_validation_cache.json`, invalidated when the file set OR
    the threshold changes. Raises RuntimeError if any file is corrupt.
    """
    VALIDATION_WIN = 512

    _val_cache_path = s2_train_dir / "s2_validation_cache.json"
    _val_cache_key  = sorted(Path(p).name for p in s2_processed)

    def _load_validation_cache():
        if not _val_cache_path.exists():
            return None
        try:
            with open(_val_cache_path) as f:
                c = json.load(f)
            if c.get("files_key") == _val_cache_key and c.get("threshold") == min_valid_frac:
                return c
        except Exception:
            pass
        return None

    _cached = _load_validation_cache()
    if _cached:
        log.info(f"S2 validation cache hit ({len(_cached['valid'])} valid files)")
        valid_s2  = [p for p in s2_processed if Path(p).name in set(_cached["valid"])]
        _corrupt_names = set(_cached.get("corrupt", []))
        _nodata_names  = {r[0] for r in _cached.get("no_data", [])}
        corrupt  = [(p, "") for p in s2_processed if Path(p).name in _corrupt_names]
        no_data  = [(p, r[1]) for p in s2_processed
                    for r in _cached.get("no_data", []) if r[0] == Path(p).name]
    else:
        corrupt  = []
        no_data  = []
        valid_s2 = []

        def _check_file(path):
            try:
                with rasterio.open(path) as src:
                    h, w      = src.height, src.width
                    n_bands   = src.count
                    sz        = min(VALIDATION_WIN, w // 4, h // 4)
                    valid_px, total_px = 0, 0
                    check_bands = sorted({1, n_bands // 2, n_bands})
                    for band in check_bands:
                        for gy in range(3):
                            for gx in range(3):
                                ox = int((gx + 0.5) * w / 3) - sz // 2
                                oy = int((gy + 0.5) * h / 3) - sz // 2
                                ox = max(0, min(ox, w - sz))
                                oy = max(0, min(oy, h - sz))
                                win  = rasterio.windows.Window(ox, oy, sz, sz)
                                data = src.read(band, window=win).astype(np.float32)
                                ok   = (data != S2_NODATA) & np.isfinite(data)
                                valid_px += ok.sum()
                                total_px += ok.size
                return path, valid_px / total_px, None
            except Exception as e:
                return path, 0.0, str(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_check_file, p): p for p in s2_processed}
            for fut in as_completed(futures):
                path, frac, err = fut.result()
                if err:
                    corrupt.append((path, err))
                elif frac < min_valid_frac:
                    no_data.append((path, frac))
                else:
                    valid_s2.append(path)

        valid_s2.sort()

        try:
            with open(_val_cache_path, "w") as f:
                json.dump({
                    "files_key": _val_cache_key,
                    "threshold": min_valid_frac,
                    "valid":     [Path(p).name for p in valid_s2],
                    "corrupt":   [Path(p).name for p, _ in corrupt],
                    "no_data":   [[Path(p).name, frac] for p, frac in no_data],
                }, f)
            log.info(f"S2 validation cached → {_val_cache_path.name}")
        except Exception as e:
            log.warning(f"Could not write validation cache: {e}")

    if corrupt:
        log.error(f"Found {len(corrupt)} corrupt S2 file(s) — re-download before training:")
        for p, err in corrupt:
            log.error(f"  {Path(p).name}  ({err})")
        raise RuntimeError(
            f"{len(corrupt)} corrupt S2 file(s) detected. "
            "Re-download:  python stages/fetch_data.py --folder-id FOLDER_ID --years <year> --overwrite"
        )
    if no_data:
        log.warning(f"Excluding {len(no_data)} date(s) below {min_valid_frac*100:.0f}% valid pixels (high cloud / partial capture):")
        for p, frac in no_data:
            log.warning(f"  {Path(p).name}  ({frac*100:.2f}% valid)")
    log.info(f"{len(valid_s2)} S2 dates valid for training ({len(no_data)} low-validity excluded, threshold={min_valid_frac*100:.0f}%)")
    return valid_s2
