"""Per-band S2 normalization stats + per-channel expansion.

Three modes (NORM_MODES): percentile (P2/P98 clip → [0,1], main), minmax
([0,1]), zscore ((x-mean)/std, no clip). Stats are computed per S2 band from a
random pixel sample across the training scene, cached per mode, then expanded
to per-channel for the selected (date × band) inputs.
"""

import logging
from pathlib import Path

import numpy as np
import rasterio

from config import N_BANDS_PER_DATE, S2_NODATA, S2_BAND_NAMES

log = logging.getLogger(__name__)


NORM_MODES = ("percentile", "minmax", "zscore")


def _sample_per_band(s2_paths, n_samples_per_file=50_000, seed=42):
    """Return list[np.ndarray] — one array of valid samples per S2 band."""
    rng = np.random.default_rng(seed)
    samples: list = [[] for _ in range(N_BANDS_PER_DATE)]
    for path in s2_paths:
        try:
            with rasterio.open(path) as src:
                h, w, nb = src.height, src.width, src.count
                if nb != N_BANDS_PER_DATE:
                    continue
                n_pick = min(n_samples_per_file, h * w)
                ys = rng.integers(0, h, n_pick)
                xs = rng.integers(0, w, n_pick)
                for b in range(1, nb + 1):
                    arr = src.read(b)
                    vals = arr[ys, xs].astype(np.float32)
                    vals = vals[np.isfinite(vals) & (vals != S2_NODATA)]
                    samples[b - 1].append(vals)
        except Exception as e:
            log.warning(f"  [norm] {Path(path).name}: read failed — {e}")
    return [np.concatenate(s) if s else np.array([0.0, 10000.0], dtype=np.float32)
            for s in samples]


def compute_per_band_percentiles(s2_paths, n_samples_per_file=50_000,
                                  percentiles=(2.0, 98.0), seed=42):
    """Compute (p_lo, p_hi) per S2 band. Default P2/P98 (ablation baseline).

    Returns: (lo, hi), each shape (N_BANDS_PER_DATE,) float32.
    """
    log.info(f"  [norm:percentile] P{percentiles[0]}/P{percentiles[1]}  "
             f"{n_samples_per_file} px/file × {len(s2_paths)} files …")
    samples = _sample_per_band(s2_paths, n_samples_per_file, seed)
    lo  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    hi  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    for b in range(N_BANDS_PER_DATE):
        lo[b], hi[b] = np.percentile(samples[b], percentiles)
        log.info(f"    {S2_BAND_NAMES[b]}: lo={lo[b]:.1f}  hi={hi[b]:.1f}")
    return lo, hi


def compute_per_band_minmax(s2_paths, n_samples_per_file=50_000, seed=42):
    """Compute (min, max) per S2 band for min-max normalization → [0, 1].

    Returns: (lo, hi), each shape (N_BANDS_PER_DATE,) float32.
    """
    log.info(f"  [norm:minmax] {n_samples_per_file} px/file × {len(s2_paths)} files …")
    samples = _sample_per_band(s2_paths, n_samples_per_file, seed)
    lo  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    hi  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    for b in range(N_BANDS_PER_DATE):
        lo[b] = float(samples[b].min())
        hi[b] = float(samples[b].max())
        log.info(f"    {S2_BAND_NAMES[b]}: min={lo[b]:.1f}  max={hi[b]:.1f}")
    return lo, hi


def compute_per_band_zscore(s2_paths, n_samples_per_file=50_000, seed=42):
    """Compute (mean, std) per S2 band for z-score normalization: (x - mean) / std.

    Returns: (mean, std), each shape (N_BANDS_PER_DATE,) float32.
    """
    log.info(f"  [norm:zscore] {n_samples_per_file} px/file × {len(s2_paths)} files …")
    samples = _sample_per_band(s2_paths, n_samples_per_file, seed)
    lo  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    hi  = np.zeros(N_BANDS_PER_DATE, dtype=np.float32)
    for b in range(N_BANDS_PER_DATE):
        lo[b] = float(samples[b].mean())
        hi[b] = float(samples[b].std()) or 1.0
        log.info(f"    {S2_BAND_NAMES[b]}: mean={lo[b]:.1f}  std={hi[b]:.1f}")
    return lo, hi


def load_or_compute_norm_stats(norm_mode, s2_paths, cache_dir):
    """Load or compute (lo, hi) normalization stats for the given norm_mode.

    Returns (lo, hi) each shape (N_BANDS_PER_DATE,) float32.
    Cache keyed by norm_mode — different modes never share a cache file.
    """
    assert norm_mode in NORM_MODES, f"norm_mode must be one of {NORM_MODES}"
    cache_path = Path(cache_dir) / f"norm_stats_{norm_mode}.npz"
    if cache_path.exists():
        d = np.load(str(cache_path))
        log.info(f"  [norm:{norm_mode}] Loaded from cache → {cache_path.name}")
        return d["lo"].astype(np.float32), d["hi"].astype(np.float32)
    if norm_mode == "percentile":
        lo, hi = compute_per_band_percentiles(s2_paths)
    elif norm_mode == "minmax":
        lo, hi = compute_per_band_minmax(s2_paths)
    else:  # zscore
        lo, hi = compute_per_band_zscore(s2_paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), lo=lo, hi=hi)
    log.info(f"  [norm:{norm_mode}] Cached → {cache_path.name}")
    return lo, hi


# Keep old name as alias for backward compat
def load_or_compute_band_percentiles(s2_paths, cache_path):
    return load_or_compute_norm_stats("percentile", s2_paths, Path(cache_path).parent)


def _channel_to_band_idx(dataset_band_indices):
    """Map each selected channel → its S2 band index (0..N_BANDS_PER_DATE-1)."""
    if dataset_band_indices is None:
        # All channels of all files — band cycles every N_BANDS_PER_DATE channels
        return None
    return np.asarray([bi % N_BANDS_PER_DATE for bi in dataset_band_indices], dtype=np.int64)


def _per_channel_percentiles(band_indices, plo_per_band, phi_per_band):
    """Expand (N_BANDS,) per-band (lo, hi) stats to (n_ch,) per-channel via band lookup.

    lo/hi are the norm_mode stats per band — for percentile mode that's P2/P98
    """
    band_idx_per_ch = _channel_to_band_idx(band_indices)
    if band_idx_per_ch is None:
        raise ValueError("band_indices required for per-band percentile lookup")
    return plo_per_band[band_idx_per_ch], phi_per_band[band_idx_per_ch]
