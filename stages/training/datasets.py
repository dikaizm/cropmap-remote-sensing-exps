"""Patch datasets: class-balanced sampling weights, augmentation, preload cache,
on-the-fly normalisation, and full-pass channel stats.
"""

import os
import time
import json
import hashlib
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import rasterio

from config import N_BANDS_PER_DATE, PRELOAD_RAM_BUDGET_GB
from stages.training.normalization import (
    NORM_MODES, _per_channel_percentiles,
)
from stages.training import run_state

log = logging.getLogger(__name__)

# Fraction of the resolved RAM budget reserved for concurrent file-read buffers vs.
# the normalisation-chunk pass (each runs at a different point in __init__, never
# overlapping, but both must fit individually under the same ceiling).
_READ_BUDGET_FRAC  = 0.5
_CHUNK_BUDGET_FRAC = 0.5
# Only claim this fraction of currently-available memory when auto-detecting —
# leaves headroom for the OS page cache, mmap pages of the output buffer, and
# whatever else the user is running.
_AVAIL_SAFETY_FRAC = 0.7
_MIN_BUDGET_BYTES  = 2e9   # floor: below this, thread/chunk sizing degenerates


def _resolve_ram_budget() -> float:
    """Return the cache-build RAM budget in bytes, resolved at build time.

    Priority: PRELOAD_RAM_BUDGET_GB env var > config.PRELOAD_RAM_BUDGET_GB >
    auto-detect. Auto-detect claims _AVAIL_SAFETY_FRAC of *currently available*
    memory (not total — respects other running processes), so the budget adapts
    to the machine and its load instead of assuming a fixed ceiling.
    """
    explicit = os.environ.get("PRELOAD_RAM_BUDGET_GB") or PRELOAD_RAM_BUDGET_GB
    if explicit:
        return max(float(explicit) * 1e9, _MIN_BUDGET_BYTES)
    try:
        import psutil
        avail = psutil.virtual_memory().available
    except ImportError:
        try:    # Linux/macOS fallback: conservative half of total physical RAM
            avail = (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) // 2
        except (ValueError, OSError):
            avail = 8e9
    return max(avail * _AVAIL_SAFETY_FRAC, _MIN_BUDGET_BYTES)


def _ram_safe_read_threads(budget_bytes: float, per_file_bytes: float,
                           n_files: int, cpu_cap: int) -> int:
    """Cap concurrent read-threads so simultaneous full-file buffers stay under budget.

    Each thread holds one decoded float32 file array in RAM at a time
    (see out_dtype=float32 in _read_one_file — no float64 intermediate).
    """
    by_ram = max(1, int(budget_bytes * _READ_BUDGET_FRAC // max(per_file_bytes, 1)))
    return max(1, min(n_files, cpu_cap, by_ram))


def _ram_safe_chunk(budget_bytes: float, n_ch: int, ps: int) -> int:
    """Cap normalisation CHUNK (patches/iter) so each chunk's float32 copy fits budget."""
    per_patch_bytes = n_ch * ps * ps * 4  # float32 working copy
    return max(1, int(budget_bytes * _CHUNK_BUDGET_FRAC // max(per_patch_bytes, 1)))


# ── Class-weighted patch sampler ──────────────────────────────────────────────

def _patch_weights(datasets: list) -> np.ndarray:
    """
    Compute a weight per patch across a list of RasterPatchDataset objects.
    Weight = sum over classes of (patch_pixel_count[c] / global_pixel_count[c]).
    Rare-class patches get higher weight → balanced mini-batches.
    Uses the in-memory _cdl array — no S2 I/O.
    """
    ps = datasets[0].patch_size

    # Pass 1: global class pixel counts
    global_counts: dict[int, int] = {}
    for ds in datasets:
        cdl = ds._cdl
        remap = ds._remap_lut
        for r, c in ds.patches:
            patch_cdl = cdl[r:r + ps, c:c + ps]
            remapped  = remap[np.clip(patch_cdl, 0, 255)]
            for cls_id in np.unique(remapped):
                if cls_id == 0:
                    continue
                global_counts[int(cls_id)] = global_counts.get(int(cls_id), 0) + int((remapped == cls_id).sum())

    if not global_counts:
        # Fallback: uniform weights
        return np.ones(sum(len(ds.patches) for ds in datasets), dtype=np.float32)

    # Pass 2: per-patch weight
    weights = []
    for ds in datasets:
        cdl   = ds._cdl
        remap = ds._remap_lut
        for r, c in ds.patches:
            patch_cdl = cdl[r:r + ps, c:c + ps]
            remapped  = remap[np.clip(patch_cdl, 0, 255)]
            w = 0.0
            for cls_id in np.unique(remapped):
                if cls_id == 0:
                    continue
                cnt = int((remapped == cls_id).sum())
                w  += cnt / global_counts[int(cls_id)]
            weights.append(w if w > 0 else 1e-6)

    return np.array(weights, dtype=np.float64)


# ── Augmentation wrapper ───────────────────────────────────────────────────────

class AugmentedSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies geometric + spectral augmentations to (img, mask).

    Spectral augmentation is *per-band* (B1..B12) not per-channel. All channels
    belonging to the same S2 band (e.g., all B4 dates) share the same scale and
    offset within one augmentation. This is physically faithful — atmospheric
    scattering, sensor calibration, and BRDF effects are band-specific but
    time-consistent for a fixed sensor.

    Designed to improve spatial generalisation by simulating cross-area
    reflectance variation (different atmospheric/illumination conditions in
    held-out areas).

    Args:
        subset:        underlying torch Dataset / Subset producing (img, mask).
        band_indices:  global band index per channel of img — used to map each
                       channel to its S2 band (B1..B12). If None, falls back to
                       per-channel augmentation.
        band_scale:    per-band multiplicative scale range (default ±15%).
        band_offset:   per-band additive offset range on normalised reflectance
                       (default ±5%, simulates haze/aerosol).
        brightness:    global multiplicative scale applied to all channels
                       (default ±10%, simulates illumination).
        gamma:         per-band gamma correction range (1±0.15), simulates
                       nonlinear sensor/atmosphere response.
        noise_std:     additive Gaussian noise sigma (default 0.05).
        drop_p:        per-channel random dropout probability (default 0.05).
        erase_p:       random erasing probability (default 0.5).
    """

    def __init__(self, subset, band_indices=None,
                 band_scale=0.15, band_offset=0.05, brightness=0.10,
                 gamma=0.15, noise_std=0.05, drop_p=0.05, erase_p=0.5):
        self.subset      = subset
        self.band_scale  = band_scale
        self.band_offset = band_offset
        self.brightness  = brightness
        self.gamma_range = gamma
        self.noise_std   = noise_std
        self.drop_p      = drop_p
        self.erase_p     = erase_p

        # Pre-compute per-channel → per-band lookup (LongTensor, K,)
        if band_indices is not None:
            ch2band = np.asarray(
                [int(bi) % N_BANDS_PER_DATE for bi in band_indices],
                dtype=np.int64,
            )
            self.ch2band = torch.from_numpy(ch2band)   # (K,)
            self.n_bands = N_BANDS_PER_DATE
        else:
            self.ch2band = None
            self.n_bands = None

    def __len__(self):
        return len(self.subset)

    def _per_channel_from_band(self, per_band_vals):
        """Expand per-band values (n_bands,) → per-channel (K, 1, 1) via lookup."""
        if self.ch2band is None:
            raise RuntimeError("band_indices not set; per-band augmentation unavailable")
        return per_band_vals[self.ch2band].view(-1, 1, 1)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]   # img: (C,H,W) float [0,1] (percentile-normalised), mask: (H,W)

        # ── Geometric ────────────────────────────────────────────────────────
        if torch.rand(1).item() > 0.5:
            img  = torch.flip(img,  [-1])
            mask = torch.flip(mask, [-1])
        if torch.rand(1).item() > 0.5:
            img  = torch.flip(img,  [-2])
            mask = torch.flip(mask, [-2])
        k = torch.randint(0, 4, (1,)).item()
        if k:
            img  = torch.rot90(img,  k, [-2, -1])
            mask = torch.rot90(mask, k, [-2, -1])

        C = img.shape[0]

        # ── Per-band spectral augmentation ──────────────────────────────────
        if self.ch2band is not None:
            # Per-band scale: simulates atmospheric/sensor variation per wavelength
            band_scale  = 1.0 + (torch.rand(self.n_bands) - 0.5) * 2.0 * self.band_scale
            band_offset = (torch.rand(self.n_bands) - 0.5) * 2.0 * self.band_offset
            scale       = self._per_channel_from_band(band_scale)
            offset      = self._per_channel_from_band(band_offset)
            img = img * scale + offset

            # Per-band gamma: nonlinear response variation
            band_gamma = 1.0 + (torch.rand(self.n_bands) - 0.5) * 2.0 * self.gamma_range
            gamma      = self._per_channel_from_band(band_gamma)
            img = img.clamp(min=0.0).pow(gamma)
        else:
            # Fallback per-channel (no band_indices supplied)
            scale  = 1.0 + (torch.rand(C, 1, 1) - 0.5) * 2.0 * self.band_scale
            offset = (torch.rand(C, 1, 1) - 0.5) * 2.0 * self.band_offset
            img = img * scale + offset

        # ── Global brightness (illumination simulation) ─────────────────────
        brightness = 1.0 + (torch.rand(1).item() - 0.5) * 2.0 * self.brightness
        img = img * brightness

        # ── Per-channel random dropout ──────────────────────────────────────
        if self.drop_p > 0:
            drop_mask = (torch.rand(C, 1, 1) > self.drop_p).float()
            img = img * drop_mask

        # ── Gaussian noise ──────────────────────────────────────────────────
        if self.noise_std > 0:
            img = img + torch.randn_like(img) * self.noise_std

        # ── Random erasing ──────────────────────────────────────────────────
        if torch.rand(1).item() < self.erase_p:
            H, W = img.shape[-2], img.shape[-1]
            rh   = int(H * (0.1 + 0.1 * torch.rand(1).item()))
            rw   = int(W * (0.1 + 0.1 * torch.rand(1).item()))
            r0   = torch.randint(0, H - rh + 1, (1,)).item()
            c0   = torch.randint(0, W - rw + 1, (1,)).item()
            img[:, r0:r0 + rh, c0:c0 + rw] = 0.0

        return img, mask


# ── In-memory dataset cache ───────────────────────────────────────────────────

class PreloadedDataset(torch.utils.data.Dataset):
    """Builds a persistent disk cache of all patches; loads imgs via memory-map.

    Reads each TIF file once in full (parallel threads) instead of per-patch
    window reads → ~30–60s instead of 15+ min for large datasets.
    Cache key covers s2_paths/cdl_path/bands/patch_size.

    Imgs stored as float16 .npy → loaded with mmap_mode='r' so the OS pages in
    only what each minibatch needs.  Peak RAM = model + batch, not full dataset.
    Masks stored as int64 .pt (typically <1 GB, always in RAM).
    """

    def __init__(self, dataset, desc="preload", cache_dir=None, n_threads=None,
                 channel_stats=None, band_percentiles=None, norm_mode="percentile"):
        """Normalisation: per-band stats → normalised values stored as float16.

        band_percentiles: (lo, hi) each shape (N_BANDS_PER_DATE,). Required.
          Semantics depend on norm_mode:
            percentile → (p2, p98) clip to [0,1]
            minmax     → (min, max) clip to [0,1]
            zscore     → (mean, std) no clip
        channel_stats: deprecated, kept for API compat (ignored).
        norm_mode: one of NORM_MODES ("percentile", "minmax", "zscore").
        """
        assert band_percentiles is not None, "band_percentiles (lo, hi) required"
        assert norm_mode in NORM_MODES, f"norm_mode must be one of {NORM_MODES}"
        self._norm_mode = norm_mode
        imgs_path, masks_path = self._cache_paths(dataset, cache_dir, norm_mode) if cache_dir else (None, None)

        if imgs_path and imgs_path.exists() and masks_path and masks_path.exists():
            log.info(f"  [{desc}] Cache hit → mmap {imgs_path.name}")
            t0 = time.time()
            self._imgs  = np.load(str(imgs_path), mmap_mode="r")
            self._masks = torch.load(masks_path, map_location="cpu", weights_only=True)
            gb_disk = imgs_path.stat().st_size / 1e9
            log.info(f"  [{desc}] mmap ready in {time.time()-t0:.1f}s ({gb_disk:.2f} GB on disk)")
            return

        log.info(f"  [{desc}] Cache miss → preloading from {len(dataset._s2_srcs)} TIF files …")
        t0 = time.time()

        n  = len(dataset)
        ps = dataset.patch_size
        band_indices  = dataset.band_indices
        n_ch_per_file = [src.count for src in dataset._s2_srcs]
        ch_offsets    = np.cumsum([0] + n_ch_per_file).tolist()
        n_ch          = len(band_indices) if band_indices is not None else ch_offsets[-1]

        # file_extraction[fi] = [(output_col, local_band_idx_1based), ...]
        file_extraction: dict = {}
        targets = band_indices if band_indices is not None else list(range(ch_offsets[-1]))
        for out_pos, gi in enumerate(targets):
            for fi in range(len(n_ch_per_file)):
                if ch_offsets[fi] <= gi < ch_offsets[fi + 1]:
                    file_extraction.setdefault(fi, []).append((out_pos, gi - ch_offsets[fi] + 1))
                    break

        patches = dataset.patches
        nodata  = dataset.nodata

        # Allocate buf as a disk-backed float16 memmap — never occupies RAM regardless
        # of channel count. 70ch × 1800 patches × 256² × float32 ≈ 33 GB; float16
        # memmap keeps peak RAM to ~O(one TIF file) during the fill loop.
        _buf_path = (imgs_path.with_suffix(".tmp.npy") if imgs_path
                     else Path(run_state.PRELOAD_CACHE_DIR) / f"_tmp_{os.getpid()}.npy")
        _buf_path.parent.mkdir(parents=True, exist_ok=True)
        buf = np.lib.format.open_memmap(
            str(_buf_path), mode="w+", dtype=np.float16, shape=(n, n_ch, ps, ps)
        )
        gb_alloc = buf.nbytes / 1e9
        log.info(f"  [{desc}] Buf: {n}×{n_ch}×{ps}×{ps} float16 = {gb_alloc:.1f} GB on disk")

        def _read_one_file(fi):
            extractions = file_extraction[fi]
            local_idxs  = [e[1] for e in extractions]
            out_cols    = [e[0] for e in extractions]
            try:
                with rasterio.open(dataset.s2_paths[fi]) as src:
                    # out_dtype=float32 decodes directly to float32 — avoids a transient
                    # float64 full-resolution copy (source rasters are float64) that would
                    # otherwise ~3x peak RAM per in-flight read thread.
                    arr = src.read(indexes=local_idxs, out_dtype=np.float32)
                arr[arr == nodata]      = 0.0
                arr[~np.isfinite(arr)]  = 0.0
                return fi, arr, out_cols
            except Exception as e:
                log.warning(f"  [{desc}] read failed file {fi}: {e}")
                return fi, None, out_cols

        # Single-threaded write to memmap — concurrent writes to overlapping patches
        # cause data races; read threads are fine, write serialised via main thread.
        _budget = _resolve_ram_budget()
        if n_threads:
            _n_threads = n_threads
        else:
            with rasterio.open(dataset.s2_paths[0]) as _src0:
                _per_file_bytes = _src0.width * _src0.height * _src0.count * 4  # float32
            _n_threads = _ram_safe_read_threads(
                _budget, _per_file_bytes, len(file_extraction), os.cpu_count() or 8,
            )
        log.info(f"  [{desc}] Using {_n_threads} read threads for {len(file_extraction)} files "
                 f"(RAM budget={_budget/1e9:.1f}GB)")
        with ThreadPoolExecutor(max_workers=_n_threads) as pool:
            for fi, arr, out_cols in pool.map(_read_one_file, list(file_extraction.keys())):
                if arr is None:
                    continue
                for ci, out_pos in enumerate(out_cols):
                    band_plane = arr[ci]
                    for pi, (r, c) in enumerate(patches):
                        buf[pi, out_pos, :, :] = band_plane[r:r+ps, c:c+ps]
                del arr

        # Per-band normalisation using norm_mode stats.
        lo_per_ch, hi_per_ch = _per_channel_percentiles(band_indices, *band_percentiles)
        denom = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
        lo_b  = lo_per_ch[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)
        d_b   = denom[np.newaxis, :, np.newaxis, np.newaxis]
        # Re-resolve: read buffers are freed by now, so available memory has shifted.
        _budget = _resolve_ram_budget()
        CHUNK = _ram_safe_chunk(_budget, n_ch, ps)
        log.info(f"  [{desc}] Normalising with norm_mode={norm_mode} … "
                 f"(chunk={CHUNK} patches, RAM budget={_budget/1e9:.1f}GB)")
        for start in range(0, n, CHUNK):
            end   = min(start + CHUNK, n)
            chunk = buf[start:end].astype(np.float32)
            chunk = (chunk - lo_b) / d_b
            if norm_mode != "zscore":
                chunk = np.clip(chunk, 0.0, 1.0)
            buf[start:end] = chunk.astype(np.float16)
        buf.flush()

        masks = [
            torch.from_numpy(
                dataset._remap_lut[np.clip(dataset._cdl[r:r+ps, c:c+ps], 0, 255)].astype(np.int64)
            )
            for r, c in patches
        ]
        self._masks = torch.stack(masks)

        elapsed = time.time() - t0
        log.info(f"  [{desc}] Preloaded in {elapsed:.1f}s — {gb_alloc:.1f} GB float16 on disk")

        if imgs_path:
            del buf  # close write-mode memmap before rename (WSL/NTFS: open handle blocks rename+reopen)
            _buf_path.rename(imgs_path)
            torch.save(self._masks, masks_path)
            log.info(f"  [{desc}] Cached → {imgs_path.name} + {masks_path.name}")
            self._imgs = np.load(str(imgs_path), mmap_mode="r")
        else:
            self._imgs = np.array(buf)   # no cache dir: load into RAM
            del buf
            _buf_path.unlink(missing_ok=True)

    @staticmethod
    def _cache_paths(dataset, cache_dir, norm_mode):
        key = {
            "s2":             sorted(os.path.basename(str(p)) for p in dataset.s2_paths),
            "cdl":            os.path.basename(str(dataset.cdl_path)),
            "ps":             dataset.patch_size,
            "bands":          list(dataset.band_indices) if dataset.band_indices is not None else None,
            "stride":         getattr(dataset, "stride", None),
            "min_valid_frac": getattr(dataset, "min_valid_frac", None),
            "n_patches":      len(dataset.patches),
            "norm":           f"norm_v2_{norm_mode}",  # invalidates pre-norm_mode caches
        }
        h = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
        base = Path(cache_dir) / f"preload_{h}"
        return base.with_suffix(".npy"), base.with_name(base.name + "_masks.pt")

    def __len__(self):
        return len(self._masks)

    def __getitem__(self, idx):
        # np array (memmap or plain) → float32 tensor; .copy() required for mmap slices
        img = torch.tensor(self._imgs[idx], dtype=torch.float32)
        return img, self._masks[idx]


# ── No-preload: on-the-fly normalisation ─────────────────────────────────────

class NormalizedDataset(torch.utils.data.Dataset):
    """Per-band normalisation wrapper — on-the-fly, no disk cache.

    norm_mode: "percentile" (P2/P98, clip [0,1]) | "minmax" (clip [0,1]) | "zscore" (no clip).
    band_percentiles: (lo, hi) per-band stats (semantics depend on norm_mode).
    """

    def __init__(self, dataset, channel_stats=None, band_percentiles=None,
                 norm_mode="percentile"):
        assert band_percentiles is not None, "band_percentiles (lo, hi) required"
        assert norm_mode in NORM_MODES, f"norm_mode must be one of {NORM_MODES}"
        self.dataset   = dataset
        self.norm_mode = norm_mode
        lo_per_ch, hi_per_ch = _per_channel_percentiles(dataset.band_indices, *band_percentiles)
        denom = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
        self.lo    = torch.tensor(lo_per_ch.astype(np.float32)).view(-1, 1, 1)
        self.denom = torch.tensor(denom).view(-1, 1, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img, dtype=torch.float32)
        img = (img.float() - self.lo) / self.denom
        if self.norm_mode != "zscore":
            img = img.clamp(0.0, 1.0)
        return img, mask


def _compute_channel_stats_full(dataset, n_threads=None):
    """Full-pass per-channel mean/std — same math as PreloadedDataset, no memmap.

    Reads each TIF file once (parallel), extracts patches in chunks, accumulates
    ch_sums / ch_sums2 in float64. Peak RAM = one TIF in memory + ~64 MB per chunk.
    Identical stats to PreloadedDataset; avoids the disk-backed memmap allocation
    that causes OOM at high channel counts.
    """
    n   = len(dataset)
    ps  = dataset.patch_size
    band_indices  = dataset.band_indices
    n_ch_per_file = [src.count for src in dataset._s2_srcs]
    ch_offsets    = np.cumsum([0] + n_ch_per_file).tolist()
    n_ch          = len(band_indices) if band_indices is not None else ch_offsets[-1]
    patches       = dataset.patches
    nodata        = dataset.nodata

    # Same file_extraction map as PreloadedDataset
    file_extraction: dict = {}
    targets = band_indices if band_indices is not None else list(range(ch_offsets[-1]))
    for out_pos, gi in enumerate(targets):
        for fi in range(len(n_ch_per_file)):
            if ch_offsets[fi] <= gi < ch_offsets[fi + 1]:
                file_extraction.setdefault(fi, []).append((out_pos, gi - ch_offsets[fi] + 1))
                break

    ch_sums  = np.zeros(n_ch, dtype=np.float64)
    ch_sums2 = np.zeros(n_ch, dtype=np.float64)
    ch_cnt   = n * ps * ps

    log.info(f"  [stats] Full-pass channel stats: {n} patches, {len(file_extraction)} TIF files …")

    def _read_one_file(fi):
        extractions = file_extraction[fi]
        local_idxs  = [e[1] for e in extractions]
        out_cols    = [e[0] for e in extractions]
        try:
            with rasterio.open(dataset.s2_paths[fi]) as src:
                arr = src.read(indexes=local_idxs).astype(np.float32)
            arr[arr == nodata]     = 0.0
            arr[~np.isfinite(arr)] = 0.0
            return fi, arr, out_cols
        except Exception as e:
            log.warning(f"  [stats] read failed file {fi}: {e}")
            return fi, None, out_cols

    PCHUNK = 64  # patches per accumulation chunk; peak RAM = PCHUNK × ps² × float64 ≈ 67 MB
    _n_threads = n_threads or min(len(file_extraction), os.cpu_count() or 8)
    with ThreadPoolExecutor(max_workers=_n_threads) as pool:
        for fi, arr, out_cols in pool.map(_read_one_file, list(file_extraction.keys())):
            if arr is None:
                continue
            for ci, out_pos in enumerate(out_cols):
                plane = arr[ci]  # (H, W) float32
                for start in range(0, len(patches), PCHUNK):
                    pslice = patches[start:start + PCHUNK]
                    batch  = np.stack([plane[r:r + ps, c:c + ps] for r, c in pslice])
                    flat   = batch.astype(np.float64).ravel()
                    ch_sums[out_pos]  += flat.sum()
                    ch_sums2[out_pos] += (flat * flat).sum()
            del arr

    means = (ch_sums / ch_cnt).astype(np.float32)
    stds  = np.sqrt(np.maximum(ch_sums2 / ch_cnt - means.astype(np.float64) ** 2, 0)).astype(np.float32)
    stds  = np.where(stds < 1.0, 1.0, stds)
    log.info(f"  [stats]  mean [{means.min():.1f}, {means.max():.1f}]"
             f"  std [{stds.min():.1f}, {stds.max():.1f}]")
    return means, stds
