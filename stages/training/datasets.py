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

# Fraction of the resolved RAM budget reserved for concurrent per-shard file-read
# buffers vs. a shard's own normalisation working copy (each runs at a different
# point per shard, never overlapping, but both must fit individually under the
# same ceiling).
_READ_BUDGET_FRAC  = 0.5
_CHUNK_BUDGET_FRAC = 0.5   # also governs shard size — see _ram_safe_shard_size
# Only claim this fraction of currently-available memory when auto-detecting —
# leaves headroom for the OS page cache, mmap pages of the output buffer, and
# whatever else the user is running. Kept conservative because some environments
# (WSL2 VMs in particular) hard-kill on overcommit instead of swapping gracefully —
# there is no cushion to lean on if the estimate runs a little hot.
_AVAIL_SAFETY_FRAC = 0.5
_MIN_BUDGET_BYTES  = 2e9   # floor: below this, thread/chunk sizing degenerates
# Decoded-array byte estimates below are the theoretical minimum (raw float32
# buffer); actual peak runs higher — rasterio block-decompression scratch space
# for the read path, and one extra transient buffer during the astype(float16)
# cast at the end of each normalisation chunk. Both paths are inflated by this
# factor rather than assumed exact, since a hard OOM-kill has no recovery path.
_SAFETY_MULT = 1.5


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
    per_file_bytes is inflated by _SAFETY_MULT to cover rasterio's internal
    decode/decompression scratch space, which isn't visible from array size alone.
    """
    by_ram = max(1, int(budget_bytes * _READ_BUDGET_FRAC // max(per_file_bytes * _SAFETY_MULT, 1)))
    return max(1, min(n_files, cpu_cap, by_ram))


def _flush_and_evict(buf, path) -> None:
    """Force writeback of the memmap's dirty pages and tell the OS it can drop them
    from the page cache.

    A large output memmap (10s of GB) accumulates resident dirty/cached pages over
    the life of the fill+normalise loop even though nothing in the *thread/chunk*
    budgeting holds it in RAM directly — measured peak RSS ran well above the
    computed thread+chunk budget on a run where this was the only unaccounted
    factor. This is especially dangerous on WSL2, where dirty-page writeback over
    some mount types lags and there is little to no swap cushion — a hot estimate
    ends in a hard SIGKILL, not graceful degradation. No-op on platforms without
    posix_fadvise (e.g. macOS) — best-effort only, never raises.
    """
    buf.flush()
    if not hasattr(os, "posix_fadvise"):
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)  # length=0 means "to EOF"
        finally:
            os.close(fd)
    except OSError:
        pass


def _ram_safe_shard_size(budget_bytes: float, n_ch: int, ps: int) -> int:
    """Cap shard size (patches/shard) so a shard's float32 working copy fits budget.

    Inflated by _SAFETY_MULT: even with in-place arithmetic, the shard's float32
    working copy and its float16 cast briefly coexist at the end of normalisation.
    """
    per_patch_bytes = n_ch * ps * ps * 4 * _SAFETY_MULT  # float32 working copy + cast overlap
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

class _ShardedMemmap:
    """Read-only view over N separately memory-mapped .npy shard files, indexed
    as one contiguous array along axis 0.

    Each shard is mmap'd lazily on first access to it, so opening a
    _ShardedMemmap costs nothing up front. The point of splitting into shards
    (vs. one big memmap) is bounding *build-time* peak RAM/disk-cache pressure —
    see the shard-per-row-band fill loop in PreloadedDataset.__init__. At read
    time this behaves like a single memmap: the OS still only pages in what's
    actually touched.
    """

    def __init__(self, shard_paths, shard_sizes):
        self._paths   = list(shard_paths)
        self._offsets = np.cumsum([0] + list(shard_sizes)).tolist()
        self._mmaps   = [None] * len(self._paths)

    def __len__(self):
        return self._offsets[-1]

    def __getitem__(self, idx):
        shard_idx = int(np.searchsorted(self._offsets, idx, side="right") - 1)
        local_idx = idx - self._offsets[shard_idx]
        if self._mmaps[shard_idx] is None:
            self._mmaps[shard_idx] = np.load(str(self._paths[shard_idx]), mmap_mode="r")
        return self._mmaps[shard_idx][local_idx]

    def total_bytes(self):
        return sum(p.stat().st_size for p in self._paths)


class PreloadedDataset(torch.utils.data.Dataset):
    """Builds a persistent disk cache of all patches, split into RAM-budget-sized
    shards; loads imgs via per-shard memory-map.

    Patches are generated in row-major order (see RasterPatchDataset), so a
    contiguous *index* range is also a contiguous *spatial* row-band — each
    shard is built from a single windowed read per source file (covering just
    that row-band) rather than reading the whole raster, so splitting into
    shards does not multiply total I/O.

    Building one shard at a time (read → normalise → flush → close → next)
    bounds peak memory to O(one shard) regardless of total dataset size —
    a single big memmap's dirty/cached pages otherwise accumulate as resident
    memory over the whole build (see _flush_and_evict), which is what caused
    hard OOM-kills on memory-constrained machines (WSL2 in particular — little
    to no swap cushion, a hot estimate ends in SIGKILL, not degradation).

    Cache key covers s2_paths/cdl_path/bands/patch_size. A manifest file is
    written last (after every shard + the masks file succeed) so its mere
    presence is proof the cache is complete — safe to resume/rebuild after an
    interrupted build without special-casing partial state.

    Imgs stored as float16 .npy shards → loaded with mmap_mode='r' so the OS
    pages in only what each minibatch needs. Peak RAM = model + batch, not
    full dataset. Masks stored as int64 .pt (typically <1 GB, always in RAM).
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
        base_path, masks_path = self._cache_paths(dataset, cache_dir, norm_mode) if cache_dir else (None, None)
        manifest_path = base_path.with_name(base_path.name + "_manifest.json") if base_path else None

        if manifest_path and manifest_path.exists() and masks_path and masks_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            shard_paths = [base_path.with_name(f"{base_path.name}_shard{i:04d}.npy")
                           for i in range(manifest["n_shards"])]
            if all(p.exists() for p in shard_paths):
                log.info(f"  [{desc}] Cache hit → mmap {len(shard_paths)} shard(s)")
                t0 = time.time()
                self._imgs  = _ShardedMemmap(shard_paths, manifest["shard_sizes"])
                self._masks = torch.load(masks_path, map_location="cpu", weights_only=True)
                gb_disk = self._imgs.total_bytes() / 1e9
                log.info(f"  [{desc}] {len(shard_paths)} shard(s) ready in {time.time()-t0:.1f}s "
                         f"({gb_disk:.2f} GB on disk)")
                return
            log.warning(f"  [{desc}] Manifest found but shard file(s) missing — rebuilding")

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

        patches = dataset.patches   # row-major (see class docstring) — shards stay spatially contiguous
        nodata  = dataset.nodata

        lo_per_ch, hi_per_ch = _per_channel_percentiles(band_indices, *band_percentiles)
        denom = np.maximum(hi_per_ch - lo_per_ch, 1.0).astype(np.float32)
        lo_b  = lo_per_ch[np.newaxis, :, np.newaxis, np.newaxis].astype(np.float32)
        d_b   = denom[np.newaxis, :, np.newaxis, np.newaxis]

        _budget = _resolve_ram_budget()
        shard_size = _ram_safe_shard_size(_budget, n_ch, ps)
        shard_ranges = [(s, min(s + shard_size, n)) for s in range(0, n, shard_size)]
        n_shards = len(shard_ranges)
        gb_total = n * n_ch * ps * ps * 2 / 1e9   # float16
        log.info(f"  [{desc}] {n}×{n_ch}×{ps}×{ps} float16 = {gb_total:.1f} GB total, "
                 f"{n_shards} shard(s) of <= {shard_size} patches "
                 f"(RAM budget={_budget/1e9:.1f}GB)")

        def _tmp_shard_path(i):
            return (base_path.with_name(f"{base_path.name}_shard{i:04d}.tmp.npy") if base_path
                    else Path(run_state.PRELOAD_CACHE_DIR) / f"_tmp_{os.getpid()}_shard{i:04d}.npy")

        final_shard_paths: list = []
        shard_sizes: list = []
        for shard_idx, (s0, s1) in enumerate(shard_ranges):
            shard_patches = patches[s0:s1]
            shard_n = s1 - s0
            r_min = min(r for r, c in shard_patches)
            r_max = max(r for r, c in shard_patches) + ps
            win_h = r_max - r_min

            shard_tmp_path = _tmp_shard_path(shard_idx)
            shard_tmp_path.parent.mkdir(parents=True, exist_ok=True)
            shard_buf = np.lib.format.open_memmap(
                str(shard_tmp_path), mode="w+", dtype=np.float16, shape=(shard_n, n_ch, ps, ps)
            )

            def _read_one_file_window(fi, _r_min=r_min, _win_h=win_h):
                extractions = file_extraction[fi]
                local_idxs  = [e[1] for e in extractions]
                out_cols    = [e[0] for e in extractions]
                try:
                    with rasterio.open(dataset.s2_paths[fi]) as src:
                        # Windowed to this shard's row-band only — out_dtype=float32
                        # decodes directly, avoiding a transient float64 copy
                        # (source rasters are float64).
                        win = rasterio.windows.Window(0, _r_min, src.width, _win_h)
                        arr = src.read(indexes=local_idxs, window=win, out_dtype=np.float32)
                    arr[arr == nodata]     = 0.0
                    arr[~np.isfinite(arr)] = 0.0
                    return fi, arr, out_cols
                except Exception as e:
                    log.warning(f"  [{desc}] shard {shard_idx}: read failed file {fi}: {e}")
                    return fi, None, out_cols

            if n_threads:
                _n_threads = n_threads
            else:
                with rasterio.open(dataset.s2_paths[0]) as _src0:
                    _per_file_bytes = win_h * _src0.width * _src0.count * 4   # float32, windowed
                _n_threads = _ram_safe_read_threads(
                    _budget, _per_file_bytes, len(file_extraction), os.cpu_count() or 8,
                )
            if shard_idx == 0:
                log.info(f"  [{desc}] Using {_n_threads} read threads per shard "
                         f"(RAM budget={_budget/1e9:.1f}GB)")

            with ThreadPoolExecutor(max_workers=_n_threads) as pool:
                for fi, arr, out_cols in pool.map(_read_one_file_window, list(file_extraction.keys())):
                    if arr is None:
                        continue
                    for ci, out_pos in enumerate(out_cols):
                        band_plane = arr[ci]   # (win_h, width)
                        for pi, (r, c) in enumerate(shard_patches):
                            lr = r - r_min
                            shard_buf[pi, out_pos, :, :] = band_plane[lr:lr+ps, c:c+ps]
                    del arr

            # Normalise this shard in-place — already sized to fit the RAM budget,
            # so (unlike the old single-buffer design) no further internal chunking
            # is needed here. In-place ops (`-=`, `/=`, `out=`) avoid the extra
            # full-size float32 temporaries that non-in-place ops would allocate.
            chunk = shard_buf[:].astype(np.float32)
            chunk -= lo_b
            chunk /= d_b
            if norm_mode != "zscore":
                np.clip(chunk, 0.0, 1.0, out=chunk)
            shard_buf[:] = chunk.astype(np.float16)
            del chunk

            shard_buf.flush()
            _flush_and_evict(shard_buf, shard_tmp_path)
            del shard_buf   # close write-mode memmap before rename (WSL/NTFS: open handle blocks rename)

            if base_path:
                final_path = base_path.with_name(f"{base_path.name}_shard{shard_idx:04d}.npy")
                shard_tmp_path.rename(final_path)
            else:
                final_path = shard_tmp_path
            final_shard_paths.append(final_path)
            shard_sizes.append(shard_n)
            log.info(f"  [{desc}] shard {shard_idx+1}/{n_shards} done "
                     f"({shard_n} patches, rows [{r_min}:{r_max}])")

        masks = [
            torch.from_numpy(
                dataset._remap_lut[np.clip(dataset._cdl[r:r+ps, c:c+ps], 0, 255)].astype(np.int64)
            )
            for r, c in patches
        ]
        self._masks = torch.stack(masks)

        elapsed = time.time() - t0
        log.info(f"  [{desc}] Preloaded in {elapsed:.1f}s — {gb_total:.1f} GB float16 "
                 f"across {n_shards} shard(s)")

        if base_path:
            torch.save(self._masks, masks_path)
            manifest_tmp = manifest_path.with_suffix(".tmp.json")
            with open(manifest_tmp, "w") as f:
                json.dump({"n_shards": n_shards, "shard_sizes": shard_sizes,
                           "n_ch": n_ch, "ps": ps}, f)
            os.replace(manifest_tmp, manifest_path)   # atomic — presence = "build complete"
            log.info(f"  [{desc}] Cached → {n_shards} shard(s) + {masks_path.name} + {manifest_path.name}")
            self._imgs = _ShardedMemmap(final_shard_paths, shard_sizes)
        else:
            # No cache dir: materialise fully in RAM (matches pre-sharding behaviour
            # for this — in practice unused — code path; every real caller passes
            # cache_dir=run_state.PRELOAD_CACHE_DIR).
            self._imgs = np.concatenate([np.load(str(p)) for p in final_shard_paths])
            for p in final_shard_paths:
                p.unlink(missing_ok=True)

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
        return base, base.with_name(base.name + "_masks.pt")

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
