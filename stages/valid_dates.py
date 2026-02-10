"""Shared per-date scene-usability filter.

Drops S2 acquisition TIFs whose valid-pixel fraction (non-nodata, finite) falls
below ``S2_MIN_VALID_FRAC``. Used by the training pipeline AND the standalone
band-selection scripts so both operate on an identical set of valid dates.

The valid-fraction estimate uses the same sampled 3x3 grid of 512px windows over
bands {1, mid, last} as the training-time validation, so results match exactly.
"""

import json
import logging
from pathlib import Path

import numpy as np
import rasterio

from crop_mapping_pipeline.config import S2_NODATA, S2_MIN_VALID_FRAC

log = logging.getLogger(__name__)

_WIN = 512


def valid_fraction(path: str) -> float:
    """Fraction of non-nodata, finite pixels (sampled). Returns 0.0 if unreadable."""
    try:
        with rasterio.open(path) as src:
            h, w, nb = src.height, src.width, src.count
            sz = min(_WIN, w // 4, h // 4)
            valid_px = total_px = 0
            for band in sorted({1, nb // 2, nb}):
                for gy in range(3):
                    for gx in range(3):
                        ox = max(0, min(int((gx + 0.5) * w / 3) - sz // 2, w - sz))
                        oy = max(0, min(int((gy + 0.5) * h / 3) - sz // 2, h - sz))
                        win  = rasterio.windows.Window(ox, oy, sz, sz)
                        data = src.read(band, window=win).astype(np.float32)
                        ok   = (data != S2_NODATA) & np.isfinite(data)
                        valid_px += int(ok.sum())
                        total_px += int(ok.size)
        return valid_px / total_px if total_px else 0.0
    except Exception as e:
        log.warning(f"valid_fraction: unreadable {Path(path).name} ({e}) → 0.0")
        return 0.0


def filter_valid_s2_dates(s2_paths, min_valid_frac: float = S2_MIN_VALID_FRAC,
                          cache_dir=None):
    """Return (valid_paths, dropped) where dropped = [(name, frac), ...].

    Caches per-file fractions in ``<cache_dir>/s2_validity_cache.json``, keyed by
    file set + threshold; threshold changes re-filter from cached fractions without
    recompute.
    """
    s2_paths   = sorted(s2_paths)
    key        = [Path(p).name for p in s2_paths]
    cache_path = Path(cache_dir) / "s2_validity_cache.json" if cache_dir else None

    fracs = None
    if cache_path and cache_path.exists():
        try:
            c = json.load(open(cache_path))
            if c.get("files_key") == key:
                fracs = c["fracs"]
        except Exception:
            fracs = None

    if fracs is None:
        fracs = {Path(p).name: float(valid_fraction(p)) for p in s2_paths}
        if cache_path:
            try:
                json.dump({"files_key": key, "fracs": fracs}, open(cache_path, "w"))
            except Exception as e:
                log.warning(f"could not write {cache_path.name}: {e}")

    valid   = [p for p in s2_paths if fracs.get(Path(p).name, 0.0) >= min_valid_frac]
    dropped = sorted([(n, f) for n, f in fracs.items() if f < min_valid_frac])
    if dropped:
        log.warning(f"Excluding {len(dropped)} date(s) below {min_valid_frac*100:.0f}% valid pixels:")
        for n, f in dropped:
            log.warning(f"  {n}  ({f*100:.1f}% valid)")
    return valid, dropped
