"""Check if histogram matching is needed between train and spatial test areas.

Compares per-band pixel distributions across train/test_a/test_b for a
representative S2 date. Outputs:
  - Console: mean/std shift per band per area
  - figures/histogram_check/  histograms + KL-divergence heatmap

Usage:
    python stages/check_histogram.py
    python stages/check_histogram.py --data-dir /Volumes/T7/.../data/processed
    python stages/check_histogram.py --date 20240730   # pick specific date
    python stages/check_histogram.py --n-sample 100000
"""

import argparse
import sys
from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_TRAIN_DIR, PROCESSED_DIR, S2_BAND_NAMES, S2_NODATA, FIGURES_DIR,
    SPATIAL_TEST_AREAS,
)

SHIFT_WARN_MEAN = 0.15   # >15% relative mean shift → warn (histogram matching likely needed)
SHIFT_WARN_STD  = 0.20   # >20% relative std shift  → warn


# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_date(path: Path) -> str | None:
    m = re.search(r"_(\d{4}_\d{2}_\d{2})(_processed)?\.tif$", path.name)
    if m:
        return m.group(1).replace("_", "")
    m = re.search(r"_(\d{8})[_\.]", path.name)
    return m.group(1) if m else None


def find_matching_file(s2_dir: Path, date: str) -> Path | None:
    """Find S2 file in dir matching given date string (YYYYMMDD)."""
    for f in sorted(s2_dir.glob("*.tif")):
        if f.name.startswith("._"):
            continue
        if _parse_date(f) == date:
            return f
    return None


def pick_reference_date(s2_dir: Path) -> str | None:
    """Pick date with most files across all areas; falls back to July dates."""
    dates = [_parse_date(f) for f in sorted(s2_dir.glob("*.tif")) if not f.name.startswith("._")]
    dates = [d for d in dates if d]
    if not dates:
        return None
    # prefer peak-season July date
    july = [d for d in dates if d[4:6] == "07"]
    return july[len(july)//2] if july else dates[len(dates)//2]


def sample_pixels(tif_path: Path, n: int, seed: int = 42) -> np.ndarray | None:
    """Sample n pixels from all bands. Returns (n, B) float32 array or None."""
    try:
        with rasterio.open(tif_path) as src:
            arr = src.read().astype(np.float32)   # (B, H, W)
    except Exception as e:
        print(f"  [ERROR] Cannot read {tif_path.name}: {e}")
        return None

    B, H, W = arr.shape
    arr_2d = arr.reshape(B, -1).T   # (H*W, B)
    nodata_mask = np.any(arr_2d == S2_NODATA, axis=1) | np.any(~np.isfinite(arr_2d), axis=1)
    valid = arr_2d[~nodata_mask]
    if len(valid) == 0:
        print(f"  [WARN] No valid pixels in {tif_path.name}")
        return None

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(valid), min(n, len(valid)), replace=False)
    return valid[idx]


# ── Statistics ────────────────────────────────────────────────────────────────

def area_stats(samples: np.ndarray, band_names: list[str]) -> pd.DataFrame:
    rows = []
    for i, band in enumerate(band_names):
        vals = samples[:, i]
        rows.append({
            "band":  band,
            "mean":  float(np.mean(vals)),
            "std":   float(np.std(vals)),
            "p5":    float(np.percentile(vals, 5)),
            "p95":   float(np.percentile(vals, 95)),
        })
    return pd.DataFrame(rows).set_index("band")


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 100) -> float:
    """Symmetric KL divergence between two 1-D arrays."""
    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())
    edges = np.linspace(lo, hi, bins + 1)
    ph, _ = np.histogram(p, bins=edges, density=True)
    qh, _ = np.histogram(q, bins=edges, density=True)
    eps = 1e-10
    ph = ph + eps;  qh = qh + eps
    ph /= ph.sum(); qh /= qh.sum()
    return float(0.5 * (np.sum(ph * np.log(ph / qh)) + np.sum(qh * np.log(qh / ph))))


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_histograms(area_samples: dict[str, np.ndarray], band_names: list[str], out_dir: Path):
    """One PNG per band — overlapping histograms for all areas."""
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = ["steelblue", "tomato", "seagreen", "orange"]
    for i, band in enumerate(band_names):
        fig, ax = plt.subplots(figsize=(7, 3))
        for j, (area_name, samples) in enumerate(area_samples.items()):
            if samples is None:
                continue
            vals = samples[:, i]
            ax.hist(vals, bins=120, alpha=0.55, color=colors[j % len(colors)],
                    label=area_name, density=True)
        ax.set(title=f"Band {band} — pixel distribution", xlabel="Reflectance", ylabel="Density")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"hist_{band}.png", dpi=120)
        plt.close()
    print(f"Histograms saved → {out_dir}")


def plot_kl_heatmap(kl_df: pd.DataFrame, out_dir: Path):
    """Heatmap of symmetric KL divergence: area-pairs × bands."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6, len(kl_df.columns) * 0.7), max(3, len(kl_df) * 0.5)))
    data = kl_df.values.astype(float)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(kl_df.columns))); ax.set_xticklabels(kl_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(kl_df.index)));   ax.set_yticklabels(kl_df.index)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            ax.text(c, r, f"{data[r,c]:.3f}", ha="center", va="center", fontsize=7,
                    color="white" if data[r,c] > data.max()*0.6 else "black")
    plt.colorbar(im, ax=ax, label="Symmetric KL divergence")
    ax.set_title("Band distribution shift: train vs test areas\n(higher = more different)")
    plt.tight_layout()
    path = out_dir / "kl_divergence_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"KL heatmap saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check histogram shift between train and test areas")
    parser.add_argument("--data-dir",  default=None, help="Override processed data root")
    parser.add_argument("--date",      default=None, help="S2 date YYYYMMDD (auto-pick if omitted)")
    parser.add_argument("--n-sample",  type=int, default=50_000, help="Pixels to sample per area")
    parser.add_argument("--bands",     nargs="+", default=None, help="Subset of bands to check (default: all)")
    args = parser.parse_args()

    data_root = Path(args.data_dir) if args.data_dir else PROCESSED_DIR
    train_dir = data_root / "s2" / "2024"
    fig_dir   = (Path(args.data_dir) if args.data_dir else FIGURES_DIR) / "histogram_check"

    test_areas = [
        {"name": a["name"], "s2_dir": data_root / "s2" / a["name"]}
        for a in SPATIAL_TEST_AREAS
    ]

    band_names = args.bands if args.bands else S2_BAND_NAMES

    # ── Pick reference date ────────────────────────────────────────────────
    date = args.date or pick_reference_date(train_dir)
    if not date:
        print(f"[ERROR] No S2 files found in {train_dir}")
        sys.exit(1)
    print(f"Reference date: {date}")

    # ── Load samples per area ─────────────────────────────────────────────
    areas = [{"name": "train", "s2_dir": train_dir}] + test_areas
    area_samples: dict[str, np.ndarray | None] = {}

    for area in areas:
        s2_dir = Path(area["s2_dir"])
        tif    = find_matching_file(s2_dir, date)
        if tif is None:
            print(f"  [{area['name']}] No file for date {date} in {s2_dir} — skipping")
            area_samples[area["name"]] = None
            continue
        print(f"  [{area['name']}] Loading {tif.name} ...")
        area_samples[area["name"]] = sample_pixels(tif, args.n_sample)

    # ── Per-band statistics ───────────────────────────────────────────────
    train_samples = area_samples.get("train")
    if train_samples is None:
        print("[ERROR] No train samples — cannot compare")
        sys.exit(1)

    band_idx = {b: i for i, b in enumerate(S2_BAND_NAMES)}
    selected_idx = [band_idx[b] for b in band_names if b in band_idx]

    train_stats = area_stats(train_samples, S2_BAND_NAMES)

    print(f"\n{'Area':<14} {'Band':<6} {'Mean':>8} {'Std':>8} {'ΔMean%':>8} {'ΔStd%':>8}  Status")
    print("─" * 68)

    needs_matching = False
    for area_name, samples in area_samples.items():
        if area_name == "train" or samples is None:
            continue
        area_st = area_stats(samples, S2_BAND_NAMES)
        for band in band_names:
            if band not in train_stats.index:
                continue
            t_mean = train_stats.loc[band, "mean"]
            t_std  = train_stats.loc[band, "std"]
            a_mean = area_st.loc[band, "mean"]
            a_std  = area_st.loc[band, "std"]
            d_mean = abs(a_mean - t_mean) / (abs(t_mean) + 1e-9)
            d_std  = abs(a_std  - t_std)  / (abs(t_std)  + 1e-9)
            warn   = d_mean > SHIFT_WARN_MEAN or d_std > SHIFT_WARN_STD
            if warn:
                needs_matching = True
            status = "⚠ SHIFT" if warn else "ok"
            print(f"{area_name:<14} {band:<6} {a_mean:>8.1f} {a_std:>8.1f} "
                  f"{d_mean*100:>7.1f}% {d_std*100:>7.1f}%  {status}")
        print()

    # ── KL divergence ─────────────────────────────────────────────────────
    kl_rows = {}
    for area_name, samples in area_samples.items():
        if area_name == "train" or samples is None:
            continue
        pair = f"train vs {area_name}"
        kl_rows[pair] = {}
        for band in band_names:
            if band not in band_idx:
                continue
            bi = band_idx[band]
            kl_rows[pair][band] = kl_divergence(train_samples[:, bi], samples[:, bi])

    kl_df = pd.DataFrame(kl_rows).T
    print("\nSymmetric KL divergence (train vs test areas):")
    print(kl_df.round(4).to_string())

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n" + "═" * 68)
    if needs_matching:
        print("VERDICT: Distribution shift detected — histogram matching RECOMMENDED")
        print("         Consider: skimage.exposure.match_histograms per band")
    else:
        print("VERDICT: Distributions close — histogram matching likely NOT needed")
    print("═" * 68)

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_histograms(area_samples, band_names, fig_dir)
    if not kl_df.empty:
        plot_kl_heatmap(kl_df, fig_dir)


if __name__ == "__main__":
    main()
