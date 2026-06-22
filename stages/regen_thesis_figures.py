"""Regenerate data exploration figures for thesis_v6 from 2024 S2 + CDL data.

Reads raw S2 tifs and processed CDL from T7 (or override paths via env vars),
produces figures at thesis_v6/figures/:

  s2_rgb_temporal.png        — true color (B4/B3/B2) at 6 phenological dates
  s2_band_grid.png           — all 11 bands at peak date (29-Jul)
  s2_ndvi_temporal.png       — mean NDVI profile across 25 dates
  s2_ndvi_per_class.png      — NDVI per crop class across 25 dates
  s2_spectral_profile.png    — per-band mean reflectance at 3 dates
  s2_data_coverage.png       — valid-pixel % per date
  cdl_label_map.png          — CDL 2024 map (8 crops + background)
  cdl_class_distribution_area.png — top-20 CDL class area bar chart

Usage:
  python stages/regen_thesis_figures.py
  python stages/regen_thesis_figures.py --only ndvi rgb
  python stages/regen_thesis_figures.py --s2-dir /custom/path --cdl /custom.tif
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT.parent))

from crop_mapping_pipeline.config import (
    S2_BAND_NAMES, KEEP_CLASSES, CDL_CLASS_NAMES,
)
from crop_mapping_pipeline.utils.constants import USDA_CDL_COLORS, USDA_CDL_NAMES

# ── default paths ─────────────────────────────────────────────────────────────
DEFAULT_S2_DIR  = Path("/Volumes/T7/research-crop-mapping-geoai/data/raw_v6/s2/train")
DEFAULT_CDL     = Path("/Volumes/T7/research-crop-mapping-geoai/data/raw_v6/cdl/cdl_train.tif")
FIGS_DIR        = _ROOT.parent / "documents" / "thesis_v6" / "figures"

# 6 phenological dates for RGB temporal grid
RGB_DATES = ["2024_01_16", "2024_03_16", "2024_05_15", "2024_07_29", "2024_09_27", "2024_10_27"]
# Peak NDVI date for band grid
PEAK_DATE = "2024_07_29"
# 3 dates for spectral profile (winter / spring / summer)
SPECTRAL_DATES = {"Januari": "2024_01_16", "Mei": "2024_05_15", "Juli": "2024_07_29"}


def list_s2_files(s2_dir: Path) -> list[Path]:
    """Return sorted list of *_processed.tif or *.tif S2 files in s2_dir."""
    tifs = sorted([p for p in s2_dir.glob("*.tif") if not p.name.startswith("._")])
    return tifs


def date_from_path(p: Path) -> str:
    """Extract YYYY_MM_DD from filename like S2H_2024_2024_07_29.tif."""
    parts = p.stem.split("_")
    if len(parts) >= 5:
        return "_".join(parts[-3:])
    return p.stem


def downsample(arr: np.ndarray, factor: int = 8) -> np.ndarray:
    """Subsample 2D array by factor."""
    return arr[::factor, ::factor]


def read_band(path: Path, band_idx: int, ds_factor: int = 8) -> np.ndarray:
    """Read one band, downsampled."""
    with rasterio.open(path) as src:
        arr = src.read(band_idx, out_shape=(src.height // ds_factor, src.width // ds_factor))
    arr = arr.astype(np.float32)
    arr[arr == src.nodata] = np.nan if src.nodata is not None else np.nan
    arr[arr <= 0] = np.nan
    return arr


def normalize_for_display(arr: np.ndarray, pmin: float = 2, pmax: float = 98) -> np.ndarray:
    """Stretch array to [0,1] using percentile clipping."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo, hi = np.percentile(valid, [pmin, pmax])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)


# ── individual figure generators ──────────────────────────────────────────────

def fig_rgb_temporal(s2_files: list[Path], out_path: Path):
    """6 RGB panels at phenological dates."""
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    titles = ["Dorman (Jan)", "Persiapan (Mar)", "Pertumbuhan (Mei)",
              "Puncak (Jul)", "Pematangan (Sep)", "Panen (Okt)"]
    for i, (date_str, title) in enumerate(zip(RGB_DATES, titles)):
        match = [p for p in s2_files if date_str in p.name]
        if not match:
            print(f"  WARN: no file for {date_str}")
            axes[i].axis("off")
            continue
        path = match[0]
        b4 = read_band(path, S2_BAND_NAMES.index("B4") + 1)
        b3 = read_band(path, S2_BAND_NAMES.index("B3") + 1)
        b2 = read_band(path, S2_BAND_NAMES.index("B2") + 1)
        rgb = np.stack([normalize_for_display(b4),
                        normalize_for_display(b3),
                        normalize_for_display(b2)], axis=-1)
        axes[i].imshow(rgb)
        axes[i].set_title(f"{title}\n{date_str.replace('_', '-')}", fontsize=10)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_band_grid(s2_files: list[Path], out_path: Path):
    """11 bands at peak date in a 3x4 grid."""
    match = [p for p in s2_files if PEAK_DATE in p.name]
    if not match:
        print(f"  WARN: no peak date file ({PEAK_DATE})")
        return
    path = match[0]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for i, bname in enumerate(S2_BAND_NAMES):
        arr = read_band(path, i + 1)
        axes[i].imshow(normalize_for_display(arr), cmap="gray")
        axes[i].set_title(bname, fontsize=11)
        axes[i].axis("off")
    axes[-1].axis("off")  # 12th panel empty (11 bands)
    fig.suptitle(f"Komposit 11 band Sentinel-2 ({PEAK_DATE.replace('_', '-')})",
                 fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_ndvi_temporal(s2_files: list[Path], out_path: Path):
    """Mean NDVI across all 25 dates."""
    means = []
    dates = []
    for p in s2_files:
        nir = read_band(p, S2_BAND_NAMES.index("B8") + 1)
        red = read_band(p, S2_BAND_NAMES.index("B4") + 1)
        ndvi = (nir - red) / (nir + red + 1e-6)
        means.append(np.nanmean(ndvi))
        dates.append(date_from_path(p))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(len(means)), means, marker="o", color="#2E7D32")
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([d.replace("2024_", "") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("NDVI rata-rata")
    ax.set_xlabel("Tanggal Akuisisi (2024)")
    ax.set_title("Profil NDVI Temporal Sacramento Valley 2024 (25 tanggal)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def _load_cdl(cdl_path: Path, ds_factor: int = 8) -> np.ndarray:
    with rasterio.open(cdl_path) as src:
        arr = src.read(1, out_shape=(src.height // ds_factor, src.width // ds_factor))
    return arr


def fig_ndvi_per_class(s2_files: list[Path], cdl_path: Path, out_path: Path):
    """Mean NDVI per crop class across 25 dates."""
    cdl = _load_cdl(cdl_path, ds_factor=8)
    dates = [date_from_path(p) for p in s2_files]
    per_class = {cid: [] for cid in KEEP_CLASSES}

    for p in s2_files:
        nir = read_band(p, S2_BAND_NAMES.index("B8") + 1)
        red = read_band(p, S2_BAND_NAMES.index("B4") + 1)
        ndvi = (nir - red) / (nir + red + 1e-6)
        h = min(cdl.shape[0], ndvi.shape[0])
        w = min(cdl.shape[1], ndvi.shape[1])
        for cid in KEEP_CLASSES:
            mask = (cdl[:h, :w] == cid)
            if mask.sum() == 0:
                per_class[cid].append(np.nan)
            else:
                per_class[cid].append(np.nanmean(ndvi[:h, :w][mask]))

    fig, ax = plt.subplots(figsize=(12, 5))
    for cid in KEEP_CLASSES:
        name = CDL_CLASS_NAMES.get(cid, str(cid))
        color = USDA_CDL_COLORS.get(cid, None)
        ax.plot(range(len(dates)), per_class[cid], marker="o", label=name, color=color, lw=1.5)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.replace("2024_", "") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("NDVI rata-rata")
    ax.set_xlabel("Tanggal Akuisisi (2024)")
    ax.set_title("Profil NDVI per Kelas Tanaman (2024)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_spectral_profile(s2_files: list[Path], out_path: Path):
    """Per-band mean reflectance at 3 dates."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"Januari": "#1E88E5", "Mei": "#F4511E", "Juli": "#43A047"}
    for season, date_str in SPECTRAL_DATES.items():
        match = [p for p in s2_files if date_str in p.name]
        if not match:
            continue
        path = match[0]
        means = []
        for i in range(len(S2_BAND_NAMES)):
            arr = read_band(path, i + 1)
            means.append(np.nanmean(arr))
        ax.plot(S2_BAND_NAMES, means, marker="o", label=f"{season} ({date_str.replace('_','-')})",
                color=colors[season], lw=2)
    ax.set_xlabel("Band Sentinel-2")
    ax.set_ylabel("Reflektansi rata-rata")
    ax.set_title("Profil Spektral per Band pada Tiga Tanggal Representatif (2024)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_data_coverage(s2_files: list[Path], out_path: Path):
    """Valid pixel % per date."""
    coverages = []
    dates = []
    for p in s2_files:
        with rasterio.open(p) as src:
            arr = src.read(1, out_shape=(src.height // 8, src.width // 8))
            nodata = src.nodata if src.nodata is not None else -9999
        valid_pct = 100.0 * (arr != nodata).sum() / arr.size
        coverages.append(valid_pct)
        dates.append(date_from_path(p))

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(coverages)), coverages, color="#3F51B5", edgecolor="black", lw=0.5)
    ax.axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Ambang batas 70%")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d.replace("2024_", "") for d in dates], rotation=45, ha="right")
    ax.set_ylabel("% Piksel Valid")
    ax.set_xlabel("Tanggal Akuisisi (2024)")
    ax.set_title("Cakupan Piksel Valid per Tanggal Akuisisi (2024)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_cdl_label_map(cdl_path: Path, out_path: Path):
    """CDL 2024 map with 8 crops + background."""
    cdl = _load_cdl(cdl_path, ds_factor=4)
    # Build colormap from KEEP_CLASSES
    display = np.zeros_like(cdl, dtype=np.int32)
    legend_colors = ["#BFBFBF"]  # background = gray
    legend_labels = ["Background"]
    for i, cid in enumerate(KEEP_CLASSES, start=1):
        display[cdl == cid] = i
        legend_colors.append(USDA_CDL_COLORS.get(cid, "#c8c8c8"))
        legend_labels.append(CDL_CLASS_NAMES.get(cid, str(cid)))
    cmap = ListedColormap(legend_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(legend_colors) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("Peta Label USDA CDL 2024 — 6 Kelas Tanaman + Background")
    ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend_colors]
    ax.legend(handles, legend_labels, loc="lower right", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_cdl_class_distribution(cdl_path: Path, out_path: Path, top_n: int = 20):
    """Top-N CDL class area bar chart."""
    with rasterio.open(cdl_path) as src:
        cdl = src.read(1)
        # pixel area in m^2 from transform
        tx = src.transform
        px_area_m2 = abs(tx.a * tx.e)
    unique, counts = np.unique(cdl, return_counts=True)
    df = sorted(zip(counts, unique), reverse=True)[:top_n + 1]  # +1 in case background present
    df = [(c, u) for c, u in df if u != 0][:top_n]

    names, areas_ha, colors = [], [], []
    for count, cid in df:
        area_ha = count * px_area_m2 / 1e4
        names.append(USDA_CDL_NAMES.get(int(cid), f"ID {cid}"))
        areas_ha.append(area_ha)
        colors.append(USDA_CDL_COLORS.get(int(cid), "#969696"))

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(names))
    ax.barh(y, areas_ha, color=colors, edgecolor="black", lw=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Area (hektar)")
    ax.set_title(f"Top-{top_n} Kelas CDL 2024 berdasarkan Cakupan Area di Area Studi")
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_cdl_patch_detail(s2_files: list[Path], cdl_path: Path, out_path: Path):
    """Two 256×256 px training patches: true color vs CDL label (2-row × 2-col grid)."""
    PATCH = 256
    # (row_start, col_start) in full-res pixels — chosen for crop diversity
    PATCH_ORIGINS = [(1800, 2200), (3200, 1400)]
    PATCH_LABELS  = ["Patch A", "Patch B"]

    match = [p for p in s2_files if PEAK_DATE in p.name]
    if not match:
        print(f"  WARN: no file for {PEAK_DATE}")
        return
    path = match[0]

    # Read full-res RGB bands
    def read_full(band_name):
        with rasterio.open(path) as src:
            idx = S2_BAND_NAMES.index(band_name) + 1
            arr = src.read(idx).astype(np.float32)
            nd = src.nodata
        if nd is not None:
            arr[arr == nd] = np.nan
        arr[arr <= 0] = np.nan
        return arr

    b4 = read_full("B4"); b3 = read_full("B3"); b2 = read_full("B2")

    # Read full-res CDL
    with rasterio.open(cdl_path) as src:
        cdl_full = src.read(1)

    # Build CDL colormap
    legend_colors = ["#BFBFBF"]
    legend_labels  = ["Background"]
    for cid in KEEP_CLASSES:
        legend_colors.append(USDA_CDL_COLORS.get(cid, "#c8c8c8"))
        legend_labels.append(CDL_CLASS_NAMES.get(cid, str(cid)))
    cmap = ListedColormap(legend_colors)
    norm = BoundaryNorm(np.arange(-0.5, len(legend_colors) + 0.5, 1), cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for row, ((r0, c0), label) in enumerate(zip(PATCH_ORIGINS, PATCH_LABELS)):
        r1, c1 = r0 + PATCH, c0 + PATCH

        # RGB patch
        rgb = np.stack([normalize_for_display(b4[r0:r1, c0:c1]),
                        normalize_for_display(b3[r0:r1, c0:c1]),
                        normalize_for_display(b2[r0:r1, c0:c1])], axis=-1)
        axes[row, 0].imshow(rgb, interpolation="nearest")
        axes[row, 0].set_title(f"{label} — True Color (B4/B3/B2)", fontsize=9)
        axes[row, 0].axis("off")

        # CDL patch
        cdl_p = cdl_full[r0:r1, c0:c1]
        display = np.zeros_like(cdl_p, dtype=np.int32)
        for i, cid in enumerate(KEEP_CLASSES, start=1):
            display[cdl_p == cid] = i
        axes[row, 1].imshow(display, cmap=cmap, norm=norm, interpolation="nearest")
        axes[row, 1].set_title(f"{label} — Label CDL 2024", fontsize=9)
        axes[row, 1].axis("off")

    # Shared legend on last CDL panel
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in legend_colors]
    axes[1, 1].legend(handles, legend_labels, loc="lower right", fontsize=7, framealpha=0.9)

    plt.suptitle(f"Contoh patch pelatihan 256×256 px ({PEAK_DATE.replace('_', '-')})", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


def fig_rgb_single(s2_files: list[Path], out_path: Path):
    """Single true-color (B4/B3/B2) image at peak date."""
    match = [p for p in s2_files if PEAK_DATE in p.name]
    if not match:
        print(f"  WARN: no file for {PEAK_DATE}")
        return
    path = match[0]
    b4 = read_band(path, S2_BAND_NAMES.index("B4") + 1)
    b3 = read_band(path, S2_BAND_NAMES.index("B3") + 1)
    b2 = read_band(path, S2_BAND_NAMES.index("B2") + 1)
    rgb = np.stack([normalize_for_display(b4),
                    normalize_for_display(b3),
                    normalize_for_display(b2)], axis=-1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.set_title(f"True Color (B4/B3/B2) — {PEAK_DATE.replace('_', '-')}", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {out_path.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

FIGURES = {
    "rgb":      ("s2_rgb_temporal.png",        "fig_rgb_temporal"),
    "rgb_single":   ("s2_rgb_single.png",          "fig_rgb_single"),
    "cdl_patch":    ("s2_cdl_patch_detail.png",    "fig_cdl_patch_detail"),
    "bands":    ("s2_band_grid.png",           "fig_band_grid"),
    "ndvi":     ("s2_ndvi_temporal.png",       "fig_ndvi_temporal"),
    "ndvi_cls": ("s2_ndvi_per_class.png",      "fig_ndvi_per_class"),
    "spectral": ("s2_spectral_profile.png",    "fig_spectral_profile"),
    "coverage": ("s2_data_coverage.png",       "fig_data_coverage"),
    "cdl_map":  ("cdl_label_map.png",          "fig_cdl_label_map"),
    "cdl_dist": ("cdl_class_distribution_area.png", "fig_cdl_class_distribution"),
}


def main():
    parser = argparse.ArgumentParser(description="Regenerate thesis_v6 data exploration figures from 2024 data")
    parser.add_argument("--s2-dir", type=Path, default=DEFAULT_S2_DIR,
                        help="Directory with 2024 S2 tif files")
    parser.add_argument("--cdl", type=Path, default=DEFAULT_CDL,
                        help="Processed CDL 2024 GeoTIFF")
    parser.add_argument("--out-dir", type=Path, default=FIGS_DIR,
                        help="Output directory for figures")
    parser.add_argument("--only", nargs="+", choices=list(FIGURES.keys()), default=None,
                        help="Generate only specific figures (default: all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.s2_dir.exists():
        print(f"ERROR: S2 dir not found: {args.s2_dir}")
        sys.exit(1)

    s2_files = list_s2_files(args.s2_dir)
    print(f"S2 files: {len(s2_files)} from {args.s2_dir}")
    print(f"CDL    : {args.cdl}")
    print(f"Output : {args.out_dir}")

    targets = args.only if args.only else list(FIGURES.keys())
    for key in targets:
        fname, fn_name = FIGURES[key]
        out = args.out_dir / fname
        print(f"\n→ {key} → {fname}")
        fn = globals()[fn_name]
        if key in ("cdl_map", "cdl_dist"):
            fn(args.cdl, out)
        elif key in ("ndvi_cls", "cdl_patch"):
            fn(s2_files, args.cdl, out)
        else:
            fn(s2_files, out)


if __name__ == "__main__":
    main()
