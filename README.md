# crop_mapping_pipeline

End-to-end pipeline for crop type mapping using multi-temporal Sentinel-2 imagery and USDA Cropland Data Layer (CDL) labels. Covers data processing, band scoring, and segmentation model training.

**Study area:** Sacramento Valley, California  
**Labels:** 8 crop classes (v6.1) + background — `NUM_CLASSES = 9`  
**Satellite:** Sentinel-2 SR Harmonized, 10 bands per date, EPSG:4326, ~10 m

---

## CDL Classes (v6.1)

Selected using the CalCROP21 criterion: ≥1,000,000 pixels in the 2024 study area (10 m grid).

| CDL ID | Class | Pixel count (approx) |
|--------|-------|----------------------|
| 75 | Almonds | 4.62 M |
| 3 | Rice | 6.18 M |
| 76 | Walnuts | 2.17 M |
| 54 | Tomatoes | 2.83 M |
| 24 | Winter Wheat | 1.74 M |
| 69 | Grapes | 1.59 M |
| 1 | Corn | 1.48 M |
| 36 | Alfalfa | 1.06 M |

All other classes (incl. Fallow/Idle CDL-61) → background (class 0).

---

## Project Structure

```
crop_mapping_pipeline/
├── pipeline.py                   # CLI orchestrator — all stages
├── config.py                     # All hyperparameters, paths, GDrive IDs
├── requirements.txt
├── environment.yml               # conda environment spec
├── check_data.sh                 # quick data-presence sanity check
├── configs/
│   └── hp_grid_*.json            # hyperparameter grid configs for search
├── stages/
│   ├── fetch_data_v6.py          # download processed S2 + CDL from Google Drive (v6 layout)
│   ├── process_data_v6.py        # process raw S2 + CDL, upload to GDrive, delete raw
│   ├── refine_cdl.py             # boundary erosion / majority filter on CDL raster
│   ├── verify_tiles.py           # tile integrity check
│   ├── valid_dates.py            # per-date S2 scene-usability filter (≥50% valid pixels)
│   ├── band_scoring.py           # GSI + RF importance scoring entry point
│   ├── feature_analysis_v2.py    # band selection CLI: selector choice + output
│   ├── spatial_split.py          # spatial block (grid) train/val/test split
│   ├── train_segmentation.py     # Stage 3: train all experiments × architectures
│   ├── select_naive_mt_dates.py  # select phenological dates for mt_base
│   ├── upload_models.py          # upload trained models to Google Drive
│   ├── ndvi_disagreement_analysis.py  # CDL vs prediction disagreement via NDVI
│   ├── check_histogram.py        # train/test band distribution shift check
│   ├── visualize_split.py        # generate split map figure for thesis
│   ├── regen_thesis_figures.py   # regenerate thesis figures from saved outputs
│   ├── experiments/
│   │   ├── base.py               # shared utilities (parse_date, build_local_band_map)
│   │   ├── exp_a.py              # single-date experiment (peak NDVI, all bands)
│   │   ├── exp_b.py              # mt_base experiment (4 phenological dates, all VEGE_BANDS)
│   │   ├── exp_select_direct.py  # GSI/RF-direct channel → local index mapper
│   │   └── registry.py           # experiment registry (ExperimentConfig dataclass)
│   ├── losses/
│   │   ├── wce.py                # Weighted Cross-Entropy (baseline loss)
│   │   ├── focal_tversky.py      # Focal Tversky (recall-biased, α=0.7 β=0.3 γ=0.75)
│   │   └── dynamic_balanced.py   # DECB-CE (Zhou et al. 2023, per-batch effective-number)
│   └── selections/
│       ├── gsi_direct.py         # GSI: all dates × bands, per-crop SI_global ranking
│       ├── rf_direct.py          # RF: multi-class RF, per-crop class-conditional MDI
│       ├── single_date_gsi.py    # GSI scoped to peak-NDVI date only
│       ├── single_date_rf.py     # RF scoped to peak-NDVI date only
│       ├── naive_mt_gsi.py       # GSI scoped to 4 phenological dates
│       ├── naive_mt_rf.py        # RF scoped to 4 phenological dates
│       ├── rf_band_only.py       # RF importance collapsed to band dimension only
│       └── band_scoring/
│           └── gsi/v3.py         # per-crop GSI computation (vectorised, ε=1e-6)
├── models/
│   ├── cbam.py                   # CBAM attention module
│   ├── deeplabv3plus.py          # DeepLabV3+ with CBAM (ResNet-50 encoder)
│   └── segformer.py              # SegFormer wrapper (MiT-B2 encoder)
└── utils/
    ├── band_selection.py         # Legacy GSI + RF utilities (notebooks)
    ├── constants.py              # CDL class colors and names
    ├── general.py                # GDrive download helpers
    ├── label.py                  # Label remapping, majority filter, erosion
    ├── mlflow_utils.py           # MLflow artifact-logging patch
    └── check_corrupt_files.py    # Identify corrupt/truncated TIF files
```

---

## Requirements

- Python ≥ 3.10
- CUDA GPU recommended for Stage 3 training (tested on RTX 2000 Ada 16 GB)
- Band scoring (GSI/RF) is CPU-only and can run locally
- `MLFLOW_DISABLE_TELEMETRY=true` must be set before importing mlflow (prevents hang on MLflow 3.x)

---

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate cropmap
```

Or manually:

```bash
conda create -n cropmap python=3.10
conda activate cropmap
pip install -r requirements.txt
```

**GPU (CUDA 12.4):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Configure `config.py`

Set GDrive folder IDs for your processed data:

```python
# v6.1 processed S2 (single parent folder, year subfolder s2/2024/)
GDRIVE_PROCESSED_S2_V6_FOLDER_IDS = {"2024": "<S2_2024_FOLDER_ID>"}
GDRIVE_PROCESSED_CDL_FOLDER_ID_V6 = "<CDL_PROCESSED_FOLDER_ID>"

# MLflow
MLFLOW_TRACKING_URI = "https://your-mlflow-server"
```

**GDrive OAuth token** (for uploads from `process_data_v6.py`):
```bash
python stages/process_data_v6.py --auth   # generates ssh/gdrive_token.pickle locally
# then scp ssh/gdrive_token.pickle to server
```

---

## Pipeline Stages

### Stage 0 — Fetch (`stages/fetch_data_v6.py`)

Downloads processed Sentinel-2 GeoTIFFs and CDL rasters from Google Drive.

```bash
python stages/fetch_data_v6.py                        # download all
python stages/fetch_data_v6.py --years 2024           # specific year
python stages/fetch_data_v6.py --verify-only          # check what is present
```

**Outputs:** `data/processed/s2/2024/*_processed.tif`, `data/processed/cdl/cdl_2024_study_area_filtered.tif`

---

### Stage 0b — Process (`stages/process_data_v6.py`)

Processes raw GEE-exported S2 TIFs and raw CDL. Designed for storage-constrained servers — processes one year, uploads to GDrive, deletes raw.

**CDL:** Reproject EPSG:5070 → EPSG:4326, clip to S2 grid, resample to ~10 m (nearest-neighbour), filter to `KEEP_CLASSES`.  
**S2:** Replace negative / NaN / Inf → NoData sentinel (`-9999`), cast to `float32`.

```bash
python stages/process_data_v6.py --years 2024 --delete --shutdown
python stages/process_data_v6.py --auth     # generate OAuth token first (run locally)
```

---

### Stage 1 — Band Scoring (`stages/band_scoring.py` / `stages/feature_analysis_v2.py`)

Scores all temporal-spectral channels per crop. Two selectors available:

| Selector | Method | Reference |
|----------|--------|-----------|
| `gsi_direct` | Per-crop SI_global (GSI) over all dates × bands | Jakubowski et al. 2013 |
| `rf_direct` | Multi-class RF, class-conditional MDI per crop | Wei et al. 2023 |

Both are **CPU-only** and can run locally on subsampled pixels (`SAMPLE_FRACTION=0.20`).

```bash
# GSI scoring
python stages/feature_analysis_v2.py --stage select --selector gsi_direct --top-k 20

# RF scoring
python stages/feature_analysis_v2.py --stage select --selector rf_direct --top-k 20

# Score threshold variant (per-crop normalised ≥ 0.5)
python stages/feature_analysis_v2.py --stage select --selector gsi_direct --score-threshold 0.5
python stages/feature_analysis_v2.py --stage select --selector rf_direct --score-threshold 0.5
```

**Outputs:** `data/processed/select_gsi_direct_k20.json`, `select_rf_direct_k20.json` (+ `_bands.txt`, per-class CSVs)

---

### Stage 2 — Spatial Split (`stages/spatial_split.py`)

Computes the train/val/test patch assignment using spatial block stratification. Independent of band selection — identical split used by all experiments.

```bash
python stages/spatial_split.py                        # uses config defaults
python stages/spatial_split.py --year 2024 --block-size 1024
```

**Split:** 70 / 15 / 15 (train/val/test) at block level (BLOCK_SIZE = 1024 px = 4×4 patches at PATCH_SIZE=256). No train patch is spatially adjacent to a val/test patch.

---

### Stage 3 — Training (`stages/train_segmentation.py`)

Trains up to 8 experiments: 4 band configs × 2 architectures.

| Experiment | Input | Band selection | Purpose |
|------------|-------|----------------|---------|
| `single_date` | Peak NDVI date | None (all 10 bands) | Single-date baseline |
| `mt_base` | 4 phenological dates | None (all VEGE_BANDS) | Multi-temporal naive baseline |
| `gsi` | Multi-temporal | GSI-direct top-K union | Spectral-temporal GSI selection |
| `rf` | Multi-temporal | RF-direct top-K union | Spectral-temporal RF selection |

**Architectures:** `deeplabv3plus_cbam` (ResNet-50 encoder) | `segformer` (MiT-B2 encoder)

**Data:** 2024 single year, spatial block split (70/15/15).  
**Losses:** `wce` (default) | `focal_tversky` | `dynamic_balanced`  
**Optimizer:** AdamW + PolynomialLR decay (power=0.9), optional linear warmup.

```bash
python stages/train_segmentation.py                            # all experiments, both archs
python stages/train_segmentation.py --exp single_date gsi rf  # subset
python stages/train_segmentation.py --exp gsi --arch segformer
python stages/train_segmentation.py --force                    # re-run even if checkpoint exists
python stages/train_segmentation.py --data-dir /mnt/data
```

**Outputs per experiment** (saved to `ml_models/<run>/<exp>/` + logged to MLflow):
- `best_model.pth`, `last_model.pth`
- `training_history.csv`, `training_curves.png`
- `test_per_class_iou.csv`, `confusion_matrix.png`, `test_segmentation_map.png`
- `test_pred_map.npy`, `test_gt_map.npy` (for NDVI disagreement analysis)

---

## Running via `pipeline.py`

```bash
python pipeline.py --stages fetch feature train
python pipeline.py --stages all --shutdown       # full run + shutdown RunPod pod
python pipeline.py --stages train --force
python pipeline.py --data-dir /mnt/data
```

Valid stages: `fetch`, `fetch-processed`, `score`, `train`, `all`

---

## Key Hyperparameters

All hyperparameters live in `config.py`.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `S2_BAND_NAMES` | 10 bands (B2–B12, excl. B1/B9/B10) | Land bands per date |
| `S2_MIN_VALID_FRAC` | 0.50 | Min valid-pixel fraction per date to include |
| `KEEP_CLASSES` | `[1, 3, 24, 36, 54, 69, 75, 76]` | 8 crop CDL IDs (v6.1) |
| `NUM_CLASSES` | 9 | 0=background + 1–8 crops |
| `SAMPLE_FRACTION` | 0.20 | Fraction of labeled pixels for GSI/RF scoring |
| `SELECT_TOP_K_PER_CROP` | 20 | Channels per crop before union |
| `PATCH_SIZE` | 256 | Spatial patch size (px) |
| `STRIDE` | 256 | Patch stride (no overlap) |
| `BLOCK_SIZE` | 1024 | Block side for spatial split (= 4×4 patches) |
| `BATCH_SIZE` | 8 | Training batch size |
| `MAX_EPOCHS` | 150 | Maximum training epochs |
| `EARLY_STOP` | 20 | Early stopping patience (epochs) |
| `EARLY_STOP_DELTA` | 0.001 | Min mIoU gain to reset patience |
| `VAL_FRAC / TEST_FRAC` | 0.15 / 0.15 | Spatial split fractions |
| `TRAIN_YEARS` | `["2024"]` | Training year(s) |
| `TEST_YEAR` | `"2024"` | Test year (same-area spatial split) |
| `RF_N_ESTIMATORS` | 500 | RF trees for band scoring |
| `RF_MAX_PIXELS` | 1,000,000 | Pixel sample cap for RF training |

**Architecture LR / weight-decay:**

| Arch | LR | Weight decay | Encoder |
|------|----|-------------|---------|
| `deeplabv3plus_cbam` | 1e-4 | 1e-4 | ResNet-50 |
| `segformer` | 6e-5 | 1e-2 | MiT-B2 |

---

## Expected Outputs

```
data/processed/
├── s2/2024/                          # processed S2 GeoTIFFs (*_processed.tif)
├── cdl/                              # CDL filtered rasters
├── select_gsi_direct_k20.json        # GSI-direct selection (top-20 per crop)
├── select_gsi_direct_k20_bands.txt   # union channel list (plain text)
├── select_rf_direct_k20.json         # RF-direct selection (top-20 per crop)
├── select_rf_direct_k20_bands.txt
├── phenol_dates.json                 # 4 phenological dates for mt_base
└── preload_cache/                    # pre-loaded tensor cache (gitignored)

ml_models/
├── <run_id>/
│   ├── exp_single_date/
│   ├── exp_mt_base/
│   ├── exp_gsi/
│   └── exp_rf/
│       ├── best_model.pth
│       ├── last_model.pth
│       ├── training_history.csv
│       ├── training_curves.png
│       ├── test_per_class_iou.csv
│       ├── confusion_matrix.png
│       ├── test_segmentation_map.png
│       ├── test_pred_map.npy
│       └── test_gt_map.npy

logs/
└── pipeline_YYYYMMDD_HHMMSS.log
```

---

## MLflow

- **Remote:** `https://mlflow-geoai.stelarea.com`
- **Experiments:**
  - `cropmap_pipeline_runs` — pipeline-level logs
  - `cropmap_feature_selection_s2` — band scoring runs
  - `cropmap_segmentation_s2_v6.1_same_area` — Stage 3 training runs

Set `MLFLOW_DISABLE_TELEMETRY=true` before importing mlflow to avoid background-thread hang on MLflow 3.x.

---

## SSH to RunPod

```bash
ssh -i ssh/runpod-cropmap <pod-user>@ssh.runpod.io
```

Auto-shutdown after training: pass `--shutdown` to `pipeline.py` (uses RunPod GraphQL `podStop` mutation via `RUNPOD_API_KEY` + `RUNPOD_POD_ID` from `.env`).
