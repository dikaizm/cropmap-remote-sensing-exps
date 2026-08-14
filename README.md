# cropmap-remote-sensing-exps

Crop-type mapping from multi-temporal Sentinel-2 imagery and USDA Cropland Data Layer (CDL) labels. Covers band selection (GSI / Random Forest) and segmentation model training (DeepLabV3+CBAM, SegFormer).

**Study area:** Sacramento Valley, California
**Labels:** 8 crop classes + background — Corn, Rice, Winter Wheat, Alfalfa, Tomatoes, Grapes, Almonds, Walnuts (v6.1 CalCROP21-style selection: every CDL class with ≥1,000,000 pixels at 10 m in the study area)
**Reference year:** 2024 (native 10 m USDA CDL; single-year, spatially-split train/val/test)

---

There is no `run.sh` or `Makefile` in this repo — the pipeline is driven directly via `python pipeline.py` or by running individual `stages/**/*.py` scripts.

---

## Requirements

- Python 3.11
- CUDA-capable GPU recommended for band selection (RF) and training (tested with CUDA 12.4 wheels; MPS supported on Apple Silicon with a compatibility workaround in `models/segformer.py`)
- [`geoai`](https://github.com/opengeos/geoai) — auto-discovered as a sibling checkout (`../geoai`, matching the superproject's git-submodule layout) or via the `GEOAI_PATH` env var. No manual `PYTHONPATH` export needed — `config.py` adds it to `sys.path` on import.

---

## Setup

### 1. Create an environment

**Conda (recommended for local/CPU dev):**
```bash
conda env create -f environment.yml
conda activate cropmap
```

**Pip (matches the pinned GPU training environment):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Make `geoai` available

```bash
git clone https://github.com/opengeos/geoai.git ../geoai
# or: export GEOAI_PATH=/path/to/geoai
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

`RUNPOD_API_KEY` is only needed for the `--shutdown` flag (stops the RunPod pod via its GraphQL API 8 minutes after the pipeline finishes; `RUNPOD_POD_ID` is set automatically by RunPod). Without RunPod env vars, `--shutdown` falls back to `sudo shutdown -h +8`.

### 4. Configure `config.py`

Fill in the Google Drive folder/file IDs (`GDRIVE_PROCESSED_S2_FOLDER_IDS`, `GDRIVE_PROCESSED_CDL_FOLDER_ID_V6`, `GDRIVE_MODELS_FOLDER_ID`, etc.) and set `MLFLOW_TRACKING_URI`. GDrive uploads use OAuth (`GDRIVE_OAUTH_TOKEN`, generated via `--auth` on `fetch_data.py` / `process_data.py`), not a service account.

---

## CDL Classes

| CDL ID | Class | ~Pixels (10 m, study area) |
|---|---|---|
| 3 | Rice | 6.18M |
| 75 | Almonds | 4.62M |
| 54 | Tomatoes | 2.83M |
| 76 | Walnuts | 2.17M |
| 24 | Winter Wheat | 1.74M |
| 69 | Grapes | 1.59M |
| 1 | Corn | 1.48M |
| 36 | Alfalfa | 1.06M |

Any class below the ~1M-pixel threshold (Pistachios, Prunes, Sunflower, Safflower, Fallow/Idle, …) is remapped to background (class 0). `NUM_CLASSES = 9` (0=background, 1–8=crops).

---

## Running the Pipeline

### `pipeline.py` — orchestrator

```bash
python pipeline.py --stages all                     # fetch + feature + train
python pipeline.py --stages fetch --years 2024       # download processed S2 + CDL only
python pipeline.py --stages feature                  # GSI + RF band selection only
python pipeline.py --stages train                    # training only
python pipeline.py --stages feature train            # skip fetch
python pipeline.py --force                           # re-run stages even if outputs exist
python pipeline.py --data-dir /mnt/data              # override data/processed root
python pipeline.py --stages train --shutdown         # stop the server after finishing
```

Logs go to `logs/pipeline_YYYYMMDD_HHMMSS.log` and are uploaded to MLflow (`cropmap_pipeline_runs`) along with per-stage timing.

### Stage 0 — Fetch / Process (`stages/data/`)

```bash
# Download GEE-exported per-date S2 tifs from Google Drive
python stages/data/fetch_data.py --folder-id <FOLDER_ID> --years 2024
python stages/data/fetch_data.py --verify-only
python stages/data/fetch_data.py --auth              # generate OAuth token

# Upload raw S2 as-is; reproject + class-filter + confidence-mask CDL
python stages/data/process_data.py --years 2024
python stages/data/process_data.py --years 2024 --skip-upload --skip-delete
python stages/data/process_data.py --cdl-only --conf-threshold 55
```

S2 tifs are uploaded unmodified (no NoData assignment or merging — GEE already exports one clean file per date). CDL handling differs by year: 2022/2023 reproject the 30 m CDL and apply a 3×3 majority filter to smooth resampling artifacts; 2024+ uses USDA's native 10 m CDL (no majority filter). Both years apply the NASS confidence-layer mask (default threshold 55, per Maleki et al. 2024) before filtering to `KEEP_CLASSES`.

### Stage 1 — Band Selection (`stages/selection/`)

Single-stage, threshold-based selection — no CNN oracle or forward-selection loop.

```bash
python stages/selection/band_scoring.py --selector gsi_direct --score-threshold 0.5
python stages/selection/band_scoring.py --selector rf_direct  --score-threshold 0.5
```

- **GSI-direct** (`gsi_selection.py`) — per-crop Global Separability Index (Li et al. 2023) computed per (date × band) channel; top-K union across crops (`TOP_K_PER_CROP=20`).
- **RF-direct** (`feature_importance_selection.py`) — one multi-class Random Forest (`RF_N_ESTIMATORS=500`), per-crop importance via class-conditional Gini-decrease decomposition (Wei et al. 2023 / Asam et al. 2022).

Outputs: `data/processed/select_gsi_direct_s{threshold}.json` / `select_rf_direct_s{threshold}.json` (+ matching `_bands.txt`).

### Stage 2 — Training (`stages/training/train_segmentation.py`)

Five experiment configurations, each trainable on either architecture:

| Experiment | Dates | Band selection | Purpose |
|---|---|---|---|
| `single_date` | peak NDVI | none (all bands) | Baseline — isolates temporal signal |
| `mt_ndvi` | 4 calendar dates | none | Multi-temporal baseline, no selection |
| `gsi` | GSI-selected | GSI | GSI spectral-temporal selection |
| `rf` | RF-selected | RF | RF spectral-temporal selection |
| `full` | all dates | none (all bands) | Full-stack upper-bound baseline |

Architectures: `deeplabv3plus_cbam` (ResNet-50 + CBAM, SGD) and `segformer` (mit_b2, AdamW).

```bash
python stages/training/train_segmentation.py                          # all experiments × both archs
python stages/training/train_segmentation.py --exp single_date
python stages/training/train_segmentation.py --exp gsi --arch segformer
python stages/training/train_segmentation.py --loss dynamic_balanced  # thesis primary loss (DECB-CE)
python stages/training/train_segmentation.py --hp-grid configs/hp_grid_final.json
python stages/training/train_segmentation.py --seed-grid 1 2 3
python stages/training/train_segmentation.py --exp gsi --random-channel-order  # channel-order sanity check
python stages/training/train_segmentation.py --build-cache-only       # preload cache only, no training
python stages/training/train_segmentation.py --force
```

Other notable flags: `--loss {wce,focal_tversky,dynamic_balanced}`, `--norm {percentile,minmax,zscore}`, `--no-aug`, `--no-preload`, `--use-cloud-preload` / `--upload-cache-gdrive`, `--eval-only <ckpt>`, `--batch-size`, `--epochs`, `--data-dir`, `--shutdown`.

**Split:** spatial block split within 2024 (`BLOCK_SIZE=1024` px = 4×4 patches), 70/15/15 train/val/test, class-balanced greedy stratification with a per-class pixel-floor repair pass (`stages/data/spatial_split.py`) — prevents patch-adjacency leakage between splits.

**Outputs per run** (`ml_models/`, logged to MLflow experiment `cropmap_segmentation_s2*`): best/last checkpoints, `training_history.csv`, training curve, per-class IoU, confusion matrix, segmentation map visualizations.

### Inference / Evaluation (`stages/infer/`)

Tiled inference and evaluation over held-out spatial test blocks — backs `notebooks/06_inference_test.ipynb`. `model_io.py` loads a checkpoint and runs tiled inference on one block; `metrics.py` aggregates per-block and full-test-set IoU/F1; `viz.py` plots RGB/GT/prediction/per-crop-IoU panels.

### Tools (`stages/tools/`)

```bash
python stages/tools/materialize_spatial_split.py --s2-dir <dir> --cdl <path> --out <dir>
python stages/tools/visualize_split.py --data-dir data/processed
python stages/tools/verify_tiles.py --data-dir data/processed --years 2024
python stages/tools/upload_models.py --experiment cropmap_segmentation_s2_v2 --dry-run
python stages/tools/collect_seed_grid_metrics.py
```

---

## Notebooks

Numbered in intended reading order, but not all are live:

1. **`01_fetch_data.ipynb`** — unused placeholder, no code.
2. **`02_image_processing.ipynb`** — exploratory CDL reprojection/filtering; a manual prototype of what `stages/data/process_data.py` now does, not a call into it.
3. **`03_data_exploration.ipynb`** — regenerates the thesis "Eksplorasi Data" figures (CDL class distribution, S2 coverage/RGB/NDVI/spectral profiles, band correlation). Self-contained.
4. **`04_feature_selection_scenarios.ipynb`** — compares the four channel-selection scenarios (`single_date`, `mt_ndvi`, `gsi`, `rf`). Reimplements the GSI/RF scoring math inline for visibility, but imports `save_selection` / `get_train_year_inputs` from `stages/selection/` for real.
5. **`05_segmentation_training.ipynb`** — trains both architectures across all four scenarios; genuinely imports `stages/training/` internals (`experiment_plan`, `losses`, `normalization`, `spatial_split`) rather than duplicating them.
6. **`06_inference_test.ipynb`** — loads a held-out spatial test block, runs tiled inference with both architectures, and evaluates across all test blocks; directly uses `stages/infer/`.

Notebooks 04–06 import repo modules via a runtime `sys.modules["cropmap_pipeline"]` shim pointed at this repo root — there is no actual `cropmap_pipeline/` package directory on disk.

---

## Key Hyperparameters

All defined in `config.py`.

| Parameter | Value | Used by | Description |
|---|---|---|---|
| `S2_MIN_VALID_FRAC` | 0.50 | data | Min valid-pixel fraction to keep a date |
| `SAMPLE_FRACTION` | 0.20 | selection | Fraction of labeled pixels sampled for GSI |
| `TOP_K_PER_CROP` | 20 | selection | Channels selected per crop before union |
| `RF_N_ESTIMATORS` | 500 | selection | Trees in the multi-class RF |
| `RF_MAX_PIXELS` | 1,000,000 | selection | Pixel sample cap for RF fitting |
| `PATCH_SIZE` / `STRIDE` | 256 / 256 | training | Patch size and stride (no pre-chipping to disk) |
| `BATCH_SIZE` | 8 | training | Training batch size |
| `MAX_EPOCHS` | 150 | training | Max training epochs |
| `EARLY_STOP` | 20 | training | Early-stopping patience (epochs) |
| `BLOCK_SIZE` | 1024 | training | Spatial split block size (px) |
| `VAL_FRAC` / `TEST_FRAC` | 0.15 / 0.15 | training | Split fractions (70/15/15) |
| `SEED` | 42 | training | Default random seed |
| `TRAIN_YEARS` / `TEST_YEAR` | ["2024"] / "2024" | training | Single-year, spatially-split |

---

## MLflow

Default tracking URI: `MLFLOW_TRACKING_URI` in `config.py`. Key experiments:

- `cropmap_pipeline_runs` — pipeline log uploads
- `cropmap_feature_selection_s2` — GSI/RF selection runs
- `cropmap_segmentation_s2*` — training runs (several versioned variants exist from earlier pipeline iterations; `_v6_same_area` / `_v6.1_same_area` are current)
