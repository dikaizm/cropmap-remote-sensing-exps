# crop_mapping_pipeline

End-to-end pipeline for crop type mapping using multi-temporal Sentinel-2 imagery and USDA Cropland Data Layer (CDL) labels. Covers data processing, band selection, and segmentation model training.

**Study area:** Sacramento Valley, California
**Labels:** 10 crop classes (Rice, Sunflower, Winter Wheat, Alfalfa, Other Hay, Tomatoes, Grapes, Almonds, Walnuts, Plums) + background

---

## Project Structure

```
crop_mapping_pipeline/
├── pipeline.py              # CLI entry point — orchestrates all stages
├── config.py                # All hyperparameters, file paths, and GDrive IDs
├── requirements.txt
├── stages/
│   ├── fetch_data.py        # Stage 0: download processed S2 + CDL from Google Drive
│   ├── process_data.py      # Stage 0.5: process raw S2 + CDL, upload to GDrive, delete raw
│   ├── feature_analysis.py  # Stage 1+2: band selection (GSI ranking + CNN forward selection)
│   └── train_segmentation.py  # Stage 3: train Exp A/B/C × 2 architectures
├── models/
│   ├── cbam.py              # CBAM attention module
│   ├── deeplabv3plus.py     # DeepLabV3+ with CBAM
│   └── segformer.py         # SegFormer wrapper
└── utils/
    ├── band_selection.py    # GSI, RF importance, joint score, top-k selection
    ├── constants.py         # CDL class colors and names
    ├── general.py           # Download helpers
    └── label.py             # Label remapping utilities
```

---

## Requirements

- Python >= 3.10
- CUDA-capable GPU recommended for Stage 2 and Stage 3 (tested on RTX 4090, 24 GB VRAM)
- Stage 1 is CPU-only and can be run locally
- [`geoai`](https://github.com/opengeos/geoai) library available on `PYTHONPATH`

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd crop_mapping_pipeline

python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

**CPU (Stage 1 / local only):**
```bash
pip install -r requirements.txt
```

**GPU (CUDA 12.4, for Stage 2 + Stage 3 on server):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**GDrive upload (for `process_data.py`):**
```bash
pip install google-api-python-client google-auth
```

### 3. Add geoai to PYTHONPATH

`geoai` is required by Stage 3 for patch dataset loading and training utilities.

```bash
git clone https://github.com/opengeos/geoai.git /path/to/geoai
export PYTHONPATH="/path/to/geoai:$PYTHONPATH"
```

Add the export to your `.bashrc` / `.zshrc` for persistence.

### 4. Configure `config.py`

**Google Drive download IDs** — for `fetch_data.py` (downloading processed files to server).
S2 files use **one GDrive folder per year** so they can be downloaded and deleted year by year:
```python
GDRIVE_FILES = {
    "s2_2022":  {"type": "folder", "id": "<S2_2022_FOLDER_ID>", ...},
    "s2_2023":  {"type": "folder", "id": "<S2_2023_FOLDER_ID>", ...},
    "s2_2024":  {"type": "folder", "id": "<S2_2024_FOLDER_ID>", ...},
    "cdl_2022": {"type": "file",   "id": "<CDL_2022_FILE_ID>",  ...},
    "cdl_2023": {"type": "file",   "id": "<CDL_2023_FILE_ID>",  ...},
    "cdl_2024": {"type": "file",   "id": "<CDL_2024_FILE_ID>",  ...},
}
```

**Google Drive upload IDs** — for `process_data.py` (uploading processed files from server).
S2 uses the same per-year folder IDs as `GDRIVE_FILES` above — set them once and both scripts share them:
```python
GDRIVE_PROCESSED_S2_FOLDER_IDS = {
    "2022": "<S2_2022_FOLDER_ID>",
    "2023": "<S2_2023_FOLDER_ID>",
    "2024": "<S2_2024_FOLDER_ID>",
}
GDRIVE_PROCESSED_CDL_FOLDER_ID = "<CDL_PROCESSED_FOLDER_ID>"
GDRIVE_CREDENTIALS             = Path("ssh/gdrive_service_account.json")
```

**MLflow:**
```python
MLFLOW_TRACKING_URI = "http://your-mlflow-server"
```

---

## Deployment Workflows

This pipeline supports two deployment patterns depending on whether processed files are already on Google Drive.

### Workflow A — Processed files already on GDrive

Use this when `*_processed.tif` S2 files and filtered CDL files are already uploaded to GDrive (e.g. processed locally first).

```
[Local]   Stage 1  →  stage1v2_candidates.json
              ↓ copy JSON to server
[Server]  Stage 2  →  stage3_exp_c_bands.txt
[Server]  Stage 3  →  trained models
```

### Workflow B — Raw files only, process on server

Use this when only raw GEE-exported S2 TIFs exist. Processes and uploads year by year to keep disk usage low.

```
[Server]  process_data.py --years 2022  →  processed TIFs → upload to GDrive → delete raw
[Server]  process_data.py --years 2023  →  ...
[Server]  process_data.py --years 2024  →  ...
[Local]   Stage 1  →  stage1v2_candidates.json
              ↓ copy JSON to server
[Server]  Stage 2  →  stage3_exp_c_bands.txt
[Server]  Stage 3  →  trained models
```

---

## Running the Pipeline

### Recommended: run locally (Stage 1) + server (Stage 2 + 3)

**Step 1 — Run Stage 1 locally (CPU, no GPU needed)**
```bash
PYTHONPATH=. python crop_mapping_pipeline/stages/feature_analysis.py \
    --stage 1 \
    --data-dir /path/to/local/processed
```
Output: `data/processed/stage1v2_candidates.json`

**Step 2 — Copy handoff file to server**
```bash
scp data/processed/stage1v2_candidates.json user@gpu-server:/data/processed/
```

**Step 3 — Run Stage 2 on server (GPU)**
```bash
PYTHONPATH=. python crop_mapping_pipeline/stages/feature_analysis.py \
    --stage 2 \
    --data-dir /data/processed
```
Output: `data/processed/stage3_exp_c_bands.txt`

**Step 4 — Run Stage 3 on server (GPU)**
```bash
./run.sh --stages train --shutdown
```

---

### Option A — run.sh launcher (recommended for server)

```bash
chmod +x run.sh

./run.sh                                      # run all pipeline stages
./run.sh --stages fetch                       # download processed data only
./run.sh --stages feature                     # band selection only (Stage 1+2)
./run.sh --stages train                       # model training only
./run.sh --stages feature train               # skip fetch
./run.sh --stages train --force               # force re-run even if outputs exist
./run.sh --stages all --data-dir /mnt/data    # override data path
./run.sh --stages train --shutdown            # shut down server 8 min after finishing
```

Logs are written to `logs/run_YYYYMMDD_HHMMSS.log`. The process PID is saved to `logs/pipeline_YYYYMMDD_HHMMSS.pid`.

Monitor a running job:
```bash
tail -f logs/run_<timestamp>.log
```

Stop a running job:
```bash
kill $(cat logs/pipeline_<timestamp>.pid)
```

### Option B — Direct Python

```bash
PYTHONPATH=. python -m crop_mapping_pipeline.pipeline                         # all stages
PYTHONPATH=. python -m crop_mapping_pipeline.pipeline --stages fetch feature  # fetch + selection
PYTHONPATH=. python -m crop_mapping_pipeline.pipeline --stages train          # training only
PYTHONPATH=. python -m crop_mapping_pipeline.pipeline --data-dir /mnt/data    # custom data dir
```

### Option C — Run stages individually

```bash
# Processing (Workflow B only)
python crop_mapping_pipeline/stages/process_data.py --years 2022
python crop_mapping_pipeline/stages/process_data.py --years 2022 2023 2024 --skip-upload
python crop_mapping_pipeline/stages/process_data.py --years 2022 --skip-delete   # keep raw

# Fetch (download processed files from GDrive)
python crop_mapping_pipeline/stages/fetch_data.py                          # all years
python crop_mapping_pipeline/stages/fetch_data.py --years 2022             # one year only
python crop_mapping_pipeline/stages/fetch_data.py --years 2022 2023        # multiple years
python crop_mapping_pipeline/stages/fetch_data.py --years 2022 --delete    # download then delete
python crop_mapping_pipeline/stages/fetch_data.py --delete                 # delete all downloaded files
python crop_mapping_pipeline/stages/fetch_data.py --verify-only            # check what's present

# Feature analysis
python crop_mapping_pipeline/stages/feature_analysis.py --stage 1              # Stage 1 only (local)
python crop_mapping_pipeline/stages/feature_analysis.py --stage 2              # Stage 2 only (server)
python crop_mapping_pipeline/stages/feature_analysis.py --force                # re-run both

# Training
python crop_mapping_pipeline/stages/train_segmentation.py
python crop_mapping_pipeline/stages/train_segmentation.py --exp A B
python crop_mapping_pipeline/stages/train_segmentation.py --arch segformer
python crop_mapping_pipeline/stages/train_segmentation.py --skip-viz
```

---

## Pipeline Stages

### Stage 0 — Fetch (`stages/fetch_data.py`)

Downloads processed Sentinel-2 GeoTIFFs and CDL rasters from Google Drive using `gdown`. Use this when processed files are already on GDrive.

**Inputs:** `GDRIVE_FILES` IDs in `config.py`
**Outputs:** `data/processed/s2/*_processed.tif`, `data/processed/cdl/cdl_*_study_area_filtered.tif`

---

### Stage 0.5 — Process (`stages/process_data.py`)

Processes raw GEE-exported S2 TIFs and raw CDL rasters year by year. Designed for storage-constrained GPU servers — processes one year, uploads to GDrive, then deletes raw files before moving to the next year.

**CDL processing:**
1. Reproject EPSG:5070 → EPSG:4326, clip to S2 grid, resample to ~10 m (nearest neighbour to preserve labels)
2. Filter to `KEEP_CLASSES`, set all other pixels to 0 (background)

**S2 processing:**
- Replace negative / NaN / Inf pixels with NoData sentinel (`-9999`), cast to `float32`

**GDrive upload setup:**
1. Create a Google Cloud service account and enable the Drive API
2. Share your GDrive processed folders with the service-account email
3. Download the JSON key → `ssh/gdrive_service_account.json`
4. Fill in `GDRIVE_PROCESSED_S2_FOLDER_ID` and `GDRIVE_PROCESSED_CDL_FOLDER_ID` in `config.py`

**Inputs:** Raw S2 TIFs (`S2H_YYYY_YYYY_MM_DD.tif`), raw CDL TIFs (`YYYY_30m_cdls/*.tif`)
**Outputs:** Processed TIFs uploaded to GDrive + deleted locally

---

### Stage 1 — GSI Ranking (`stages/feature_analysis.py --stage 1`)

Ranks all input channels per crop using the SIglobal metric (per-crop GSI). Produces a ranked candidate list of up to `TOP_K_PER_CROP=20` bands per crop. **CPU-only — run locally.**

**Inputs:** S2 + CDL from 2022 (training reference year)
**Outputs:** `data/processed/stage1v2_candidates.json` — handoff file for Stage 2

> Copy this file to the GPU server before running Stage 2.

---

### Stage 2 — CNN Forward Selection (`stages/feature_analysis.py --stage 2`)

Per-crop binary CNN forward selection in Stage 1 rank order. A lightweight U-Net (ResNet-18) is trained for each candidate band; a band is accepted if IoU gain ≥ `S2_DELTA=0.005`. Stops after `S2_NO_IMPROVE=5` consecutive rejections. **Requires GPU.**

MLflow logs one **nested run per crop** under a parent Stage 2 run, with a clean IoU-vs-band-rank curve per crop.

**Inputs:** `stage1v2_candidates.json` + S2 + CDL from 2022
**Outputs:**
- `data/processed/stage2v2_per_crop_results.csv` — per-crop K*, key dates, key bands, IoU
- `data/processed/stage3_exp_c_bands.txt` — union of selected bands (Stage 3 Exp C input)

> `stage3_exp_c_bands.txt` **must exist** before running Stage 3. Training will fail with a clear error if it is missing.

---

### Stage 3 — Training (`stages/train_segmentation.py`)

Trains 6 experiments: **Exp A / B / C** × **DeepLabV3+CBAM / SegFormer**. **Requires GPU.**

| Config | Input | Channels | Purpose |
|--------|-------|----------|---------|
| Exp A | Single date (Jul 30, peak season) | 9 | Baseline — no temporal info |
| Exp B | 4 phenological dates (Jan/Mar/Jul/Nov) | 36 | Temporal, no band selection |
| Exp C | Stage 2 output | K* ≤ 36 | Proposed method |

**Train/test split:** 2022 + 2023 → train, 2024 → test (temporal split).
**Loss:** weighted cross-entropy (inverse class frequency).
**Optimizer:** AdamW + PolynomialLR, early stopping patience = 10 epochs.
**Patches:** 256×256, stride=256, generated on-the-fly (no pre-chipping to disk).

**Outputs per experiment** (logged to MLflow + saved locally):
- `best_model.pth` — best validation mIoU checkpoint
- `last_model.pth` — end-of-training checkpoint
- `training_history.csv` + `training_curves.png`
- `test_per_class_iou.csv`
- `confusion_matrix.png`
- `test_segmentation_map.png`

---

## Key Hyperparameters

All hyperparameters are in `config.py`.

| Parameter | Value | Stage | Description |
|-----------|-------|-------|-------------|
| `SAMPLE_FRACTION` | 0.05 | 1 | Fraction of crop pixels sampled for GSI |
| `TOP_K_PER_CROP` | 20 | 1 | Candidate list size per crop |
| `S2_DELTA` | 0.005 | 2 | Min IoU gain to accept a band |
| `S2_NO_IMPROVE` | 5 | 2 | Consecutive rejections before stopping |
| `S2_MAX_BANDS` | 20 | 2 | Max bands selected per crop |
| `S2_EPOCHS` | 15 | 2 | Epochs per band evaluation |
| `PATCH_SIZE` | 256 | 3 | Spatial patch size (pixels) |
| `BATCH_SIZE` | 8 | 3 | Training batch size |
| `MAX_EPOCHS` | 100 | 3 | Max training epochs |
| `EARLY_STOP` | 10 | 3 | Early stopping patience |
| `TRAIN_YEARS` | 2022, 2023 | 3 | Training years |
| `TEST_YEAR` | 2024 | 3 | Test year |

---

## Expected Outputs

After a full pipeline run:

```
data/processed/
├── s2/                              # processed S2 GeoTIFFs (*_processed.tif)
├── cdl/                             # CDL filtered rasters
├── stage1v2_candidates.json         # Stage 1 → Stage 2 handoff
├── stage2v2_per_crop_results.csv    # per-crop band selection results
└── stage3_exp_c_bands.txt           # union band list for Exp C

ml_models/
├── expA_deeplabv3plus_cbam/
│   ├── best_model.pth
│   ├── last_model.pth
│   ├── training_history.csv
│   ├── training_curves.png
│   ├── test_per_class_iou.csv
│   ├── confusion_matrix.png
│   └── test_segmentation_map.png
├── expA_segformer/
├── expB_deeplabv3plus_cbam/
├── expB_segformer/
├── expC_deeplabv3plus_cbam/
└── expC_segformer/

logs/
├── run_YYYYMMDD_HHMMSS.log
└── pipeline_YYYYMMDD_HHMMSS.pid
```
