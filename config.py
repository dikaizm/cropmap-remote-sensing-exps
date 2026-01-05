"""
Pipeline configuration — edit GDrive folder IDs and paths before running.
All path settings can be overridden via --data-dir in pipeline.py.
"""

import numpy as np
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent   # crop_mapping_pipeline/

# ── Data paths ─────────────────────────────────────────────────────────────────
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
CDL_DIR          = PROCESSED_DIR / "cdl"
MODELS_DIR       = PROJECT_ROOT / "ml_models"
FIGURES_DIR      = PROJECT_ROOT / "documents" / "thesis" / "figures"
LOGS_DIR         = PROJECT_ROOT / "logs"
PRELOAD_CACHE_DIR = PROCESSED_DIR / "preload_cache"
PRELOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# S2 data organised by role, not year
S2_TRAIN_DIR = PROCESSED_DIR / "s2" / "2024"   # main training area (all dates, flat; v6.1 processed S2)
S2_PROCESSED_DIR = S2_TRAIN_DIR                  # backwards-compat alias

CDL_TRAIN  = CDL_DIR / "cdl_2024_study_area_filtered.tif"   # matches process_data_v6.py output naming
CDL_BY_YEAR = {"2024": CDL_TRAIN}        # legacy lookup used internally

# ── S2 metadata ────────────────────────────────────────────────────────────────
S2_BAND_NAMES    = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]   # v6.1: 10 land bands (B1/B9/B10 atmospheric 60m excluded)
N_BANDS_PER_DATE = len(S2_BAND_NAMES)
S2_NODATA        = -9999.0
# Per-date scene-usability threshold: drop downloaded date TIFs whose valid-pixel
# fraction (non-nodata, finite) falls below this. Follows the CalCROP21 grid-curation
# criterion (>=50% non-unknown pixels). Excludes residual high-cloud/partial-capture
# dates (e.g. 2024-01-31 = 14.8% valid, 2024-12-11 = 4.8%) from selection + training.
S2_MIN_VALID_FRAC = 0.50
# 9 vegetation bands — legacy, kept for GSI/RF feature analysis (excludes B8A)
VEGE_BANDS       = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

# ── CDL classes ────────────────────────────────────────────────────────────────
# v6.1: CalCROP21-style selection — keep every crop class with >= 1,000,000 pixels
# in the study area (10 m grid, 2024 reprojected CDL). 8 crops pass the threshold.
# Counts (10 m px): Rice 6.18M, Almonds 4.62M, Tomatoes 2.83M, Walnuts 2.17M,
#   Winter Wheat 1.74M, Grapes 1.59M, Corn 1.48M, Alfalfa 1.06M.
# Below threshold → background: Pistachios(204) 0.57M, Prunes(210) 0.54M,
#   Sunflower(6) 0.50M, Safflower(33) 0.43M, …  Fallow/Idle(61) → background.
KEEP_CLASSES = [1, 3, 24, 36, 54, 69, 75, 76]  # Corn, Rice, WinterWheat, Alfalfa, Tomatoes, Grapes, Almonds, Walnuts
CLASS_REMAP  = {cls_id: i + 1 for i, cls_id in enumerate(KEEP_CLASSES)}
NUM_CLASSES  = len(KEEP_CLASSES) + 1   # 9: 0=bg + 1–8=crops

CDL_CLASS_NAMES = {
    1:  "Corn",      3:  "Rice",     24: "Winter Wheat",  36: "Alfalfa",
    54: "Tomatoes",  69: "Grapes",   75: "Almonds",       76: "Walnuts",
}

REMAP_LUT = np.zeros(256, dtype=np.int64)
for _cdl_id, _model_id in CLASS_REMAP.items():
    if _cdl_id < 256:
        REMAP_LUT[_cdl_id] = _model_id

# ── Google Drive upload (processed files → GDrive) ────────────────────────────
# Used by process_data.py after local processing.
# GDRIVE_CREDENTIALS: path to a Google service-account JSON key file.
#   Create one at: console.cloud.google.com → IAM → Service Accounts → Keys
#   Share the target GDrive folders with the service-account email.
GDRIVE_CREDENTIALS  = Path(__file__).parent / "ssh" / "gdrive_service_account.json"
GDRIVE_OAUTH_SECRET = Path(__file__).parent / "ssh" / next(
    (f.name for f in (Path(__file__).parent / "ssh").glob("client_secret_*.json")),
    "client_secret.json",
)
GDRIVE_OAUTH_TOKEN  = Path(__file__).parent / "ssh" / "gdrive_token.pickle"
GDRIVE_RAW_S2_V2_FOLDER_ID = "1yZmKDjGnXZH6622d8SU4GDUB1z940HwY"
GDRIVE_RAW_S2_V5_FOLDER_ID = "1HZOB1b8eq9sF9dtYhppYQC0jsGPuBZZM"

GDRIVE_RAW_S2_V5_FOLDER_IDS = {
    "2022": "14PE8DRpDJqUlux__bBqd-6oAofJrDioU",
    "2023": "1kP7qv9zvjZ8YRlxhFrrwC0fC3GhHK55S",
    "2024": "1YLfx6b5CXbkeR4lvG2hyoky5KICqHSJr",
}

GDRIVE_PROCESSED_S2_FOLDER_IDS = {
    "2022": "1mgiE8vHXiKZHN-zRc68zYLQOAMtO8hst",
    "2023": "1loxQTczrQ_oje6D3dYxzcU-Eo_tPNfnl",
    "2024": "1Dp--kFrQfqFS7C9osEREy9EZKt7KnN_4",
}
# v6.1 processed S2 (single-year 2024, single parent folder containing s2/ + cdl/).
GDRIVE_PROCESSED_S2_V6_FOLDER_ID  = "1efS4GdRy-RmMrWIs3d2KJ2B4o-2f_mws"
GDRIVE_PROCESSED_S2_V6_FOLDER_IDS = {"2024": GDRIVE_PROCESSED_S2_V6_FOLDER_ID}
# Cloud-built portable preload cache (preload_*.npy + *_masks.pt). GDrive folder
# used by both `--preload-cache-gdrive` (download a prebuilt cache instead of
# rebuilding locally) and `--build-cache-only` auto-upload. Empty = disabled.
GDRIVE_PRELOAD_CACHE_FOLDER_ID    = "1bT-iZ3stuMuzrL0y_Exx-jp97P3aj_BG"
GDRIVE_PROCESSED_CDL_FOLDER_ID_V5 = "1L2vIVTJAuWCpLY9g4wmsWcAF6pXPZsnY"
GDRIVE_RAW_CDL_FOLDER_ID           = ""   # optional GDrive fallback; USDA NASS used by default
CDL_DOWNLOAD_URLS = {
    "2022": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2022_30m_cdls.zip",
    "2023": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2023_30m_cdls.zip",
    "2024": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2024_30m_cdls.zip",
}
# Native 10m CDL (random-forest, Sentinel-2+Landsat fusion, no 30m resampling).
# Only available from 2024 onward — used by process_data_v6.py for the test year.
CDL_DOWNLOAD_URLS_10M = {
    "2024": "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2024_10m_cdls.zip",
}
GDRIVE_PROCESSED_CDL_FOLDER_ID    = "1limegK5Eu3NpNOKHG9xDPe8RoW1B7qMQ"
GDRIVE_PROCESSED_CDL_FOLDER_ID_V6 = "1oztNLt4a2YS4CzL5cIahW0rvKZB0GARW"
# V2 study area processed data — single parent folder; year subfolders created automatically
GDRIVE_PROCESSED_V2_FOLDER_ID     = "1RepvRly_kh4z54Jum-3F_RBzxsw3wxcS"
GDRIVE_PROCESSED_V3_FOLDER_ID     = "1WyMw6j1jRdTeIMrG0rkbRv712RBw5rz_"
GDRIVE_PROCESSED_V5_FOLDER_ID     = "1uIYK2dgfmAKyiw1E-wt0qYvIx0OJhLwy"
GDRIVE_MODELS_FOLDER_ID        = "1R6VbWAJpwEe83iCZX0x2O_m8zLetkH9J"

# Spatial test area S2 folders (processed, flat — one file per date)
GDRIVE_S2_TEST_A_FOLDER_ID = "1i1tlLlfuKu8NB1lnKb1bZTFciJa5A2tt"
GDRIVE_S2_TEST_B_FOLDER_ID = "1_RP6y_NsmN7OVkruQg6X3WrWbVOgjF8H"

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI        = "https://mlflow-geoai.stelarea.com"
MLFLOW_EXPERIMENT_PIPELINE = "cropmap_pipeline_runs"
MLFLOW_EXPERIMENT_FEATURE  = "cropmap_feature_selection_s2"
MLFLOW_EXPERIMENT_TRAIN_V6_1_SAME_AREA = "cropmap_segmentation_s2_v6.1_same_area"   # current thesis (v6.1)

# ── GSI scoring hyperparameters ───────────────────────────────────────────────
SAMPLE_FRACTION = 0.20   # 20% of labeled crop pixels for GSI computation
TOP_K_PER_CROP  = 20     # top-K channels per crop before union

# ── Training hyperparameters ───────────────────────────────────────────────────
TRAIN_YEARS    = ["2024"]
TEST_YEAR      = "2024"
PATCH_SIZE     = 256
STRIDE         = 256
MIN_VALID_FRAC = 0.1
BATCH_SIZE     = 8
MAX_EPOCHS     = 150
EARLY_STOP     = 20
EARLY_STOP_DELTA = 0.001   # min mIoU improvement to reset patience
VAL_FRAC       = 0.15      # fraction → val
TEST_FRAC      = 0.15      # fraction → test  (70/15/15)
SEED           = 42

# ── Train/val/test split strategy ──────────────────────────────────────────
# Spatial block (grid) split — patches grouped into BLOCK_SIZE×BLOCK_SIZE blocks;
# each block assigned wholly to one split (train/val/test) via class-balanced
# greedy stratification. Prevents patch-adjacency spatial leakage (no train patch
# spatially adjacent to a val/test patch).
BLOCK_SIZE = 1024          # px per block side = 4×4 patches (PATCH_SIZE=256)
# Per-split, per-crop minimum pixel fraction (of that crop's total) the repair pass
# enforces, so no split gets a crop only as a token sliver. 0 disables (presence-only).
MIN_CLASS_FRAC = 0.05

# Scheduler: PolynomialLR decay with optional linear warmup.
# Both tunable via --hp-grid (scheduler/warmup hyperparameter search).
SCHED_POWER    = 0.9       # PolynomialLR power (1.0 = linear decay)
WARMUP_EPOCHS  = 0         # linear-warmup epochs before polynomial decay (0 = no warmup)
WARMUP_START_FACTOR = 0.1  # initial lr multiplier at epoch 0 during warmup

ARCH_CFG = {
    "deeplabv3plus_cbam": {"lr": 1e-4, "weight_decay": 1e-4, "encoder": "resnet50"},
    "segformer":          {"lr": 6e-5, "weight_decay": 1e-2, "encoder": "mit_b2"},
}

# ── Band scoring hyperparameters ───────────────────────────────────────────────
TOP_DATES_PER_CROP = 10   # top dates per crop kept as candidates
TOP_BANDS_PER_CROP = 9    # top bands per crop (= len(VEGE_BANDS))
# Aliases used in scoring code
MAX_DATES_PER_CROP = TOP_DATES_PER_CROP
MAX_BANDS_PER_CROP = TOP_BANDS_PER_CROP

GSI_CANDIDATES_JSON = PROCESSED_DIR / "s2" / "train" / "gsi_candidates.json"

# ── Band selection outputs ──────────────────────────────────────────────────────
# GSI: rank all (date × band) channels by per-crop SI_global, take top-K union
# RF:  rank all (date × band) channels by per-crop RF importance, take top-K union
SELECT_TOP_K_PER_CROP    = 20   # channels selected per crop before union
SELECT_GSI_DIRECT_JSON   = PROCESSED_DIR / "select_gsi_direct.json"
SELECT_GSI_DIRECT_BANDS  = PROCESSED_DIR / "select_gsi_direct_bands.txt"
SELECT_RF_DIRECT_JSON    = PROCESSED_DIR / "select_rf_direct.json"
SELECT_RF_DIRECT_BANDS   = PROCESSED_DIR / "select_rf_direct_bands.txt"

# ── RF selector hyperparameters ────────────────────────────────────────────────
RF_N_ESTIMATORS       = 500       # trees in the multi-class RF, per Asam et al. 2022 (rs14132981)
RF_MAX_PIXELS         = 1_000_000 # pixel sample cap (crop + rest) — no-cap was too slow on CPU
RF_IMPORTANCE_THRESH  = 0.10      # keep dates/bands with importance >= 10% of max
