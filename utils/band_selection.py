"""
3-Stage Band Selection for Semantic Segmentation
----------------------------------------------------------------------------
Stage 1 — Filter Preselection (cheap):
  Compute GSI + RF importance per band, rank by joint score, keep top K.

Stage 2 — CNN Forward Selection (wrapper):
  Iteratively add bands guided by a lightweight segmentation model's mIoU.

Stage 3 — Full Model Validation:
  Train full models (U-Net, DeepLabV3+, SegFormer) on each band subset
  and compare mIoU, training time, and GPU memory.
"""

import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm


# ============================================================
# STEP 1 — Load raster data and sample pixels
# ============================================================

def load_raster_data(image_path, label_path, sample_fraction=0.01, valid_classes=None):
    with rasterio.open(image_path) as src_img:
        img = src_img.read()
        band_count = img.shape[0]
        band_names = [f"B{i+1}" for i in range(band_count)]

    with rasterio.open(label_path) as src_lbl:
        lbl = src_lbl.read(1)

    img_2d = img.reshape(band_count, -1).T
    lbl_1d = lbl.flatten()

    mask = np.isfinite(img_2d).all(axis=1) & np.isfinite(lbl_1d)
    if valid_classes:
        mask &= np.isin(lbl_1d, valid_classes)
    else:
        mask &= (lbl_1d != 0)  # exclude background

    img_2d = img_2d[mask]
    lbl_1d = lbl_1d[mask]

    n = int(len(lbl_1d) * sample_fraction)
    idx = np.random.choice(len(lbl_1d), n, replace=False)
    img_sample = img_2d[idx]
    lbl_sample = lbl_1d[idx]

    df = pd.DataFrame(img_sample, columns=band_names)
    df.insert(0, "class_label", lbl_sample.astype(int))

    print(f"Sampled {len(df):,} pixels from {band_count} bands.")
    return df, band_names


# ============================================================
# STEP 2 — Compute Global Separation Index (GSI)
# ============================================================

def calculate_gsi(data: pd.DataFrame, class_col: str) -> pd.DataFrame:
    bands = [col for col in data.columns if col != class_col]
    classes = [c for c in data[class_col].unique() if c != 0]
    gsi_results = []

    for s in classes:
        class_s = data[data[class_col] == s]
        mean_s = class_s[bands].mean()
        std_s = class_s[bands].std()

        si_values = []
        for o in [c for c in classes if c != s]:
            class_o = data[data[class_col] == o]
            mean_o = class_o[bands].mean()
            std_o = class_o[bands].std()

            si = abs(mean_s - mean_o) / (1.96 * (std_s + std_o))
            si_values.append(si)

        gsi = pd.concat(si_values, axis=1).mean(axis=1)
        gsi_results.append(gsi.rename(s))

    gsi_df = pd.DataFrame(gsi_results).T
    gsi_df.index = bands
    return gsi_df


# ============================================================
# STAGE 1 — RF Importance & Joint Score
# ============================================================

def compute_rf_importance(df: pd.DataFrame, class_col: str, n_estimators: int = 200) -> pd.Series:
    """
    Compute Random Forest feature importance (mean decrease in impurity)
    for each spectral band.

    Returns a Series indexed by band name, normalized to [0, 1].
    """
    bands = [col for col in df.columns if col != class_col]
    X = df[bands].values
    y = df[class_col].values

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    importance = pd.Series(clf.feature_importances_, index=bands)
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-9)
    return importance.sort_values(ascending=False)


def compute_joint_score(gsi_mean: pd.Series, rf_importance: pd.Series,
                        alpha: float = 0.5) -> pd.Series:
    """
    Compute a joint ranking score combining GSI and RF importance:

        Score = alpha * GSI_norm + (1 - alpha) * RF_importance_norm

    Both inputs are normalized to [0, 1] before combining.

    Parameters
    ----------
    gsi_mean      : Series indexed by band name (mean GSI across classes)
    rf_importance : Series indexed by band name (normalized RF importance)
    alpha         : weight for GSI (default 0.5 = equal weighting)

    Returns
    -------
    Series indexed by band name, sorted descending by joint score.
    """
    gsi_norm = (gsi_mean - gsi_mean.min()) / (gsi_mean.max() - gsi_mean.min() + 1e-9)

    all_bands = gsi_norm.index.union(rf_importance.index)
    gsi_norm = gsi_norm.reindex(all_bands, fill_value=0.0)
    rf_norm  = rf_importance.reindex(all_bands, fill_value=0.0)

    joint = alpha * gsi_norm + (1 - alpha) * rf_norm
    return joint.sort_values(ascending=False)


def select_top_k(joint_score: pd.Series, k: int = 25) -> list:
    """Return the top K band names by joint score."""
    return joint_score.head(k).index.tolist()


# ============================================================
# STEP 3 — Cluster correlated bands & select representatives
# ============================================================

def cluster_correlated_bands_and_select(df_bands: pd.DataFrame,
                                        gsi_series: pd.Series,
                                        corr_thresh: float = 0.9,
                                        n_per_cluster: int = 1,
                                        max_bands: int = None,
                                        allow_variable_per_cluster: bool = True):
    """
    Cluster correlated bands and select representative ones based on GSI.

    Instead of deleting correlated bands, this groups them and
    keeps the highest-GSI band(s) from each group.
    """
    bands = gsi_series.index.tolist()
    corr = df_bands[bands].corr().abs()
    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)

    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method='average')
    dist_thresh = 1.0 - corr_thresh
    cluster_ids = fcluster(Z, t=dist_thresh, criterion='distance')

    clusters = {}
    for b, cid in zip(bands, cluster_ids):
        clusters.setdefault(cid, []).append(b)

    selected = []
    for cid, members in clusters.items():
        members_sorted = sorted(members, key=lambda b: gsi_series[b], reverse=True)
        if allow_variable_per_cluster:
            k = max(1, int(np.ceil(len(members) * n_per_cluster)))
        else:
            k = n_per_cluster
        k = min(k, len(members_sorted))
        selected.extend(members_sorted[:k])

    selected_sorted = sorted(set(selected), key=lambda b: gsi_series[b], reverse=True)
    if max_bands is not None and len(selected_sorted) > max_bands:
        selected_sorted = selected_sorted[:max_bands]

    print(f"Clustered into {len(clusters)} groups, selected {len(selected_sorted)} representative bands.")
    return selected_sorted, clusters


# ============================================================
# STEP 4 — Evaluate candidate band combinations (with tqdm)
# ============================================================

def evaluate_combinations(df: pd.DataFrame, class_col: str, band_list, max_size=5, cv_folds=3):
    """
    Evaluate candidate band combinations using RandomForest + CV, with progress bar.
    """
    combos = []
    for r in range(2, min(max_size, len(band_list)) + 1):
        combos += list(combinations(band_list, r))

    results = []
    y = df[class_col]

    print(f"Evaluating {len(combos)} band combinations...")
    for combo in tqdm(combos, desc="Evaluating combinations", ncols=100):
        X = df[list(combo)].values
        Xs = StandardScaler().fit_transform(X)
        clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
        score = cross_val_score(clf, Xs, y, cv=cv_folds, scoring=make_scorer(f1_score, average="macro"))
        results.append({
            "combo": combo,
            "mean_f1": score.mean(),
            "std_f1": score.std()
        })

    results_df = pd.DataFrame(results).sort_values("mean_f1", ascending=False)
    return results_df


# ============================================================
# STEP 5 — Visualization
# ============================================================

def plot_gsi(gsi_df: pd.DataFrame, save_path=None):
    plt.figure(figsize=(10, 6))
    gsi_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Global Separation Index (GSI) per Band")
    plt.xlabel("Spectral Band")
    plt.ylabel("GSI Value")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(title="Main Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# ============================================================
# STEP 6 — Main execution
# ============================================================

if __name__ == "__main__":
    image_path = "sentinel_stack.tif"
    label_path = "labels.tif"
    sample_fraction = 0.005  # 0.5% pixels

    df, band_names = load_raster_data(image_path, label_path, sample_fraction)
    class_col = "class_label"

    print("Calculating GSI...")
    gsi_df = calculate_gsi(df, class_col)
    gsi_mean = gsi_df.mean(axis=1).sort_values(ascending=False)

    print("\nTop 10 bands by mean GSI:")
    print(gsi_mean.head(10))
    plot_gsi(gsi_df, save_path="gsi_plot.png")

    # ✅ Cluster-based band selection
    selected_bands, clusters = cluster_correlated_bands_and_select(
        df_bands=df,
        gsi_series=gsi_mean,
        corr_thresh=0.9,       # allow moderate correlation
        n_per_cluster=1,
        max_bands=None,
        allow_variable_per_cluster=True
    )

    print("\nSelected representative bands:", selected_bands)
    print("Cluster composition:")
    for cid, members in clusters.items():
        print(f"Cluster {cid}: {members}")

    print("\nEvaluating candidate combinations...")
    results_df = evaluate_combinations(df, class_col, selected_bands, max_size=5, cv_folds=3)

    print("\nTop 10 band combinations:")
    print(results_df.head(10))
    results_df.to_csv("band_combination_results.csv", index=False)
    print("\n✅ Results saved to 'band_combination_results.csv'")