#!/usr/bin/env python3
"""Collect and analyze seed-grid run metrics from MLflow experiment 16.

Outputs:
  seed_grid_raw.csv        — one row per finished child run
  seed_grid_summary.csv    — mean ± std per (exp, arch) across seeds
  (stdout)                 — summary tables
"""

import os, re, sys
import pandas as pd
import numpy as np

os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
from mlflow.tracking import MlflowClient

TRACKING_URI  = "https://mlflow-geoai.stelarea.com"
EXPERIMENT_ID = "16"
OUT_DIR       = os.path.join(os.path.dirname(__file__), "..", "documents", "thesis")

CROPS = ["corn", "rice", "winter_wheat", "alfalfa", "tomatoes", "grapes", "almonds", "walnuts"]

# Regex: exp_{expkey}[_seed{N}]_{arch}
# expkey = single_date | mt_base | gsi | rf
# seed   = optional integer
# arch   = deeplabv3plus_cbam | segformer
_RE = re.compile(
    r"^exp_(?P<exp>single_date|mt_base|gsi|rf)"
    r"(?:_seed(?P<seed>\d+))?"
    r"_(?P<arch>deeplabv3plus_cbam|segformer)$"
)


def _parse_name(run_name: str) -> dict | None:
    m = _RE.match(run_name)
    if not m:
        return None
    return {
        "exp":  m.group("exp"),
        "seed": int(m.group("seed")) if m.group("seed") else None,
        "arch": m.group("arch"),
    }


def collect(client: MlflowClient) -> pd.DataFrame:
    runs = client.search_runs(
        experiment_ids=[EXPERIMENT_ID],
        filter_string="status = 'FINISHED'",
        max_results=500,
    )
    print(f"Finished runs fetched: {len(runs)}")

    rows = []
    for r in runs:
        # Only child runs carry per-run metrics
        if not r.data.tags.get("mlflow.parentRunId"):
            continue
        parsed = _parse_name(r.info.run_name)
        if not parsed:
            continue

        m = r.data.metrics
        p = r.data.params

        row = {
            "run_id":   r.info.run_id,
            "run_name": r.info.run_name,
            "exp":      parsed["exp"],
            "seed":     parsed["seed"],
            "arch":     parsed["arch"],
            # Core test metrics
            "test_miou": m.get("test_miou"),
            "test_mf1":  m.get("test_mf1"),
            "test_oa":   m.get("test_oa"),
            # Best val (what early-stopping saw)
            "best_val_miou": m.get("best_val_miou"),
            "best_val_mf1":  m.get("best_val_mf1"),
            "best_val_oa":   m.get("best_val_oa"),
            # Training info
            "total_epochs":        m.get("total_epochs"),
            "train_time_min":      m.get("train_time_total_min"),
            "inference_ms_avg":    m.get("inference_time_ms_avg"),
            "n_channels":          int(p.get("in_channels", 0)),
        }
        # Per-class test IoU
        for crop in CROPS:
            row[f"iou_{crop}"] = m.get(f"test_iou_{crop}")
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Parsed child runs:     {len(df)}")
    if df.empty:
        return df

    # Runs without seed tag = baseline (seed=None); seed-grid runs have integer seed
    seed_runs = df[df["seed"].notna()].copy()
    base_runs = df[df["seed"].isna()].copy()
    print(f"  seed-grid runs: {len(seed_runs)}")
    print(f"  baseline runs:  {len(base_runs)}")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std per (exp, arch) across seeds."""
    seed_df = df[df["seed"].notna()].copy()
    if seed_df.empty:
        return pd.DataFrame()

    metric_cols = ["test_miou", "test_mf1", "test_oa"] + [f"iou_{c}" for c in CROPS]
    agg = {}
    for col in metric_cols:
        if col in seed_df.columns:
            agg[col + "_mean"] = seed_df.groupby(["exp", "arch"])[col].mean()
            agg[col + "_std"]  = seed_df.groupby(["exp", "arch"])[col].std()
    summary = pd.concat(agg, axis=1).reset_index()
    summary["n_seeds"] = seed_df.groupby(["exp", "arch"])["seed"].nunique().values
    return summary


def print_table(df: pd.DataFrame) -> None:
    """Print mean ± std table to stdout."""
    if df.empty:
        print("No seed-grid data to summarize.")
        return

    print("\n" + "=" * 80)
    print("SEED-GRID STABILITY — test mIoU / mF1 / OA  (mean ± std across seeds)")
    print("=" * 80)
    for _, row in df.sort_values(["exp", "arch"]).iterrows():
        miou = f"{row['test_miou_mean']:.4f} ± {row['test_miou_std']:.4f}"
        mf1  = f"{row['test_mf1_mean']:.4f} ± {row['test_mf1_std']:.4f}"
        oa   = f"{row['test_oa_mean']:.4f} ± {row['test_oa_std']:.4f}"
        print(f"  {row['exp']:<12} {row['arch']:<25}  n={int(row['n_seeds'])}  "
              f"mIoU={miou}  mF1={mf1}  OA={oa}")

    print("\n" + "-" * 80)
    print("PER-CLASS IoU mean ± std")
    print("-" * 80)
    for _, row in df.sort_values(["exp", "arch"]).iterrows():
        print(f"\n  {row['exp']} / {row['arch']}")
        for crop in CROPS:
            mean = row.get(f"iou_{crop}_mean")
            std  = row.get(f"iou_{crop}_std")
            if mean is not None and not np.isnan(mean):
                print(f"    {crop:<15} {mean:.4f} ± {std:.4f}")


def main():
    client = MlflowClient(TRACKING_URI)
    df     = collect(client)
    if df.empty:
        print("No data collected.")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "seed_grid_raw.csv")
    df.to_csv(raw_path, index=False)
    print(f"\nRaw data → {raw_path}")

    summary = summarize(df)
    if not summary.empty:
        sum_path = os.path.join(OUT_DIR, "seed_grid_summary.csv")
        summary.to_csv(sum_path, index=False)
        print(f"Summary  → {sum_path}")
        print_table(summary)

    # Also show individual seed-grid runs sorted
    seed_df = df[df["seed"].notna()].sort_values(["exp", "arch", "seed"])
    if not seed_df.empty:
        print("\n" + "=" * 80)
        print("INDIVIDUAL SEED RUNS")
        print("=" * 80)
        cols = ["exp", "arch", "seed", "test_miou", "test_mf1", "test_oa",
                "best_val_miou", "total_epochs", "n_channels"]
        print(seed_df[cols].to_string(index=False, float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
