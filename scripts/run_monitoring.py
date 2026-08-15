from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


REFERENCE_PATH = Path("data/train_dataset.parquet")
CURRENT_PATH = Path("data/current_batch.parquet")
OUTPUT_PATH = Path("data/monitoring_report.csv")


def _safe_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.replace([np.inf, -np.inf], np.nan).dropna()
    actual = actual.replace([np.inf, -np.inf], np.nan).dropna()

    if expected.empty or actual.empty:
        return 0.0

    # Квантили по reference
    try:
        _, bin_edges = pd.qcut(expected, q=bins, retbins=True, duplicates="drop")
    except ValueError:
        return 0.0

    if len(bin_edges) < 3:
        return 0.0

    expected_bins = pd.cut(expected, bins=bin_edges, include_lowest=True)
    actual_bins = pd.cut(actual, bins=bin_edges, include_lowest=True)

    expected_dist = expected_bins.value_counts(normalize=True, sort=False)
    actual_dist = actual_bins.value_counts(normalize=True, sort=False)

    expected_dist, actual_dist = expected_dist.align(actual_dist, fill_value=0.0001)

    psi = ((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)).sum()
    return float(psi)


def _psi_level(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "warning"
    return "drift"


def main() -> None:
    if not REFERENCE_PATH.exists():
        print(f"Reference dataset not found: {REFERENCE_PATH}")
        return

    if not CURRENT_PATH.exists():
        print(f"Current batch not found: {CURRENT_PATH}")
        print("Put fresh scored batch into data/current_batch.parquet")
        return

    ref = pd.read_parquet(REFERENCE_PATH)
    cur = pd.read_parquet(CURRENT_PATH)

    drop_cols = ["Target", "ApplicationId", "ec.ApplicationId", "BirthDate"]
    ref = ref.drop(columns=drop_cols, errors="ignore")
    cur = cur.drop(columns=drop_cols, errors="ignore")

    common_cols = [c for c in ref.columns if c in cur.columns]
    if not common_cols:
        print("No common columns found between reference and current batch.")
        return

    report_rows = []
    for col in common_cols:
        if col == "Target":
            continue

        ref_col = ref[col]
        cur_col = cur[col]

        if pd.api.types.is_numeric_dtype(ref_col) and pd.api.types.is_numeric_dtype(cur_col):
            psi = _safe_psi(ref_col, cur_col)
            report_rows.append(
                {
                    "feature": col,
                    "type": "numeric",
                    "psi": psi,
                    "status": _psi_level(psi),
                }
            )
        else:
            # Для категориальных: PSI по top categories + other
            ref_s = ref_col.astype(str).fillna("Unknown")
            cur_s = cur_col.astype(str).fillna("Unknown")

            top_categories = ref_s.value_counts(normalize=True).head(10).index.tolist()
            ref_bucket = ref_s.where(ref_s.isin(top_categories), other="__OTHER__")
            cur_bucket = cur_s.where(cur_s.isin(top_categories), other="__OTHER__")

            categories = sorted(set(ref_bucket.unique()) | set(cur_bucket.unique()))
            ref_dist = ref_bucket.value_counts(normalize=True).reindex(categories, fill_value=0.0001)
            cur_dist = cur_bucket.value_counts(normalize=True).reindex(categories, fill_value=0.0001)

            psi = float(((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum())
            report_rows.append(
                {
                    "feature": col,
                    "type": "categorical",
                    "psi": psi,
                    "status": _psi_level(psi),
                }
            )

    report = pd.DataFrame(report_rows).sort_values("psi", ascending=False)
    report.to_csv(OUTPUT_PATH, index=False)

    print(report.to_string(index=False))
    print(f"\nSaved to {OUTPUT_PATH}")

    drift_features = report[report["status"] == "drift"]
    if not drift_features.empty:
        print("\nDrift detected in features:")
        print(drift_features["feature"].tolist())


if __name__ == "__main__":
    main()