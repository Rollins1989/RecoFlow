"""
Drift detection: training distribution vs. rolling current distribution for
features, predictions, and labels. On threshold breach, raises an alert that
(in production) would trigger the retrain -> offline eval -> approval ->
promote pipeline described in docs/DESIGN.md §7.

Uses population stability index (PSI) — a standard, dependency-light drift
metric — plus a simple mean-shift check for numeric features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_ALERT_THRESHOLD = 0.2  # > 0.2 is conventionally "significant drift"


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected, actual = np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)

    def _hist(values):
        counts, _ = np.histogram(values, bins=breakpoints)
        pct = counts / max(len(values), 1)
        return np.clip(pct, 1e-6, None)

    e_pct, a_pct = _hist(expected), _hist(actual)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def detect_feature_drift(train_df: pd.DataFrame, current_df: pd.DataFrame,
                           numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        if col not in train_df.columns or col not in current_df.columns:
            continue
        psi = population_stability_index(train_df[col].dropna(), current_df[col].dropna())
        rows.append({
            "feature": col, "psi": round(psi, 4),
            "alert": psi > PSI_ALERT_THRESHOLD,
            "train_mean": float(train_df[col].mean()),
            "current_mean": float(current_df[col].mean()),
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def detect_ctr_drift(train_events: pd.DataFrame, current_events: pd.DataFrame) -> dict:
    def ctr(events):
        impressions = (events["event"] == "impression").sum()
        clicks = (events["event"] == "click").sum()
        return clicks / impressions if impressions else 0.0

    train_ctr, current_ctr = ctr(train_events), ctr(current_events)
    rel_change = abs(current_ctr - train_ctr) / train_ctr if train_ctr else 0.0
    return {
        "train_ctr": round(train_ctr, 4), "current_ctr": round(current_ctr, 4),
        "relative_change": round(rel_change, 4), "alert": rel_change > 0.25,
    }


def should_trigger_retrain(feature_drift_df: pd.DataFrame, ctr_drift: dict) -> bool:
    return bool(feature_drift_df["alert"].any()) or ctr_drift["alert"]
