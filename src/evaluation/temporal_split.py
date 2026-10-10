"""Chronological train/validation/test splits for interaction data.

Rows are sorted by timestamp with a stable sort. Callers must fit models and
historical features on the training partition only; splitting the events does
not automatically prevent leakage in separately precomputed features.
"""
from __future__ import annotations

import pandas as pd


def temporal_split(events: pd.DataFrame, train_frac: float = 0.7,
                   valid_frac: float = 0.15) -> dict:
    """Split interactions by time without shuffling.

    Fractions refer to rows after chronological sorting. The final partition
    receives any rounding remainder. Raises clear errors for invalid inputs.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    if not 0 <= valid_frac < 1:
        raise ValueError("valid_frac must be between 0 and 1")
    if train_frac + valid_frac >= 1:
        raise ValueError("train_frac + valid_frac must be less than 1")
    if events.empty:
        raise ValueError("events must contain at least one row")
    if "timestamp" not in events.columns:
        raise ValueError("events must include a timestamp column")

    ordered = events.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise ValueError("timestamp contains missing or invalid values")
    ordered = ordered.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    valid_end = int(n_rows * (train_frac + valid_frac))
    if train_end < 1:
        raise ValueError("training partition would be empty; provide more rows")
    if valid_frac > 0 and valid_end <= train_end:
        raise ValueError("validation partition would be empty; provide more rows")

    train = ordered.iloc[:train_end].copy()
    valid = ordered.iloc[train_end:valid_end].copy()
    test = ordered.iloc[valid_end:].copy()
    if test.empty:
        raise ValueError("test partition would be empty; provide more rows")

    train_cutoff = train["timestamp"].max()
    valid_cutoff = valid["timestamp"].max() if not valid.empty else train_cutoff
    return {
        "train": train,
        "valid": valid,
        "test": test,
        "train_cutoff": train_cutoff,
        "valid_cutoff": valid_cutoff,
        "test_cutoff": test["timestamp"].min(),
    }


def leave_one_out_split(events: pd.DataFrame):
    """Hold out each user's latest purchase, if present, as a test interaction."""
    if "timestamp" not in events.columns or "user_id" not in events.columns:
        raise ValueError("events must include user_id and timestamp columns")
    ordered = events.copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], errors="coerce")
    if ordered["timestamp"].isna().any():
        raise ValueError("timestamp contains missing or invalid values")
    purchases = ordered[ordered["event"] == "purchase"].sort_values(
        "timestamp", kind="mergesort"
    )
    test_rows = purchases.groupby("user_id", sort=False).tail(1)
    train = ordered.drop(index=test_rows.index).copy()
    return train, test_rows.copy()
