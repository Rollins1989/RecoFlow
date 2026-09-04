"""
Temporal split: never randomly shuffle interactions for recommendation
evaluation — train on the past, evaluate on the future, to avoid leakage and
to reflect how the system is actually used in production.
"""
from __future__ import annotations

import pandas as pd


def temporal_split(events: pd.DataFrame, train_frac: float = 0.7,
                     valid_frac: float = 0.15):
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events = events.sort_values("timestamp")

    n = len(events)
    train_end = int(n * train_frac)
    valid_end = int(n * (train_frac + valid_frac))

    train = events.iloc[:train_end]
    valid = events.iloc[train_end:valid_end]
    test = events.iloc[valid_end:]

    train_cutoff = train["timestamp"].max()
    valid_cutoff = valid["timestamp"].max() if len(valid) else train_cutoff
    return {
        "train": train, "valid": valid, "test": test,
        "train_cutoff": train_cutoff, "valid_cutoff": valid_cutoff,
    }


def leave_one_out_split(events: pd.DataFrame):
    """For each user, hold out their single most-recent purchase (if any) as
    the test item; everything else is train. Standard CF eval protocol."""
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    purchases = events[events["event"] == "purchase"].sort_values("timestamp")

    test_rows = purchases.groupby("user_id").tail(1)
    train = events.drop(test_rows.index)
    return train, test_rows
