"""
Feature engineering shared by retrieval, ranking, and the online feature store
(src/feature_store). Keeping this pure-pandas/numpy means the exact same code
path can run offline (training) and be mirrored by Feast online features
(training-serving consistency).
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd

EVENT_WEIGHTS = {
    "impression": 0.1, "click": 1.0, "search": 0.2, "category_visit": 0.3,
    "wishlist": 2.0, "add_to_cart": 3.0, "remove_from_cart": -1.5, "purchase": 5.0,
}


def add_implicit_weight(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["weight"] = events["event"].map(EVENT_WEIGHTS).fillna(0.0)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    return events


def build_item_features(products: pd.DataFrame, events: pd.DataFrame,
                         as_of: datetime | None = None) -> pd.DataFrame:
    """Popularity, freshness, CTR, purchase-rate per item, as of a given time
    (temporal-safe: only uses events strictly before `as_of`)."""
    events = add_implicit_weight(events)
    if as_of is not None:
        events = events[events["timestamp"] < as_of]

    agg = events.groupby("item_id").agg(
        impressions=("event", lambda s: (s == "impression").sum()),
        clicks=("event", lambda s: (s == "click").sum()),
        purchases=("event", lambda s: (s == "purchase").sum()),
        add_to_carts=("event", lambda s: (s == "add_to_cart").sum()),
        total_weight=("weight", "sum"),
        last_interaction=("timestamp", "max"),
    ).reset_index()

    items = products.merge(agg, on="item_id", how="left").fillna({
        "impressions": 0, "clicks": 0, "purchases": 0, "add_to_carts": 0,
        "total_weight": 0.0,
    })
    items["ctr"] = items["clicks"] / items["impressions"].replace(0, np.nan)
    items["ctr"] = items["ctr"].fillna(0.0)
    items["purchase_rate"] = items["purchases"] / items["impressions"].replace(0, np.nan)
    items["purchase_rate"] = items["purchase_rate"].fillna(0.0)

    now = as_of or events["timestamp"].max() or pd.Timestamp.now()
    created = pd.to_datetime(items["created_ts"])
    items["age_days"] = (pd.Timestamp(now) - created).dt.days.clip(lower=0)
    items["freshness"] = np.exp(-items["age_days"] / 30.0)  # decays over ~30 days
    return items


def build_user_category_affinity(events: pd.DataFrame, products: pd.DataFrame,
                                   as_of: datetime | None = None) -> pd.DataFrame:
    events = add_implicit_weight(events)
    if as_of is not None:
        events = events[events["timestamp"] < as_of]
    merged = events.merge(products[["item_id", "category", "brand", "price"]],
                           on="item_id", how="left")
    affinity = (merged.groupby(["user_id", "category"])["weight"].sum()
                .reset_index())
    # normalize per user to [0, 1]
    totals = affinity.groupby("user_id")["weight"].transform("sum").replace(0, np.nan)
    affinity["affinity"] = (affinity["weight"] / totals).fillna(0.0)
    return affinity


def build_user_features(users: pd.DataFrame, events: pd.DataFrame,
                          products: pd.DataFrame,
                          as_of: datetime | None = None) -> pd.DataFrame:
    events = add_implicit_weight(events)
    if as_of is not None:
        events = events[events["timestamp"] < as_of]
    merged = events.merge(products[["item_id", "price"]], on="item_id", how="left")

    agg = merged.groupby("user_id").agg(
        n_views=("event", lambda s: (s == "click").sum()),
        n_purchases=("event", lambda s: (s == "purchase").sum()),
        n_events=("event", "count"),
        avg_price_seen=("price", "mean"),
        last_active=("timestamp", "max"),
    ).reset_index()

    out = users.merge(agg, on="user_id", how="left").fillna({
        "n_views": 0, "n_purchases": 0, "n_events": 0, "avg_price_seen": 0.0,
    })
    out["is_cold_start"] = out["n_events"] < 3
    return out


def context_features(timestamp: pd.Timestamp, device: str | None = None) -> dict:
    ts = pd.Timestamp(timestamp)
    return {
        "hour_of_day": ts.hour,
        "day_of_week": ts.dayofweek,
        "is_weekend": int(ts.dayofweek >= 5),
        "device": device or "unknown",
    }
