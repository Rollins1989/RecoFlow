"""Baseline 1 — Popularity. Most-purchased / most-clicked / trending products."""
from __future__ import annotations

import numpy as np
import pandas as pd


class PopularityRecommender:
    """Two variants: all-time popularity and recency-weighted 'trending'."""

    def __init__(self, half_life_days: float = 7.0):
        self.half_life_days = half_life_days
        self.scores_: pd.Series | None = None
        self.trending_: pd.Series | None = None

    def fit(self, events: pd.DataFrame, as_of: pd.Timestamp | None = None):
        events = events.copy()
        events["timestamp"] = pd.to_datetime(events["timestamp"])
        if as_of is not None:
            events = events[events["timestamp"] < as_of]

        weight_map = {"purchase": 5.0, "add_to_cart": 3.0, "wishlist": 2.0, "click": 1.0}
        events["w"] = events["event"].map(weight_map).fillna(0.0)

        # all-time popularity
        self.scores_ = events.groupby("item_id")["w"].sum().sort_values(ascending=False)

        # recency-weighted trending (exponential decay by half-life)
        now = as_of or events["timestamp"].max()
        age_days = (now - events["timestamp"]).dt.total_seconds() / 86400.0
        decay = np.power(0.5, age_days / self.half_life_days)
        events["decayed_w"] = events["w"] * decay
        self.trending_ = events.groupby("item_id")["decayed_w"].sum().sort_values(ascending=False)
        return self

    def recommend(self, k: int = 10, trending: bool = False) -> list[tuple[int, float]]:
        series = self.trending_ if trending else self.scores_
        top = series.head(k)
        return list(zip(top.index.tolist(), top.values.tolist()))

    def candidates(self, n: int = 200, trending: bool = False) -> pd.DataFrame:
        series = self.trending_ if trending else self.scores_
        top = series.head(n)
        return pd.DataFrame({"item_id": top.index, "popularity_score": top.values})
