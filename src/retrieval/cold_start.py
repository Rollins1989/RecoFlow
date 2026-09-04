"""
Cold-start strategy (see docs/DESIGN.md §5).

New user  -> popularity + trending + context + first-session behavior.
New product -> content embedding + category + brand + price bracket, boosted
               by the freshness term in re-ranking until it earns interactions.
"""
from __future__ import annotations

import pandas as pd

COLD_START_EVENT_THRESHOLD = 3
NEW_PRODUCT_AGE_DAYS = 14


def is_cold_start_user(user_history: list[int]) -> bool:
    return len(user_history) < COLD_START_EVENT_THRESHOLD


def is_cold_start_item(item_row: pd.Series) -> bool:
    return item_row.get("age_days", 0) <= NEW_PRODUCT_AGE_DAYS or item_row.get("impressions", 0) < 5


def cold_start_user_candidates(pop_recommender, n: int = 200) -> pd.DataFrame:
    """First-session fallback: pure trending, re-scored after every new event
    (call again once the session has >=1 click, which flips is_cold_start_user
    to False as soon as enough events accumulate)."""
    return pop_recommender.candidates(n=n, trending=True)


def boost_new_products(candidates: pd.DataFrame, items: pd.DataFrame,
                        boost_factor: float = 1.5) -> pd.DataFrame:
    """Multiplicatively boost candidates that are new products, so they get a
    fair shot at surfacing despite having little to no interaction history."""
    merged = candidates.merge(items[["item_id", "age_days", "impressions"]],
                               on="item_id", how="left")
    is_new = (merged["age_days"] <= NEW_PRODUCT_AGE_DAYS) | (merged["impressions"] < 5)
    merged.loc[is_new, "score"] = merged.loc[is_new, "score"] * boost_factor
    return merged.drop(columns=["age_days", "impressions"])
