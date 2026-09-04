"""
Learn-to-rank stage: 1,000 candidates -> LightGBM LambdaMART ranker -> top-100.

Formulates recommendation as "how should these candidates be ordered for this
user", not binary classification. Trained with LightGBM's lambdarank
objective (LambdaMART) against graded relevance labels derived from implicit
feedback (purchase > add_to_cart > wishlist > click > impression).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

RELEVANCE_LABELS = {
    "purchase": 4, "add_to_cart": 3, "wishlist": 2, "click": 1, "impression": 0,
}

FEATURE_COLUMNS = [
    "cf_score", "content_score", "popularity_score", "category_affinity",
    "item_ctr", "item_purchase_rate", "freshness", "price", "price_diff_from_pref",
    "brand_affinity", "hours_since_last_interaction", "user_n_purchases",
    "user_n_views", "is_weekend", "hour_of_day",
]


def build_training_frame(events: pd.DataFrame, items: pd.DataFrame,
                          user_affinity: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    """Every (user, item, event) impression becomes a labeled training row.
    Group id = user_id (LightGBM ranks candidates within each group)."""
    events = events.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["label"] = events["event"].map(RELEVANCE_LABELS).fillna(0).astype(int)

    # collapse to one row per (user,item,session): take the max label reached
    grouped = (events.groupby(["user_id", "item_id", "session_id"])
               .agg(label=("label", "max"), timestamp=("timestamp", "min"),
                    device=("device", "first"))
               .reset_index())

    df = grouped.merge(items, on="item_id", how="left", suffixes=("", "_item"))
    df = df.merge(user_affinity.rename(columns={"category": "cat_affinity_cat"}),
                   left_on=["user_id", "category"],
                   right_on=["user_id", "cat_affinity_cat"], how="left")
    df["category_affinity"] = df["affinity"].fillna(0.0)

    df = df.merge(users[["user_id", "price_sensitivity"]], on="user_id", how="left")
    df["price_diff_from_pref"] = (df["price"] - df["price"].median()) * df["price_sensitivity"].fillna(0.5)

    df["item_ctr"] = df.get("ctr", 0.0)
    df["item_purchase_rate"] = df.get("purchase_rate", 0.0)
    df["brand_affinity"] = 0.0  # placeholder: could be learned per-user-brand like category_affinity
    df["hours_since_last_interaction"] = 0.0
    df["user_n_purchases"] = 0.0
    df["user_n_views"] = 0.0
    df["is_weekend"] = df["timestamp"].dt.dayofweek >= 5
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["cf_score"] = 0.0
    df["content_score"] = 0.0
    df["popularity_score"] = df.get("total_weight", 0.0)
    df["freshness"] = df.get("freshness", 0.0)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def train_lightgbm_ranker(train_df: pd.DataFrame, valid_df: pd.DataFrame | None = None,
                            params: dict | None = None) -> lgb.Booster:
    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 10,
        "verbose": -1,
    }
    params = {**default_params, **(params or {})}

    def to_dataset(df):
        group_sizes = df.groupby("user_id", sort=False).size().values
        return lgb.Dataset(df[FEATURE_COLUMNS], label=df["label"], group=group_sizes)

    train_set = to_dataset(train_df)
    valid_sets = [train_set]
    valid_names = ["train"]
    if valid_df is not None and len(valid_df) > 0:
        valid_sets.append(to_dataset(valid_df))
        valid_names.append("valid")

    model = lgb.train(params, train_set, num_boost_round=200,
                       valid_sets=valid_sets, valid_names=valid_names,
                       callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])
    return model


def score_candidates(model: lgb.Booster, candidate_features: pd.DataFrame) -> np.ndarray:
    for col in FEATURE_COLUMNS:
        if col not in candidate_features.columns:
            candidate_features[col] = 0.0
    return model.predict(candidate_features[FEATURE_COLUMNS])
