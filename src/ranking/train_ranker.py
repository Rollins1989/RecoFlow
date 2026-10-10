"""
Learn-to-rank stage: 1,000 candidates -> LightGBM LambdaMART ranker -> top-100.

Formulates recommendation as "how should these candidates be ordered for this
user", not binary classification. Trained with LightGBM's lambdarank
objective (LambdaMART) against graded relevance labels derived from implicit
feedback (purchase > add_to_cart > wishlist > click > impression).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

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
    """Build point-in-time training rows using only events before each session.

    Items must include catalogue metadata. The precomputed affinity argument is
    retained for call compatibility but is deliberately not used: a full-period
    aggregate would leak future interactions into earlier training examples.
    """
    from collections import defaultdict
    from src.features.feature_engineering import EVENT_WEIGHTS

    raw = events.copy()
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["label"] = raw["event"].map(RELEVANCE_LABELS).fillna(0).astype(int)
    grouped = (
        raw.groupby(["user_id", "item_id", "session_id"], dropna=False)
        .agg(label=("label", "max"), timestamp=("timestamp", "min"),
             device=("device", "first"))
        .reset_index()
        .sort_values(["timestamp", "user_id", "item_id"], kind="stable")
        .reset_index(drop=True)
    )

    product_cols = [c for c in
                    ("item_id", "category", "brand", "price", "created_ts")
                    if c in items.columns]
    products = items[product_cols].drop_duplicates("item_id").set_index("item_id")
    history = raw.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cursor = 0
    item_stats = defaultdict(lambda: {"impressions": 0, "clicks": 0,
                                      "purchases": 0, "total_weight": 0.0})
    user_category_weight = defaultdict(float)
    user_total_weight = defaultdict(float)
    rows = []

    for target in grouped.itertuples(index=False):
        # Strictly earlier timestamps only; equal-time events are excluded to
        # avoid order-dependent leakage among simultaneous prediction rows.
        while cursor < len(history) and history.iloc[cursor]["timestamp"] < target.timestamp:
            event = history.iloc[cursor]
            cursor += 1
            weight = float(EVENT_WEIGHTS.get(event["event"], 0.0))
            item_id = event["item_id"]
            stat = item_stats[item_id]
            stat["total_weight"] += weight
            if event["event"] == "impression":
                stat["impressions"] += 1
            elif event["event"] == "click":
                stat["clicks"] += 1
            elif event["event"] == "purchase":
                stat["purchases"] += 1
            product = products.loc[item_id] if item_id in products.index else None
            if product is not None and pd.notna(product.get("category")):
                key = (event["user_id"], product["category"])
                user_category_weight[key] += weight
                user_total_weight[event["user_id"]] += weight

        product = products.loc[target.item_id] if target.item_id in products.index else {}
        category = product.get("category", "")
        stat = item_stats[target.item_id]
        impressions = stat["impressions"]
        created = pd.to_datetime(product.get("created_ts"), errors="coerce")
        age_days = max(0.0, (target.timestamp - created).total_seconds() / 86400.0) if pd.notna(created) else 0.0
        total_user_weight = user_total_weight[target.user_id]
        affinity_weight = user_category_weight[(target.user_id, category)]
        rows.append({
            "user_id": target.user_id,
            "item_id": target.item_id,
            "session_id": target.session_id,
            "label": int(target.label),
            "timestamp": target.timestamp,
            "category": category,
            "price": product.get("price", 0.0),
            "category_affinity": affinity_weight / total_user_weight if total_user_weight > 0 else 0.0,
            "item_ctr": stat["clicks"] / impressions if impressions else 0.0,
            "item_purchase_rate": stat["purchases"] / impressions if impressions else 0.0,
            "popularity_score": stat["total_weight"],
            "freshness": float(np.exp(-age_days / 30.0)),
            "cf_score": 0.0,
            "content_score": 0.0,
            "price_diff_from_pref": 0.0,
            "brand_affinity": 0.0,
            "hours_since_last_interaction": 0.0,
            "user_n_purchases": 0.0,
            "user_n_views": 0.0,
            "is_weekend": 0.0,
            "hour_of_day": 0.0,
        })

    frame = pd.DataFrame(rows)
    for col in FEATURE_COLUMNS:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame.sort_values(["user_id", "timestamp"]).reset_index(drop=True)



def build_candidate_frame(pool: pd.DataFrame, items: pd.DataFrame,
                          user_affinity: pd.DataFrame) -> pd.DataFrame:
    """Build inference features using the same feature names and fallback policy as training.

    Retrieval currently exposes a blended score rather than calibrated per-source
    scores, so cf/content scores are deliberately zero instead of copying the
    blended score into both columns. Placeholder features stay zero in both paths
    until they are backed by real historical/online data.
    """
    if pool.empty:
        return pool.copy()
    features = pool.merge(items, on="item_id", how="left", suffixes=("", "_item"))
    if "category" in features.columns and not user_affinity.empty:
        affinity = user_affinity[["category", "affinity"]].drop_duplicates("category")
        features = features.merge(affinity, on="category", how="left")
    else:
        features["affinity"] = 0.0

    features["category_affinity"] = features["affinity"].fillna(0.0)
    features["cf_score"] = 0.0
    features["content_score"] = 0.0
    features["popularity_score"] = features.get(
        "total_weight", pd.Series(0.0, index=features.index)
    )
    features["item_ctr"] = features.get("ctr", pd.Series(0.0, index=features.index))
    features["item_purchase_rate"] = features.get(
        "purchase_rate", pd.Series(0.0, index=features.index)
    )
    for column in ("price_diff_from_pref", "brand_affinity",
                   "hours_since_last_interaction", "user_n_purchases",
                   "user_n_views", "is_weekend", "hour_of_day"):
        features[column] = 0.0
    for column in FEATURE_COLUMNS:
        if column not in features:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    return features


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
