"""
Offline evaluation harness: runs each retrieval/ranking stage against the
temporal test set and produces the comparison table from docs/DESIGN.md §6.

    Model                  Recall@10  NDCG@10  Coverage  Diversity
    Popularity
    Content
    CF
    Hybrid
    Ranker
    Ranker + Re-ranking
"""
from __future__ import annotations

import pandas as pd

from src.evaluation.metrics import (catalog_coverage, evaluate_ranked_lists,
                                     intra_list_diversity)


def build_relevance_lookup(test_events: pd.DataFrame) -> dict[int, set[int]]:
    """user_id -> set of item_ids the user actually engaged with positively
    (click/wishlist/add_to_cart/purchase) in the held-out test period."""
    positive = test_events[test_events["event"].isin(
        ["click", "wishlist", "add_to_cart", "purchase"])]
    return positive.groupby("user_id")["item_id"].apply(set).to_dict()


def evaluate_recommender(recommend_fn, user_ids: list[int], relevance_lookup: dict,
                          embeddings=None, item_id_to_idx=None,
                          k: int = 10, total_catalog_size: int = 0) -> dict:
    """`recommend_fn(user_id) -> list[item_id]` (already top-k ordered)."""
    per_user_rel, per_user_n_rel = {}, {}
    all_recommended, diversities = set(), []

    for user_id in user_ids:
        relevant_items = relevance_lookup.get(user_id, set())
        if not relevant_items:
            continue
        recs = recommend_fn(user_id)
        rels = [1.0 if item in relevant_items else 0.0 for item in recs]
        per_user_rel[user_id] = rels
        per_user_n_rel[user_id] = len(relevant_items)
        all_recommended.update(recs)

        if embeddings is not None and item_id_to_idx is not None and recs:
            diversities.append(intra_list_diversity(recs, embeddings, item_id_to_idx))

    metrics = evaluate_ranked_lists(per_user_rel, per_user_n_rel, k=k)
    metrics["coverage"] = catalog_coverage(all_recommended, total_catalog_size)
    metrics["diversity"] = float(sum(diversities) / len(diversities)) if diversities else 0.0
    metrics["n_users_evaluated"] = len(per_user_rel)
    return metrics


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    cols = [c for c in ["recall@10", "ndcg@10", "map@10", "mrr", "coverage", "diversity"]
            if c in df.columns]
    return df[cols].round(4)
