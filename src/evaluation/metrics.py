"""Ranking + business + diversity metrics for offline evaluation."""
from __future__ import annotations

import numpy as np


def dcg_at_k(relevances: list[float], k: int) -> float:
    relevances = np.asarray(relevances[:k], dtype=float)
    if relevances.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def average_precision_at_k(relevances: list[float], k: int) -> float:
    """Binary relevance (relevance > 0) average precision."""
    hits, sum_prec = 0, 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        if rel > 0:
            hits += 1
            sum_prec += hits / i
    n_relevant = sum(1 for r in relevances if r > 0)
    return sum_prec / n_relevant if n_relevant > 0 else 0.0


def reciprocal_rank(relevances: list[float]) -> float:
    for i, rel in enumerate(relevances, start=1):
        if rel > 0:
            return 1.0 / i
    return 0.0


def recall_at_k(relevances: list[float], n_relevant_total: int, k: int) -> float:
    if n_relevant_total == 0:
        return 0.0
    hits = sum(1 for r in relevances[:k] if r > 0)
    return hits / n_relevant_total


def intra_list_diversity(item_ids: list[int], embeddings: np.ndarray,
                           item_id_to_idx: dict) -> float:
    """1 - average pairwise cosine similarity among recommended items."""
    idxs = [item_id_to_idx[i] for i in item_ids if i in item_id_to_idx]
    if len(idxs) < 2:
        return 0.0
    vecs = embeddings[idxs]
    sim_matrix = vecs @ vecs.T
    n = len(idxs)
    off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
    avg_sim = off_diag_sum / (n * (n - 1))
    return float(1 - avg_sim)


def catalog_coverage(recommended_item_ids: set[int], total_catalog_size: int) -> float:
    if total_catalog_size == 0:
        return 0.0
    return len(recommended_item_ids) / total_catalog_size


def evaluate_ranked_lists(per_user_relevances: dict[int, list[float]],
                            per_user_n_relevant: dict[int, int] | None = None,
                            k: int = 10) -> dict:
    """per_user_relevances: {user_id: [relevance of item at rank 1, 2, ...]}"""
    ndcgs, maps, mrrs, recalls = [], [], [], []
    for user_id, rels in per_user_relevances.items():
        ndcgs.append(ndcg_at_k(rels, k))
        maps.append(average_precision_at_k(rels, k))
        mrrs.append(reciprocal_rank(rels))
        if per_user_n_relevant:
            recalls.append(recall_at_k(rels, per_user_n_relevant.get(user_id, 0), k))

    result = {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"map@{k}": float(np.mean(maps)) if maps else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
    }
    if recalls:
        result[f"recall@{k}"] = float(np.mean(recalls))
    return result
