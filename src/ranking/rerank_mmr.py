"""
Re-ranking stage: ranking_score + diversity (MMR) + freshness + business rules.

Solves the "iPhone, iPhone case, iPhone charger, iPhone cable..." problem —
technically high-scoring, terrible recommendations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mmr_rerank(candidates: pd.DataFrame, embeddings: np.ndarray, item_id_to_idx: dict,
               lambda_relevance: float = 0.7, k: int = 10) -> pd.DataFrame:
    """
    Maximal Marginal Relevance: iteratively pick the candidate that maximizes
        lambda * relevance(i) - (1 - lambda) * max_sim(i, already_selected)
    `candidates` must have columns [item_id, score] sorted or not (we re-sort).
    """
    cands = candidates.copy().reset_index(drop=True)
    if cands.empty:
        return cands

    scores = cands["score"].to_numpy(dtype=float)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)

    selected_idx: list[int] = []
    remaining = list(range(len(cands)))

    while remaining and len(selected_idx) < k:
        best_i, best_val = None, -1e18
        for i in remaining:
            item_id = cands.loc[i, "item_id"]
            emb_i = embeddings[item_id_to_idx[item_id]] if item_id in item_id_to_idx else None
            if emb_i is None or not selected_idx:
                max_sim = 0.0
            else:
                sims = []
                for j in selected_idx:
                    jid = cands.loc[j, "item_id"]
                    emb_j = embeddings[item_id_to_idx[jid]] if jid in item_id_to_idx else None
                    if emb_j is not None:
                        sims.append(float(np.dot(emb_i, emb_j)))
                max_sim = max(sims) if sims else 0.0
            val = lambda_relevance * scores[i] - (1 - lambda_relevance) * max_sim
            if val > best_val:
                best_val, best_i = val, i
        selected_idx.append(best_i)
        remaining.remove(best_i)

    return cands.loc[selected_idx].reset_index(drop=True)


def apply_freshness_boost(df: pd.DataFrame, freshness_col: str = "freshness",
                            weight: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    df["score"] = df["score"] + weight * df.get(freshness_col, 0.0)
    return df


def apply_business_rules(df: pd.DataFrame, exclude_purchased: set[int] | None = None,
                           exclude_out_of_stock: set[int] | None = None,
                           max_per_category: int | None = None,
                           category_col: str = "category") -> pd.DataFrame:
    """Filter already-purchased/OOS items, cap per-category exposure."""
    df = df.copy()
    if exclude_purchased:
        df = df[~df["item_id"].isin(exclude_purchased)]
    if exclude_out_of_stock:
        df = df[~df["item_id"].isin(exclude_out_of_stock)]
    if max_per_category and category_col in df.columns:
        df = (df.sort_values("score", ascending=False)
              .groupby(category_col, group_keys=False)
              .apply(lambda g: g.head(max_per_category)))
    return df.sort_values("score", ascending=False).reset_index(drop=True)
