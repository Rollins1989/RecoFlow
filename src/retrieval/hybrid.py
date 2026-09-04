"""
Hybrid retrieval: blend CF, content-based, trending, and profile-based
candidate generators into one deduplicated candidate pool (~1000 items),
tagging each candidate with its source(s) for the debug panel and for
bandit-driven source weighting later.
"""
from __future__ import annotations

import pandas as pd

from src.retrieval.cold_start import is_cold_start_user
from src.retrieval.collaborative_filtering import ImplicitALS
from src.retrieval.content_based import ContentBasedRecommender
from src.retrieval.popularity import PopularityRecommender

DEFAULT_POOL_SPLIT = {"cf": 400, "content": 300, "trending": 200, "personalized": 100}


class HybridRetriever:
    def __init__(self, pop: PopularityRecommender, cf: ImplicitALS,
                 content: ContentBasedRecommender,
                 pool_split: dict | None = None):
        self.pop = pop
        self.cf = cf
        self.content = content
        self.pool_split = pool_split or DEFAULT_POOL_SPLIT

    def get_candidates(self, user_id: int, user_history: list[int],
                        n_total: int | None = None) -> pd.DataFrame:
        split = self.pool_split
        frames = []

        if is_cold_start_user(user_history):
            # Cold-start user: skip CF entirely, lean on popularity + content.
            trend = self.pop.candidates(n=split["trending"] + split["cf"] // 2, trending=True)
            trend["source"] = "trending"
            frames.append(trend.rename(columns={"popularity_score": "score"}))
        else:
            cf_cands = self.cf.candidates(user_id, n=split["cf"])
            if not cf_cands.empty:
                cf_cands["source"] = "cf"
                frames.append(cf_cands.rename(columns={"cf_score": "score"}))

            trend = self.pop.candidates(n=split["trending"], trending=True)
            trend["source"] = "trending"
            frames.append(trend.rename(columns={"popularity_score": "score"}))

        if user_history:
            content_cands = self.content.candidates_for_user_history(
                user_history, n=split["content"])
            if not content_cands.empty:
                content_cands["source"] = "content"
                frames.append(content_cands.rename(columns={"content_score": "score"}))

        pop_cands = self.pop.candidates(n=split["personalized"], trending=False)
        pop_cands["source"] = "personalized"
        frames.append(pop_cands.rename(columns={"popularity_score": "score"}))

        pool = pd.concat(frames, ignore_index=True)
        # normalize scores per source to [0,1] before dedup so no one source dominates
        pool["score"] = pool.groupby("source")["score"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9))

        agg = (pool.groupby("item_id")
               .agg(score=("score", "max"),
                    sources=("source", lambda s: sorted(set(s))))
               .reset_index())
        agg = agg.sort_values("score", ascending=False)
        if n_total:
            agg = agg.head(n_total)
        return agg.reset_index(drop=True)
