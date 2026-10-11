"""Combine recommendation candidates from several retrieval strategies.

Each strategy contributes a scored list. Scores are normalised within their
source before duplicate items are merged, so one model's score scale does not
automatically outweigh another's.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.retrieval.cold_start import is_cold_start_user
from src.retrieval.collaborative_filtering import ImplicitALS
from src.retrieval.content_based import ContentBasedRecommender
from src.retrieval.popularity import PopularityRecommender

DEFAULT_POOL_SPLIT = {
    "cf": 400,
    "content": 300,
    "trending": 200,
    "personalized": 100,
}
RESULT_COLUMNS = ["item_id", "score", "sources"]


class HybridRetriever:
    """Build a single ranked candidate pool from multiple recommenders."""

    def __init__(
        self,
        pop: PopularityRecommender,
        cf: ImplicitALS,
        content: ContentBasedRecommender,
        pool_split: Mapping[str, int] | None = None,
    ) -> None:
        self.pop = pop
        self.cf = cf
        self.content = content
        self.pool_split = {**DEFAULT_POOL_SPLIT, **(pool_split or {})}

    def get_candidates(
        self,
        user_id: int,
        user_history: Sequence[int],
        n_total: int | None = None,
    ) -> pd.DataFrame:
        """Return unique item candidates ordered by their merged score."""
        if n_total is not None and n_total <= 0:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        candidates = self._collect_candidates(user_id, user_history)
        if not candidates:
            return pd.DataFrame(columns=RESULT_COLUMNS)

        pool = pd.concat(candidates, ignore_index=True)
        pool["score"] = pool.groupby("source")["score"].transform(
            self._normalise_scores
        )

        ranked = (
            pool.groupby("item_id", as_index=False)
            .agg(
                score=("score", "max"),
                sources=("source", lambda values: sorted(set(values))),
            )
            .sort_values("score", ascending=False)
            .reset_index(drop=True)
        )

        if n_total is not None:
            ranked = ranked.head(n_total).reset_index(drop=True)

        return ranked

    def _collect_candidates(
        self, user_id: int, user_history: Sequence[int]
    ) -> list[pd.DataFrame]:
        """Fetch candidate lists and standardise their score/source columns."""
        split = self.pool_split
        frames: list[pd.DataFrame] = []

        if is_cold_start_user(list(user_history)):
            # With little history, popularity is a safer fallback than CF.
            trending_count = split["trending"] + split["cf"] // 2
            frames.append(
                self._with_source(
                    self.pop.candidates(n=trending_count, trending=True),
                    score_column="popularity_score",
                    source="trending",
                )
            )
        else:
            collaborative = self.cf.candidates(user_id, n=split["cf"])
            if not collaborative.empty:
                frames.append(
                    self._with_source(
                        collaborative,
                        score_column="cf_score",
                        source="cf",
                    )
                )

            frames.append(
                self._with_source(
                    self.pop.candidates(n=split["trending"], trending=True),
                    score_column="popularity_score",
                    source="trending",
                )
            )

        if user_history:
            content_candidates = self.content.candidates_for_user_history(
                list(user_history), n=split["content"]
            )
            if not content_candidates.empty:
                frames.append(
                    self._with_source(
                        content_candidates,
                        score_column="content_score",
                        source="content",
                    )
                )

        frames.append(
            self._with_source(
                self.pop.candidates(
                    n=split["personalized"], trending=False
                ),
                score_column="popularity_score",
                source="personalized",
            )
        )

        return [frame for frame in frames if not frame.empty]

    @staticmethod
    def _with_source(
        candidates: pd.DataFrame, score_column: str, source: str
    ) -> pd.DataFrame:
        """Return a copy with the source-specific score renamed consistently."""
        result = candidates.rename(columns={score_column: "score"}).copy()
        result["source"] = source
        return result

    @staticmethod
    def _normalise_scores(scores: pd.Series) -> pd.Series:
        """Scale scores to [0, 1], including the all-equal-score case."""
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return pd.Series(0.0, index=scores.index)
        return (scores - scores.min()) / score_range
