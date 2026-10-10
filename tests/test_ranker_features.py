"""Regression tests for ranker serving-feature construction."""
import pandas as pd

from src.ranking.train_ranker import FEATURE_COLUMNS, build_candidate_frame


def test_candidate_features_use_real_item_signals_and_neutral_fallbacks():
    pool = pd.DataFrame({
        "item_id": [1, 2],
        "score": [0.9, 0.4],
        "sources": [["cf", "content"], ["trending"]],
    })
    items = pd.DataFrame({
        "item_id": [1, 2],
        "category": ["books", "games"],
        "price": [100.0, 200.0],
        "total_weight": [5.0, 2.0],
        "ctr": [0.2, 0.0],
        "purchase_rate": [0.1, 0.0],
        "freshness": [0.8, 0.4],
    })
    affinity = pd.DataFrame({
        "category": ["books"],
        "affinity": [0.75],
    })

    result = build_candidate_frame(pool, items, affinity)

    assert result["item_id"].tolist() == [1, 2]
    assert result["category_affinity"].tolist() == [0.75, 0.0]
    assert result["popularity_score"].tolist() == [5.0, 2.0]
    assert result["item_ctr"].tolist() == [0.2, 0.0]
    assert result["item_purchase_rate"].tolist() == [0.1, 0.0]
    assert (result["cf_score"] == 0.0).all()
    assert (result["content_score"] == 0.0).all()
    assert set(FEATURE_COLUMNS).issubset(result.columns)
    assert not result[FEATURE_COLUMNS].isna().any().any()


def test_empty_candidate_pool_is_returned_empty():
    result = build_candidate_frame(
        pd.DataFrame(columns=["item_id", "score"]),
        pd.DataFrame(columns=["item_id"]),
        pd.DataFrame(columns=["category", "affinity"]),
    )
    assert result.empty
