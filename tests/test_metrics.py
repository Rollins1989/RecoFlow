import numpy as np

from src.evaluation.metrics import (average_precision_at_k, catalog_coverage,
                                     ndcg_at_k, reciprocal_rank, recall_at_k)


def test_ndcg_perfect_order_is_one():
    rels = [1, 1, 0, 0]
    assert ndcg_at_k(rels, k=4) == 1.0


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_worse_order_scores_lower():
    good = ndcg_at_k([1, 0, 0], k=3)
    bad = ndcg_at_k([0, 0, 1], k=3)
    assert good > bad


def test_average_precision():
    ap = average_precision_at_k([1, 0, 1, 0], k=4)
    assert 0 < ap <= 1


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_recall_at_k():
    assert recall_at_k([1, 0, 1], n_relevant_total=4, k=3) == 0.5
    assert recall_at_k([0, 0], n_relevant_total=0, k=2) == 0.0


def test_catalog_coverage():
    assert catalog_coverage({1, 2, 3}, 10) == 0.3
    assert catalog_coverage(set(), 10) == 0.0
