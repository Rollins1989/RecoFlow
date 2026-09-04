import os

import pandas as pd
import pytest

from src.retrieval.cold_start import is_cold_start_user
from src.retrieval.collaborative_filtering import ImplicitALS
from src.retrieval.content_based import ContentBasedRecommender
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.popularity import PopularityRecommender
from src.ranking.rerank_mmr import apply_business_rules, mmr_rerank

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "events.csv")),
    reason="run data/generate_synthetic_data.py first",
)


@pytest.fixture(scope="module")
def dataset():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    return users, products, events


def test_popularity_recommends_correct_count(dataset):
    _, _, events = dataset
    pop = PopularityRecommender().fit(events)
    recs = pop.recommend(k=10)
    assert len(recs) <= 10
    assert len(set(r[0] for r in recs)) == len(recs)  # no duplicates


def test_content_based_similar_items_excludes_self(dataset):
    _, products, _ = dataset
    cb = ContentBasedRecommender().fit(products)
    item_id = int(products.iloc[0]["item_id"])
    sims = cb.similar_items(item_id, k=5)
    assert all(i != item_id for i, _ in sims)


def test_cf_candidates_for_known_user(dataset):
    _, _, events = dataset
    cf = ImplicitALS(n_factors=8, n_iter=3).fit(events)
    active_user = events["user_id"].value_counts().index[0]
    cands = cf.candidates(int(active_user), n=50)
    assert not cands.empty


def test_cold_start_detection():
    assert is_cold_start_user([]) is True
    assert is_cold_start_user([1, 2]) is True
    assert is_cold_start_user([1, 2, 3, 4]) is False


def test_hybrid_pool_has_no_duplicate_items(dataset):
    users, products, events = dataset
    pop = PopularityRecommender().fit(events)
    cf = ImplicitALS(n_factors=8, n_iter=3).fit(events)
    cb = ContentBasedRecommender().fit(products)
    hybrid = HybridRetriever(pop, cf, cb)

    user_id = int(events["user_id"].iloc[0])
    history = events[events["user_id"] == user_id]["item_id"].tolist()
    pool = hybrid.get_candidates(user_id, history, n_total=100)
    assert pool["item_id"].is_unique


def test_business_rules_exclude_purchased(dataset):
    _, products, events = dataset
    df = pd.DataFrame({"item_id": products["item_id"].head(20),
                        "score": range(20), "category": products["category"].head(20)})
    purchased = {int(products.iloc[0]["item_id"]), int(products.iloc[1]["item_id"])}
    filtered = apply_business_rules(df, exclude_purchased=purchased)
    assert not any(i in purchased for i in filtered["item_id"])


def test_mmr_rerank_returns_k_items(dataset):
    _, products, _ = dataset
    cb = ContentBasedRecommender().fit(products)
    pool = pd.DataFrame({
        "item_id": products["item_id"].head(30),
        "score": range(30, 0, -1),
    })
    item_id_to_idx = {int(i): idx for idx, i in enumerate(cb.embedder.item_ids_)}
    result = mmr_rerank(pool, cb.embedder.embeddings_, item_id_to_idx, k=10)
    assert len(result) == 10
    assert result["item_id"].is_unique
