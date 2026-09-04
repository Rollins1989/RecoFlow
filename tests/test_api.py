import os

import pytest
from fastapi.testclient import TestClient

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "events.csv")),
    reason="run data/generate_synthetic_data.py first",
)


@pytest.fixture(scope="module")
def client():
    from src.serving.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "degraded")


def test_recommendations_valid_user(client):
    r = client.get("/recommendations/1?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == 1
    assert len(body["recommendations"]) <= 5
    ids = [x["item_id"] for x in body["recommendations"]]
    assert len(ids) == len(set(ids))  # no duplicates


def test_recommendations_nonexistent_user_still_responds(client):
    r = client.get("/recommendations/999999999?limit=5")
    assert r.status_code == 200  # cold-start path, not an error


def test_recommendations_cache_hit_on_second_call(client):
    client.get("/recommendations/2?limit=5")
    r2 = client.get("/recommendations/2?limit=5")
    assert r2.json()["cache_hit"] is True


def test_similar_unknown_item_404(client):
    r = client.get("/similar/99999999")
    assert r.status_code == 404


def test_post_event_valid(client):
    r = client.post("/events", json={
        "user_id": 1, "item_id": 2, "event": "click", "session_id": "s1", "device": "mobile",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "accepted"


def test_post_event_malformed_rejected(client):
    r = client.post("/events", json={"user_id": 1, "item_id": 2, "event": "not_a_real_event"})
    assert r.status_code == 422


def test_search_returns_results(client):
    r = client.post("/search", json={"query": "wireless headphones", "limit": 5})
    assert r.status_code == 200
    assert "results" in r.json()


def test_user_profile_cold_start_user(client):
    r = client.get("/user/999999999/profile")
    assert r.status_code == 200
    assert r.json()["is_cold_start"] is True


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    assert "model_version" in r.json()


def test_metrics_endpoint_exposed(client):
    r = client.get("/metrics")
    assert r.status_code == 200


# ---- infra-unavailable graceful degradation ---------------------------------
def test_cache_falls_back_to_memory_when_redis_down():
    from src.serving.cache import RecommendationCache
    cache = RecommendationCache(url="redis://nonexistent-host:6379/0")
    assert cache.backend == "memory"
    cache.set(1, 10, [{"item_id": 1}])
    assert cache.get(1, 10) == [{"item_id": 1}]


def test_vector_store_falls_back_to_memory_when_qdrant_down():
    import numpy as np
    from src.serving.vector_store import VectorStore
    vs = VectorStore(dim=4, url="http://nonexistent-host:6333")
    assert vs.backend == "memory"
    vs.upsert(np.array([1, 2]), np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float))
    results = vs.search(np.array([1, 0, 0, 0], dtype=float), k=1)
    assert results[0][0] == 1
