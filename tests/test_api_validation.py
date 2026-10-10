"""API contract checks for query bounds and payload validation."""
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
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_recommendations_reject_invalid_limit(client, limit):
    response = client.get(f"/recommendations/1?limit={limit}")
    assert response.status_code == 422


@pytest.mark.parametrize("limit", [0, -1, 101])
def test_similar_rejects_invalid_limit(client, limit):
    response = client.get(f"/similar/1?limit={limit}")
    assert response.status_code == 422


def test_search_rejects_empty_query(client):
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 422



def test_recommendations_include_catalogue_metadata(client):
    response = client.get("/recommendations/1?limit=5")
    assert response.status_code == 200
    payload = response.json()
    for recommendation in payload["recommendations"]:
        assert recommendation["title"]
        assert "category" in recommendation
        assert "brand" in recommendation
        assert "price" in recommendation


def test_search_results_include_catalogue_metadata(client):
    response = client.post("/search", json={"query": "book", "limit": 5})
    assert response.status_code == 200
    for result in response.json()["results"]:
        assert result["title"]
        assert "category" in result
        assert "brand" in result
        assert "price" in result
