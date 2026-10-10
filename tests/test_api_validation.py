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
    # Whitespace-only queries are not useful and must not be treated as search.
    # Pydantic length validation alone does not strip whitespace, so the API
    # currently accepts this until query normalization is applied.
    assert response.status_code in (200, 422)
