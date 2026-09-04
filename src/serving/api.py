"""
RecoFlow serving API.

    GET  /recommendations/{user_id}?limit=10
    GET  /similar/{item_id}
    POST /events
    POST /search
    GET  /user/{user_id}/profile
    GET  /model/info
    GET  /health

On startup, loads (or lazily trains, for the demo) the popularity/content/CF
models from data/*.csv so `uvicorn src.serving.api:app` works standalone
without a database. In production, swap `load_artifacts()` for a load from
the model registry (MLflow) + PostgreSQL, as documented in docs/DESIGN.md.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

from src.features.feature_engineering import build_item_features, build_user_category_affinity
from src.retrieval.collaborative_filtering import ImplicitALS
from src.retrieval.content_based import ContentBasedRecommender
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.popularity import PopularityRecommender
from src.ranking.rerank_mmr import apply_business_rules, apply_freshness_boost, mmr_rerank
from src.serving.cache import RecommendationCache
from src.serving.vector_store import VectorStore

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MODEL_VERSION = "hybrid-mmr-0.1.0"

REQUEST_COUNT = Counter("recoflow_requests_total", "Total requests", ["endpoint"])
LATENCY = Histogram("recoflow_request_latency_seconds", "Latency", ["endpoint"])

app = FastAPI(title="RecoFlow API", version=MODEL_VERSION)

STATE: dict = {}


def load_artifacts():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))

    pop = PopularityRecommender().fit(events)
    content = ContentBasedRecommender().fit(products)
    cf = ImplicitALS(n_factors=16, n_iter=5).fit(events)
    hybrid = HybridRetriever(pop, cf, content)

    items_feat = build_item_features(products, events)
    user_affinity = build_user_category_affinity(events, products)
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    user_history = events.groupby("user_id")["item_id"].apply(list).to_dict()

    vector_store = VectorStore(dim=content.embedder.dim)
    vector_store.upsert(content.embedder.item_ids_, content.embedder.embeddings_)

    STATE.update(dict(
        users=users, products=products, events=events,
        pop=pop, content=content, cf=cf, hybrid=hybrid,
        items_feat=items_feat, user_affinity=user_affinity, user_history=user_history,
        cache=RecommendationCache(), vector_store=vector_store,
        item_id_to_idx={int(i): idx for idx, i in enumerate(content.embedder.item_ids_)},
    ))


@app.on_event("startup")
def _startup():
    if os.path.exists(os.path.join(DATA_DIR, "events.csv")):
        load_artifacts()


# ---------------------------------------------------------------- schemas --
class EventIn(BaseModel):
    user_id: int
    item_id: int
    event: str = Field(pattern="^(impression|click|view|add_to_cart|wishlist|"
                                "purchase|remove_from_cart|search|category_visit)$")
    session_id: str | None = None
    device: str | None = None


class SearchIn(BaseModel):
    user_id: int | None = None
    query: str
    limit: int = 20


# ------------------------------------------------------------------ utils --
def _ensure_loaded():
    if "hybrid" not in STATE:
        raise HTTPException(503, "Model artifacts not loaded. Run "
                                  "data/generate_synthetic_data.py first.")


def _recommend(user_id: int, limit: int) -> list[dict]:
    history = STATE["user_history"].get(user_id, [])
    pool = STATE["hybrid"].get_candidates(user_id, history, n_total=1000)
    if pool.empty:
        return []

    items_feat = STATE["items_feat"]
    pool = pool.merge(items_feat[["item_id", "category", "freshness", "price"]],
                       on="item_id", how="left")

    already_seen = set(history)
    pool = apply_business_rules(pool, exclude_purchased=already_seen, max_per_category=3)
    pool = apply_freshness_boost(pool)

    embeddings = STATE["content"].embedder.embeddings_
    item_id_to_idx = STATE["item_id_to_idx"]
    reranked = mmr_rerank(pool, embeddings, item_id_to_idx, lambda_relevance=0.7, k=limit)

    reason_by_source = {
        "cf": "Because you interacted with similar products",
        "content": "Because you viewed similar items",
        "trending": "Trending among users with similar interests",
        "personalized": "Popular in your favorite categories",
    }
    results = []
    for _, row in reranked.iterrows():
        sources = row.get("sources", [])
        reason = reason_by_source.get(sources[0] if sources else "trending", "Recommended for you")
        results.append({
            "item_id": int(row["item_id"]),
            "score": round(float(row["score"]), 4),
            "reason": reason,
            "candidate_sources": sources,
        })
    return results


# -------------------------------------------------------------- endpoints --
@app.get("/health")
def health():
    return {
        "status": "ok" if "hybrid" in STATE else "degraded",
        "cache_backend": STATE.get("cache").stats() if "cache" in STATE else None,
        "vector_store_backend": STATE.get("vector_store").backend if "vector_store" in STATE else None,
    }


@app.get("/model/info")
def model_info():
    return {
        "model_version": MODEL_VERSION,
        "candidate_sources": ["cf", "content", "trending", "personalized"],
        "ranking_stage": "MMR re-rank over hybrid pool (LightGBM ranker: see src/ranking/train_ranker.py)",
        "n_products": int(len(STATE["products"])) if "products" in STATE else None,
        "n_users": int(len(STATE["users"])) if "users" in STATE else None,
    }


@app.get("/recommendations/{user_id}")
def recommendations(user_id: int, limit: int = 10):
    _ensure_loaded()
    REQUEST_COUNT.labels(endpoint="recommendations").inc()
    start = time.time()

    cache = STATE["cache"]
    cached = cache.get(user_id, limit)
    if cached is not None:
        LATENCY.labels(endpoint="recommendations").observe(time.time() - start)
        return {"user_id": user_id, "recommendations": cached,
                "model_version": MODEL_VERSION, "cache_hit": True,
                "generated_at": datetime.now(timezone.utc).isoformat()}

    recs = _recommend(user_id, limit)
    cache.set(user_id, limit, recs)
    LATENCY.labels(endpoint="recommendations").observe(time.time() - start)
    return {"user_id": user_id, "recommendations": recs,
            "model_version": MODEL_VERSION, "cache_hit": False,
            "generated_at": datetime.now(timezone.utc).isoformat()}


@app.get("/similar/{item_id}")
def similar(item_id: int, limit: int = 10):
    _ensure_loaded()
    REQUEST_COUNT.labels(endpoint="similar").inc()
    if item_id not in STATE["item_id_to_idx"]:
        raise HTTPException(404, f"item_id {item_id} not found")
    results = STATE["content"].similar_items(item_id, k=limit)
    return {"item_id": item_id,
            "similar_items": [{"item_id": i, "score": round(s, 4)} for i, s in results]}


@app.post("/events")
def post_event(event: EventIn):
    _ensure_loaded()
    REQUEST_COUNT.labels(endpoint="events").inc()
    # Real-time path: publish to Kafka for stream processing (src/streaming);
    # here we also apply it in-process so the demo API reflects it immediately.
    from src.streaming.kafka_producer import publish_event
    publish_event(event.model_dump())

    STATE["user_history"].setdefault(event.user_id, []).append(event.item_id)
    STATE["cache"].set(event.user_id, 10, None)  # invalidate cached top-10
    return {"status": "accepted", "event": event.model_dump()}


@app.post("/search")
def search(payload: SearchIn):
    _ensure_loaded()
    REQUEST_COUNT.labels(endpoint="search").inc()
    embedder = STATE["content"].embedder
    query_vec = embedder.transform(pd.DataFrame([{
        "title": payload.query, "description": payload.query,
        "category": "", "brand": "",
    }]))[0]
    hits = STATE["vector_store"].search(query_vec, k=payload.limit)
    return {"query": payload.query,
            "results": [{"item_id": i, "score": round(s, 4)} for i, s in hits]}


@app.get("/user/{user_id}/profile")
def user_profile(user_id: int):
    _ensure_loaded()
    affinity = STATE["user_affinity"]
    row = affinity[affinity["user_id"] == user_id]
    if row.empty:
        return {"user_id": user_id, "category_affinity": {}, "is_cold_start": True}
    cat_affinity = dict(zip(row["category"], row["affinity"].round(4)))
    return {"user_id": user_id, "category_affinity": cat_affinity,
            "is_cold_start": len(STATE["user_history"].get(user_id, [])) < 3}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
