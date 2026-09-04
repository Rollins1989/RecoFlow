# RecoFlow — Real-Time Personalized Recommendation & Ranking Platform

RecoFlow is a production-style recommendation platform for an e-commerce/marketplace
domain. It goes from raw behavioral events (`impression`, `click`, `view`,
`add_to_cart`, `wishlist`, `purchase`, `remove_from_cart`, `search`,
`category_visit`) all the way to a served, re-ranked, explainable Top-N list —
with retraining, drift detection, and online experimentation on top.

This repo implements the **full progression (Version 1 → Version 6)** described in
`docs/DESIGN.md`, but keeps every stage runnable independently so you can demo any
layer in isolation.

```
Version 1 — Core ML          popularity → content-based → CF → hybrid → offline eval
Version 2 — Ranking          candidate generation → LightGBM ranker → NDCG/MAP/Recall → MMR re-ranking
Version 3 — Production       FastAPI, PostgreSQL, Redis, Docker, Next.js
Version 4 — Real-time        Kafka events → stream processing → live feature updates
Version 5 — MLOps            MLflow, DVC, Feast, GitHub Actions, Prometheus/Grafana, drift detection
Version 6 — Advanced         Multi-armed bandits (ε-greedy / UCB / Thompson), A/B test simulation
```

## Quickstart (no infra — pure ML core)

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
python run_pipeline.py
```

`run_pipeline.py` runs the entire offline path end-to-end on synthetic data: EDA →
popularity → content-based → CF → hybrid → temporal split → LightGBM ranker →
MMR re-ranking → offline evaluation table → bandit simulation → A/B simulation.
It requires no Kafka/Redis/Qdrant/Postgres and prints the same evaluation table
described in the design doc.

## Quickstart (full stack, with infra)

```bash
docker compose up -d          # postgres, redis, qdrant, kafka, zookeeper, prometheus, grafana
python data/generate_synthetic_data.py --load-db
python src/ranking/train_ranker.py
uvicorn src.serving.api:app --reload --port 8000
cd frontend && npm install && npm run dev
```

Every serving-layer client (`src/serving/cache.py`, `src/serving/vector_store.py`)
degrades gracefully to an in-process fallback when its backing service is
unreachable, per the "infra unavailable" test requirements in the design doc —
so the API stays usable even with partial infra up.

## Repository layout

```
data/                       synthetic e-commerce dataset generator
src/db/schema.sql           PostgreSQL schema (users, products, events, profiles, ...)
src/features/               feature engineering (user/item/context features)
src/retrieval/              popularity, content-based, CF, hybrid, cold-start
src/ranking/                LightGBM learn-to-rank + MMR re-ranking
src/evaluation/             NDCG/MAP/MRR/Recall, temporal split, A/B simulation
src/bandits/                epsilon-greedy, UCB, Thompson sampling
src/serving/                FastAPI app, Redis cache, Qdrant client
src/streaming/               Kafka producer/consumer, live profile updater
src/feature_store/          Feast feature repo (offline/online consistency)
src/mlops/                  MLflow tracking, DVC pipeline
src/monitoring/             Prometheus config, Grafana dashboard JSON, drift detection
tests/                      pytest suite (retrieval, ranking, API, infra-down cases)
frontend/                   Next.js + TypeScript shopping UI with a debug panel
.github/workflows/ci.yml    lint → unit tests → integration tests → build → scan → deploy
```

## API surface

```
GET  /recommendations/{user_id}?limit=10
GET  /similar/{item_id}
POST /events
POST /search
GET  /user/{user_id}/profile
GET  /model/info
GET  /health
```

See `docs/DESIGN.md` for the full architecture diagram, feature list, cold-start
strategy, and evaluation methodology.
