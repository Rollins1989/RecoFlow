# RecoFlow

Production-style real-time recommendation and ranking platform for e-commerce applications.

## Problem

Traditional recommendation systems can struggle with personalization, ranking quality, cold-start users, changing user behavior, and serving recommendations in real time.

## Solution

RecoFlow combines multiple recommendation strategies, machine-learning ranking, real-time user updates, and online experimentation to generate personalized Top-N recommendations.

## Key Features

* Hybrid recommendation: popularity, content-based, and collaborative filtering
* LightGBM learning-to-rank with MMR re-ranking
* Real-time user profile and feature updates
* Cold-start recommendation strategies
* Multi-armed bandits for adaptive recommendations
* A/B testing and offline evaluation
* FastAPI recommendation API
* Redis caching and Qdrant vector search
* Kafka event streaming
* MLflow, DVC, and Feast for MLOps
* Prometheus/Grafana monitoring and drift detection
* Dockerized infrastructure
* Next.js + TypeScript frontend
* Automated testing and GitHub Actions CI/CD

## Tech Stack

**Python, Scikit-learn, LightGBM, FastAPI, PostgreSQL, Redis, Qdrant, Kafka, Feast, MLflow, DVC, Prometheus, Grafana, Docker, Next.js, TypeScript, Pytest, GitHub Actions**

## Architecture

```text
User Events
    ↓
Feature Engineering
    ↓
Candidate Generation
    ↓
ML Ranking
    ↓
MMR Re-ranking
    ↓
Personalized Top-N Recommendations
    ↓
Feedback → Real-time Updates
```

## Quick Start

```bash
pip install -r requirements.txt
python data/generate_synthetic_data.py
python run_pipeline.py
```

For the complete production stack:

```bash
docker compose up -d
```

## API

```text
GET  /recommendations/{user_id}
GET  /similar/{item_id}
POST /events
POST /search
GET  /user/{user_id}/profile
GET  /model/info
GET  /health
```

See `docs/DESIGN.md` for the complete architecture and implementation details.

