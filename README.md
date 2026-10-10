# RecoFlow

**A reproducible recommendation-systems engineering project** combining hybrid candidate retrieval, learning-to-rank, diversity-aware re-ranking, offline evaluation, and an API demo.

> **Scope note:** RecoFlow is a research/demo prototype trained on synthetic e-commerce interactions. Its infrastructure adapters and simulations are not evidence of production deployment or real-world conversion lift.

## What it demonstrates

- Popularity, content-based, and implicit-feedback collaborative retrieval.
- Hybrid candidate aggregation and diversity-aware MMR re-ranking.
- A LightGBM learning-to-rank experiment.
- Chronological train/validation/test splitting and ranking metrics.
- FastAPI endpoints for recommendations, similar items, search, events, and model information.
- Optional Redis/Qdrant/Kafka integrations with local fallbacks where implemented.
- Pytest coverage and GitHub Actions checks.

## Architecture

```text
Synthetic interactions + product catalogue
                  |
        Chronological split
                  |
       +----------+-----------+
       |          |           |
   Popularity   Content    Implicit CF
       +----------+-----------+
                  |
          Hybrid candidates
                  |
        Learning-to-rank*
                  |
        MMR / business rules
                  |
         Offline evaluation
                  |
          FastAPI demo
```

`*` The ranker is an experimental stage. Review the benchmark output and limitations before interpreting it as an improvement over the baseline.

## Quick start: offline pipeline

Python 3.11 is the supported baseline for the container and CI configuration.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python data/generate_synthetic_data.py --n-users 500 --n-items 200 --n-events 12000
python run_pipeline.py
```

The pipeline reads CSVs from `data/`. The data files are generated locally and intentionally excluded from Git. Use a fresh environment if dependency installation fails; optional infrastructure dependencies may require additional system resources.

## Run the API demo

Generate the data first, then run:

```bash
uvicorn src.serving.api:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for interactive API documentation.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Process/model status |
| GET | `/recommendations/{user_id}?limit=10` | Personalized recommendations |
| GET | `/similar/{item_id}?limit=10` | Similar catalogue items |
| POST | `/events` | Submit a user interaction |
| POST | `/search` | Search the catalogue by text |
| GET | `/user/{user_id}/profile` | Category-affinity profile |
| GET | `/model/info` | Model and catalogue metadata |
| GET | `/metrics` | Prometheus metrics |

Example event:

```json
{
  "user_id": 1,
  "item_id": 42,
  "event": "click",
  "session_id": "demo-session",
  "device": "desktop"
}
```

## Evaluation methodology

The offline pipeline reports NDCG@10, Recall@10, MAP@10, MRR, catalogue coverage, and intra-list diversity for multiple recommendation stages. Data is ordered chronologically rather than randomly shuffled.

Interpret results with these constraints in mind:

- The dataset is synthetic; scores do not predict commercial performance.
- Offline metrics measure agreement with held-out positive interactions, not causal conversion lift.
- A simulated A/B test or bandit run is a software experiment, not a real online experiment.
- Training features must be computed only from information available before the relevant prediction cutoff.
- Baseline and treatment models must use the same evaluation users, candidate eligibility rules, and test period for a fair comparison.

## Tests and quality checks

```bash
python data/generate_synthetic_data.py --n-users 200 --n-items 100 --n-events 4000
pytest -q
ruff check src tests
```

Some integration tests require local service configuration. Check the GitHub Actions run for the exact status; a workflow definition alone is not proof that deployment or all integrations succeeded.

## Repository map

- `src/retrieval/`: retrieval strategies and hybrid candidate generation
- `src/ranking/`: ranker and MMR re-ranking
- `src/evaluation/`: metrics, temporal splits, and simulations
- `src/serving/`: FastAPI, cache, and vector-store adapters
- `src/features/`: feature engineering
- `src/monitoring/`: drift and service metrics
- `src/streaming/`: event producer/consumer adapters
- `data/generate_synthetic_data.py`: reproducible synthetic data generator
- `run_pipeline.py`: offline experiment entry point
- `frontend/`: Next.js demo interface

## Current limitations

- Synthetic data only; no external customer data or real user study.
- The in-memory fallback is for local demos and is not durable across restarts.
- Kafka publishing is best-effort; successful HTTP acceptance should not be interpreted as durable event delivery unless the configured broker confirms it.
- No production deployment target is configured by this repository's CI workflow.
- Model quality and serving latency should be measured on a documented machine and dataset before publishing performance claims.

## License

No license is declared yet. Add a license file before inviting reuse or external contributions.
