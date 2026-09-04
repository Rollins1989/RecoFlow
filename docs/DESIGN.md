# RecoFlow — Design Doc

## 1. Architecture

```
USER → Next.js Frontend → API Gateway → FastAPI
                                          │
                         ┌────────────────┴────────────────┐
                         ▼                                  ▼
                 User Profile Service                 Event Service
                         │                                  │
                         ▼                                  ▼
                    PostgreSQL                            Kafka
                         │                                  │
                         │                                  ▼
                         │                          Event Processing
                         │                                  │
                         └────────────────┬─────────────────┘
                                           ▼
                                    Feature Store (Feast)
                                           │
                                           ▼
                                  Candidate Retrieval
                              ┌────────────┼────────────┐
                              ▼            ▼             ▼
                          CF Model    ANN Search     Popularity
                              └────────────┼────────────┘
                                           ▼
                                    ~1,000 candidates
                                           │
                                           ▼
                                LightGBM Ranking Model
                                           │
                                           ▼
                                       Re-ranking
                              ┌────────────┼────────────┐
                              ▼            ▼             ▼
                         Diversity    Freshness    Business Rules
                              └────────────┼────────────┘
                                           ▼
                                        Top-10
                                           │
                                           ▼
                                          User
                                           │
                                           ▼
                                      Interaction
                                           │
                                           ▼
                                          Kafka
                                           │
                                           ▼
                              Analytics / Retrain (drift → MLflow)
```

## 2. Candidate generation (hybrid)

Blend of independently-scored candidate generators, deduplicated into one pool:

| Source                | Share of pool |
|------------------------|--------------|
| Collaborative filtering | 400 |
| Content-based retrieval | 300 |
| Trending / popularity   | 200 |
| Personalized (profile)  | 100 |

## 3. Implicit feedback strength

`purchase (5) > add_to_cart (3) > wishlist (2) > click (1) > impression (0.1)`
`remove_from_cart` and long `search`-without-click sessions apply negative/neutral
signal. Used both as the CF confidence weight and as the ranking label.

## 4. Ranking features

user-item interaction history · user-category affinity · item popularity ·
item freshness · price · discount · brand affinity · content similarity ·
collaborative score · time since last interaction · view count · purchase count ·
context (hour-of-day, day-of-week, device, session length, current category).

## 5. Cold-start strategy

**New user** (no history): popularity + trending + context (time/device) +
first-session behavior re-scored after every event in that session.

**New product** (no interactions): content embedding similarity + category +
brand + price bracket, boosted in the "freshness" re-ranking term until it
accumulates enough interactions to enter CF.

## 6. Evaluation

- **Offline**: NDCG@10, MAP@10, MRR, Recall@10, intra-list diversity, catalog
  coverage — computed at every stage (popularity → content → CF → hybrid →
  ranker → ranker+re-rank) so improvements are attributable to a specific stage.
- **Temporal split**: train on past interactions, evaluate on future interactions
  (no random shuffling — avoids leakage). Leave-one-out supported for CF.
- **A/B simulation**: control vs. treatment model compared on CTR, conversion,
  revenue-per-recommendation and offline metrics simultaneously, since offline
  metrics do not necessarily move online business metrics in the same direction.
- **Bandit layer**: exploration/exploitation across candidate generators/model
  versions via ε-greedy, UCB1, and Thompson Sampling — used for the "which
  ranker version serves this request" decision, not for user-facing product
  choice.

## 7. Drift & retraining

`training distribution` vs. `rolling current distribution` compared (feature +
prediction + label drift). Threshold breach → alert → scheduled retrain →
offline re-evaluation → manual approval gate → promote to production model
alias in MLflow.
