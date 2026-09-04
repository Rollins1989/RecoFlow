"""
Runs the entire RecoFlow offline path end-to-end on synthetic data, with no
Kafka/Redis/Qdrant/Postgres required:

  EDA -> popularity -> content-based -> CF -> hybrid -> temporal split
      -> LightGBM ranker -> MMR re-ranking -> offline evaluation table
      -> bandit simulation -> A/B test simulation

Usage:
  python data/generate_synthetic_data.py     # once
  python run_pipeline.py
"""
from __future__ import annotations

import os

import pandas as pd

from src.bandits.bandits import UCB1Bandit, simulate_bandit
from src.evaluation.ab_test_simulation import simulate_engagement, summarize_ab_test
from src.evaluation.offline_eval import build_relevance_lookup, evaluate_recommender, results_table
from src.evaluation.temporal_split import temporal_split
from src.features.feature_engineering import build_item_features, build_user_category_affinity
from src.ranking.rerank_mmr import mmr_rerank
from src.ranking.train_ranker import (build_training_frame, score_candidates,
                                       train_lightgbm_ranker)
from src.retrieval.collaborative_filtering import ImplicitALS
from src.retrieval.content_based import ContentBasedRecommender
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.popularity import PopularityRecommender

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def main():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    products = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    events = pd.read_csv(os.path.join(DATA_DIR, "events.csv"))
    events["timestamp"] = pd.to_datetime(events["timestamp"])

    print(f"\n=== EDA ===\nusers={len(users)} products={len(products)} events={len(events)}")
    print(events["event"].value_counts().to_string())

    # ---- Version 1: temporal split, popularity / content / CF / hybrid ----
    split = temporal_split(events, train_frac=0.7, valid_frac=0.15)
    train, valid, test = split["train"], split["valid"], split["test"]
    print(f"\ntrain={len(train)} valid={len(valid)} test={len(test)} "
          f"(cutoff train={split['train_cutoff']})")

    pop = PopularityRecommender().fit(train, as_of=split["train_cutoff"])
    content = ContentBasedRecommender().fit(products)
    cf = ImplicitALS(n_factors=16, n_iter=6).fit(train)
    hybrid = HybridRetriever(pop, cf, content)

    user_history = train.groupby("user_id")["item_id"].apply(list).to_dict()
    relevance_lookup = build_relevance_lookup(test)
    eval_users = list(relevance_lookup.keys())[:300]  # keep eval fast

    embeddings = content.embedder.embeddings_
    item_id_to_idx = {int(i): idx for idx, i in enumerate(content.embedder.item_ids_)}
    n_catalog = len(products)

    def pop_reco(u):
        return [i for i, _ in pop.recommend(k=10)]

    def content_reco(u):
        hist = user_history.get(u, [])
        if not hist:
            return [i for i, _ in pop.recommend(k=10)]
        return content.candidates_for_user_history(hist, n=10)["item_id"].tolist()

    def cf_reco(u):
        c = cf.candidates(u, n=10)
        return c["item_id"].tolist() if not c.empty else [i for i, _ in pop.recommend(k=10)]

    def hybrid_reco(u):
        pool = hybrid.get_candidates(u, user_history.get(u, []), n_total=10)
        return pool["item_id"].tolist()

    results = {}
    for name, fn in [("Popularity", pop_reco), ("Content", content_reco),
                      ("CF", cf_reco), ("Hybrid", hybrid_reco)]:
        results[name] = evaluate_recommender(
            fn, eval_users, relevance_lookup, embeddings, item_id_to_idx,
            k=10, total_catalog_size=n_catalog)

    # ---- Version 2: LightGBM ranker + MMR re-ranking ----
    items_feat = build_item_features(products, train, as_of=split["train_cutoff"])
    user_affinity = build_user_category_affinity(train, products, as_of=split["train_cutoff"])
    train_frame = build_training_frame(train, items_feat, user_affinity, users)
    valid_frame = build_training_frame(valid, items_feat, user_affinity, users)

    if train_frame["user_id"].nunique() > 1 and len(train_frame) > 50:
        model = train_lightgbm_ranker(train_frame, valid_frame)

        def ranker_reco(u, rerank=False):
            hist = user_history.get(u, [])
            pool = hybrid.get_candidates(u, hist, n_total=200)
            if pool.empty:
                return [i for i, _ in pop.recommend(k=10)]
            feat = pool.merge(items_feat, on="item_id", how="left")
            feat = feat.merge(
                user_affinity[user_affinity["user_id"] == u][["category", "affinity"]],
                on="category", how="left")
            feat["category_affinity"] = feat["affinity"].fillna(0.0)
            feat["cf_score"] = feat["score"]
            feat["content_score"] = feat["score"]
            feat["popularity_score"] = feat.get("total_weight", 0.0)
            feat["price_diff_from_pref"] = 0.0
            feat["brand_affinity"] = 0.0
            feat["hours_since_last_interaction"] = 0.0
            feat["user_n_purchases"] = 0.0
            feat["user_n_views"] = 0.0
            feat["is_weekend"] = 0
            feat["hour_of_day"] = 12
            feat["score"] = score_candidates(model, feat)

            if rerank:
                reranked = mmr_rerank(feat[["item_id", "score"]], embeddings, item_id_to_idx, k=10)
                return reranked["item_id"].tolist()
            return feat.sort_values("score", ascending=False).head(10)["item_id"].tolist()

        results["Ranker"] = evaluate_recommender(
            lambda u: ranker_reco(u, rerank=False), eval_users, relevance_lookup,
            embeddings, item_id_to_idx, k=10, total_catalog_size=n_catalog)
        results["Ranker + Re-ranking"] = evaluate_recommender(
            lambda u: ranker_reco(u, rerank=True), eval_users, relevance_lookup,
            embeddings, item_id_to_idx, k=10, total_catalog_size=n_catalog)
    else:
        print("\n[skip] Not enough training rows for LightGBM ranker on this sample size.")
        ranker_reco = hybrid_reco

    print("\n=== Offline evaluation (Recall@10 / NDCG@10 / MAP@10 / MRR / Coverage / Diversity) ===")
    print(results_table(results).to_string())

    # ---- Version 6: bandit simulation over candidate-source mixes ----
    print("\n=== Bandit simulation (which candidate-mix arm wins) ===")
    true_probs = {"cf-heavy": 0.18, "content-heavy": 0.14, "trending-heavy": 0.10, "hybrid": 0.22}
    bandit = UCB1Bandit(arms=list(true_probs))
    bandit_result = simulate_bandit(bandit, true_probs, n_rounds=3000)
    print(f"pulls={bandit_result['pull_counts']} "
          f"final_cumulative_regret={bandit_result['final_cumulative_regret']:.2f}")

    # ---- Version 6: A/B test simulation, control=Hybrid vs treatment=Ranker ----
    print("\n=== A/B test simulation: control=Hybrid vs treatment=Ranker+Rerank ===")
    prices = dict(zip(products["item_id"], products["price"]))
    control_fn = hybrid_reco
    treatment_fn = (lambda u: ranker_reco(u, rerank=True)) if "Ranker" in results else hybrid_reco
    sim_df = simulate_engagement(eval_users, relevance_lookup, control_fn, treatment_fn, prices)
    ab_summary = summarize_ab_test(sim_df)
    for k, v in ab_summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
