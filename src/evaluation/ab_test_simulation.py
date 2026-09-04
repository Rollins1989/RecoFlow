"""
A/B test simulation — Version 6's move from offline ML metrics into product
experimentation. Randomly assigns users to control/treatment, simulates
engagement using a simple stochastic response model driven by each
recommender's offline relevance signal, and runs a two-proportion z-test on
CTR / conversion.

Offline metrics (NDCG, diversity, ...) do NOT necessarily move online business
metrics (CTR, conversion, revenue) in the same direction — this module
reports both side by side so that disconnect is visible, not hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def assign_variants(user_ids: list[int], seed: int = 42) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    return {u: ("treatment" if rng.random() < 0.5 else "control") for u in user_ids}


def simulate_engagement(user_ids: list[int], relevance_lookup: dict,
                          recommend_fn_control, recommend_fn_treatment,
                          prices: dict[int, float], seed: int = 42) -> pd.DataFrame:
    """For each user, simulate whether each recommended item is clicked /
    converted, using a noisy function of true relevance (ground truth from
    the held-out relevance_lookup) as the click/purchase probability."""
    rng = np.random.default_rng(seed)
    variants = assign_variants(user_ids, seed=seed)
    rows = []

    for user_id in user_ids:
        variant = variants[user_id]
        recommend_fn = recommend_fn_treatment if variant == "treatment" else recommend_fn_control
        relevant_items = relevance_lookup.get(user_id, set())
        recs = recommend_fn(user_id)

        clicked, converted, revenue = 0, 0, 0.0
        for item in recs:
            base_p_click = 0.35 if item in relevant_items else 0.03
            p_click = float(np.clip(base_p_click + rng.normal(0, 0.05), 0, 1))
            did_click = rng.random() < p_click
            clicked += int(did_click)
            if did_click:
                base_p_conv = 0.25 if item in relevant_items else 0.02
                did_convert = rng.random() < base_p_conv
                converted += int(did_convert)
                if did_convert:
                    revenue += prices.get(item, 0.0)

        rows.append({
            "user_id": user_id, "variant": variant, "n_shown": len(recs),
            "clicks": clicked, "conversions": converted, "revenue": revenue,
        })
    return pd.DataFrame(rows)


def summarize_ab_test(sim_df: pd.DataFrame) -> dict:
    summary = {}
    for variant in ["control", "treatment"]:
        v = sim_df[sim_df["variant"] == variant]
        n_shown = v["n_shown"].sum()
        summary[variant] = {
            "n_users": len(v),
            "ctr": v["clicks"].sum() / n_shown if n_shown else 0.0,
            "conversion_rate": v["conversions"].sum() / n_shown if n_shown else 0.0,
            "revenue_per_user": v["revenue"].mean() if len(v) else 0.0,
            "total_revenue": v["revenue"].sum(),
        }

    c, t = sim_df[sim_df["variant"] == "control"], sim_df[sim_df["variant"] == "treatment"]
    c_clicks, c_shown = c["clicks"].sum(), c["n_shown"].sum()
    t_clicks, t_shown = t["clicks"].sum(), t["n_shown"].sum()
    count = np.array([c_clicks, t_clicks])
    nobs = np.array([c_shown, t_shown])
    if nobs.min() > 0:
        p_pool = count.sum() / nobs.sum()
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
        z = (count[1] / nobs[1] - count[0] / nobs[0]) / se if se > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    else:
        z, p_value = 0.0, 1.0

    summary["ctr_z_stat"] = float(z)
    summary["ctr_p_value"] = float(p_value)
    summary["significant_at_0.05"] = bool(p_value < 0.05)
    return summary
