"""
Exploration/exploitation for RecoFlow.

Used to decide *which candidate-generation strategy / model version* serves
a given request (not which literal product to show a user) — e.g. arms could
be {"ranker-v1", "ranker-v2", "ranker-v3"} or {"cf-heavy", "content-heavy",
"trending-heavy"} candidate mixes. Each arm's reward is the observed
click/purchase signal from serving that arm.
"""
from __future__ import annotations

import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, arms: list[str], epsilon: float = 0.1, seed: int = 42):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}
        self.rng = np.random.default_rng(seed)

    def select_arm(self) -> str:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.arms)
        return max(self.values, key=self.values.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UCB1Bandit:
    def __init__(self, arms: list[str]):
        self.arms = arms
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}
        self.total_pulls = 0

    def select_arm(self) -> str:
        for a in self.arms:
            if self.counts[a] == 0:
                return a  # ensure every arm tried once
        self.total_pulls = sum(self.counts.values())
        ucb_scores = {
            a: self.values[a] + np.sqrt(2 * np.log(self.total_pulls) / self.counts[a])
            for a in self.arms
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class ThompsonSamplingBandit:
    """Beta-Bernoulli Thompson Sampling — reward must be in {0, 1} (e.g. click/no-click)."""

    def __init__(self, arms: list[str], seed: int = 42):
        self.arms = arms
        self.alpha = {a: 1.0 for a in arms}  # successes + 1
        self.beta = {a: 1.0 for a in arms}   # failures + 1
        self.counts = {a: 0 for a in arms}   # exposed for uniform reporting/regret analysis
        self.rng = np.random.default_rng(seed)

    def select_arm(self) -> str:
        samples = {a: self.rng.beta(self.alpha[a], self.beta[a]) for a in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float):
        self.counts[arm] += 1
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1


def simulate_bandit(bandit, true_reward_probs: dict[str, float], n_rounds: int = 2000,
                      seed: int = 0) -> dict:
    """Runs `n_rounds` of arm-pulling against ground-truth Bernoulli reward
    probabilities, returns cumulative regret + per-arm pull counts."""
    rng = np.random.default_rng(seed)
    best_prob = max(true_reward_probs.values())
    cumulative_regret = 0.0
    regret_curve = []

    for _ in range(n_rounds):
        arm = bandit.select_arm()
        reward = 1.0 if rng.random() < true_reward_probs[arm] else 0.0
        bandit.update(arm, reward)
        cumulative_regret += best_prob - true_reward_probs[arm]
        regret_curve.append(cumulative_regret)

    return {
        "final_cumulative_regret": cumulative_regret,
        "pull_counts": dict(bandit.counts),
        "regret_curve": regret_curve,
    }
