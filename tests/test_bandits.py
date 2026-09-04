from src.bandits.bandits import (EpsilonGreedyBandit, ThompsonSamplingBandit,
                                  UCB1Bandit, simulate_bandit)


def test_ucb1_converges_to_best_arm():
    true_probs = {"a": 0.1, "b": 0.5, "c": 0.9}
    bandit = UCB1Bandit(arms=list(true_probs))
    result = simulate_bandit(bandit, true_probs, n_rounds=3000, seed=1)
    assert max(result["pull_counts"], key=result["pull_counts"].get) == "c"


def test_thompson_sampling_converges_to_best_arm():
    true_probs = {"a": 0.2, "b": 0.5, "c": 0.85}
    bandit = ThompsonSamplingBandit(arms=list(true_probs))
    result = simulate_bandit(bandit, true_probs, n_rounds=3000, seed=2)
    assert max(result["pull_counts"], key=result["pull_counts"].get) == "c"


def test_epsilon_greedy_explores_all_arms():
    true_probs = {"a": 0.3, "b": 0.6}
    bandit = EpsilonGreedyBandit(arms=list(true_probs), epsilon=0.2)
    result = simulate_bandit(bandit, true_probs, n_rounds=500, seed=3)
    assert all(c > 0 for c in result["pull_counts"].values())


def test_regret_is_non_decreasing():
    true_probs = {"a": 0.4, "b": 0.6}
    bandit = EpsilonGreedyBandit(arms=list(true_probs), epsilon=0.1)
    result = simulate_bandit(bandit, true_probs, n_rounds=200, seed=4)
    curve = result["regret_curve"]
    assert all(curve[i] <= curve[i + 1] + 1e-9 for i in range(len(curve) - 1))
