import pandas as pd
import pytest

from src.evaluation.temporal_split import leave_one_out_split, temporal_split


def sample_events(n=20):
    return pd.DataFrame({
        "user_id": [i % 4 for i in range(n)],
        "item_id": list(range(n)),
        "event": ["click" if i % 3 else "purchase" for i in range(n)],
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
    })


def test_temporal_split_is_ordered_and_disjoint():
    result = temporal_split(sample_events(), train_frac=0.6, valid_frac=0.2)
    train, valid, test = result["train"], result["valid"], result["test"]
    assert len(train) + len(valid) + len(test) == 20
    assert train["timestamp"].max() <= valid["timestamp"].min()
    assert valid["timestamp"].max() <= test["timestamp"].min()
    assert result["train_cutoff"] == train["timestamp"].max()


@pytest.mark.parametrize("train_frac,valid_frac", [(0, 0.2), (0.7, 0.3), (1.1, 0.1), (0.7, -0.1)])
def test_temporal_split_rejects_invalid_fractions(train_frac, valid_frac):
    with pytest.raises(ValueError):
        temporal_split(sample_events(), train_frac=train_frac, valid_frac=valid_frac)


def test_temporal_split_rejects_invalid_timestamps():
    events = sample_events()
    events.loc[0, "timestamp"] = "not-a-date"
    with pytest.raises(ValueError, match="timestamp"):
        temporal_split(events)


def test_leave_one_out_holds_out_latest_purchase_per_user():
    events = sample_events()
    train, test = leave_one_out_split(events)
    assert test["user_id"].is_unique
    for _, row in test.iterrows():
        user_events = events[(events["user_id"] == row["user_id"]) & (events["event"] == "purchase")]
        assert row["timestamp"] == user_events["timestamp"].max()
        assert row.name not in train.index
