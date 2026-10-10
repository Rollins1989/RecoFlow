"""Point-in-time feature tests for the learning-to-rank pipeline."""
import pandas as pd

from src.ranking.train_ranker import build_training_frame


def test_training_features_exclude_events_at_or_after_target_timestamp():
    events = pd.DataFrame([
        {"user_id": 1, "item_id": 10, "event": "impression",
         "timestamp": "2026-01-01T10:00:00", "session_id": "s1", "device": "web"},
        {"user_id": 1, "item_id": 10, "event": "click",
         "timestamp": "2026-01-01T10:01:00", "session_id": "s2", "device": "web"},
        {"user_id": 1, "item_id": 20, "event": "purchase",
         "timestamp": "2026-01-01T10:02:00", "session_id": "s3", "device": "web"},
    ])
    items = pd.DataFrame([
        {"item_id": 10, "category": "books", "brand": "A", "price": 10.0,
         "created_ts": "2025-12-01"},
        {"item_id": 20, "category": "books", "brand": "B", "price": 20.0,
         "created_ts": "2025-12-01"},
    ])
    affinity = pd.DataFrame(columns=["user_id", "category", "affinity"])
    users = pd.DataFrame([{"user_id": 1, "price_sensitivity": 0.5}])

    frame = build_training_frame(events, items, affinity, users).set_index("timestamp")

    first = frame.loc[pd.Timestamp("2026-01-01T10:00:00")]
    assert first["popularity_score"] == 0.0
    assert first["item_ctr"] == 0.0
    assert first["category_affinity"] == 0.0

    second = frame.loc[pd.Timestamp("2026-01-01T10:01:00")]
    assert second["popularity_score"] == 0.1
    assert second["item_ctr"] == 0.0

    third = frame.loc[pd.Timestamp("2026-01-01T10:02:00")]
    assert third["popularity_score"] > 0.0
    assert third["category_affinity"] > 0.0
