"""
Stream processing: consumes `user-events`, updates the live user profile
(category affinity / recent interests) and pushes it to the online feature
store, so a search for "wireless headphones" changes next request's
recommendations without a full retrain.

Version 1: this simple Python consumer.
Version 2 (heavier volume): swap for Apache Flink or Spark Structured
Streaming consuming the same topic — the update logic in
`src/streaming/user_profile_updater.py` stays the same either way.
"""
from __future__ import annotations

import json
import os
import time

from src.streaming.kafka_producer import KAFKA_BOOTSTRAP, TOPIC_USER_EVENTS, drain_local_queue
from src.streaming.user_profile_updater import apply_event_to_profile


def run_consumer_loop(profile_store: dict, poll_interval_s: float = 1.0, max_iterations: int | None = None):
    """profile_store: mutable dict[user_id] -> profile dict (in-memory demo;
    production would upsert into Feast's online store / PostgreSQL)."""
    backend = "memory"
    consumer = None
    try:
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            TOPIC_USER_EVENTS, bootstrap_servers=KAFKA_BOOTSTRAP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=1000,
        )
        backend = "kafka"
    except Exception:
        consumer = None
        backend = "memory"

    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        events = []
        if consumer is not None:
            events = [msg.value for msg in consumer]
        else:
            events = [payload for _, payload in drain_local_queue()]

        for event in events:
            apply_event_to_profile(profile_store, event)

        iterations += 1
        if max_iterations is not None:
            continue
        time.sleep(poll_interval_s)

    return {"backend": backend, "n_users_updated": len(profile_store)}


if __name__ == "__main__":
    store: dict = {}
    print(run_consumer_loop(store, max_iterations=1))
