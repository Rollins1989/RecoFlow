"""
Publishes user events onto the `user-events` Kafka topic for real-time
personalization (Version 4). Falls back to an in-process queue when no
broker is reachable, so the API keeps working in the pure-ML-core demo.

Topics:
  user-events            raw impression/click/.../purchase events
  recommendation-events  every served recommendation set (for offline eval / drift)
  purchase-events        purchases only, for fast downstream fan-out (e.g. inventory)
"""
from __future__ import annotations

import json
import os
from collections import deque

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC_USER_EVENTS = "user-events"
TOPIC_RECO_EVENTS = "recommendation-events"
TOPIC_PURCHASE_EVENTS = "purchase-events"

_local_queue: deque = deque(maxlen=10000)
_producer = None
_backend = "memory"


def _get_producer():
    global _producer, _backend
    if _producer is not None or _backend == "memory_tried":
        return _producer
    try:
        from kafka import KafkaProducer
        _producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=1000,
        )
        _backend = "kafka"
    except Exception:
        _producer = None
        _backend = "memory_tried"
    return _producer


def publish_event(event: dict, topic: str = TOPIC_USER_EVENTS) -> str:
    producer = _get_producer()
    if producer is not None:
        try:
            producer.send(topic, event)
            producer.flush(timeout=1)
            return "kafka"
        except Exception:
            pass
    _local_queue.append((topic, event))
    return "memory"


def drain_local_queue() -> list[tuple[str, dict]]:
    """Used by src/streaming/kafka_consumer.py's in-process fallback mode and
    by tests — pops everything currently buffered."""
    items = list(_local_queue)
    _local_queue.clear()
    return items
