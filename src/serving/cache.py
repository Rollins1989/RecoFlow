"""
Recommendation cache. Wraps Redis but degrades to an in-process dict if Redis
is unreachable — satisfies the "Redis unavailable" graceful-failure test in
docs/DESIGN.md / the design brief's testing section: the API should stay up,
just slower (no cache hit-rate benefit), rather than crash.
"""
from __future__ import annotations

import json
import os
import time

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_TTL_SECONDS = 300


class RecommendationCache:
    def __init__(self, url: str = REDIS_URL, ttl: int = DEFAULT_TTL_SECONDS):
        self.ttl = ttl
        self._redis = None
        self._local: dict[str, tuple[float, str]] = {}
        self.backend = "memory"
        try:
            import redis  # noqa: local import so redis-py is optional at runtime
            client = redis.Redis.from_url(url, socket_connect_timeout=0.5)
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:
            self._redis = None
            self.backend = "memory"

    def _key(self, user_id: int, limit: int) -> str:
        return f"reco:{user_id}:{limit}"

    def get(self, user_id: int, limit: int):
        key = self._key(user_id, limit)
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                return json.loads(raw) if raw else None
            except Exception:
                self.backend = "memory"  # fall through to local for this call
        entry = self._local.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if time.time() > expires_at:
            del self._local[key]
            return None
        return json.loads(payload)

    def set(self, user_id: int, limit: int, value) -> None:
        key = self._key(user_id, limit)
        payload = json.dumps(value)
        if self._redis is not None:
            try:
                self._redis.setex(key, self.ttl, payload)
                return
            except Exception:
                self.backend = "memory"
        self._local[key] = (time.time() + self.ttl, payload)

    def stats(self) -> dict:
        return {"backend": self.backend, "local_keys": len(self._local)}
