"""TTL recommendation cache with Redis and an in-process demo fallback."""
from __future__ import annotations

import json
import os
import time
from typing import Any

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DEFAULT_TTL_SECONDS = 300


class RecommendationCache:
    def __init__(self, url: str = REDIS_URL, ttl: int = DEFAULT_TTL_SECONDS):
        if ttl <= 0:
            raise ValueError("ttl must be a positive number of seconds")
        self.ttl = ttl
        self._redis = None
        self._local: dict[str, tuple[float, str]] = {}
        self.backend = "memory"
        try:
            import redis

            client = redis.Redis.from_url(url, socket_connect_timeout=0.5)
            client.ping()
            self._redis = client
            self.backend = "redis"
        except Exception:  # noqa: BLE001 - service is optional for local demo
            self._redis = None

    def _key(self, user_id: int, limit: int) -> str:
        return f"reco:{user_id}:{limit}"

    def get(self, user_id: int, limit: int) -> Any | None:
        key = self._key(user_id, limit)
        if self._redis is not None:
            try:
                raw = self._redis.get(key)
                if raw is not None:
                    return json.loads(raw)
                return None
            except Exception:  # noqa: BLE001 - degrade gracefully
                self.backend = "memory"
        entry = self._local.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if time.time() >= expires_at:
            self._local.pop(key, None)
            return None
        return json.loads(payload)

    def set(self, user_id: int, limit: int, value: Any) -> None:
        key = self._key(user_id, limit)
        payload = json.dumps(value)
        if self._redis is not None:
            try:
                self._redis.setex(key, self.ttl, payload)
                return
            except Exception:  # noqa: BLE001 - degrade gracefully
                self.backend = "memory"
        self._local[key] = (time.time() + self.ttl, payload)

    def invalidate(self, user_id: int, limit: int | None = None) -> None:
        """Invalidate one user's cached recommendation lists.

        If limit is provided, invalidate only that list size. Otherwise clear
        every cached list size for this user in the local fallback and Redis.
        """
        if limit is not None:
            keys = [self._key(user_id, limit)]
        else:
            prefix = f"reco:{user_id}:"
            keys = [key for key in self._local if key.startswith(prefix)]
            if self._redis is not None:
                try:
                    keys.extend(key.decode() if isinstance(key, bytes) else key
                                for key in self._redis.scan_iter(match=f"{prefix}*"))
                except Exception:  # noqa: BLE001
                    self.backend = "memory"
        for key in set(keys):
            self._local.pop(key, None)
            if self._redis is not None:
                try:
                    self._redis.delete(key)
                except Exception:  # noqa: BLE001
                    self.backend = "memory"

    def stats(self) -> dict:
        return {"backend": self.backend, "local_keys": len(self._local)}
