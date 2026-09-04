"""
Vector store for semantic product retrieval. Wraps Qdrant; falls back to an
in-memory brute-force numpy cosine search if Qdrant is unreachable (per the
"Qdrant unavailable" graceful-degradation test). Both paths implement the
same `.upsert` / `.search` interface.
"""
from __future__ import annotations

import os

import numpy as np

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "recoflow_products"


class VectorStore:
    def __init__(self, dim: int, url: str = QDRANT_URL):
        self.dim = dim
        self._client = None
        self._local_ids: np.ndarray | None = None
        self._local_vecs: np.ndarray | None = None
        self.backend = "memory"
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams

            client = QdrantClient(url=url, timeout=1.0)
            client.get_collections()  # connectivity check
            if not client.collection_exists(COLLECTION_NAME):
                client.create_collection(
                    COLLECTION_NAME,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                )
            self._client = client
            self.backend = "qdrant"
        except Exception:
            self._client = None
            self.backend = "memory"

    def upsert(self, item_ids: np.ndarray, embeddings: np.ndarray) -> None:
        if self._client is not None:
            try:
                from qdrant_client.http.models import PointStruct
                points = [
                    PointStruct(id=int(i), vector=embeddings[idx].tolist())
                    for idx, i in enumerate(item_ids)
                ]
                self._client.upsert(COLLECTION_NAME, points=points)
                return
            except Exception:
                self.backend = "memory"
        self._local_ids = np.asarray(item_ids)
        self._local_vecs = np.asarray(embeddings)

    def search(self, query_vec: np.ndarray, k: int = 20) -> list[tuple[int, float]]:
        if self._client is not None:
            try:
                hits = self._client.search(COLLECTION_NAME, query_vector=query_vec.tolist(), limit=k)
                return [(int(h.id), float(h.score)) for h in hits]
            except Exception:
                self.backend = "memory"
        if self._local_vecs is None:
            return []
        sims = self._local_vecs @ query_vec
        order = np.argsort(-sims)[:k]
        return [(int(self._local_ids[i]), float(sims[i])) for i in order]
