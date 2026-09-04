"""
Content-based retrieval.

Products are embedded from title/description/category/brand/attributes. In
this repo we default to a TF-IDF + SVD embedding (fast, dependency-light, no
network/model download needed) behind the exact same interface a
sentence-transformers embedding would use — swap `TfidfProductEmbedder` for
`SentenceTransformerProductEmbedder` in production without touching callers.

Nearest-neighbor search is delegated to `src/serving/vector_store.py`
(Qdrant-backed, with a numpy fallback), so this module and the API share one
ANN implementation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def _product_text(row) -> str:
    return f"{row['title']} {row['description']} {row['category']} {row['brand']}"


class TfidfProductEmbedder:
    """Lightweight stand-in for a 768-dim sentence-transformer embedding."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
        self.svd = TruncatedSVD(n_components=dim, random_state=42)
        self.item_ids_: np.ndarray | None = None
        self.embeddings_: np.ndarray | None = None

    def fit_transform(self, products: pd.DataFrame) -> np.ndarray:
        texts = products.apply(_product_text, axis=1)
        tfidf = self.vectorizer.fit_transform(texts)
        n_features = tfidf.shape[1]
        n_samples = tfidf.shape[0]
        effective_dim = max(2, min(self.dim, n_features - 1, n_samples - 1))
        if effective_dim != self.svd.n_components:
            self.svd = TruncatedSVD(n_components=effective_dim, random_state=42)
        emb = self.svd.fit_transform(tfidf)
        emb = normalize(emb)
        self.item_ids_ = products["item_id"].to_numpy()
        self.embeddings_ = emb
        return emb

    def transform(self, products: pd.DataFrame) -> np.ndarray:
        texts = products.apply(_product_text, axis=1)
        tfidf = self.vectorizer.transform(texts)
        return normalize(self.svd.transform(tfidf))


class ContentBasedRecommender:
    def __init__(self, embedder: TfidfProductEmbedder | None = None):
        self.embedder = embedder or TfidfProductEmbedder()
        self.products_: pd.DataFrame | None = None

    def fit(self, products: pd.DataFrame):
        self.products_ = products.reset_index(drop=True)
        self.embedder.fit_transform(self.products_)
        return self

    def similar_items(self, item_id: int, k: int = 20) -> list[tuple[int, float]]:
        emb = self.embedder.embeddings_
        ids = self.embedder.item_ids_
        idx = np.where(ids == item_id)[0]
        if len(idx) == 0:
            return []
        query = emb[idx[0]]
        sims = emb @ query  # cosine sim, embeddings are normalized
        order = np.argsort(-sims)
        results = [(int(ids[i]), float(sims[i])) for i in order if ids[i] != item_id][:k]
        return results

    def candidates_for_user_history(self, liked_item_ids: list[int], n: int = 200) -> pd.DataFrame:
        """Average the embeddings of items the user liked, retrieve nearest neighbors."""
        emb = self.embedder.embeddings_
        ids = self.embedder.item_ids_
        idxs = [np.where(ids == i)[0][0] for i in liked_item_ids if i in ids]
        if not idxs:
            return pd.DataFrame(columns=["item_id", "content_score"])
        profile_vec = emb[idxs].mean(axis=0)
        profile_vec = profile_vec / (np.linalg.norm(profile_vec) + 1e-9)
        sims = emb @ profile_vec
        order = np.argsort(-sims)[:n]
        return pd.DataFrame({
            "item_id": ids[order].astype(int),
            "content_score": sims[order],
        })
