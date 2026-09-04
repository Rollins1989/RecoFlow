"""
Collaborative filtering on implicit feedback.

Implements Weighted Regularized Matrix Factorization (the "implicit ALS"
algorithm from Hu, Koren & Volinsky 2008) from scratch in numpy/scipy so the
repo has zero exotic native-build dependencies. The `implicit` package's ALS
or a BPR loss are drop-in swaps behind the same `.fit/.recommend` interface
noted in the docstring below — this is the "Version 1 baseline you'd swap for
a bigger library later" implementation the design doc calls for.

Confidence follows Hu et al.: c_ui = 1 + alpha * r_ui, where r_ui is the
implicit-feedback weight (purchase=5, add_to_cart=3, wishlist=2, click=1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

EVENT_WEIGHTS = {"click": 1.0, "wishlist": 2.0, "add_to_cart": 3.0, "purchase": 5.0}


class ImplicitALS:
    def __init__(self, n_factors: int = 32, alpha: float = 40.0, reg: float = 0.05,
                 n_iter: int = 10, seed: int = 42):
        self.n_factors = n_factors
        self.alpha = alpha
        self.reg = reg
        self.n_iter = n_iter
        self.rng = np.random.default_rng(seed)
        self.user_ids_: np.ndarray | None = None
        self.item_ids_: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def _build_matrix(self, events: pd.DataFrame) -> csr_matrix:
        events = events[events["event"].isin(EVENT_WEIGHTS)].copy()
        events["w"] = events["event"].map(EVENT_WEIGHTS)
        agg = events.groupby(["user_id", "item_id"])["w"].sum().reset_index()

        self.user_ids_ = np.sort(agg["user_id"].unique())
        self.item_ids_ = np.sort(agg["item_id"].unique())
        u_idx = {u: i for i, u in enumerate(self.user_ids_)}
        i_idx = {it: i for i, it in enumerate(self.item_ids_)}

        rows = agg["user_id"].map(u_idx).to_numpy()
        cols = agg["item_id"].map(i_idx).to_numpy()
        vals = agg["w"].to_numpy()
        return csr_matrix((vals, (rows, cols)),
                           shape=(len(self.user_ids_), len(self.item_ids_)))

    def fit(self, events: pd.DataFrame):
        R = self._build_matrix(events)
        n_users, n_items = R.shape
        f = self.n_factors
        self.U = 0.01 * self.rng.standard_normal((n_users, f))
        self.V = 0.01 * self.rng.standard_normal((n_items, f))

        C = R.copy()
        C.data = 1.0 + self.alpha * C.data  # confidence
        P = R.copy()
        P.data = np.ones_like(P.data)  # preference indicator (1 where observed)

        reg_eye = self.reg * np.eye(f)
        for _ in range(self.n_iter):
            VtV = self.V.T @ self.V
            for u in range(n_users):
                start, end = C.indptr[u], C.indptr[u + 1]
                item_idx = C.indices[start:end]
                if len(item_idx) == 0:
                    continue
                c_u = C.data[start:end]
                Vu = self.V[item_idx]
                Cu_minus_I = np.diag(c_u - 1.0)
                A = VtV + Vu.T @ Cu_minus_I @ Vu + reg_eye
                b = Vu.T @ (c_u * 1.0)
                self.U[u] = np.linalg.solve(A, b)

            Ct = C.tocsc()
            UtU = self.U.T @ self.U
            for it in range(n_items):
                start, end = Ct.indptr[it], Ct.indptr[it + 1]
                user_idx = Ct.indices[start:end]
                if len(user_idx) == 0:
                    continue
                c_i = Ct.data[start:end]
                Ui = self.U[user_idx]
                Ci_minus_I = np.diag(c_i - 1.0)
                A = UtU + Ui.T @ Ci_minus_I @ Ui + reg_eye
                b = Ui.T @ (c_i * 1.0)
                self.V[it] = np.linalg.solve(A, b)
        return self

    def score_all_items(self, user_id: int) -> pd.Series | None:
        if self.user_ids_ is None or user_id not in self.user_ids_:
            return None
        u = np.where(self.user_ids_ == user_id)[0][0]
        scores = self.V @ self.U[u]
        return pd.Series(scores, index=self.item_ids_).sort_values(ascending=False)

    def candidates(self, user_id: int, n: int = 400) -> pd.DataFrame:
        scores = self.score_all_items(user_id)
        if scores is None:
            return pd.DataFrame(columns=["item_id", "cf_score"])
        top = scores.head(n)
        return pd.DataFrame({"item_id": top.index, "cf_score": top.values})
