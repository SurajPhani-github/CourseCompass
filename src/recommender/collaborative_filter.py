"""
collaborative_filter.py
========================
Implements a lightweight collaborative filtering recommendation layer
using the implicit library (ALS - Alternating Least Squares).

This module reads synthetic learner interactions, builds a user-item
interaction matrix (using completion_rate as implicit confidence),
trains an ALS model, and provides peer-based course suggestions.

When interaction data is sparse (cold-start), falls back gracefully
to the existing content-based ranker.
"""

import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
MODEL_CACHE = ROOT / "data" / "processed" / "cf_als_model.pkl"


class CollaborativeFilter:
    """
    Alternating Least Squares collaborative filtering over learner-course
    interactions using completion rate as implicit feedback signal.
    """

    def __init__(self, interactions_df: pd.DataFrame):
        self.interactions = interactions_df
        self.model = None
        self.user_map = {}  # learner_id -> matrix row index
        self.item_map = {}  # course_id -> matrix col index
        self.reverse_item_map = {}  # col index -> course_id
        self.interaction_matrix = None

    def _build_interaction_matrix(self):
        """
        Constructs a sparse user-item matrix where cell values represent
        completion_rate (0.0 to 1.0) as a confidence signal.
        """
        df = self.interactions.copy()

        # Build mappings
        unique_users = df["learner_id"].unique()
        unique_items = df["course_id"].unique()

        self.user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_map = {cid: idx for idx, cid in enumerate(unique_items)}
        self.reverse_item_map = {idx: cid for cid, idx in self.item_map.items()}

        # Create sparse matrix
        rows = df["learner_id"].map(self.user_map).values
        cols = df["course_id"].map(self.item_map).values
        
        # Use completion_rate as confidence. Higher completion = stronger signal.
        vals = df["completion_rate"].fillna(0.3).values.astype(np.float32)

        n_users = len(unique_users)
        n_items = len(unique_items)

        self.interaction_matrix = csr_matrix(
            (vals, (rows, cols)), shape=(n_users, n_items)
        )
        log.info(f"Built interaction matrix: {n_users} users × {n_items} items")

    def train(self, factors: int = 50, iterations: int = 15, force_rebuild: bool = False):
        """
        Trains the ALS model. Caches to disk for fast reuse.
        Uses a pure numpy SVD fallback if the `implicit` library is unavailable.
        """
        if not force_rebuild and MODEL_CACHE.exists():
            try:
                with open(MODEL_CACHE, "rb") as f:
                    cached = pickle.load(f)
                self.model = cached["model"]
                self.user_map = cached["user_map"]
                self.item_map = cached["item_map"]
                self.reverse_item_map = cached["reverse_item_map"]
                self.interaction_matrix = cached["interaction_matrix"]
                log.info("Loaded cached CF model.")
                return
            except Exception as e:
                log.warning(f"Cache load failed, retraining: {e}")

        self._build_interaction_matrix()

        try:
            from implicit.als import AlternatingLeastSquares
            self.model = AlternatingLeastSquares(
                factors=factors,
                iterations=iterations,
                regularization=0.1,
                random_state=42,
            )
            self.model.fit(self.interaction_matrix)
            log.info("Trained implicit ALS model.")
        except ImportError:
            log.warning("implicit library not installed. Using SVD fallback.")
            # SVD-based fallback
            dense = self.interaction_matrix.toarray()
            U, sigma, Vt = np.linalg.svd(dense, full_matrices=False)
            k = min(factors, len(sigma))
            self.model = {
                "type": "svd",
                "U": U[:, :k],
                "sigma": sigma[:k],
                "Vt": Vt[:k, :],
            }

        # Cache
        try:
            with open(MODEL_CACHE, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "user_map": self.user_map,
                    "item_map": self.item_map,
                    "reverse_item_map": self.reverse_item_map,
                    "interaction_matrix": self.interaction_matrix,
                }, f)
        except Exception as e:
            log.warning(f"Could not cache CF model: {e}")

    def recommend_for_user(self, learner_id: str, top_n: int = 10) -> list[dict]:
        """
        Returns top-N collaborative recommendations for a learner.

        Returns a list of dicts: [{"course_id": ..., "cf_score": ...}, ...]
        If the learner is unknown (cold-start), returns an empty list.
        """
        if self.model is None:
            log.warning("CF model not trained. Call .train() first.")
            return []

        if learner_id not in self.user_map:
            log.info(f"Cold-start: learner {learner_id} not in CF matrix.")
            return []

        user_idx = self.user_map[learner_id]

        # Get already-interacted items to filter out
        interacted = set(self.interaction_matrix[user_idx].nonzero()[1])

        if isinstance(self.model, dict) and self.model.get("type") == "svd":
            # SVD fallback scoring
            user_vec = self.model["U"][user_idx] * self.model["sigma"]
            scores = user_vec @ self.model["Vt"]
            ranked = np.argsort(scores)[::-1]
        else:
            # implicit library model
            try:
                ids, scores = self.model.recommend(
                    user_idx,
                    self.interaction_matrix[user_idx],
                    N=top_n + len(interacted),
                    filter_already_liked_items=True,
                )
                results = []
                for item_idx, score in zip(ids, scores):
                    cid = self.reverse_item_map.get(item_idx)
                    if cid is not None:
                        results.append({"course_id": cid, "cf_score": float(score)})
                return results[:top_n]
            except Exception as e:
                log.error(f"implicit recommend failed: {e}")
                return []

        # For SVD fallback
        results = []
        for item_idx in ranked:
            if item_idx in interacted:
                continue
            cid = self.reverse_item_map.get(item_idx)
            if cid is not None:
                results.append({"course_id": cid, "cf_score": float(scores[item_idx])})
            if len(results) >= top_n:
                break

        return results
