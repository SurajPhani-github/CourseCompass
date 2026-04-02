"""
content_similarity.py
======================
TF-IDF + cosine similarity engine.

Builds a per-course similarity lookup from enriched_courses combined_text
(title + slug + skills + domain). Caches the index to disk for speed.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_PATH = ROOT / "data" / "processed" / "enriched_courses.csv"
CACHE_PATH = ROOT / "data" / "processed" / ".tfidf_cache.pkl"


class ContentSimilarityEngine:
    """
    Builds TF-IDF vectors from enriched course text and provides fast
    nearest-neighbour lookup by cosine similarity.
    """

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None          # sparse (n_courses × vocab)
        self.course_ids: list[str] = []
        self.id_to_idx: dict[str, int] = {}

    def fit(self, enriched: pd.DataFrame) -> None:
        """Fit TF-IDF on the combined_text column."""
        log.info("Fitting TF-IDF vectorizer …")
        texts = (
            enriched["combined_text"].fillna("") + " " +
            enriched["skills_tags"].fillna("") + " " +
            enriched["inferred_domain"].fillna("")
        ).tolist()

        self.vectorizer = TfidfVectorizer(
            max_features=25_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.course_ids = enriched["course_id"].astype(str).tolist()
        self.id_to_idx = {cid: idx for idx, cid in enumerate(self.course_ids)}
        log.info(f"  TF-IDF matrix: {self.tfidf_matrix.shape}, vocab: {len(self.vectorizer.vocabulary_)}")

    def get_similar(
        self,
        course_id: str,
        top_n: int = 20,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Return list of (course_id, similarity_score) for the top_n most similar
        courses to the given course_id. Excludes the source course itself.
        """
        if course_id not in self.id_to_idx:
            return []
        idx = self.id_to_idx[course_id]
        course_vec = self.tfidf_matrix[idx]
        sims = cosine_similarity(course_vec, self.tfidf_matrix).flatten()
        sims[idx] = 0.0  # exclude self
        if exclude:
            for eid in exclude:
                eidx = self.id_to_idx.get(eid)
                if eidx is not None:
                    sims[eidx] = 0.0

        top_indices = np.argpartition(sims, -min(top_n, len(sims)))[-top_n:]
        top_indices = sorted(top_indices, key=lambda i: sims[i], reverse=True)
        return [(self.course_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def get_multi_similar(
        self,
        course_ids: list[str],
        top_n: int = 30,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        Average similarity across multiple seed courses (for learner recs).
        """
        valid_ids = [c for c in course_ids if c in self.id_to_idx]
        if not valid_ids:
            return []

        indices = [self.id_to_idx[c] for c in valid_ids]
        seed_vecs = self.tfidf_matrix[indices]
        # average of seed vectors
        avg_vec = np.asarray(seed_vecs.mean(axis=0))
        sims = cosine_similarity(avg_vec, self.tfidf_matrix).flatten()

        # zero out seeds and excluded
        for i in indices:
            sims[i] = 0.0
        if exclude:
            for eid in exclude:
                eidx = self.id_to_idx.get(eid)
                if eidx is not None:
                    sims[eidx] = 0.0

        top_indices = np.argpartition(sims, -min(top_n, len(sims)))[-top_n:]
        top_indices = sorted(top_indices, key=lambda i: sims[i], reverse=True)
        return [(self.course_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def query_text(
        self,
        query: str,
        top_n: int = 30,
        exclude: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Find courses most similar to a free-text query string."""
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        if exclude:
            for eid in exclude:
                eidx = self.id_to_idx.get(eid)
                if eidx is not None:
                    sims[eidx] = 0.0
        top_indices = np.argpartition(sims, -min(top_n, len(sims)))[-top_n:]
        top_indices = sorted(top_indices, key=lambda i: sims[i], reverse=True)
        return [(self.course_ids[i], float(sims[i])) for i in top_indices if sims[i] > 0]

    def save_cache(self) -> None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "tfidf_matrix": self.tfidf_matrix,
                "course_ids": self.course_ids,
                "id_to_idx": self.id_to_idx,
            }, f)
        log.info(f"  TF-IDF cache saved → {CACHE_PATH}")

    def load_cache(self) -> bool:
        if not CACHE_PATH.exists():
            return False
        log.info(f"Loading TF-IDF cache from {CACHE_PATH} …")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.tfidf_matrix = data["tfidf_matrix"]
        self.course_ids = data["course_ids"]
        self.id_to_idx = data["id_to_idx"]
        log.info(f"  Loaded. Matrix shape: {self.tfidf_matrix.shape}")
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Singleton loader
# ──────────────────────────────────────────────────────────────────────────────
_ENGINE: ContentSimilarityEngine | None = None


def get_engine(enriched: pd.DataFrame | None = None, force_rebuild: bool = False) -> ContentSimilarityEngine:
    """
    Returns the global engine, building or loading from cache as needed.
    If enriched is provided and no cache exists, builds and caches.
    """
    global _ENGINE
    if _ENGINE is not None and not force_rebuild:
        return _ENGINE

    engine = ContentSimilarityEngine()
    if not force_rebuild and engine.load_cache():
        _ENGINE = engine
        return _ENGINE

    if enriched is None:
        log.info(f"Loading enriched courses for engine build …")
        enriched = pd.read_csv(ENRICHED_PATH, low_memory=False)

    engine.fit(enriched)
    engine.save_cache()
    _ENGINE = engine
    return _ENGINE
