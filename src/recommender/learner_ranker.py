"""
learner_ranker.py
==================
Inference-time wrapper around the trained LightGBM ranker.

Design
------
* `preferred_difficulty` and `pace_preference` are computed dynamically from
  the learner's interaction history at inference time — identical to the
  training pipeline. They are NEVER read from a static CSV column.
* The caller can pass `user_pref_override` to inject a session-level
  preference (e.g., the learner selected "Advanced" in the UI today).
  This overrides the inferred value for this request only.
* Falls back gracefully (returns None) if model file is missing or
  lightgbm/joblib are not installed.

Usage
-----
    from src.recommender.learner_ranker import LearnerRanker
    ranker = LearnerRanker()
    if ranker.is_available():
        scores = ranker.score_candidates(
            profile=profile_series,
            candidates_df=enriched_candidates,
            interactions_df=interactions,
            transitions_df=transitions,
            user_pref_override={"preferred_difficulty": 3.0, "pace_preference": 2.0}
        )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.recommender.ranker_features import (
    compute_dynamic_preferences,
    DIFFICULTY_MAP,
    WORKLOAD_MAP,
)

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[2]
MODEL_DIR  = ROOT / "models"
MODEL_PATH = MODEL_DIR / "lgbm_ranker.pkl"

# ── Lazy imports ──────────────────────────────────────────────────────────────
try:
    import joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False


class LearnerRanker:
    """
    Loads the saved LightGBM model and scores candidate courses for a learner.
    Singleton — loads the model once and reuses across calls.
    """

    _instance: Optional["LearnerRanker"] = None

    def __new__(cls) -> "LearnerRanker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self) -> None:
        if self._loaded:
            return
        if not _JOBLIB_OK:
            log.warning("joblib not available — ML ranker disabled.")
            self._model: object       = None
            self._feat_cols: list[str] = []
            self._loaded = True
            return
        if not MODEL_PATH.exists():
            log.info(f"ML model not found at {MODEL_PATH}. Run: python train_ranker.py")
            self._model      = None
            self._feat_cols  = []
            self._loaded     = True
            return
        try:
            artifact       = joblib.load(MODEL_PATH)
            self._model    = artifact["model"]
            self._feat_cols = artifact["feature_cols"]
            auc            = artifact.get("auc", "?")
            log.info(f"✅ ML ranker loaded (AUC={auc}, {len(self._feat_cols)} features)")
        except Exception as e:
            log.warning(f"Failed to load ML model: {e}")
            self._model     = None
            self._feat_cols = []
        self._loaded = True

    def is_available(self) -> bool:
        """True if model is loaded and ready."""
        self._load()
        return self._model is not None

    # ── Feature row construction ──────────────────────────────────────────────

    def _build_feature_row(
        self,
        profile:           pd.Series,
        course:            pd.Series,
        dyn_prefs:         dict,
        prev_course_id:    Optional[str],
        transitions_df:    pd.DataFrame,
    ) -> dict:
        """
        Build feature dict for one (learner, candidate_course) pair.
        Exactly mirrors the column layout from ranker_features.generate_training_data().

        Parameters
        ----------
        dyn_prefs : output of compute_dynamic_preferences() — already accounts
                    for recency-decay and any UI override applied upstream.
        """
        pref_diff = dyn_prefs.get("preferred_difficulty", 2.0)
        pref_pace = dyn_prefs.get("pace_preference",      2.0)

        row: dict = {
            # Dynamic preferences (computed, not stored)
            "preferred_difficulty":            pref_diff,
            "pace_preference":                 pref_pace,
            "is_progressing":                  int(dyn_prefs.get("is_progressing", False)),
            # Static profile signals (real column names)
            "proficiency_score":               float(profile.get("proficiency_score",              0.5)),
            "workload_tolerance_score":        float(profile.get("workload_tolerance_score",       0.5)),
            "consistency_index":               float(profile.get("consistency_index",              0.5)),
            "curiosity_index":                 float(profile.get("curiosity_index",               0.5)),
            "completion_likelihood_baseline":  float(profile.get("completion_likelihood_baseline", 0.5)),
            "avg_completion_rate":             float(profile.get("avg_completion_rate",            0.5)),
            "avg_quiz_score":                  float(profile.get("avg_quiz_score",                0.5)),
            "avg_engagement_score":            float(profile.get("avg_engagement_score",          50.0)),
            "total_courses_completed":         float(profile.get("total_courses_completed",        0.0)),
        }

        # Course static features
        c_diff = float(DIFFICULTY_MAP.get(str(course.get("difficulty_level", "Intermediate")), 2.0))
        c_wl   = float(WORKLOAD_MAP .get(str(course.get("workload_bucket",   "Medium")),       2.0))
        row.update({
            "course_difficulty_val":    c_diff,
            "course_workload_val":      c_wl,
            "popularity_proxy":         float(course.get("popularity_proxy", 0.5)),
            "quality_proxy":            float(course.get("quality_proxy",    0.5)),
            "estimated_duration_hours": float(course.get("estimated_duration_hours", 12.0)),
        })

        # Affinity / delta
        row["difficulty_delta"]   = abs(c_diff - pref_diff)
        row["pace_delta"]         = abs(c_wl   - pref_pace)

        learner_dom   = str(profile.get("dominant_domain",  ""))
        secondary_dom = str(profile.get("secondary_domain", ""))
        course_dom    = str(course.get("inferred_domain",   ""))
        row["is_domain_match"]    = 1 if course_dom == learner_dom    else 0
        row["is_secondary_match"] = 1 if course_dom == secondary_dom  else 0

        # Transition signal
        trans_score = 0.0
        if prev_course_id and not transitions_df.empty:
            mask = (
                (transitions_df["prev_course_id"].astype(str) == str(prev_course_id)) &
                (transitions_df["next_course_id"].astype(str) == str(course.get("course_id", "")))
            )
            hits = transitions_df[mask]
            if not hits.empty:
                trans_score = float(hits["high_success_transition_score"].max())
        row["high_success_transition_score"] = trans_score

        # Behavioral context (unknown at inference → neutral defaults)
        row["streak_flag"]    = 0.0
        row["session_order"]  = 1.0
        row["recency_weight"] = 1.0   # treating inference as most-recent event

        return row

    # ── Public API ────────────────────────────────────────────────────────────

    def score_candidates(
        self,
        profile:             pd.Series,
        candidates_df:       pd.DataFrame,
        interactions_df:     pd.DataFrame,
        transitions_df:      pd.DataFrame,
        user_pref_override:  Optional[dict] = None,
    ) -> Optional[np.ndarray]:
        """
        Score each candidate course for a learner using the ML model.

        Parameters
        ----------
        profile              : Single-row Series from learner_profiles
        candidates_df        : Candidate courses DataFrame (from enriched_courses)
        interactions_df      : Full interactions DataFrame
        transitions_df       : Course transitions DataFrame
        user_pref_override   : Optional session-level preference from the UI.
                               Keys: 'preferred_difficulty' (1-3 float),
                                     'pace_preference' (1-3 float).
                               These override the inferred values for THIS call only.
                               Example: {"preferred_difficulty": 3.0}  ← "I want Advanced"

        Returns
        -------
        np.ndarray of predicted success probabilities (shape: n_candidates),
        or None if model unavailable.
        """
        self._load()
        if not self.is_available():
            return None

        learner_id = str(profile.get("learner_id", ""))

        # ── Compute dynamic preferences (always fresh from history) ───────────
        dyn_prefs = compute_dynamic_preferences(learner_id, interactions_df)

        # Apply UI session override (does NOT persist — only for this request)
        if user_pref_override:
            for key, val in user_pref_override.items():
                if key in dyn_prefs:
                    dyn_prefs[key] = float(val)
            log.debug(
                f"UI preference override applied for {learner_id}: {user_pref_override}"
            )

        # Most recent course (for transition signal)
        prev_course_id: Optional[str] = None
        if learner_id and not interactions_df.empty:
            li = interactions_df[interactions_df["learner_id"] == learner_id]
            if not li.empty:
                prev_course_id = str(
                    li.sort_values("timestamp", ascending=False).iloc[0]["course_id"]
                )

        # ── Build feature matrix ──────────────────────────────────────────────
        rows = []
        for _, course in candidates_df.iterrows():
            row = self._build_feature_row(
                profile, course, dyn_prefs, prev_course_id, transitions_df
            )
            rows.append(row)

        if not rows:
            return None

        feat_df = pd.DataFrame(rows)
        # Align columns to training order
        for col in self._feat_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        feat_df = feat_df[self._feat_cols].astype(float)

        try:
            probs = self._model.predict_proba(feat_df)[:, 1]
            return probs
        except Exception as e:
            log.warning(f"ML scoring failed: {e}")
            return None
