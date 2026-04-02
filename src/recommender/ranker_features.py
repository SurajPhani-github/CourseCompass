"""
ranker_features.py
==================
Feature engineering for the LightGBM ranking/classification model.

Key design principles
---------------------
* `preferred_difficulty` and `pace_preference` are NEVER stored statically.
  They are ALWAYS computed dynamically from interaction history so they
  evolve naturally as a learner progresses from Beginner → Advanced.
* An optional `user_pref_override` dict can inject explicit UI-provided
  preferences (e.g., learner says "I want Advanced courses today").
* All column names match the actual processed CSV schemas exactly.

Data sources and their actual schemas
--------------------------------------
profiles:      learner_id, proficiency_score, workload_tolerance_score,
               avg_completion_rate, avg_quiz_score, avg_engagement_score,
               total_courses_completed, consistency_index, curiosity_index,
               completion_likelihood_baseline, dominant_domain, secondary_domain,
               estimated_proficiency, workload_tolerance, momentum_trend
interactions:  learner_id, timestamp, course_id, difficulty_at_time,
               workload_bucket_at_time, completion_rate, learning_outcome,
               session_order, streak_flag, engagement_score, quiz_score
courses:       course_id, difficulty_level, workload_bucket, inferred_domain,
               popularity_proxy, quality_proxy, estimated_duration_hours
transitions:   prev_course_id, next_course_id, high_success_transition_score,
               transition_probability
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

# ── Ordinal maps ──────────────────────────────────────────────────────────────
DIFFICULTY_MAP   = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
DIFFICULTY_UNMAP = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
WORKLOAD_MAP     = {"Light": 1, "Medium": 2, "Heavy": 3}
OUTCOME_WEIGHT   = {"excellent": 1.0, "good": 0.75, "average": 0.5,
                    "weak": 0.25,  "dropped": 0.0}


# ── Dynamic preference computation ───────────────────────────────────────────

def compute_dynamic_preferences(
    learner_id:       str,
    interactions_df:  pd.DataFrame,
    decay_halflife:   float = 90.0,
    n_recent:         int   = 20,
) -> dict:
    """
    Compute `preferred_difficulty` and `pace_preference` dynamically from a
    learner's interaction history. Values update automatically as the learner
    progresses, without ever being stored in a CSV.

    Parameters
    ----------
    learner_id      : Target learner
    interactions_df : Full interactions dataframe
    decay_halflife  : Recency half-life in days (90 = recent matters 2× more)
    n_recent        : Number of most-recent interactions to consider

    Returns
    -------
    dict with keys:
        preferred_difficulty  (float, 1–3)
        pace_preference       (float, 1–3)
        preferred_difficulty_label   (str)
        pace_preference_label        (str)
        is_progressing        (bool)  True if difficulty trend is increasing
    """
    li = interactions_df[interactions_df["learner_id"] == learner_id].copy()
    if li.empty:
        return {
            "preferred_difficulty":       2.0,
            "pace_preference":            2.0,
            "preferred_difficulty_label": "Intermediate",
            "pace_preference_label":      "Medium",
            "is_progressing":             False,
        }

    li["timestamp"] = pd.to_datetime(li["timestamp"], errors="coerce")
    li = li.sort_values("timestamp", ascending=False).head(n_recent)

    t_max   = li["timestamp"].max()
    days_ago = (t_max - li["timestamp"]).dt.days.fillna(0).clip(lower=0)
    recency_w = np.exp(-days_ago / decay_halflife)

    # Map difficulty strings → numerics
    li["diff_val"] = li["difficulty_at_time"].map(DIFFICULTY_MAP)
    li["wl_val"]   = li["workload_bucket_at_time"].map(WORKLOAD_MAP)

    # Success-weight: only weight by interactions the learner engaged well with
    li["success_w"] = li["learning_outcome"].map(OUTCOME_WEIGHT).fillna(0.5)
    combined_w      = recency_w * li["success_w"]

    # Weighted average difficulty of courses they actually completed successfully
    if li["diff_val"].notna().any():
        pref_diff = float(
            np.average(li["diff_val"].fillna(2.0), weights=combined_w)
        )
    else:
        pref_diff = 2.0

    # Pace: weighted avg workload of their recent courses
    if li["wl_val"].notna().any():
        pref_pace = float(
            np.average(li["wl_val"].fillna(2.0), weights=combined_w)
        )
    else:
        pref_pace = 2.0

    # Progression detection: is the learner trending toward harder content?
    # Compare average difficulty of first-half vs second-half of recent history
    is_progressing = False
    if len(li) >= 6:
        half        = len(li) // 2
        older_half  = li.iloc[half:]["diff_val"].mean()
        newer_half  = li.iloc[:half]["diff_val"].mean()
        is_progressing = bool(newer_half > older_half + 0.1)

    # Map back to nearest label
    def nearest_label(val: float, mapping: dict) -> str:
        return min(mapping.items(), key=lambda kv: abs(kv[1] - val))[0]

    return {
        "preferred_difficulty":       round(pref_diff, 3),
        "pace_preference":            round(pref_pace, 3),
        "preferred_difficulty_label": nearest_label(pref_diff, DIFFICULTY_MAP),
        "pace_preference_label":      nearest_label(pref_pace, WORKLOAD_MAP),
        "is_progressing":             is_progressing,
    }


def compute_all_learner_preferences(
    interactions_df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """
    Batch-compute dynamic preferences for every learner.
    Returns a DataFrame indexed by learner_id with computed pref columns.
    Used by the training pipeline to generate per-learner dynamic features.
    """
    learner_ids = interactions_df["learner_id"].unique()
    records = []
    for lid in learner_ids:
        prefs = compute_dynamic_preferences(lid, interactions_df, **kwargs)
        prefs["learner_id"] = lid
        records.append(prefs)
    return pd.DataFrame(records)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(DATA_DIR / "synthetic_learner_interactions.csv", low_memory=False)
    courses      = pd.read_csv(DATA_DIR / "enriched_courses.csv",               low_memory=False)
    profiles     = pd.read_csv(DATA_DIR / "learner_profiles.csv",               low_memory=False)
    transitions  = pd.read_csv(DATA_DIR / "course_transitions.csv",             low_memory=False)
    return interactions, courses, profiles, transitions


def _recency_weights(df: pd.DataFrame, halflife: float = 90.0) -> pd.Series:
    """Row-level exponential decay weight based on timestamp."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    t_max    = df["timestamp"].max()
    days_ago = (t_max - df["timestamp"]).dt.days.fillna(0).clip(lower=0)
    return np.exp(-days_ago / halflife)


# ── Main training feature pipeline ───────────────────────────────────────────

def generate_training_data(
    user_pref_override: Optional[dict] = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame]:
    """
    Build the full feature matrix for training the LightGBM ranker.

    Parameters
    ----------
    user_pref_override : Optional dict to override computed preferences for
                         specific learners. Key = learner_id,
                         value = {preferred_difficulty: float, pace_preference: float}
                         This mirrors the UI "session preference" override.

    Returns
    -------
    X            : feature DataFrame
    y            : binary success label
    feature_cols : ordered list of feature column names
    df           : full merged DataFrame (for debugging)
    """
    log.info("Loading base data for ML Ranker…")
    interactions, courses, profiles, transitions = load_data()

    # ── 1. Compute dynamic preferences for all learners ──────────────────────
    log.info("Computing dynamic preferences from interaction history…")
    dyn_prefs = compute_all_learner_preferences(interactions)
    # dyn_prefs has: learner_id, preferred_difficulty, pace_preference, is_progressing

    # ── 2. Label: positive = completion ≥ 0.8 AND outcome ∈ {good, excellent} ─
    df = interactions.copy()
    df["is_success"] = (
        (df["completion_rate"] >= 0.8) &
        (df["learning_outcome"].isin(["good", "excellent"]))
    ).astype(int)

    # Recency weight per row (used as sample weight)
    df["recency_weight"] = _recency_weights(df)

    # ── 3. Merge profiles ─────────────────────────────────────────────────────
    log.info("Merging learner profiles…")
    profile_cols = [
        "learner_id",
        "proficiency_score",           # numeric proficiency (0-1 scale typically)
        "workload_tolerance_score",    # numeric workload tolerance
        "completion_likelihood_baseline",
        "avg_completion_rate",
        "avg_quiz_score",
        "avg_engagement_score",
        "total_courses_completed",
        "consistency_index",
        "curiosity_index",
        "dominant_domain",
        "secondary_domain",
    ]
    existing_profile_cols = [c for c in profile_cols if c in profiles.columns]
    df = df.merge(profiles[existing_profile_cols], on="learner_id", how="left")

    # ── 4. Attach dynamic preferences (always computed, never static) ─────────
    df = df.merge(dyn_prefs[["learner_id", "preferred_difficulty", "pace_preference",
                              "is_progressing"]], on="learner_id", how="left")

    # Apply UI-style per-learner override if provided (training-time injection)
    if user_pref_override:
        for lid, overrides in user_pref_override.items():
            mask = df["learner_id"] == lid
            for key, val in overrides.items():
                if key in df.columns:
                    df.loc[mask, key] = val

    df["preferred_difficulty"] = df["preferred_difficulty"].fillna(2.0)
    df["pace_preference"]      = df["pace_preference"].fillna(2.0)
    df["is_progressing"]       = df["is_progressing"].fillna(False).astype(int)

    # ── 5. Merge course features ──────────────────────────────────────────────
    log.info("Merging course features…")
    course_cols = [
        "course_id", "difficulty_level", "workload_bucket",
        "inferred_domain", "popularity_proxy", "quality_proxy",
        "estimated_duration_hours",
    ]
    existing_course_cols = [c for c in course_cols if c in courses.columns]
    df = df.merge(courses[existing_course_cols], on="course_id", how="left",
                  suffixes=("", "_course"))

    # ── 6. Ordinal encoding ───────────────────────────────────────────────────
    df["course_difficulty_val"] = df["difficulty_level"].map(DIFFICULTY_MAP).fillna(2.0)
    df["course_workload_val"]   = df["workload_bucket"].map(WORKLOAD_MAP).fillna(2.0)
    df["proficiency_score"]     = df["proficiency_score"].fillna(0.5)
    df["workload_tolerance_score"] = df["workload_tolerance_score"].fillna(0.5)

    # ── 7. Delta / affinity features ─────────────────────────────────────────
    df["difficulty_delta"]  = (df["course_difficulty_val"] - df["preferred_difficulty"]).abs()
    df["pace_delta"]        = (df["course_workload_val"]   - df["pace_preference"]).abs()

    # Domain match
    dom_profile_col = "dominant_domain"
    dom_course_col  = "inferred_domain" if "inferred_domain" in df.columns else "inferred_domain_course"
    df["is_domain_match"]     = (df[dom_profile_col] == df[dom_course_col]).astype(int)
    df["is_secondary_match"]  = (df["secondary_domain"] == df[dom_course_col]).astype(int)

    # ── 8. Transition features ────────────────────────────────────────────────
    log.info("Computing transition features…")
    df = df.sort_values(["learner_id", "timestamp"])
    df["prev_course_id"] = df.groupby("learner_id")["course_id"].shift(1)

    trans_join = transitions[["prev_course_id", "next_course_id", "high_success_transition_score"]].copy()
    trans_join = trans_join.rename(columns={"next_course_id": "course_id"})
    df = df.merge(trans_join, on=["prev_course_id", "course_id"], how="left")
    df["high_success_transition_score"] = df["high_success_transition_score"].fillna(0.0)

    # ── 9. Row-level behavioral signals ──────────────────────────────────────
    df["streak_flag"]   = df.get("streak_flag",   pd.Series(0, index=df.index)).fillna(0.0)
    df["session_order"] = df.get("session_order", pd.Series(1, index=df.index)).fillna(1.0)

    # ── 10. Fill remaining NaNs ───────────────────────────────────────────────
    fill_defaults = {
        "consistency_index":             0.5,
        "popularity_proxy":              0.5,
        "quality_proxy":                 0.5,
        "avg_completion_rate":           0.5,
        "avg_quiz_score":                0.5,
        "avg_engagement_score":          50.0,
        "total_courses_completed":       0.0,
        "curiosity_index":               0.5,
        "completion_likelihood_baseline": 0.5,
        "estimated_duration_hours":      12.0,
        "recency_weight":                1.0,
        "proficiency_score":             0.5,
        "workload_tolerance_score":      0.5,
    }
    for col, default in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)
        else:
            df[col] = default

    # ── 11. Feature list ──────────────────────────────────────────────────────
    feature_cols = [
        # Dynamic user preference (computed, not stored)
        "preferred_difficulty",       # rolling inferred from history
        "pace_preference",            # rolling inferred from history
        "is_progressing",             # learner trending toward harder courses?
        # Static profile signals
        "proficiency_score",
        "workload_tolerance_score",
        "consistency_index",
        "curiosity_index",
        "completion_likelihood_baseline",
        "avg_completion_rate",
        "avg_quiz_score",
        "avg_engagement_score",
        "total_courses_completed",
        # Course static features
        "course_difficulty_val",
        "course_workload_val",
        "popularity_proxy",
        "quality_proxy",
        "estimated_duration_hours",
        # Affinity / match features
        "difficulty_delta",
        "pace_delta",
        "is_domain_match",
        "is_secondary_match",
        # Sequence signal
        "high_success_transition_score",
        # Behavioral context
        "streak_flag",
        "session_order",
        "recency_weight",
    ]

    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols].copy().astype(float)
    y = df["is_success"].copy()

    log.info(
        f"Feature matrix: {X.shape} | "
        f"Positive rate: {y.mean():.2%} | "
        f"Features ({len(feature_cols)}): {feature_cols}"
    )
    return X, y, feature_cols, df


if __name__ == "__main__":
    X, y, features, df = generate_training_data()
    print("Features:", features)
    print("Shape:", X.shape)
    print("Label balance:", y.value_counts().to_dict())

    # Show sample of dynamic preference vs eventual success
    sample = df[["learner_id", "preferred_difficulty", "pace_preference",
                 "is_progressing", "is_success"]].head(10)
    print("\nSample rows:\n", sample.to_string())
