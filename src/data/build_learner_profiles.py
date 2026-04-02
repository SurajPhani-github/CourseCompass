"""
build_learner_profiles.py
==========================
Step 3 of the data pipeline.

Reads  : data/processed/synthetic_learner_interactions.csv
         data/processed/enriched_courses.csv
Writes : data/processed/learner_profiles.csv

Aggregates raw interactions into per-learner profile features used by recs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
INTERACTIONS_PATH = ROOT / "data" / "processed" / "synthetic_learner_interactions.csv"
ENRICHED_PATH = ROOT / "data" / "processed" / "enriched_courses.csv"
OUT_PATH = ROOT / "data" / "processed" / "learner_profiles.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Proficiency estimation
# ──────────────────────────────────────────────────────────────────────────────

def estimate_proficiency(avg_quiz: float, avg_completion: float, n_completed: int,
                          avg_diff: float, recent_momentum: float) -> tuple[str, float]:
    """
    Weighted rule-based proficiency score → label.
    Weights: quiz 30%, completion 25%, count 20%, difficulty 15%, momentum 10%.
    """
    quiz_norm = avg_quiz / 100.0
    completion_norm = avg_completion
    # n_completed: cap at 30 for scoring, normalize to [0, 1]
    count_norm = min(n_completed, 30) / 30.0
    # difficulty: 1–3, normalize to [0, 1]
    diff_norm = (avg_diff - 1) / 2.0
    momentum_norm = max(0.0, min(1.0, recent_momentum))

    score = (
        0.30 * quiz_norm +
        0.25 * completion_norm +
        0.20 * count_norm +
        0.15 * diff_norm +
        0.10 * momentum_norm
    )
    score = float(np.clip(score, 0, 1))

    if score < 0.45:
        label = "Beginner"
    elif score < 0.72:
        label = "Intermediate"
    else:
        label = "Advanced"

    return label, round(score, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Workload tolerance
# ──────────────────────────────────────────────────────────────────────────────

def estimate_workload_tolerance(interactions: pd.DataFrame, enriched: pd.DataFrame) -> tuple[str, float]:
    """Infer workload tolerance from completion rates on medium/heavy courses."""
    if len(interactions) == 0:
        return "Light", 1.0

    merged = interactions.merge(
        enriched[["course_id", "workload_bucket", "workload_score", "estimated_duration_hours"]],
        on="course_id", how="left"
    )
    # avg time spent as fraction of expected duration
    merged["time_fraction"] = merged["time_spent_minutes"] / (merged["estimated_duration_hours"] * 60 + 0.1)

    heavy_comp = merged.loc[merged["workload_bucket"] == "Heavy", "completion_rate"].mean()
    medium_comp = merged.loc[merged["workload_bucket"] == "Medium", "completion_rate"].mean()
    heavy_n = (merged["workload_bucket"] == "Heavy").sum()
    medium_n = (merged["workload_bucket"] == "Medium").sum()

    avg_wl_score = merged["workload_score"].mean() if "workload_score" in merged else 2.0

    if (not np.isnan(heavy_comp) and heavy_comp > 0.55 and heavy_n >= 2):
        return "Heavy", 3.0
    elif (not np.isnan(medium_comp) and medium_comp > 0.55 and medium_n >= 2):
        return "Medium", 2.0
    elif avg_wl_score < 1.8:
        return "Light", 1.0
    else:
        return "Medium", 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Momentum trend
# ──────────────────────────────────────────────────────────────────────────────

def compute_momentum(interactions: pd.DataFrame) -> tuple[str, float]:
    """
    Compare recent 5 vs previous 5 on engagement, completion, quiz.
    Returns (trend_label, score_delta).
    """
    if len(interactions) < 5:
        return "stable", 0.0

    sorted_int = interactions.sort_values("timestamp")
    recent = sorted_int.tail(5)
    prev = sorted_int.iloc[-10:-5] if len(sorted_int) >= 10 else sorted_int.head(min(5, len(sorted_int) - 5))

    if len(prev) == 0:
        return "stable", 0.0

    def safe_mean(df: pd.DataFrame, col: str) -> float:
        v = df[col].dropna()
        return float(v.mean()) if len(v) > 0 else 0.0

    r_eng = safe_mean(recent, "engagement_score") / 100
    p_eng = safe_mean(prev, "engagement_score") / 100
    r_comp = safe_mean(recent, "completion_rate")
    p_comp = safe_mean(prev, "completion_rate")
    r_quiz = safe_mean(recent, "quiz_score") / 100
    p_quiz = safe_mean(prev, "quiz_score") / 100

    delta = (0.4 * (r_eng - p_eng) + 0.4 * (r_comp - p_comp) + 0.2 * (r_quiz - p_quiz))

    if delta > 0.05:
        return "improving", round(delta, 4)
    elif delta < -0.05:
        return "declining", round(delta, 4)
    return "stable", round(delta, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Profile builder
# ──────────────────────────────────────────────────────────────────────────────

def build_profile(learner_id: str, interactions: pd.DataFrame, enriched: pd.DataFrame) -> dict:
    interactions = interactions.sort_values("timestamp")

    n_int = len(interactions)
    n_courses_seen = interactions["course_id"].nunique()
    completed = interactions[interactions["event_type"] == "completed"]
    n_completed = len(completed)

    avg_time = float(interactions["time_spent_minutes"].mean()) if n_int > 0 else 0.0
    avg_eng = float(interactions["engagement_score"].mean()) if n_int > 0 else 0.0
    avg_quiz = float(interactions["quiz_score"].dropna().mean()) if interactions["quiz_score"].notna().any() else 0.0
    avg_comp = float(interactions["completion_rate"].mean()) if n_int > 0 else 0.0

    # Completion consistency (std dev of completion rates, inverted)
    comp_std = float(interactions["completion_rate"].std()) if n_int > 1 else 0.0
    completion_consistency = round(1.0 - min(comp_std, 1.0), 4)

    # Domain distributions
    domain_counts = interactions["domain_at_time"].value_counts()
    dominant_domain = domain_counts.index[0] if len(domain_counts) > 0 else "General Studies"
    secondary_domain = domain_counts.index[1] if len(domain_counts) > 1 else dominant_domain
    domain_diversity = round(float(len(domain_counts)) / max(n_courses_seen, 1), 4)

    # Preferred content type
    ct_counts = interactions["content_type"].value_counts()
    preferred_content_type = ct_counts.index[0] if len(ct_counts) > 0 else "video"

    # Workload tolerance
    wl_tolerance, wl_score = estimate_workload_tolerance(interactions, enriched)

    # Proficiency from completed courses
    completed_merged = completed.merge(
        enriched[["course_id", "difficulty_score"]],
        on="course_id", how="left"
    )
    avg_completed_diff = float(completed_merged["difficulty_score"].mean()) if n_completed > 0 else 1.0

    # Momentum
    momentum_trend, momentum_delta = compute_momentum(interactions)
    # Normalize delta to [0, 1] for proficiency calc
    recent_momentum = max(0.0, min(1.0, 0.5 + momentum_delta))

    proficiency_label, prof_score = estimate_proficiency(
        avg_quiz, avg_comp, n_completed, avg_completed_diff, recent_momentum
    )

    # Recent interaction stats
    recent_5 = interactions.tail(5)
    recent_3 = interactions.tail(3)
    recent_avg_eng = float(recent_5["engagement_score"].mean()) if len(recent_5) > 0 else avg_eng
    recent_avg_comp = float(recent_5["completion_rate"].mean()) if len(recent_5) > 0 else avg_comp
    recent_domain_focus = (
        recent_5["domain_at_time"].value_counts().index[0]
        if len(recent_5) > 0 else dominant_domain
    )

    recent_3_courses = "|".join([str(c) for c in recent_3["course_id"].tolist() if str(c) != "nan"])
    recent_5_courses = "|".join([str(c) for c in recent_5["course_id"].tolist() if str(c) != "nan"])

    # Completion likelihood baseline
    completion_likelihood = round(min(1.0, avg_comp * 0.6 + prof_score * 0.4), 4)

    # Dropout risk profile
    dropout_risk_vals = interactions["dropout_risk_signal"].dropna()
    dropout_risk = float(dropout_risk_vals.mean()) if len(dropout_risk_vals) > 0 else 0.5
    if dropout_risk > 0.65:
        dropout_profile = "high"
    elif dropout_risk > 0.40:
        dropout_profile = "medium"
    else:
        dropout_profile = "low"

    # Curiosity index (exploration of diverse domains)
    curiosity_index = round(min(1.0, domain_diversity + interactions["revisit_flag"].mean() * 0.2), 4)

    # Consistency index
    streak_rate = float(interactions["streak_flag"].mean()) if n_int > 0 else 0.0
    consistency_index = round(min(1.0, completion_consistency * 0.6 + streak_rate * 0.4), 4)

    # Last active
    last_active = interactions["timestamp"].max()

    return {
        "learner_id": learner_id,
        "total_interactions": n_int,
        "total_courses_seen": n_courses_seen,
        "total_courses_completed": n_completed,
        "avg_time_spent_minutes": round(avg_time, 2),
        "avg_engagement_score": round(avg_eng, 2),
        "avg_quiz_score": round(avg_quiz, 2),
        "avg_completion_rate": round(avg_comp, 4),
        "completion_consistency": completion_consistency,
        "dominant_domain": dominant_domain,
        "secondary_domain": secondary_domain,
        "domain_diversity": domain_diversity,
        "preferred_content_type": preferred_content_type,
        "workload_tolerance": wl_tolerance,
        "workload_tolerance_score": wl_score,
        "estimated_proficiency": proficiency_label,
        "proficiency_score": prof_score,
        "momentum_trend": momentum_trend,
        "recent_avg_engagement": round(recent_avg_eng, 2),
        "recent_avg_completion": round(recent_avg_comp, 4),
        "recent_domain_focus": recent_domain_focus,
        "recent_3_courses": recent_3_courses,
        "recent_5_courses": recent_5_courses,
        "completion_likelihood_baseline": completion_likelihood,
        "dropout_risk_profile": dropout_profile,
        "curiosity_index": curiosity_index,
        "consistency_index": consistency_index,
        "last_active_at": str(last_active),
    }


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading interactions from {INTERACTIONS_PATH}")
    interactions_df = pd.read_csv(INTERACTIONS_PATH, low_memory=False)
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    log.info(f"Loaded {len(interactions_df):,} interactions for {interactions_df['learner_id'].nunique()} learners")

    log.info(f"Loading enriched courses from {ENRICHED_PATH}")
    enriched = pd.read_csv(ENRICHED_PATH, low_memory=False)

    profiles: list[dict] = []
    learner_ids = sorted(interactions_df["learner_id"].unique())
    for i, lid in enumerate(learner_ids):
        learner_interactions = interactions_df[interactions_df["learner_id"] == lid].copy()
        profile = build_profile(lid, learner_interactions, enriched)
        profiles.append(profile)
        if (i + 1) % 200 == 0:
            log.info(f"  Profiled {i + 1}/{len(learner_ids)} learners …")

    profiles_df = pd.DataFrame(profiles)
    profiles_df.to_csv(OUT_PATH, index=False)
    log.info(f"Saved {len(profiles_df):,} learner profiles → {OUT_PATH}")

    # Summary stats
    log.info(f"  Proficiency distribution:\n{profiles_df['estimated_proficiency'].value_counts().to_string()}")
    log.info(f"  Momentum distribution:\n{profiles_df['momentum_trend'].value_counts().to_string()}")
    log.info(f"  Dominant domain top-10:\n{profiles_df['dominant_domain'].value_counts().head(10).to_string()}")


if __name__ == "__main__":
    main()
