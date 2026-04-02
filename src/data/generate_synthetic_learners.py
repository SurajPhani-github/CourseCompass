"""
generate_synthetic_learners.py
================================
Step 2 of the data pipeline.

Reads  : data/processed/enriched_courses.csv
Writes : data/processed/synthetic_learner_interactions.csv

Generates 1500 synthetic learners, each with a realistic archetype,
producing 5–40 chronological course interactions per learner.
"""

from __future__ import annotations

import uuid
import logging
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

ROOT = Path(__file__).resolve().parents[2]
ENRICHED_PATH = ROOT / "data" / "processed" / "enriched_courses.csv"
OUT_PATH = ROOT / "data" / "processed" / "synthetic_learner_interactions.csv"

N_LEARNERS = 1500
SIM_START = datetime(2023, 1, 1)
SIM_END = datetime(2024, 12, 31)

# ──────────────────────────────────────────────────────────────────────────────
# Learner archetypes
# Each archetype defines distributions over hidden traits.
# ──────────────────────────────────────────────────────────────────────────────

ARCHETYPES = [
    {
        "name": "Focused Beginner",
        "weight": 0.15,
        "proficiency_start": (0.1, 0.35),   # (low, high) uniform range
        "n_interactions": (5, 15),
        "exploration_tendency": 0.10,        # probability of exploring secondary domain
        "consistency_level": (0.6, 0.9),     # probability of completing a course
        "engagement_baseline": (50, 75),
        "quiz_strength": (45, 70),
        "workload_pref": ["Light", "Medium"],
        "preferred_difficulty": ["Beginner"],
        "preferred_content": ["video", "reading"],
    },
    {
        "name": "Career Switcher",
        "weight": 0.15,
        "proficiency_start": (0.3, 0.55),
        "n_interactions": (15, 35),
        "exploration_tendency": 0.25,
        "consistency_level": (0.5, 0.85),
        "engagement_baseline": (55, 80),
        "quiz_strength": (50, 75),
        "workload_pref": ["Medium", "Heavy"],
        "preferred_difficulty": ["Beginner", "Intermediate"],
        "preferred_content": ["video", "practice", "quiz"],
    },
    {
        "name": "High Performer Specialist",
        "weight": 0.12,
        "proficiency_start": (0.6, 0.85),
        "n_interactions": (20, 40),
        "exploration_tendency": 0.10,
        "consistency_level": (0.8, 0.98),
        "engagement_baseline": (70, 95),
        "quiz_strength": (75, 98),
        "workload_pref": ["Medium", "Heavy"],
        "preferred_difficulty": ["Intermediate", "Advanced"],
        "preferred_content": ["quiz", "practice", "project"],
    },
    {
        "name": "Casual Explorer",
        "weight": 0.15,
        "proficiency_start": (0.2, 0.50),
        "n_interactions": (5, 20),
        "exploration_tendency": 0.55,
        "consistency_level": (0.3, 0.65),
        "engagement_baseline": (35, 65),
        "quiz_strength": (35, 60),
        "workload_pref": ["Light"],
        "preferred_difficulty": ["Beginner"],
        "preferred_content": ["video", "reading"],
    },
    {
        "name": "Certification Chaser",
        "weight": 0.12,
        "proficiency_start": (0.4, 0.70),
        "n_interactions": (20, 40),
        "exploration_tendency": 0.08,
        "consistency_level": (0.75, 0.96),
        "engagement_baseline": (60, 88),
        "quiz_strength": (70, 95),
        "workload_pref": ["Medium", "Heavy"],
        "preferred_difficulty": ["Intermediate", "Advanced"],
        "preferred_content": ["quiz", "practice", "reading"],
    },
    {
        "name": "Creative Learner",
        "weight": 0.10,
        "proficiency_start": (0.2, 0.55),
        "n_interactions": (8, 25),
        "exploration_tendency": 0.35,
        "consistency_level": (0.50, 0.80),
        "engagement_baseline": (55, 80),
        "quiz_strength": (40, 70),
        "workload_pref": ["Light", "Medium"],
        "preferred_difficulty": ["Beginner", "Intermediate"],
        "preferred_content": ["video", "project"],
    },
    {
        "name": "Academic Builder",
        "weight": 0.11,
        "proficiency_start": (0.45, 0.75),
        "n_interactions": (15, 35),
        "exploration_tendency": 0.20,
        "consistency_level": (0.70, 0.95),
        "engagement_baseline": (65, 90),
        "quiz_strength": (65, 90),
        "workload_pref": ["Medium", "Heavy"],
        "preferred_difficulty": ["Intermediate", "Advanced"],
        "preferred_content": ["reading", "quiz", "practice"],
    },
    {
        "name": "Wellness Personal Growth",
        "weight": 0.10,
        "proficiency_start": (0.15, 0.45),
        "n_interactions": (6, 20),
        "exploration_tendency": 0.30,
        "consistency_level": (0.45, 0.75),
        "engagement_baseline": (45, 72),
        "quiz_strength": (40, 65),
        "workload_pref": ["Light"],
        "preferred_difficulty": ["Beginner"],
        "preferred_content": ["video", "reading"],
    },
]

# Domains grouped by mega-category for secondary domain sampling
DOMAIN_GROUPS = {
    "Tech": [
        "Machine Learning", "Deep Learning", "Artificial Intelligence", "Data Science",
        "Software Development", "Web Development", "Mobile Development", "Cloud Computing",
        "Cybersecurity", "DevOps", "Databases", "Networking", "Programming Languages",
        "Computer Science", "Blockchain", "UI/UX Design", "Robotics", "Game Development",
    ],
    "Business": [
        "Business", "Marketing", "Finance", "Accounting", "Economics", "Project Management",
        "Human Resources", "Sales", "Leadership", "Supply Chain", "Business Analytics",
    ],
    "Science": [
        "Statistics", "Mathematics", "Data Analysis", "Physics", "Chemistry",
        "Biology", "Biotechnology", "Environmental Science",
    ],
    "Health": [
        "Health & Wellness", "Public Health", "Psychology", "Nursing Foundations",
    ],
    "Creative": [
        "Graphic Design", "Photography", "Video & Film", "Music",
        "Creative Writing", "Animation",
    ],
    "Education": [
        "Teaching & Education", "Language Learning", "Communication Skills", "Career Development",
    ],
    "Humanities": [
        "Sociology", "Philosophy", "History", "Political Science",
    ],
    "Other": [
        "Law", "Sustainability", "Real Estate", "General Studies",
    ],
}

ALL_DOMAINS = [d for group in DOMAIN_GROUPS.values() for d in group]


def pick_domain_for_archetype(archetype: dict, rng: np.random.Generator) -> str:
    """Pick a primary domain that makes sense for the archetype."""
    name = archetype["name"]
    if name in ("High Performer Specialist", "Certification Chaser", "Academic Builder"):
        group = rng.choice(["Tech", "Business", "Science"])
    elif name == "Creative Learner":
        group = rng.choice(["Creative", "Education"])
    elif name == "Wellness Personal Growth":
        group = rng.choice(["Health", "Humanities", "Education"])
    elif name == "Career Switcher":
        group = rng.choice(["Tech", "Business"])
    elif name == "Casual Explorer":
        group = rng.choice(list(DOMAIN_GROUPS.keys()))
    else:
        group = rng.choice(["Tech", "Business", "Science", "Education"])
    domains = DOMAIN_GROUPS[group]
    return domains[int(rng.integers(0, len(domains)))]


def pick_secondary_domain(primary: str, rng: np.random.Generator) -> str:
    """Pick a different domain, possibly adjacent."""
    candidates = [d for d in ALL_DOMAINS if d != primary]
    return candidates[int(rng.integers(0, len(candidates)))]


# ──────────────────────────────────────────────────────────────────────────────
# Learner trait generation
# ──────────────────────────────────────────────────────────────────────────────

def sample_archetype(rng: np.random.Generator) -> dict:
    weights = [a["weight"] for a in ARCHETYPES]
    total = sum(weights)
    probs = [w / total for w in weights]
    idx = int(rng.choice(len(ARCHETYPES), p=probs))
    return ARCHETYPES[idx]


def generate_learner_traits(learner_idx: int, rng: np.random.Generator) -> dict:
    arch = sample_archetype(rng)
    prof_lo, prof_hi = arch["proficiency_start"]
    prof = float(rng.uniform(prof_lo, prof_hi))
    n_lo, n_hi = arch["n_interactions"]
    n_interactions = int(rng.integers(n_lo, n_hi + 1))
    cons_lo, cons_hi = arch["consistency_level"]
    consistency = float(rng.uniform(cons_lo, cons_hi))
    eng_lo, eng_hi = arch["engagement_baseline"]
    quiz_lo, quiz_hi = arch["quiz_strength"]

    primary_domain = pick_domain_for_archetype(arch, rng)
    secondary_domain = pick_secondary_domain(primary_domain, rng)

    return {
        "learner_id": f"L{learner_idx:05d}",
        "archetype": arch["name"],
        "primary_domain": primary_domain,
        "secondary_domain": secondary_domain,
        "proficiency_start": prof,
        "n_interactions": n_interactions,
        "exploration_tendency": arch["exploration_tendency"],
        "consistency": consistency,
        "engagement_low": eng_lo,
        "engagement_high": eng_hi,
        "quiz_low": quiz_lo,
        "quiz_high": quiz_hi,
        "workload_pref": arch["workload_pref"],
        "preferred_difficulty": arch["preferred_difficulty"],
        "preferred_content": arch["preferred_content"],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Course sampling
# ──────────────────────────────────────────────────────────────────────────────

def candidate_courses(df: pd.DataFrame, domain: str, difficulty_level: str, workload_buckets: list[str],
                       rng: np.random.Generator, n: int = 20) -> pd.DataFrame:
    """
    Return up to n candidate courses that match domain and difficulty.
    Falls back progressively if not enough matches.
    """
    mask = (df["inferred_domain"] == domain)
    sub = df[mask]
    if len(sub) == 0:
        sub = df.copy()

    # prefer matching difficulty
    diff_sub = sub[sub["difficulty_level"] == difficulty_level]
    if len(diff_sub) >= 5:
        sub = diff_sub

    # prefer matching workload
    wl_sub = sub[sub["workload_bucket"].isin(workload_buckets)]
    if len(wl_sub) >= 5:
        sub = wl_sub

    if len(sub) > n:
        sub = sub.sample(n=n, random_state=int(rng.integers(0, 10000)))
    return sub


def proficiency_to_difficulty(prof: float) -> str:
    if prof < 0.40:
        return "Beginner"
    elif prof < 0.70:
        return "Intermediate"
    return "Advanced"


# ──────────────────────────────────────────────────────────────────────────────
# Score generation given fit
# ──────────────────────────────────────────────────────────────────────────────

def compute_interaction_scores(
    traits: dict, course: pd.Series, session_idx: int, rng: np.random.Generator
) -> dict:
    """
    Generate engagement, quiz, completion etc. based on domain/difficulty/workload fit.
    """
    # Domain fit
    domain_fit = 1.0 if course["inferred_domain"] == traits["primary_domain"] else (
        0.7 if course["inferred_domain"] == traits["secondary_domain"] else 0.4
    )
    # Difficulty fit
    current_prof = traits["proficiency_start"] + session_idx * 0.005  # gradual growth
    current_prof = min(current_prof, 0.99)
    expected_diff = proficiency_to_difficulty(current_prof)
    if course["difficulty_level"] == expected_diff:
        diff_fit = 1.0
    elif (expected_diff == "Beginner" and course["difficulty_level"] == "Intermediate") or \
         (expected_diff == "Intermediate" and course["difficulty_level"] in ["Beginner", "Advanced"]):
        diff_fit = 0.7
    else:
        diff_fit = 0.4

    # Workload fit
    wl_fit = 1.0 if course["workload_bucket"] in traits["workload_pref"] else 0.55

    # Combined fit for baseline
    fit = 0.50 * domain_fit + 0.30 * diff_fit + 0.20 * wl_fit

    # Engagement
    eng_base = traits["engagement_low"] + (traits["engagement_high"] - traits["engagement_low"]) * fit
    engagement = float(np.clip(eng_base + rng.normal(0, 8), 0, 100))

    # Quiz score
    quiz_base = traits["quiz_low"] + (traits["quiz_high"] - traits["quiz_low"]) * fit
    has_quiz = rng.random() < 0.70
    quiz_score = float(np.clip(quiz_base + rng.normal(0, 12), 0, 100)) if has_quiz else None

    # Completion rate — also depends on consistency
    base_completion = traits["consistency"] * fit
    completion = float(np.clip(base_completion + rng.normal(0, 0.12), 0, 1))

    # Dropout signal
    dropout_risk = 1.0 - (0.5 * traits["consistency"] + 0.5 * fit)

    # Proficiency signal
    prof_signal = current_prof

    # Satisfaction
    satisfaction = float(np.clip(0.5 * engagement / 100 + 0.3 * (completion) + 0.2 * fit + rng.normal(0, 0.08), 0, 1))

    return {
        "engagement_score": round(engagement, 2),
        "quiz_score": round(quiz_score, 2) if quiz_score is not None else None,
        "completion_rate": round(completion, 4),
        "dropout_risk_signal": round(dropout_risk, 4),
        "proficiency_signal": round(prof_signal, 4),
        "satisfaction_signal": round(satisfaction, 4),
        "domain_fit": domain_fit,
        "diff_fit": diff_fit,
        "wl_fit": wl_fit,
    }


def compute_learning_outcome(eng: float, quiz: float | None, comp: float) -> str:
    q = quiz if quiz is not None else 0
    if comp > 0.85 and q > 80 and eng > 75:
        return "excellent"
    elif comp > 0.70 and q > 65:
        return "good"
    elif comp < 0.20:
        return "dropped"
    elif comp > 0.50:
        return "average"
    else:
        return "weak"


def compute_event_type(completion: float, session_idx: int, revisit: bool, rng: np.random.Generator) -> str:
    if revisit:
        return "revisited"
    if completion < 0.10:
        return "enrolled" if rng.random() < 0.4 else "dropped"
    if completion < 0.40:
        return "in_progress"
    if completion < 0.80:
        return rng.choice(["in_progress", "started"])
    return "completed"


CONTENT_TYPES = ["video", "reading", "quiz", "practice", "project"]


def generate_interactions_for_learner(
    traits: dict, enriched: pd.DataFrame, rng: np.random.Generator
) -> list[dict]:
    learner_id = traits["learner_id"]
    n = traits["n_interactions"]
    rows = []

    # Start timestamp
    start_offset_days = int(rng.integers(0, 365))
    current_time = SIM_START + timedelta(days=start_offset_days)

    seen_courses: set[str] = set()
    prev_course_id: str | None = None

    for i in range(n):
        # Decide domain for this interaction
        is_secondary = rng.random() < traits["exploration_tendency"]
        domain = traits["secondary_domain"] if is_secondary else traits["primary_domain"]

        current_prof = traits["proficiency_start"] + i * 0.005
        diff_level = proficiency_to_difficulty(current_prof)

        # Occasionally step up/down difficulty for realism
        if rng.random() < 0.15 and diff_level != "Advanced":
            diff_level = {"Beginner": "Intermediate", "Intermediate": "Advanced"}.get(diff_level, diff_level)
        if rng.random() < 0.08 and diff_level != "Beginner":
            diff_level = {"Advanced": "Intermediate", "Intermediate": "Beginner"}.get(diff_level, diff_level)

        candidates = candidate_courses(enriched, domain, diff_level, traits["workload_pref"], rng, n=30)

        # Prefer unseen courses, but allow revisits
        unseen = candidates[~candidates["course_id"].isin(seen_courses)]
        if len(unseen) == 0:
            unseen = candidates  # all seen, allow any

        if len(unseen) == 0:
            continue

        course = unseen.sample(1, random_state=int(rng.integers(0, 100000))).iloc[0]
        course_id = str(course["course_id"])

        is_revisit = course_id in seen_courses
        revisit_flag = int(is_revisit)

        # Time gap
        days_gap = int(rng.integers(0, 14)) if i == 0 else int(abs(rng.normal(3, 5)))
        days_since_prev = days_gap
        current_time += timedelta(days=days_gap, hours=int(rng.integers(0, 20)))

        # Clamp to simulation end
        if current_time > SIM_END:
            current_time = SIM_END - timedelta(days=int(rng.integers(1, 30)))

        scores = compute_interaction_scores(traits, course, i, rng)
        content_type = rng.choice(
            traits["preferred_content"] if rng.random() < 0.7 else CONTENT_TYPES
        )

        time_spent = float(np.clip(
            scores["completion_rate"] * course["estimated_duration_hours"] * 60 * (0.7 + rng.random() * 0.6),
            2, 600
        ))

        event_type = compute_event_type(scores["completion_rate"], i, is_revisit, rng)
        outcome = compute_learning_outcome(scores["engagement_score"], scores["quiz_score"], scores["completion_rate"])

        # Streak: consecutive days active
        streak_flag = int(days_gap <= 1 and i > 0)

        session_id = f"S{learner_id}_{i:03d}"
        attempts_per_quiz = int(rng.integers(1, 4)) if scores["quiz_score"] is not None else 0

        rows.append({
            "learner_id": learner_id,
            "session_id": session_id,
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "course_id": course_id,
            "content_type": content_type,
            "event_type": event_type,
            "time_spent_minutes": round(time_spent, 2),
            "engagement_score": scores["engagement_score"],
            "quiz_score": scores["quiz_score"],
            "completion_rate": scores["completion_rate"],
            "attempts_per_quiz": attempts_per_quiz,
            "learning_outcome": outcome,
            "domain_at_time": str(course["inferred_domain"]),
            "difficulty_at_time": str(course["difficulty_level"]),
            "workload_bucket_at_time": str(course["workload_bucket"]),
            "session_order": i + 1,
            "days_since_prev_activity": days_since_prev if i > 0 else 0,
            "revisit_flag": revisit_flag,
            "streak_flag": streak_flag,
            "dropout_risk_signal": scores["dropout_risk_signal"],
            "proficiency_signal": scores["proficiency_signal"],
            "satisfaction_signal": scores["satisfaction_signal"],
        })

        seen_courses.add(course_id)
        prev_course_id = course_id

    return rows


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading enriched courses from {ENRICHED_PATH}")
    enriched = pd.read_csv(ENRICHED_PATH, low_memory=False)
    log.info(f"Loaded {len(enriched):,} courses")

    rng = np.random.default_rng(SEED)

    all_rows: list[dict] = []
    for i in range(1, N_LEARNERS + 1):
        traits = generate_learner_traits(i, rng)
        interactions = generate_interactions_for_learner(traits, enriched, rng)
        all_rows.extend(interactions)
        if i % 100 == 0:
            log.info(f"  Generated {i}/{N_LEARNERS} learners …")

    interactions_df = pd.DataFrame(all_rows)
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    interactions_df = interactions_df.sort_values(["learner_id", "timestamp"]).reset_index(drop=True)

    interactions_df.to_csv(OUT_PATH, index=False)
    log.info(f"Saved {len(interactions_df):,} interactions → {OUT_PATH}")
    log.info(f"Learners: {interactions_df['learner_id'].nunique()}")
    log.info(f"Unique courses referenced: {interactions_df['course_id'].nunique()}")


if __name__ == "__main__":
    main()
