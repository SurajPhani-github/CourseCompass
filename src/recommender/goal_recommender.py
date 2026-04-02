"""
goal_recommender.py
====================
Mode B: Goal-based recommendation engine.

Scoring formula:
  0.40 × domain_match
+ 0.25 × proficiency_match
+ 0.15 × progression_value
+ 0.10 × workload_fit
+ 0.05 × popularity_proxy
+ 0.05 × quality_proxy

Results are sorted in a soft easier-to-harder progression.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommender.content_similarity import ContentSimilarityEngine
from src.recommender.explainability import ComponentScores, generate_reasons

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# ── Weight constants ──────────────────────────────────────────────────────────
W_DOMAIN = 0.40
W_PROFICIENCY = 0.25
W_PROGRESSION = 0.15
W_WORKLOAD = 0.10
W_POPULARITY = 0.05
W_QUALITY = 0.05

# All known domains — used for fuzzy domain matching
ALL_DOMAINS: list[str] = []  # populated on first call


def _load_domains(enriched: pd.DataFrame) -> None:
    global ALL_DOMAINS
    ALL_DOMAINS = sorted(enriched["inferred_domain"].dropna().unique().tolist())


def _fuzzy_domain_match(query: str, enriched: pd.DataFrame, engine: ContentSimilarityEngine) -> dict[str, float]:
    """
    Returns a dict of domain → match_score (0–1).
    Uses: exact/substring match first, then TF-IDF text similarity as fallback.
    """
    query_lower = query.lower().strip()
    domain_scores: dict[str, float] = {}

    for domain in ALL_DOMAINS:
        d_lower = domain.lower()
        if query_lower == d_lower:
            domain_scores[domain] = 1.0
        elif query_lower in d_lower or d_lower in query_lower:
            domain_scores[domain] = 0.85

    if not domain_scores:
        # Try token overlap
        q_tokens = set(query_lower.split())
        for domain in ALL_DOMAINS:
            d_tokens = set(domain.lower().split())
            overlap = len(q_tokens & d_tokens) / max(len(q_tokens | d_tokens), 1)
            if overlap > 0:
                domain_scores[domain] = 0.5 + 0.35 * overlap

    if not domain_scores:
        # TF-IDF fallback: get top similar courses and infer domain
        sim_results = engine.query_text(query, top_n=20)
        top_courses = [cid for cid, _ in sim_results]
        matched = enriched[enriched["course_id"].astype(str).isin(top_courses)]
        if len(matched) > 0:
            domain_counts = matched["inferred_domain"].value_counts(normalize=True)
            for d, score in domain_counts.items():
                domain_scores[str(d)] = min(0.75, float(score))

    # Fallback: assign small score to all
    if not domain_scores:
        for domain in ALL_DOMAINS:
            domain_scores[domain] = 0.05

    # Normalize to [0, 1]
    max_score = max(domain_scores.values()) if domain_scores else 1.0
    return {d: v / max_score for d, v in domain_scores.items()}


def _proficiency_match_score(target_proficiency: str, course_difficulty: str) -> float:
    prof_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    target = prof_map.get(target_proficiency, 2)
    course = prof_map.get(course_difficulty, 2)
    delta = abs(target - course)
    if delta == 0:
        return 1.0
    elif delta == 1:
        # One step off is okay (slightly above is good, below is less useful)
        if course > target:
            return 0.50  # slightly above: stretch goal
        return 0.40  # below: too easy
    else:
        return 0.10


def _progression_value_score(course: pd.Series, target_proficiency: str) -> float:
    """
    Progression value: beginner course is valuable for beginner goal,
    advanced course is valuable for advanced goal.
    Also credit foundational and project-based courses.
    """
    diff = str(course.get("difficulty_level", "Intermediate"))
    is_found = int(course.get("is_foundational", 0))
    is_proj = int(course.get("is_project_based", 0))

    # Base: difficulty alignment with proficiency (forward-looking)
    prof_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    target_val = prof_map.get(target_proficiency, 2)
    diff_val = prof_map.get(diff, 2)

    if diff_val == target_val:
        base = 0.7
    elif diff_val == target_val + 1:
        base = 0.5  # slightly harder: good stretch
    elif diff_val == target_val - 1:
        base = 0.4  # slightly easier: still foundational value
    else:
        base = 0.2

    bonus = 0.15 * is_found + 0.15 * is_proj
    return min(1.0, base + bonus)


def _workload_fit_score(target_proficiency: str, course_wl: str) -> float:
    """Beginners prefer lighter workloads. Advanced learners can handle heavy."""
    pref = {"Beginner": "Light", "Intermediate": "Medium", "Advanced": "Medium"}
    preferred = pref.get(target_proficiency, "Medium")
    wl_map = {"Light": 1, "Medium": 2, "Heavy": 3}
    delta = abs(wl_map.get(preferred, 2) - wl_map.get(course_wl, 2))
    if delta == 0:
        return 1.0
    elif delta == 1:
        return 0.60
    else:
        return 0.20


def recommend_for_goal(
    target_domain: str,
    target_proficiency: str,
    enriched_df: pd.DataFrame,
    similarity_engine: ContentSimilarityEngine,
    top_n: int = 5,
    workload_preference: str | None = None,
) -> list[dict]:
    """
    Returns top_n courses for the given goal (domain + proficiency),
    sorted in soft easier-to-harder progression.
    """
    _load_domains(enriched_df)

    # Step 1: get domain match scores for all domains
    domain_match_map = _fuzzy_domain_match(target_domain, enriched_df, similarity_engine)

    # Step 2: get text-similarity candidates
    sim_results = similarity_engine.query_text(target_domain, top_n=200)
    sim_map: dict[str, float] = {}
    if sim_results:
        max_sim = max(s for _, s in sim_results) or 1.0
        sim_map = {cid: s / max_sim for cid, s in sim_results}

    # Step 3: candidate pool — domain-aligned courses + tf-idf candidates
    top_domains = sorted(domain_match_map.items(), key=lambda x: x[1], reverse=True)[:5]
    top_domain_names = [d for d, _ in top_domains]

    candidate_mask = enriched_df["inferred_domain"].isin(top_domain_names)
    domain_pool = enriched_df[candidate_mask]["course_id"].astype(str).tolist()

    sim_pool = [cid for cid, _ in sim_results[:100]]
    candidate_ids = set(domain_pool + sim_pool)

    candidates = enriched_df[enriched_df["course_id"].astype(str).isin(candidate_ids)].copy()
    candidates["course_id"] = candidates["course_id"].astype(str)

    if candidates.empty:
        # Last resort: return all courses from enriched sorted by popularity
        candidates = enriched_df.copy()
        candidates["course_id"] = candidates["course_id"].astype(str)

    # Step 4: score each candidate
    results: list[dict] = []
    for _, course in candidates.iterrows():
        cid = str(course["course_id"])
        course_domain = str(course.get("inferred_domain", "General Studies"))
        course_difficulty = str(course.get("difficulty_level", "Intermediate"))
        course_wl = str(course.get("workload_bucket", "Medium"))

        domain_match = domain_match_map.get(course_domain, 0.05)
        # Boost if course directly is in text-similarity results
        domain_match = min(1.0, domain_match + sim_map.get(cid, 0.0) * 0.25)

        proficiency_match = _proficiency_match_score(target_proficiency, course_difficulty)
        progression_val = _progression_value_score(course, target_proficiency)

        # Workload fit: use user preference if provided, else infer from proficiency
        wl_ref = workload_preference if workload_preference else target_proficiency
        if workload_preference:
            wl_map = {"Light": 1, "Medium": 2, "Heavy": 3}
            delta = abs(wl_map.get(workload_preference, 2) - wl_map.get(course_wl, 2))
            wl_fit = 1.0 if delta == 0 else (0.6 if delta == 1 else 0.2)
        else:
            wl_fit = _workload_fit_score(target_proficiency, course_wl)

        pop = float(course.get("popularity_proxy", 0.5))
        qual = float(course.get("quality_proxy", 0.5))

        final = (
            W_DOMAIN * domain_match +
            W_PROFICIENCY * proficiency_match +
            W_PROGRESSION * progression_val +
            W_WORKLOAD * wl_fit +
            W_POPULARITY * pop +
            W_QUALITY * qual
        )

        scores = ComponentScores(
            domain_match=round(domain_match, 4),
            proficiency_match=round(proficiency_match, 4),
            progression_value=round(progression_val, 4),
            workload_fit=round(wl_fit, 4),
            popularity_proxy=round(pop, 4),
            quality_proxy=round(qual, 4),
            final_score=round(final, 4),
            mode="B",
        )

        course_meta = course.to_dict()
        reasons = generate_reasons(scores, course_meta, target_domain, target_proficiency)

        results.append({
            "course_id": cid,
            "title": str(course.get("title", cid)),
            "url": str(course.get("url", "")),
            "inferred_domain": course_domain,
            "difficulty_level": course_difficulty,
            "workload_bucket": course_wl,
            "estimated_duration_hours": float(course.get("estimated_duration_hours", 12)),
            "skills_tags": str(course.get("skills_tags", "")),
            "popularity_proxy": pop,
            "final_score": round(final, 4),
            "component_scores": scores,
            "reasons": reasons,
        })

    # ── Soft progression sort ──────────────────────────────────────────────────
    # First: score threshold (top N*3), then sort by difficulty (easier first)
    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_pool = results[: top_n * 3]

    diff_order = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    top_pool.sort(key=lambda x: (
        diff_order.get(x["difficulty_level"], 2),
        -x["final_score"]
    ))

    return top_pool[:top_n]
