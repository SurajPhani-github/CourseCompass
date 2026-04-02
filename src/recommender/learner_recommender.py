"""
learner_recommender.py
=======================
Mode A: Hybrid per-learner recommendation engine.

Two-stage ranking
-----------------
Stage 1 — Heuristic scoring (all signals from actual profile columns):
  0.25 × content_similarity
  0.20 × domain_affinity
  0.20 × transition_score
  0.15 × difficulty_fit        (uses dynamic preferred_difficulty)
  0.10 × workload_fit          (uses dynamic pace_preference)
  0.10 × completion_likelihood

Stage 2 — ML re-ranking (LightGBM, when model is available):
  final_score = 0.65 × ml_score + 0.35 × normalised_heuristic_score

Fallback: pure heuristic score if model not found.

Dynamic preferences
-------------------
`preferred_difficulty` and `pace_preference` are NEVER read from a static
CSV. They are always computed fresh from interaction history by
compute_dynamic_preferences(). The caller can pass `user_pref_override` to
inject an explicit session-level preference from the UI (e.g. the learner
selected "I want Advanced courses today").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.recommender.content_similarity import ContentSimilarityEngine
from src.recommender.explainability import ComponentScores, generate_reasons
from src.recommender.learner_ranker import LearnerRanker
from src.recommender.ranker_features import (
    compute_dynamic_preferences,
    DIFFICULTY_MAP,
    WORKLOAD_MAP,
)

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]

# ── Heuristic weight constants ────────────────────────────────────────────────
W_CONTENT    = 0.25
W_DOMAIN     = 0.20
W_TRANSITION = 0.20
W_DIFFICULTY = 0.15
W_WORKLOAD   = 0.10
W_COMPLETION = 0.10

# ── ML blend ──────────────────────────────────────────────────────────────────
ML_BLEND        = 0.65
HEURISTIC_BLEND = 0.35

# ── Singleton ranker ──────────────────────────────────────────────────────────
_ranker: Optional[LearnerRanker] = None

def _get_ranker() -> LearnerRanker:
    global _ranker
    if _ranker is None:
        _ranker = LearnerRanker()
    return _ranker


# ── Score helpers ─────────────────────────────────────────────────────────────

def _difficulty_fit_score(pref_diff: float, course_diff: float) -> float:
    """
    Compares dynamic preferred_difficulty (float 1-3) with course difficulty.
    Uses continuous distance so partial fits are rewarded smoothly.
    """
    delta = abs(pref_diff - course_diff)
    if delta < 0.3:   return 1.0
    elif delta < 0.9: return 0.75
    elif delta < 1.5: return 0.45
    else:             return 0.15


def _workload_fit_score(pref_pace: float, course_wl: float) -> float:
    """Continuous workload fit using numeric pace_preference (1-3)."""
    delta = abs(pref_pace - course_wl)
    if delta < 0.3:   return 1.0
    elif delta < 0.9: return 0.70
    elif delta < 1.5: return 0.40
    else:             return 0.15


def _domain_affinity_score(dominant_domain: str, secondary_domain: str,
                            course_domain: str) -> float:
    if course_domain == dominant_domain:   return 1.0
    elif course_domain == secondary_domain: return 0.55
    else:                                   return 0.10


# ── Main recommendation function ──────────────────────────────────────────────

def recommend_for_learner(
    learner_id:          str,
    profiles_df:         pd.DataFrame,
    interactions_df:     pd.DataFrame,
    enriched_df:         pd.DataFrame,
    transitions_df:      pd.DataFrame,
    similarity_engine:   ContentSimilarityEngine,
    top_n:               int  = 5,
    user_pref_override:  Optional[dict] = None,
) -> list[dict]:
    """
    Returns top_n recommendation dicts with ML-ranked scores where available.

    Parameters
    ----------
    user_pref_override : Optional session preference injected from the UI.
                         Dict with keys like:
                           'preferred_difficulty'  (float 1.0–3.0 or string label)
                           'pace_preference'       (float 1.0–3.0 or string label)
                         Example: {"preferred_difficulty": 3.0}  ← "Show Advanced"
                         These do NOT persist. Applied only for this request.
    """
    # ── 1. Profile ───────────────────────────────────────────────────────────
    profile_rows = profiles_df[profiles_df["learner_id"] == learner_id]
    if profile_rows.empty:
        log.warning(f"No profile found for learner {learner_id}")
        return []
    profile = profile_rows.iloc[0]

    dominant_domain  = str(profile.get("dominant_domain",  "General Studies"))
    secondary_domain = str(profile.get("secondary_domain", ""))
    completion_baseline = float(profile.get("completion_likelihood_baseline", 0.5))

    # Map categorical proficiency/workload for course-level heuristic
    estimated_proficiency = str(profile.get("estimated_proficiency", "Intermediate"))
    workload_tolerance    = str(profile.get("workload_tolerance",    "Medium"))

    # ── 2. Dynamic preferences (auto-computed, always fresh) ─────────────────
    dyn_prefs = compute_dynamic_preferences(learner_id, interactions_df)

    # Apply UI session override (non-persistent)
    if user_pref_override:
        for key, val in user_pref_override.items():
            if key in dyn_prefs:
                # Accept both numeric (2.5) and label ("Advanced") inputs
                if isinstance(val, str) and val in DIFFICULTY_MAP:
                    dyn_prefs[key] = float(DIFFICULTY_MAP[val])
                elif isinstance(val, str) and val in WORKLOAD_MAP:
                    dyn_prefs[key] = float(WORKLOAD_MAP[val])
                else:
                    dyn_prefs[key] = float(val)

    pref_diff = dyn_prefs["preferred_difficulty"]
    pref_pace = dyn_prefs["pace_preference"]

    # ── 3. Seen courses + recent seeds ───────────────────────────────────────
    learner_interactions = interactions_df[interactions_df["learner_id"] == learner_id]
    seen_courses = set(learner_interactions["course_id"].astype(str))

    recent_courses = (
        learner_interactions.sort_values("timestamp", ascending=False)
        ["course_id"].astype(str).unique()[:5].tolist()
    )

    # ── 4. Content similarity ─────────────────────────────────────────────────
    sim_results = similarity_engine.get_multi_similar(
        recent_courses, top_n=150, exclude=seen_courses
    )
    max_sim = max((s for _, s in sim_results), default=1.0) or 1.0
    sim_map: dict[str, float] = {cid: s / max_sim for cid, s in sim_results}

    # ── 5. Transition scores ──────────────────────────────────────────────────
    transition_mask      = transitions_df["prev_course_id"].isin(recent_courses)
    relevant_transitions = transitions_df[transition_mask]
    trans_map: dict[str, float] = {}
    for _, tr in relevant_transitions.iterrows():
        nxt   = str(tr["next_course_id"])
        score = float(tr["high_success_transition_score"])
        if nxt not in trans_map or trans_map[nxt] < score:
            trans_map[nxt] = score

    # ── 6. Candidate pool ─────────────────────────────────────────────────────
    candidate_ids = set(sim_map) | set(trans_map)
    if len(candidate_ids) < 50:
        domain_pool = enriched_df[
            (enriched_df["inferred_domain"] == dominant_domain) &
            (~enriched_df["course_id"].astype(str).isin(seen_courses))
        ]["course_id"].astype(str).tolist()[:50]
        candidate_ids.update(domain_pool)

    candidate_ids -= seen_courses
    if not candidate_ids:
        return []

    candidates = enriched_df[enriched_df["course_id"].astype(str).isin(candidate_ids)].copy()
    candidates["course_id"] = candidates["course_id"].astype(str)

    # ── 7. Heuristic scoring ──────────────────────────────────────────────────
    heuristic_scores: list[float] = []
    results_pre: list[dict]       = []

    for _, course in candidates.iterrows():
        cid          = str(course["course_id"])
        course_domain = str(course.get("inferred_domain",     "General Studies"))
        course_diff   = str(course.get("difficulty_level",    "Intermediate"))
        course_wl     = str(course.get("workload_bucket",     "Medium"))

        # Convert course fields to numeric for continuous fit scoring
        c_diff_val = float(DIFFICULTY_MAP.get(course_diff, 2))
        c_wl_val   = float(WORKLOAD_MAP  .get(course_wl,   2))

        content_sim = sim_map.get(cid, 0.0)
        domain_aff  = _domain_affinity_score(dominant_domain, secondary_domain, course_domain)
        trans_score = min(1.0, trans_map.get(cid, 0.0))
        diff_fit    = _difficulty_fit_score(pref_diff, c_diff_val)
        wl_fit      = _workload_fit_score(pref_pace,  c_wl_val)
        comp_lik    = min(1.0, completion_baseline * 0.7 + diff_fit * 0.2 + wl_fit * 0.1)

        heuristic = (
            W_CONTENT    * content_sim +
            W_DOMAIN     * domain_aff  +
            W_TRANSITION * trans_score +
            W_DIFFICULTY * diff_fit    +
            W_WORKLOAD   * wl_fit      +
            W_COMPLETION * comp_lik
        )
        heuristic_scores.append(heuristic)
        results_pre.append({
            "_cid":         cid,
            "_course":      course,
            "content_sim":  content_sim,
            "domain_aff":   domain_aff,
            "trans_score":  trans_score,
            "diff_fit":     diff_fit,
            "wl_fit":       wl_fit,
            "comp_lik":     comp_lik,
            "heuristic":    heuristic,
        })

    # ── 8. ML re-ranking ──────────────────────────────────────────────────────
    ranker    = _get_ranker()
    ml_active = ranker.is_available()
    ml_scores: Optional[np.ndarray] = None

    if ml_active:
        try:
            ml_scores = ranker.score_candidates(
                profile            = profile,
                candidates_df      = candidates,
                interactions_df    = interactions_df,
                transitions_df     = transitions_df,
                user_pref_override = user_pref_override,
            )
        except Exception as e:
            log.warning(f"ML scoring error for {learner_id}: {e}")
            ml_scores = None

    # Normalise heuristic to [0, 1] for blending
    h_arr  = np.array(heuristic_scores)
    h_max  = h_arr.max() if h_arr.max() > 0 else 1.0
    h_norm = h_arr / h_max

    # ── 9. Build results ──────────────────────────────────────────────────────
    results: list[dict] = []
    for i, pre in enumerate(results_pre):
        course      = pre["_course"]
        cid         = pre["_cid"]
        course_meta = course.to_dict()

        h_score  = h_norm[i]
        ml_score = float(ml_scores[i]) if ml_scores is not None else None
        final    = (ML_BLEND * ml_score + HEURISTIC_BLEND * h_score) \
                   if ml_score is not None else pre["heuristic"]

        scores = ComponentScores(
            content_similarity   = round(pre["content_sim"], 4),
            domain_affinity      = round(pre["domain_aff"],  4),
            transition_score     = round(pre["trans_score"], 4),
            difficulty_fit       = round(pre["diff_fit"],    4),
            workload_fit         = round(pre["wl_fit"],      4),
            completion_likelihood = round(pre["comp_lik"],   4),
            ml_ranking_score     = round(ml_score, 4) if ml_score is not None else 0.0,
            ml_ranked            = ml_active,
            final_score          = round(final, 4),
            mode                 = "A",
        )

        results.append({
            "course_id":               cid,
            "title":                   str(course.get("title",  cid)),
            "url":                     str(course.get("url",    "")),
            "inferred_domain":         str(course.get("inferred_domain",        "General Studies")),
            "difficulty_level":        str(course.get("difficulty_level",       "Intermediate")),
            "workload_bucket":         str(course.get("workload_bucket",        "Medium")),
            "estimated_duration_hours": float(course.get("estimated_duration_hours", 12)),
            "skills_tags":             str(course.get("skills_tags",            "")),
            "final_score":             round(final, 4),
            "ml_score":                round(ml_score, 4) if ml_score is not None else None,
            "ml_ranked":               ml_active,
            "dyn_pref_difficulty":     dyn_prefs["preferred_difficulty_label"],
            "dyn_pref_pace":           dyn_prefs["pace_preference_label"],
            "is_progressing":          dyn_prefs["is_progressing"],
            "component_scores":        scores,
            "reasons":                 generate_reasons(scores, course_meta),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_n]
