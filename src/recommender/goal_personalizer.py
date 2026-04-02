"""
goal_personalizer.py
=====================
Enhanced Goal Mode (Phase 1.5).

Scores courses based on complex goal inputs:
- Free text learning goal
- Target domain
- Target proficiency
- Workload preference & weekly hour budget
"""

from __future__ import annotations

import logging
import pandas as pd

from src.recommender.content_similarity import ContentSimilarityEngine
from src.recommender.explainability import ComponentScores, generate_reasons
from src.recommender.goal_recommender import _fuzzy_domain_match, _proficiency_match_score, _progression_value_score, _load_domains

log = logging.getLogger(__name__)

# Weights for Phase 1.5
W_GOAL_TEXT = 0.35
W_DOMAIN = 0.20
W_PROFICIENCY = 0.20
W_PROGRESSION = 0.10
W_WORKLOAD = 0.10
W_QUALITY = 0.05

def _resolve_workload_preference(workload_pref: str, weekly_hours: int | None) -> str:
    if weekly_hours:
        if weekly_hours <= 5: return "Light"
        elif weekly_hours >= 15: return "Heavy"
        else: return "Medium"
    
    if workload_pref and workload_pref != "Any":
        return workload_pref
    return "Medium"

def personalize_goal(
    learning_goal: str,
    target_domain: str,
    target_proficiency: str,
    workload_preference: str,
    weekly_hours_budget: int | None,
    enriched_df: pd.DataFrame,
    similarity_engine: ContentSimilarityEngine,
    top_n: int = 20,
) -> list[dict]:
    _load_domains(enriched_df)
    
    # Text Relevance from learning goal
    sim_map: dict[str, float] = {}
    if learning_goal and learning_goal.strip():
        sim_results = similarity_engine.query_text(learning_goal, top_n=300)
        if sim_results:
            max_sim = max(s for _, s in sim_results) or 1.0
            sim_map = {cid: s / max_sim for cid, s in sim_results}

    # Domain Match
    domain_match_map = {}
    if target_domain and target_domain != "Any":
        domain_match_map = _fuzzy_domain_match(target_domain, enriched_df, similarity_engine)
    else:
        # If no domain picked, use text similarity to infer domain boost
        # fallback is 1.0 for all domains
        domain_match_map = {d: 1.0 for d in enriched_df["inferred_domain"].unique()}

    # Candidate pooling
    sim_pool = list(sim_map.keys())[:200] if sim_map else []
    
    if target_domain and target_domain != "Any":
        top_domains = sorted(domain_match_map.items(), key=lambda x: x[1], reverse=True)[:3]
        domain_names = [d for d, _ in top_domains]
        domain_mask = enriched_df["inferred_domain"].isin(domain_names)
        domain_pool = enriched_df[domain_mask]["course_id"].astype(str).tolist()
    else:
        domain_pool = sim_pool

    candidate_ids = set(domain_pool + sim_pool)
    candidates = enriched_df[enriched_df["course_id"].astype(str).isin(candidate_ids)].copy()
    
    if candidates.empty:
        candidates = enriched_df.copy()

    resolved_wl = _resolve_workload_preference(workload_preference, weekly_hours_budget)
    
    results: list[dict] = []
    
    # Score calculation
    for _, course in candidates.iterrows():
        cid = str(course["course_id"])
        
        goal_rel = sim_map.get(cid, 0.0) if learning_goal else 0.5
        course_domain = str(course.get("inferred_domain", "General Studies"))
        course_difficulty = str(course.get("difficulty_level", "Intermediate"))
        course_wl = str(course.get("workload_bucket", "Medium"))
        
        dom_match = domain_match_map.get(course_domain, 0.05) if target_domain and target_domain != "Any" else goal_rel
        prof_match = _proficiency_match_score(target_proficiency, course_difficulty)
        prog_val = _progression_value_score(course, target_proficiency)
        
        wl_map = {"Light": 1, "Medium": 2, "Heavy": 3}
        delta = abs(wl_map.get(resolved_wl, 2) - wl_map.get(course_wl, 2))
        wl_fit = 1.0 if delta == 0 else (0.6 if delta == 1 else 0.2)
        
        qual = float(course.get("quality_proxy", 0.5))
        pop = float(course.get("popularity_proxy", 0.5))
        
        final = (
            W_GOAL_TEXT * goal_rel +
            W_DOMAIN * dom_match +
            W_PROFICIENCY * prof_match +
            W_PROGRESSION * prog_val +
            W_WORKLOAD * wl_fit +
            W_QUALITY * max(qual, pop)
        )
        
        scores = ComponentScores(
            goal_relevance=round(goal_rel, 4),
            domain_match=round(dom_match, 4),
            proficiency_match=round(prof_match, 4),
            progression_value=round(prog_val, 4),
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
            "is_foundational": int(course.get("is_foundational", 0))
        })
        
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_n]
