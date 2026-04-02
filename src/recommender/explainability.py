"""
explainability.py
==================
Converts component scores into 2–4 human-readable reason strings.

Both Mode A (learner) and Mode B (goal) are handled here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ComponentScores:
    """Holds all sub-scores that feed into a final recommendation score."""
    # Mode A
    content_similarity:   float = 0.0
    domain_affinity:      float = 0.0
    transition_score:     float = 0.0
    difficulty_fit:       float = 0.0
    workload_fit:         float = 0.0
    completion_likelihood: float = 0.0
    # ML ranker extension
    ml_ranking_score:     float = 0.0
    ml_ranked:            bool  = False
    # Mode B
    goal_relevance:       float = 0.0
    domain_match:         float = 0.0
    proficiency_match:    float = 0.0
    progression_value:    float = 0.0
    popularity_proxy:     float = 0.0
    quality_proxy:        float = 0.0
    # shared
    final_score:          float = 0.0
    mode:                 str   = "A"  # "A" or "B"


# ──────────────────────────────────────────────────────────────────────────────
# Reason templates (Mode A — Learner)
# ──────────────────────────────────────────────────────────────────────────────

def _mode_a_reasons(scores: ComponentScores, course_meta: dict) -> list[str]:
    """Return 2–4 reasons for a Mode A recommendation based on component scores."""
    candidates: list[tuple[float, str]] = []

    # ML confidence — highest priority signal when available
    if scores.ml_ranked and scores.ml_ranking_score > 0.70:
        candidates.append((
            scores.ml_ranking_score + 0.1,          # boost to surface first
            f"🤖 {int(scores.ml_ranking_score * 100)}% predicted success — similar learners completed this next"
        ))
    elif scores.ml_ranked and scores.ml_ranking_score > 0.50:
        candidates.append((
            scores.ml_ranking_score,
            "🤖 High predicted completion likelihood based on your learning pattern"
        ))
    elif scores.ml_ranked and scores.ml_ranking_score > 0.35:
        candidates.append((
            scores.ml_ranking_score * 0.8,
            "🤖 ML model identified this as a good next step for your profile"
        ))

    # Content similarity
    if scores.content_similarity > 0.25:
        candidates.append((scores.content_similarity,
                           "📚 Similar to courses you've recently engaged with"))
    elif scores.content_similarity > 0.10:
        candidates.append((scores.content_similarity * 0.7,
                           "📚 Related to topics in your learning history"))

    # Domain affinity
    if scores.domain_affinity > 0.75:
        domain = course_meta.get("inferred_domain", "your domain")
        candidates.append((scores.domain_affinity,
                           f"🎯 Strong match with your dominant domain: {domain}"))
    elif scores.domain_affinity > 0.5:
        candidates.append((scores.domain_affinity * 0.8,
                           "🎯 Aligns with your primary area of interest"))

    # Transition score
    if scores.transition_score > 0.6:
        candidates.append((scores.transition_score,
                           "🔗 Common next step after your recent learning activity"))
    elif scores.transition_score > 0.3:
        candidates.append((scores.transition_score * 0.75,
                           "🔗 Frequently paired with courses you've taken"))

    # Difficulty fit
    diff_label = course_meta.get("difficulty_level", "")
    if scores.difficulty_fit > 0.75:
        candidates.append((scores.difficulty_fit,
                           f"📈 Matches your current proficiency level ({diff_label})"))
    elif scores.difficulty_fit > 0.5:
        candidates.append((scores.difficulty_fit * 0.8,
                           "📈 Good progression for your skill level"))

    # Workload fit
    wl_label = course_meta.get("workload_bucket", "")
    if scores.workload_fit > 0.75:
        candidates.append((scores.workload_fit,
                           f"⏱️ Fits your preferred workload pattern ({wl_label})"))
    elif scores.workload_fit > 0.45:
        candidates.append((scores.workload_fit * 0.7,
                           "⏱️ Manageable workload based on your history"))

    # Completion likelihood
    if scores.completion_likelihood > 0.70:
        candidates.append((scores.completion_likelihood,
                           "✅ High likelihood of completion based on your history"))
    elif scores.completion_likelihood > 0.50:
        candidates.append((scores.completion_likelihood * 0.6,
                           "✅ Good fit for your completion pattern"))

    # Sort by strength, take top 4, require at least 2
    candidates.sort(key=lambda x: x[0], reverse=True)
    reasons = [r for _, r in candidates[:4]]

    if len(reasons) < 2:
        reasons.append("🌟 Recommended based on your overall learning profile")

    return reasons[:4]


# ──────────────────────────────────────────────────────────────────────────────
# Reason templates (Mode B — Goal)
# ──────────────────────────────────────────────────────────────────────────────

def _mode_b_reasons(
    scores: ComponentScores,
    course_meta: dict,
    target_domain: str,
    target_proficiency: str,
) -> list[str]:
    """Return 2–4 reasons for a Mode B recommendation."""
    candidates: list[tuple[float, str]] = []

    diff_label     = course_meta.get("difficulty_level",  "")
    wl_label       = course_meta.get("workload_bucket",   "")
    is_foundational = course_meta.get("is_foundational",  0)
    is_project      = course_meta.get("is_project_based", 0)

    # Goal relevance
    if scores.goal_relevance > 0.75:
        candidates.append((scores.goal_relevance,
                           "🎯 Strong semantic match for your specific learning goal"))
    elif scores.goal_relevance > 0.4:
        candidates.append((scores.goal_relevance * 0.8,
                           "🎯 Relevant topics to your specific goal"))

    # Domain match
    if scores.domain_match > 0.80:
        candidates.append((scores.domain_match,
                           f"🎯 Strong match for your selected domain: {target_domain}"))
    elif scores.domain_match > 0.55:
        candidates.append((scores.domain_match * 0.85,
                           f"🎯 Relevant to {target_domain}"))
    else:
        candidates.append((scores.domain_match * 0.5,
                           f"📌 Covers adjacent topics in {target_domain}"))

    # Proficiency match
    if scores.proficiency_match > 0.75:
        candidates.append((scores.proficiency_match,
                           f"📊 Fits your {target_proficiency} proficiency perfectly"))
    elif scores.proficiency_match > 0.5:
        candidates.append((scores.proficiency_match * 0.8,
                           f"📊 Appropriate level for your {target_proficiency} background"))

    # Progression value
    if scores.progression_value > 0.7:
        if is_foundational:
            candidates.append((scores.progression_value,
                               "🏗️ Beginner-friendly foundation in this topic"))
        else:
            candidates.append((scores.progression_value,
                               "🚀 Good next step in structural progression toward advanced topics"))
    elif scores.progression_value > 0.4:
        candidates.append((scores.progression_value * 0.75,
                           "📖 Good next step after foundational concepts"))

    # Workload fit
    if scores.workload_fit > 0.7:
        candidates.append((scores.workload_fit,
                           f"⏱️ Suitable workload ({wl_label}) based on your time budget"))
    elif scores.workload_fit > 0.45:
        candidates.append((scores.workload_fit * 0.6,
                           "⏱️ Reasonable time investment for this topic"))

    # Popularity / quality
    if scores.popularity_proxy > 0.6 or scores.quality_proxy > 0.6:
        score_val = max(scores.popularity_proxy, scores.quality_proxy)
        candidates.append((score_val * 0.6,
                           "✅ Strong completion likelihood / low drop-off risk"))

    if is_project:
        candidates.append((0.4, "🛠️ Project-based learning — apply concepts hands-on"))

    candidates.sort(key=lambda x: x[0], reverse=True)
    reasons = [r for _, r in candidates[:4]]

    if len(reasons) < 2:
        reasons.append("🌟 Recommended for your selected goal and proficiency")

    return reasons[:4]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def generate_reasons(
    scores: ComponentScores,
    course_meta: dict,
    target_domain: str = "",
    target_proficiency: str = "",
) -> list[str]:
    """Main entry point. Dispatches to Mode A or Mode B reason generator."""
    if scores.mode == "A":
        return _mode_a_reasons(scores, course_meta)
    else:
        return _mode_b_reasons(scores, course_meta, target_domain, target_proficiency)
