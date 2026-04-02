"""
streamlit_app.py
=================
Hybrid Course Recommendation System — Streamlit UI.

Run with:
    streamlit run src/app/streamlit_app.py
from the project root (d:/PLS/).
"""

from __future__ import annotations

import time
import requests
import sys
import logging
from pathlib import Path
import textwrap

import pandas as pd
import streamlit as st

# ── Ensure project root is on path ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender.content_similarity import get_engine
from src.recommender.learner_recommender import recommend_for_learner
from src.recommender.ranker_features import compute_dynamic_preferences
from src.recommender.goal_personalizer import personalize_goal
from src.recommender.roadmap_builder import build_roadmap
from src.app.metrics import get_global_metrics
from src.recommender.skill_graph import KnowledgeGraph
from src.recommender.course_skill_mapper import CourseSkillMapper
from src.recommender.goal_decomposer import GoalDecomposer
from src.recommender.roadmap_engine import RoadmapEngine

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CourseCompass — Hybrid Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f1629 40%, #0d1f3c 100%);
    min-height: 100vh;
}

/* ── Global ── */
body {
    background-color: #0b0f19;
    color: #e2e8f0;
}

/* ── Typography & Headings ── */
.section-header {
    font-size: 1.6rem;
    font-weight: 800;
    margin: 2rem 0 1rem 0;
    color: #ffffff;
    letter-spacing: -0.02em;
}

.stat-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: rgba(99, 102, 241, 0.5);
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
    line-height: 1;
    margin-bottom: 0.3rem;
}

.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Profile summary cards ── */
.profile-card {
    background: linear-gradient(135deg, #1e2744 0%, #16213e 100%);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}

.profile-card:hover { border-color: rgba(99, 102, 241, 0.5); }

.profile-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}

.profile-value {
    color: #e2e8f0;
    font-size: 1.05rem;
    font-weight: 600;
}

.profile-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
}

.badge-beginner { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-intermediate { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.badge-advanced { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-improving { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-stable { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.3); }
.badge-declining { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-light { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.badge-medium { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }
.badge-heavy { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── Recommendation card ── */
.rec-card {
    background: linear-gradient(135deg, #1a2040 0%, #151c38 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
}

.rec-card:hover {
    transform: translateY(-3px);
    border-color: rgba(99, 102, 241, 0.55);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.15);
}

/* Number circle */
.num-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* Chat text visibility fix */
[data-testid="stChatInput"] textarea, [data-testid="stChatMessageContent"] p {
    color: #e2e8f0 !important;
    -webkit-text-fill-color: #e2e8f0 !important;
}

/* Timeline connection line */
.timeline-path {
    border-left: 2px dashed rgba(99,102,241,0.4);
    padding-left: 1.8rem;
    margin-left: 0.6rem;
    position: relative;
}

.timeline-dot {
    position: absolute;
    left: -6.5px;
    top: 5px;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #6366f1;
    box-shadow: 0 0 8px rgba(99,102,241,0.8);
}

.rec-rank {
    position: absolute;
    top: -12px;
    left: 20px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 12px;
    letter-spacing: 0.05em;
}

.rec-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0.5rem 0 0.5rem 0;
    line-height: 1.3;
}

.rec-meta {
    display: flex;
    gap: 0.6rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
    margin-top: 0.4rem;
}

.meta-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
}

.chip-domain { background: rgba(99,102,241,0.15); color: #818cf8; border: 1px solid rgba(99,102,241,0.25); }
.chip-difficulty-b { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.2); }
.chip-difficulty-i { background: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.2); }
.chip-difficulty-a { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.2); }
.chip-workload { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }
.chip-duration { background: rgba(167,139,250,0.1); color: #a78bfa; border: 1px solid rgba(167,139,250,0.2); }

/* ── Score bar ── */
.score-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.9rem;
}

.score-label {
    color: #64748b;
    font-size: 0.73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    white-space: nowrap;
    width: 70px;
}

.score-bar-bg {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.07);
    border-radius: 3px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
}

.score-num {
    color: #818cf8;
    font-size: 0.88rem;
    font-weight: 700;
    width: 38px;
    text-align: right;
}

/* ── Reason chips ── */
.reasons-container {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    margin-top: 0.5rem;
}

.reason-chip {
    display: inline-block;
    padding: 0.35rem 0.85rem;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 20px;
    font-size: 0.8rem;
    color: #c7d2fe;
    font-weight: 400;
    line-height: 1.4;
}

/* ── Section headers ── */
.section-header {
    color: #818cf8;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

/* ── History table ── */
.history-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
}

.history-table th {
    color: #64748b;
    font-weight: 600;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
    text-align: left;
}

.history-table td {
    color: #cbd5e1;
    padding: 0.7rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.history-table tr:hover td {
    background: rgba(99,102,241,0.05);
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem;
    color: #475569;
}

.empty-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}

/* ── Input styling ── */
.stSelectbox > div > div {
    background: #1e2744 !important;
    border-color: rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-weight: 500;
    font-size: 0.9rem;
    padding: 0.6rem 1.2rem;
    border-radius: 8px 8px 0 0;
}

.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.1) !important;
    color: #818cf8 !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* Subheader ── */
.sub-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #c7d2fe;
    margin: 1.2rem 0 1rem 0;
}

/* Pipeline warning */
.pipeline-warning {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #fca5a5;
    font-size: 0.9rem;
}

/* Link button styling */
.course-link {
    display: inline-block;
    margin-top: 0.5rem;
    padding: 0.35rem 1rem;
    background: rgba(99,102,241,0.15);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 500;
    text-decoration: none;
    transition: background 0.2s;
}
.course-link:hover {
    background: rgba(99,102,241,0.3);
    color: #a5b4fc;
}

/* ── AI Chat Panel ── */
.chat-panel-header {
    background: linear-gradient(135deg, #1e2744 0%, #16213e 100%);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 14px 14px 0 0;
    padding: 1rem 1.4rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0;
    position: sticky;
    top: 0;
    z-index: 10;
}

.chat-panel-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 6px #4ade80;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.chat-panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e2e8f0;
    flex: 1;
}

.chat-panel-model {
    font-size: 0.72rem;
    color: #64748b;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 0.2rem 0.6rem;
}

.chat-panel-body {
    background: rgba(10, 14, 26, 0.85);
    border: 1px solid rgba(99,102,241,0.2);
    border-top: none;
    border-radius: 0 0 14px 14px;
    padding: 1rem;
    min-height: 200px;
    margin-bottom: 0.75rem;
}

/* Force dark background on Streamlit's chat messages */
[data-testid="stChatMessageContent"] {
    background: rgba(30, 39, 68, 0.9) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
}

[data-testid="stChatMessageContent"] p {
    color: #e2e8f0 !important;
}

/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
    background: rgba(99, 102, 241, 0.2) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
}

/* Assistant message bubble */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stChatMessageContent"] {
    background: rgba(16, 30, 60, 0.9) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* Dark chat input */
[data-testid="stChatInput"] {
    background: #1e2744 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #e2e8f0 !important;
}

/* Attach button dark styling */
.stButton button {
    background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.2)) !important;
    border: 1px solid rgba(99,102,241,0.4) !important;
    color: #c7d2fe !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.4), rgba(139,92,246,0.4)) !important;
    border-color: rgba(99,102,241,0.7) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.2) !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────
PROCESSED = ROOT / "data" / "processed"

@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    enriched = pd.read_csv(PROCESSED / "enriched_courses.csv", low_memory=False)
    enriched["course_id"] = enriched["course_id"].astype(str)

    interactions = pd.read_csv(PROCESSED / "synthetic_learner_interactions.csv", low_memory=False)
    interactions["course_id"] = interactions["course_id"].astype(str)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    profiles = pd.read_csv(PROCESSED / "learner_profiles.csv", low_memory=False)

    transitions = pd.read_csv(PROCESSED / "course_transitions.csv", low_memory=False)
    transitions["prev_course_id"] = transitions["prev_course_id"].astype(str)
    transitions["next_course_id"] = transitions["next_course_id"].astype(str)

    return enriched, interactions, profiles, transitions


@st.cache_resource(show_spinner=False)
def load_engine(enriched: pd.DataFrame):
    return get_engine(enriched)

@st.cache_resource(show_spinner=False)
def load_graph_engine(enriched: pd.DataFrame, interactions: pd.DataFrame, cache_bust=10):
    import importlib
    import sys
    if "src.recommender.roadmap_engine" in sys.modules:
        importlib.reload(sys.modules["src.recommender.roadmap_engine"])
    if "src.recommender.llm_branching" in sys.modules:
        importlib.reload(sys.modules["src.recommender.llm_branching"])
    from src.recommender.roadmap_engine import RoadmapEngine
    from src.recommender.llm_branching import get_chatbot_response
    kg = KnowledgeGraph()
    mapper = CourseSkillMapper(enriched, kg)
    # The first run will build the model, encode 15k courses, and save to .pkl. Subsequent runs load the .pkl
    mapper.build_or_load_mapping(force_rebuild=False)
    decomposer = GoalDecomposer(kg, mapper)
    engine = RoadmapEngine(kg, mapper, decomposer, enriched, interactions)
    return engine


def check_pipeline_ready() -> bool:
    required = [
        PROCESSED / "enriched_courses.csv",
        PROCESSED / "synthetic_learner_interactions.csv",
        PROCESSED / "learner_profiles.csv",
        PROCESSED / "course_transitions.csv",
    ]
    return all(p.exists() for p in required)


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def difficulty_chip_class(diff: str) -> str:
    return {"Beginner": "chip-difficulty-b", "Intermediate": "chip-difficulty-i", "Advanced": "chip-difficulty-a"}.get(diff, "chip-difficulty-i")


def badge_class(value: str, badge_type: str = "proficiency") -> str:
    v = value.lower()
    if badge_type == "proficiency":
        return {"beginner": "badge-beginner", "intermediate": "badge-intermediate", "advanced": "badge-advanced"}.get(v, "badge-intermediate")
    elif badge_type == "momentum":
        return {"improving": "badge-improving", "stable": "badge-stable", "declining": "badge-declining"}.get(v, "badge-stable")
    elif badge_type == "workload":
        return {"light": "badge-light", "medium": "badge-medium", "heavy": "badge-heavy"}.get(v, "badge-medium")
    return ""


def render_rec_card(rec: dict, rank: int, mode: str = "A") -> None:
    diff = rec.get("difficulty_level", "Intermediate")
    diff_cls = difficulty_chip_class(diff)
    wl = rec.get("workload_bucket", "Medium")
    domain = rec.get("inferred_domain", "")
    dur = rec.get("estimated_duration_hours", 12)
    score = rec.get("final_score", 0.0)
    title = rec.get("title", "Untitled Course")
    url = rec.get("url", "")
    reasons = rec.get("reasons", [])
    skills = rec.get("skills_tags", "")
    ml_ranked = rec.get("ml_ranked", False)
    ml_score  = rec.get("ml_score", None)

    score_pct = int(score * 100)
    score_bar_width = int(score * 100)

    rank_labels = ["🥇 Top Match", "🥈 2nd Pick", "🥉 3rd Pick", "4th Pick", "5th Pick"]
    rank_label = rank_labels[rank - 1] if rank <= len(rank_labels) else f"#{rank}"

    # ML badge
    ml_badge = ""
    if ml_ranked and ml_score is not None:
        ml_pct = int(ml_score * 100)
        ml_badge = f'<span style="font-size:0.7rem;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white;padding:0.15rem 0.55rem;border-radius:10px;font-weight:700;margin-left:0.5rem">🤖 ML {ml_pct}%</span>'

    # Build reasons HTML
    reasons_html = ""
    for reason in reasons:
        reasons_html += f'<div class="reason-chip">{reason}</div>'

    # Skills
    skill_chips = ""
    if skills:
        for s in skills.split(", ")[:4]:
            skill_chips += f'<span class="meta-chip chip-workload">🔧 {s.strip()}</span>'

    link_html = f'<a class="course-link" href="{url}" target="_blank">🔗 View Course</a>' if url and url != "nan" else ""

    card_html = f"""
    <div class="rec-card">
        <div class="rec-rank">{rank_label}</div>
        <div class="rec-title">{title}{ml_badge}</div>
        <div class="rec-meta">
            <span class="meta-chip chip-domain">📁 {domain}</span>
            <span class="meta-chip {diff_cls}">📊 {diff}</span>
            <span class="meta-chip chip-workload">💼 {wl}</span>
            <span class="meta-chip chip-duration">⏱️ ~{dur:.0f}h</span>
        </div>
        <div class="score-row">
            <span class="score-label">Match</span>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_bar_width}%"></div></div>
            <span class="score-num">{score_pct}%</span>
        </div>
        <div class="reasons-container">{reasons_html}</div>
        <div class="rec-meta" style="margin-top:0.8rem">{skill_chips}</div>
        {link_html}
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_profile_cards(profile: pd.Series) -> None:
    prof = str(profile.get("estimated_proficiency", "Intermediate"))
    momentum = str(profile.get("momentum_trend", "stable"))
    wl = str(profile.get("workload_tolerance", "Medium"))
    domain = str(profile.get("dominant_domain", "—"))
    secondary = str(profile.get("secondary_domain", "—"))
    n_completed = int(profile.get("total_courses_completed", 0))
    avg_eng = float(profile.get("avg_engagement_score", 0))
    avg_comp = float(profile.get("avg_completion_rate", 0))
    avg_quiz = float(profile.get("avg_quiz_score", 0))
    curiosity = float(profile.get("curiosity_index", 0))
    consistency = float(profile.get("consistency_index", 0))
    recent_focus = str(profile.get("recent_domain_focus", domain))

    html = f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Proficiency</div>
            <span class="profile-badge {badge_class(prof, 'proficiency')}">{prof}</span>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Momentum</div>
            <span class="profile-badge {badge_class(momentum, 'momentum')}">{'📈' if momentum=='improving' else '📉' if momentum=='declining' else '➡️'} {momentum.capitalize()}</span>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Workload Tolerance</div>
            <span class="profile-badge {badge_class(wl, 'workload')}">{wl}</span>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Courses Completed</div>
            <div class="profile-value">🎓 {n_completed}</div>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Dominant Domain</div>
            <div class="profile-value" style="font-size:0.92rem">🎯 {domain}</div>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Secondary Interest</div>
            <div class="profile-value" style="font-size:0.92rem">🌐 {secondary}</div>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Avg Engagement</div>
            <div class="profile-value">⚡ {avg_eng:.1f}<span style="color:#475569;font-size:0.7rem">/100</span></div>
        </div>
        <div class="profile-card" style="margin-bottom: 0;">
            <div class="profile-label">Avg Completion</div>
            <div class="profile-value">✅ {avg_comp:.0%}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_history_table(learner_id: str, interactions_df: pd.DataFrame, enriched_df: pd.DataFrame) -> None:
    learner_int = interactions_df[interactions_df["learner_id"] == learner_id].copy()
    learner_int = learner_int.sort_values("timestamp", ascending=False).head(10)

    if learner_int.empty:
        st.markdown('<div class="empty-state"><div class="empty-icon">📭</div>No history available.</div>', unsafe_allow_html=True)
        return

    # Merge to get titles
    merged = learner_int.merge(
        enriched_df[["course_id", "title"]],
        on="course_id", how="left"
    )

    outcome_icons = {
        "excellent": "🏆 Excellent",
        "good": "✅ Good",
        "average": "📊 Average",
        "weak": "⚠️ Weak",
        "dropped": "❌ Dropped",
    }

    rows_html = ""
    for _, row in merged.iterrows():
        title = str(row.get("title", row["course_id"]))[:55] + ("…" if len(str(row.get("title", ""))) > 55 else "")
        domain = str(row.get("domain_at_time", ""))
        diff = str(row.get("difficulty_at_time", ""))
        outcome = str(row.get("learning_outcome", ""))
        comp = float(row.get("completion_rate", 0))
        ts = str(row["timestamp"])[:10]
        outcome_display = outcome_icons.get(outcome, outcome)
        comp_color = "#4ade80" if comp > 0.7 else "#fbbf24" if comp > 0.4 else "#f87171"
        rows_html += f"<tr><td><span style='color:#e2e8f0;font-weight:500'>{title}</span></td>" \
                     f"<td><span style='color:#818cf8'>{domain}</span></td>" \
                     f"<td>{diff}</td>" \
                     f"<td><span style='color:{comp_color};font-weight:600'>{comp:.0%}</span></td>" \
                     f"<td>{outcome_display}</td>" \
                     f"<td style='color:#475569'>{ts}</td></tr>"

    table_html = f"""
    <div style="background:rgba(15,22,45,0.6);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1rem;overflow-x:auto">
    <table class="history-table">
        <thead>
            <tr>
                <th>Course</th><th>Domain</th><th>Difficulty</th>
                <th>Completion</th><th>Outcome</th><th>Date</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <div class="app-title">🧭 CourseCompass</div>
        <div class="app-subtitle">Hybrid Course Recommendation System &nbsp;·&nbsp; ML-Ranked &nbsp;+&nbsp; TF-IDF &nbsp;+&nbsp; Knowledge Graph</div>
    </div>
    """, unsafe_allow_html=True)

    # Check pipeline
    if not check_pipeline_ready():
        st.markdown("""
        <div class="pipeline-warning">
            <strong>⚠️ Data Pipeline Not Ready</strong><br>
            Please run the data pipeline first:<br><br>
            <code>python run_pipeline.py</code><br><br>
            This will generate all required data files in <code>data/processed/</code>.
        </div>
        """, unsafe_allow_html=True)
        return

    # Load data
    with st.spinner("Loading data…"):
        try:
            enriched, interactions, profiles, transitions = load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
            
    with st.spinner("Initializing AI engines & embeddings (may take ~15s on first launch)…"):
        try:
            engine = load_engine(enriched)
            graph_engine = load_graph_engine(enriched, interactions, cache_bust=10)
        except Exception as e:
            st.error(f"Error initializing engine: {e}")
            return
    # Global stats banner
    global_metrics = get_global_metrics(enriched, interactions, profiles)
    
    # Inject KG Stats dynamically
    all_nodes = graph_engine.kg.get_all_nodes()
    global_metrics["KG Tracked Skills"] = len(all_nodes)
    
    cols = st.columns(len(global_metrics))
    for col, (label, val) in zip(cols, global_metrics.items()):
        with col:
            st.markdown(f'<div class="stat-card" style="padding:1rem 0.6rem"><div class="stat-value" style="font-size:1.3rem">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["👤  Recommend for Learner", "🎯  Explore by Goal", "📚  Course Catalog"])

    # ── TAB 1: Learner ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        col_left, col_right = st.columns([1, 2], gap="large")

        with col_left:
            st.markdown('<div class="section-header">Select Learner</div>', unsafe_allow_html=True)

            learner_ids = sorted(profiles["learner_id"].tolist())
            selected_learner = st.selectbox(
                "Learner ID",
                learner_ids,
                index=0,
                key="learner_select",
                label_visibility="collapsed",
            )

            # Learner profile quick stats
            profile_row = profiles[profiles["learner_id"] == selected_learner]
            if not profile_row.empty:
                profile = profile_row.iloc[0]
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Learning Behaviour Summary</div>', unsafe_allow_html=True)
                render_profile_cards(profile)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-header">Recent Learning History</div>', unsafe_allow_html=True)
            render_history_table(selected_learner, interactions, enriched)

        with col_right:
            st.markdown('<div class="section-header">Top 5 Recommended Next Courses</div>', unsafe_allow_html=True)

            # ── Session preference override ───────────────────────────────────
            # Compute the inferred preference first, so we can show it as default
            dyn_prefs_display = compute_dynamic_preferences(selected_learner, interactions)
            inferred_diff_label = dyn_prefs_display.get("preferred_difficulty_label", "Intermediate")
            inferred_pace_label = dyn_prefs_display.get("pace_preference_label", "Medium")
            is_progressing      = dyn_prefs_display.get("is_progressing", False)

            # Show auto-inferred status
            prog_icon = "📊 Progressing ↑" if is_progressing else "➡️ Stable"
            st.markdown(
                f'<div style="font-size:0.78rem;color:#64748b;margin-bottom:0.6rem;">'
                f'🤖 Auto-inferred preference: '
                f'<span style="color:#818cf8">Difficulty: {inferred_diff_label}</span> '
                f'· <span style="color:#a78bfa">Pace: {inferred_pace_label}</span> '
                f'· <span style="color:#4ade80">{prog_icon}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Optional explicit override
            with st.expander("🎯 Session Preference Override (optional)", expanded=False):
                st.markdown(
                    '<div style="font-size:0.8rem;color:#64748b;margin-bottom:0.8rem">'
                    'Override your auto-inferred preference for <b>this session only</b>. '
                    'Your history is unchanged. Leave at “Auto” to use model inference.'
                    '</div>',
                    unsafe_allow_html=True,
                )
                diff_options  = ["Auto (inferred)", "Beginner", "Intermediate", "Advanced"]
                pace_options  = ["Auto (inferred)", "Light", "Medium", "Heavy"]

                sel_diff = st.select_slider(
                    "Preferred Difficulty",
                    options=diff_options,
                    value="Auto (inferred)",
                    key="pref_diff_slider",
                )
                sel_pace = st.select_slider(
                    "Preferred Pace / Workload",
                    options=pace_options,
                    value="Auto (inferred)",
                    key="pref_pace_slider",
                )

            # Build override dict (None when user leaves at Auto)
            DIFF_MAP_UI = {"Beginner": 1.0, "Intermediate": 2.0, "Advanced": 3.0}
            PACE_MAP_UI = {"Light": 1.0, "Medium": 2.0, "Heavy": 3.0}
            user_pref_override: dict | None = None
            ui_overrides = {}
            if sel_diff != "Auto (inferred)":
                ui_overrides["preferred_difficulty"] = DIFF_MAP_UI[sel_diff]
            if sel_pace != "Auto (inferred)":
                ui_overrides["pace_preference"] = PACE_MAP_UI[sel_pace]
            if ui_overrides:
                user_pref_override = ui_overrides
                st.markdown(
                    f'<div style="font-size:0.78rem;background:rgba(99,102,241,0.1);'
                    f'border:1px solid rgba(99,102,241,0.3);border-radius:8px;'
                    f'padding:0.4rem 0.8rem;margin-bottom:0.6rem;color:#c7d2fe">'
                    f'⚙️ Session override active: {ui_overrides}</div>',
                    unsafe_allow_html=True,
                )

            with st.spinner("Computing recommendations…"):
                try:
                    recs = recommend_for_learner(
                        learner_id=selected_learner,
                        profiles_df=profiles,
                        interactions_df=interactions,
                        enriched_df=enriched,
                        transitions_df=transitions,
                        similarity_engine=engine,
                        top_n=5,
                        user_pref_override=user_pref_override,
                    )
                except Exception as e:
                    st.error(f"Recommendation error: {e}")
                    recs = []

            if recs:
                for i, rec in enumerate(recs, 1):
                    render_rec_card(rec, rank=i, mode="A")
            else:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <div>No recommendations found for this learner.<br>
                    <small style="color:#475569">Try selecting a different learner.</small></div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 2: Goal ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        col_input, col_results = st.columns([1, 2], gap="large")

        with col_input:
            st.markdown('<div class="section-header">Define Your Goal</div>', unsafe_allow_html=True)

            available_nodes = ["(Type custom goal...)"] + sorted([n.title for n in graph_engine.kg.get_all_nodes()])
            goal_dropdown = st.selectbox("Select a target skill / node", options=available_nodes)
            
            learning_goal = st.text_input(
                "Or manually type Target Goal / Topic",
                placeholder="e.g. Deep Learning, LLM, Computer Vision…",
                disabled=(goal_dropdown != "(Type custom goal...)")
            )
            
            if goal_dropdown != "(Type custom goal...)":
                learning_goal = goal_dropdown
            
            learner_select_goal = st.selectbox(
                "Select Learner Profile (For Personalization & Skips)",
                ["None (Guest)"] + sorted(profiles["learner_id"].tolist())
            )
            
            go_btn = st.button("🚀 Generate Curriculum Roadmap", use_container_width=True)

        with col_results:
            result = None
            if go_btn and learning_goal:
                with st.spinner(f"Traversing Knowledge Graph for '{learning_goal}'…"):
                    learner_val = None if learner_select_goal == "None (Guest)" else learner_select_goal
                    try:
                        result = graph_engine.build_personalized_roadmap(
                            learning_goal=learning_goal,
                            learner_id=learner_val
                        )
                    except Exception as e:
                        st.error(f"Error building roadmap: {e}")
                        result = {"status": "error"}
                        
            if "roadmap_result" not in st.session_state:
                st.session_state.roadmap_result = None
                
            if result is not None and result.get("status") == "success":
                st.session_state.roadmap_result = result
                st.session_state.llm_chat = []  # reset chat on new result
                
            if st.session_state.roadmap_result:
                result = st.session_state.roadmap_result

                target_node = result["target_node"]
                roadmap = result["roadmap"]
                skipped = result["skipped"]
                
                # Split layout for Roadmap and Chat pane
                col_map, col_chat = st.columns([7, 4])
                
                with col_map:
                    total_dur = result.get("metrics", {}).get("total_duration", 0)
                    st.markdown(f'<div class="section-header">Graph Path: {target_node.title}</div>', unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#94a3b8;font-size:0.9rem;margin-bottom:0.5rem'>{target_node.description}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='color:#fbbf24;font-size:0.85rem;font-weight:600;margin-bottom:1rem'>⏱️ Total Pipeline: ~{total_dur:.0f}h</div>", unsafe_allow_html=True)
                    
                    with st.expander("🛠️ Mastery Skills Required (Full Graph View)", expanded=False):
                        st.markdown(f"To master **{target_node.title}**, the Knowledge Graph tracks the following dependency chain in its ontology:")
                        prereq_graph = graph_engine.get_prerequisite_graph_for_goal(learning_goal)
                        if prereq_graph:
                            for p_nid, deps in prereq_graph.items():
                                p_node = graph_engine.kg.get_node(p_nid)
                                if p_node:
                                    dep_str = ", ".join([graph_engine.kg.get_node(d).title for d in deps if graph_engine.kg.get_node(d)]) if deps else "Foundation Skill"
                                    st.markdown(f"- **{p_node.title}** <span style='color:#64748b;font-size:0.8em'>(Depends on: {dep_str})</span>", unsafe_allow_html=True)
                        else:
                            st.write("Dynamic path (no static prerequisites).")
                            
                    st.markdown("<br>", unsafe_allow_html=True)

                    # Layout Roadmap Stages as Timeline
                    st.markdown('<div class="timeline-path">', unsafe_allow_html=True)
                    for idx, stage in enumerate(roadmap, 1):
                        node = stage["node"]
                        course = stage["course"]
                        milestone = stage.get("milestone", "Core Concept")
                        
                        stage_color = "#4ade80" if node.level == "Foundation" else "#fbbf24" if node.level in ["Core", "Intermediate"] else "#f87171"
                        
                        st.markdown(f'<div class="timeline-dot" style="background:{stage_color};box-shadow:0 0 8px {stage_color}"></div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:1.1rem;font-weight:700;margin:0 0 0.2rem 0;color:{stage_color}">Stage {idx}: {node.title} ({node.level})</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:0.85rem;color:#e2e8f0;font-weight:600;margin-bottom:0.4rem">🌟 Milestone: {milestone}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.8rem">{node.description}</div>', unsafe_allow_html=True)
                        
                        if course:
                            c_title = course["title"]
                            c_dur = course["estimated_duration_hours"]
                            c_wl = course["workload_bucket"]
                            c_url = course.get("url", "")
                            reasons = course.get("reasons", [])
                            prereqs = course.get("prereqs", [])
                            outcomes = course.get("outcome_tags", [])
                            readiness = course.get("readiness_score", 0.0)
                            
                            req_str = f"Requires: {', '.join(prereqs)}" if prereqs else "No hard prerequisites"
                            
                            reasons_html = "".join([f'<div style="display:inline-block;padding:0.2rem 0.5rem;background:rgba(255,255,255,0.05);border-radius:4px;font-size:0.75rem;margin:0.2rem 0.2rem 0 0;color:#cbd5e1">{r}</div>' for r in reasons])
                            
                            outcomes_html = "".join([f'<div style="display:inline-block;padding:0.15rem 0.45rem;background:rgba(16,185,129,0.1);border-radius:4px;font-size:0.75rem;margin:0.2rem 0.2rem 0 0;color:#34d399;border:1px solid rgba(16,185,129,0.2)">✔️ {tag}</div>' for tag in outcomes])
                            
                            rscore_color = "#34d399" if readiness > 0.8 else "#fbbf24"
                            readiness_html = f'<div style="font-size:0.8rem;color:{rscore_color};font-weight:700">Readiness: {int(readiness*100)}%</div>'
                            
                            link_html = f'<a href="{c_url}" target="_blank" style="text-decoration:none;font-size:0.85rem;color:#818cf8;font-weight:600;margin-left:auto">View Course ↗</a>' if c_url else ''
                            
                            bg   = "background:rgba(99,102,241,0.08)"
                            bord  = "border:1px solid rgba(99,102,241,0.3)"
                            html_parts = [
                                f'<div style="{bg};padding:1.2rem;border-radius:12px;{bord};margin-bottom:1.5rem">',
                                f'<div style="display:flex;justify-content:space-between;align-items:start">',
                                f'<div style="font-weight:700;font-size:1rem;color:#e2e8f0;margin-bottom:0.4rem">{c_title}</div>',
                                link_html,
                                '</div>',
                                '<div style="display:flex;gap:0.5rem;margin-bottom:0.8rem">',
                                f'<span style="font-size:0.8rem;color:#94a3b8;background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:12px">\U0001f4bc {c_wl}</span>',
                                f'<span style="font-size:0.8rem;color:#94a3b8;background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:12px">\u23f1\ufe0f ~{c_dur:.0f}h</span>',
                                f'<span style="font-size:0.8rem;color:#94a3b8;background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:12px">📚 {req_str}</span>',
                                '</div>',
                                f'<div style="margin-bottom:0.4rem">{readiness_html}</div>',
                                f'<div style="margin-bottom:0.4rem">{reasons_html}</div>',
                                f'<div>{outcomes_html}</div>',
                                '</div>',
                            ]
                            st.markdown("".join(html_parts), unsafe_allow_html=True)
                        else:
                            st.warning(f"No perfectly suitable courses mapped for {node.title}. Check catalog directly.")
                            st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True) # close timeline
                            
                    if skipped:
                        st.markdown("<hr>", unsafe_allow_html=True)
                        st.markdown('<div class="section-header">Skipped Prerequisites</div>', unsafe_allow_html=True)
                        for skip in skipped:
                            s_node = skip["node"]
                            s_reason = skip["reason"]
                            s_prereqs = skip.get("prereqs", [])
                            req_str = f"Covered Prereqs: {', '.join(s_prereqs)}" if s_prereqs else ""
                            st.markdown(f"✅ **{s_node.title}** — *{s_reason}* <span style='font-size:0.85em;color:#94a3b8'>({req_str})</span>", unsafe_allow_html=True)
                
                with col_chat:
                    # ── Chat Panel Header / Menu Bar ──────────────────────
                    st.markdown("""
                    <div class="chat-panel-header">
                        <div class="chat-panel-dot"></div>
                        <div class="chat-panel-title">🤖 CourseCompass AI</div>
                        <div class="chat-panel-model">Llama 3.3 · Groq</div>
                    </div>
                    <div class="chat-panel-body">
                    """, unsafe_allow_html=True)

                    if "llm_chat" not in st.session_state:
                        st.session_state.llm_chat = []

                    # Attach Roadmap Button
                    if st.button("📎 Attach Roadmap to Chat", use_container_width=True,
                                 help="Injects your current roadmap stages as context for the AI tutor."):
                        ctx_info = f"[SYSTEM INJECTED Roadmap Context] Goal: {target_node.title}. "
                        for i, stg in enumerate(roadmap):
                            ctx_info += f" Stage {i+1} ({stg['node'].title}): {stg['course']['title'] if stg['course'] else 'Theory only'}. "
                        st.session_state.llm_chat.append({"role": "user", "content": ctx_info})
                        st.session_state.llm_chat.append({"role": "assistant", "content": f"✅ Roadmap attached! I now have full context of your **{target_node.title}** learning path with {len(roadmap)} stages. What would you like to know?"})
                        st.rerun()

                    for msg in st.session_state.llm_chat:
                        if msg["content"].startswith("[SYSTEM"):
                            continue
                        with st.chat_message(msg["role"]):
                            st.markdown(msg["content"])

                    st.markdown("</div>", unsafe_allow_html=True)

                    if prompt := st.chat_input("Ask me anything about your roadmap..."):
                        with st.chat_message("user"):
                            st.markdown(prompt)
                        st.session_state.llm_chat.append({"role": "user", "content": prompt})

                        with st.chat_message("assistant"):
                            import sys
                            if "src.recommender.llm_branching" in sys.modules:
                                del sys.modules["src.recommender.llm_branching"]
                            from src.recommender.llm_branching import get_chatbot_response
                            with st.spinner("Thinking..."):
                                resp = get_chatbot_response(prompt, st.session_state.llm_chat, "")
                                st.markdown(resp)
                                st.session_state.llm_chat.append({"role": "assistant", "content": resp})

            if result is not None and result.get("status") == "no_match":
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🔍</div>
                    <div>No matching technical node found in the Knowledge Graph.<br>
                    <small style="color:#475569">Try semantic goals like 'LLM', 'Deep Learning', 'Data Science', 'Python'.</small></div>
                </div>
                """, unsafe_allow_html=True)
            elif not st.session_state.roadmap_result and not go_btn:
                st.markdown("""
                <div class="empty-state">
                    <div class="empty-icon">🎯</div>
                    <div>Enter an AI/ML goal and generate a prerequisite-aware graph roadmap.<br></div>
                </div>
                """, unsafe_allow_html=True)
            elif not st.session_state.roadmap_result and go_btn and not learning_goal:
                st.warning("Please enter a target goal first.")

    # ── TAB 3: Course Catalog ──────────────────────────────────────────────────
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">📚 Master Course Catalog</div>', unsafe_allow_html=True)
        st.markdown("Browse, search, and filter the raw enriched course repository.")
        
        search_query = st.text_input("🔍 Search over 15,000+ courses by title, skill, or domain...", "")
        
        # Fast text filtering
        if search_query:
            # fillna first to avoid errors on empty text fields
            mask = enriched["combined_text"].fillna("").str.contains(search_query, case=False, na=False)
            filtered = enriched[mask]
        else:
            filtered = enriched
            
        st.markdown(f"**Found {len(filtered):,} matches**")
        
        # Render as Box Mode (Cards)
        # We slice top 50 strictly for UI rendering speed, allowing user to narrow down organically
        for _, row in filtered.head(50).iterrows():
            c_title = row.get("title", "Unknown")
            c_dom = row.get("inferred_domain", "")
            c_dur = float(row.get("estimated_duration_hours", 0))
            c_skills = row.get("skills_tags", "")
            c_url = row.get("url", "")
            c_diff = row.get("difficulty_level", "")
            
            st.markdown(f"""
            <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.3);border-radius:12px;padding:1.2rem;margin-bottom:1rem;">
                <div style="display:flex;justify-content:space-between;align-items:start;margin-bottom:0.4rem;">
                    <span style="font-size:1.1rem;font-weight:bold;color:#e2e8f0;">{c_title}</span>
                    <a href="{c_url}" target="_blank" style="text-decoration:none;font-size:0.85rem;color:#818cf8;font-weight:600;min-width:100px;text-align:right;">View Course ↗</a>
                </div>
                <div style="display:flex;gap:0.5rem;margin-bottom:0.6rem;">
                    <span style="font-size:0.8rem;color:#94a3b8;background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:12px">📁 {c_dom}</span>
                    <span style="font-size:0.8rem;color:#94a3b8;background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:12px">📈 {c_diff}</span>
                    <span style="font-size:0.8rem;color:#94a3b8;background:rgba(255,255,255,0.05);padding:0.2rem 0.6rem;border-radius:12px">⏱️ ~{c_dur:.0f}h</span>
                </div>
                <div style="color:#cbd5e1;font-size:0.85rem;">💡 <b>Skills:</b> {c_skills}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if len(filtered) > 50:
            st.markdown("<div style='text-align:center;color:#64748b;font-size:0.9rem;padding:1rem;'>Showing the first 50 results (Box Mode). Please refine your search query to narrow down the remaining hits!</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
