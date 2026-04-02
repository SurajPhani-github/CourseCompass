"""Smoke test for the full ML ranking pipeline."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import pandas as pd
from pathlib import Path

DATA = Path("data/processed")
profiles     = pd.read_csv(DATA / "learner_profiles.csv")
interactions = pd.read_csv(DATA / "synthetic_learner_interactions.csv")
enriched     = pd.read_csv(DATA / "enriched_courses.csv")
transitions  = pd.read_csv(DATA / "course_transitions.csv")

interactions["timestamp"]        = pd.to_datetime(interactions["timestamp"])
enriched["course_id"]            = enriched["course_id"].astype(str)
transitions["prev_course_id"]    = transitions["prev_course_id"].astype(str)
transitions["next_course_id"]    = transitions["next_course_id"].astype(str)

print("=== Loading content engine ===")
from src.recommender.content_similarity import get_engine
engine = get_engine(enriched)

from src.recommender.learner_recommender import recommend_for_learner
from src.recommender.ranker_features import compute_dynamic_preferences

lid = profiles["learner_id"].iloc[0]
print(f"\n=== Auto-inferred prefs for {lid} ===")
prefs = compute_dynamic_preferences(lid, interactions)
print(f"  preferred_difficulty : {prefs['preferred_difficulty']} ({prefs['preferred_difficulty_label']})")
print(f"  pace_preference      : {prefs['pace_preference']} ({prefs['pace_preference_label']})")
print(f"  is_progressing       : {prefs['is_progressing']}")

print(f"\n=== Recommendations (auto pref) for {lid} ===")
recs = recommend_for_learner(lid, profiles, interactions, enriched, transitions, engine, top_n=3)
print(f"Recs returned: {len(recs)}")
for r in recs:
    ml  = r.get("ml_score")
    hs  = r.get("final_score")
    mlr = r.get("ml_ranked")
    diff = r["difficulty_level"]
    print(f"  [{diff:12}] final={hs:.3f}  ml={ml}  ml_ranked={mlr}")
    print(f"    Title: {r['title'][:60]}")
    for reason in r.get("reasons", []):
        print(f"    -> {reason}")

print(f"\n=== With UI override: Advanced ===")
recs_adv = recommend_for_learner(
    lid, profiles, interactions, enriched, transitions, engine,
    top_n=3, user_pref_override={"preferred_difficulty": 3.0}
)
for r in recs_adv:
    diff = r["difficulty_level"]
    print(f"  [{diff:12}] final={r['final_score']:.3f}  {r['title'][:55]}")

print(f"\n=== Knowledge Graph smoke test ===")
from src.recommender.skill_graph import KnowledgeGraph
kg = KnowledgeGraph()
nodes = kg.get_all_nodes()
print(f"Total nodes : {len(nodes)}")
print(f"Domains     : {kg.get_all_domains()}")
path = kg.get_full_path("llm")
print(f"Path to LLM : {path}")
search = kg.search_nodes("machine learning")
print(f"Search 'machine learning': {[(nid, n.title) for nid, n in search[:3]]}")

prereq_graph = kg.get_prerequisite_subgraph("llm")
print(f"Prereq graph for LLM: {prereq_graph}")

print("\n=== All checks PASSED ===")
