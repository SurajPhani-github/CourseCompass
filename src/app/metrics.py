"""
metrics.py
===========
Calculates global metrics to display at the top of the dashboard.
"""

from __future__ import annotations
import pandas as pd

def get_global_metrics(enriched: pd.DataFrame, interactions: pd.DataFrame, profiles: pd.DataFrame) -> dict:
    total_courses = len(enriched)
    total_enriched = len(enriched)  # Total pipeline size
    synthetic_learners = profiles["learner_id"].nunique()
    total_interactions = len(interactions)
    
    # Diversity: How many domains are represented in the dataset
    total_domains = enriched["inferred_domain"].nunique()
    
    # Recommendation coverage is essentially how many courses have good interactivity (synthetic proxy)
    coverage = len(interactions["course_id"].unique())
    
    return {
        "Total Courses": f"{total_courses:,}",
        "Enriched Courses": f"{total_enriched:,}",
        "Synthetic Learners": f"{synthetic_learners:,}",
        "Interacciones": f"{total_interactions:,}",
        "Rec Coverage": f"{coverage:,}",
        "Domains": f"{total_domains} topics",
    }
