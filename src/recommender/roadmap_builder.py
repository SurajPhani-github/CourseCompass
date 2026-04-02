"""
roadmap_builder.py
===================
Takes top candidates from Goal Mode and extracts a sequential 3-6 course roadmap.
Grouped by: Foundation -> Core -> Advanced.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

def build_roadmap(candidates: list[dict], min_courses: int = 3, max_courses: int = 6) -> dict[str, list[dict]]:
    """
    Given a pool of highly relevant candidate courses, build a structured sequential roadmap.
    Returns structurally grouped courses: {"Foundation": [], "Core": [], "Advanced": []}
    """
    roadmap = {"Foundation": [], "Core": [], "Advanced": []}
    
    # Sort candidates conceptually by difficulty first
    diff_order = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
    
    sorted_cands = sorted(candidates, key=lambda x: (diff_order.get(x["difficulty_level"], 2), -x["final_score"]))
    
    seen_ids = set()
    
    # Step 1: Find 1-2 Foundation
    for c in sorted_cands:
        if len(roadmap["Foundation"]) >= 2:
            break
        if c["difficulty_level"] == "Beginner" and c["course_id"] not in seen_ids:
            roadmap["Foundation"].append(c)
            seen_ids.add(c["course_id"])
            
    # Step 2: Find 2-3 Core
    for c in sorted_cands:
        if len(roadmap["Core"]) >= 3:
            break
        if c["difficulty_level"] == "Intermediate" and c["course_id"] not in seen_ids:
            roadmap["Core"].append(c)
            seen_ids.add(c["course_id"])
            
    # Step 3: Find 1-2 Advanced
    for c in sorted_cands:
        if len(roadmap["Advanced"]) >= 2:
            break
        if c["difficulty_level"] == "Advanced" and c["course_id"] not in seen_ids:
            roadmap["Advanced"].append(c)
            seen_ids.add(c["course_id"])
            
    # Fallback to fill gaps if we don't have enough courses
    # If a user asks for advanced only, they might skip foundation
    total_courses = sum(len(v) for v in roadmap.values())
    if total_courses < min_courses:
        # Resort back to highest relevance
        rel_cands = sorted(candidates, key=lambda x: x["final_score"], reverse=True)
        for c in rel_cands:
            if total_courses >= max_courses:
                break
            if c["course_id"] not in seen_ids:
                if c["difficulty_level"] == "Beginner":
                    roadmap["Foundation"].append(c)
                elif c["difficulty_level"] == "Intermediate":
                    roadmap["Core"].append(c)
                else:
                    roadmap["Advanced"].append(c)
                seen_ids.add(c["course_id"])
                total_courses += 1
                
    return roadmap
