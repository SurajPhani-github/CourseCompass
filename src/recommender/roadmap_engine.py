"""
roadmap_engine.py
==================
Generates the specific course sequence for a decomposed skill path.
Personalizes by skipping nodes the user has mastered (inferred from past courses).
"""

from __future__ import annotations
import logging
import pandas as pd
from typing import Any

from src.recommender.skill_graph import KnowledgeGraph, SkillNode
from src.recommender.course_skill_mapper import CourseSkillMapper
from src.recommender.goal_decomposer import GoalDecomposer
from src.recommender.content_similarity import get_engine
from src.recommender.ranker_features import compute_dynamic_preferences
from src.recommender.llm_branching import generate_roadmap_branches

log = logging.getLogger(__name__)

class RoadmapEngine:
    def __init__(self, kg: KnowledgeGraph, mapper: CourseSkillMapper, decomposer: GoalDecomposer, enriched_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.kg = kg
        self.mapper = mapper
        self.decomposer = decomposer
        self.enriched = enriched_df
        self.interactions = interactions_df
        
    def _get_user_mastered_nodes(self, learner_id: str | None) -> set[str]:
        """Infers mastered skills by looking at the user's historical course completions."""
        if not learner_id or self.interactions is None or self.interactions.empty:
            return set()
            
        # Get courses completed by user with high outcome
        user_history = self.interactions[self.interactions["learner_id"] == learner_id]
        if user_history.empty:
            return set()
            
        # Keep courses with good completion and outcome
        passed = user_history[(user_history["completion_rate"] > 0.6) & (user_history["learning_outcome"].isin(["good", "excellent"]))]
        passed_cids = passed["course_id"].astype(str).tolist()
        
        mastered_nodes = set()
        for cid in passed_cids:
            if cid in self.mapper.mapping:
                # If course maps strongly (>0.5) to a node, consider it mastered
                for nid, score in self.mapper.mapping[cid]:
                    if score > 0.50:
                        mastered_nodes.add(nid)
                        
        return mastered_nodes

    def _select_best_course_for_node(self, node_id: str, dyn_prefs: dict, used_cids: set) -> dict | None:
        """Finds the best single course to fulfill a single node, matching dynamic user prefs."""
        matches = self.mapper.get_courses_for_node(node_id, min_confidence=0.45)
        if not matches:
            # Fallback lower confidence
            matches = self.mapper.get_courses_for_node(node_id, min_confidence=0.35)
            if not matches:
                return None
                
        # Candidate courses
        candidate_ids = [cid for cid, score in matches[:25] if cid not in used_cids]
        if not candidate_ids:
            return None
            
        c_df = self.enriched[self.enriched["course_id"].astype(str).isin(candidate_ids)].copy()
        
        if c_df.empty: return None
        
        scores = []
        wl_map = {"Light": 1, "Medium": 2, "Heavy": 3}
        diff_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        
        pref_pace = float(dyn_prefs.get("pace_preference", 2.0))
        pref_diff = float(dyn_prefs.get("preferred_difficulty", 2.0))
        
        for _, row in c_df.iterrows():
            cid = str(row["course_id"])
            # Confidence score from the semantic matching
            conf = next((s for c, s in matches if c == cid), 0.5)
            
            wl = str(row.get("workload_bucket", "Medium"))
            diff = str(row.get("difficulty_level", "Intermediate"))
            
            wl_val = wl_map.get(wl, 2)
            diff_val = diff_map.get(diff, 2)
            wl_delta = abs(pref_pace - wl_val)
            diff_delta = abs(pref_diff - diff_val)

            wl_score = 1.0 if wl_delta == 0 else (0.6 if wl_delta == 1 else 0.2)
            
            # Heavy penalty if recommending Beginner to Advanced or Advanced to Beginner
            if pref_diff >= 2.5 and diff_val == 1:
                diff_score = -1.0
            elif pref_diff <= 1.5 and diff_val == 3:
                diff_score = -0.5
            else:
                diff_score = 1.0 if diff_delta == 0 else (0.6 if diff_delta == 1 else 0.2)
            
            pop = float(row.get("popularity_proxy", 0.5))
            qual = float(row.get("quality_proxy", 0.5))
            
            # Simple ranking: Semantic match is still king, but difficulty is strict now
            final = (conf * 0.40) + (diff_score * 0.25) + (wl_score * 0.15) + (qual * 0.10) + (pop * 0.10)
            
            node = self.kg.get_node(node_id)
            node_level = node.level if node else ""
            
            # Outcome tags generated dynamically from node
            outcome_tags = [kw.title() for kw in (node.keywords[:3] if node else [])]
            
            scores.append({
                "course_id": cid,
                "title": str(row.get("title", cid)),
                "difficulty_level": diff,
                "workload_bucket": wl,
                "estimated_duration_hours": float(row.get("estimated_duration_hours", 10)),
                "popularity_proxy": pop,
                "final_score": final,
                "semantic_confidence": conf,
                "readiness_score": min(0.98, final + 0.15),
                "node_aligned": node_level,
                "url": str(row.get("url", "")),
                "outcome_tags": outcome_tags,
            })
            
        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores[0]

    def _build_dynamic_fallback(self, learning_goal: str, workload_preference: str) -> tuple[SkillNode | None, list[dict]]:
        """Dynamically generates a 5-stage sequential roadmap using TF-IDF text logic for arbitrary domains."""
        sim_engine = get_engine(self.enriched)
        
        # 1. Query top highly relevant courses
        top_raw = sim_engine.query_text(learning_goal, top_n=60)
        if not top_raw:
            return None, []
        
        cids = [r[0] for r in top_raw]
        rel_map = {r[0]: r[1] for r in top_raw}
        
        c_df = self.enriched[self.enriched["course_id"].astype(str).isin(cids)].copy()
        if c_df.empty:
            return None, []
            
        c_df["search_relevance"] = c_df["course_id"].astype(str).map(rel_map)
        
        wl_map = {"Light": 1, "Medium": 2, "Heavy": 3}
        pref_val = wl_map.get(workload_preference, 2)
        
        def score_row(row):
            wl = str(row.get("workload_bucket", "Medium"))
            delta = abs(pref_val - wl_map.get(wl, 2))
            wl_score = 1.0 if delta == 0 else (0.6 if delta == 1 else 0.2)
            pop = float(row.get("popularity_proxy", 0.5))
            qual = float(row.get("quality_proxy", 0.5))
            rel = float(row["search_relevance"])
            return (rel * 0.6) + (qual * 0.2) + (pop * 0.1) + (wl_score * 0.1)

        c_df["final_score"] = c_df.apply(score_row, axis=1)
        c_df = c_df.sort_values("final_score", ascending=False)
        
        # Create difficulty buckets
        beg = c_df[c_df["difficulty_level"] == "Beginner"]
        inter = c_df[c_df["difficulty_level"] == "Intermediate"]
        adv = c_df[c_df["difficulty_level"].isin(["Advanced", "Mixed"])]
        
        target_node = SkillNode(
            node_id=f"dyn_{learning_goal.replace(' ', '_').lower()}",
            title=f"{learning_goal.title()} Specialization",
            description=f"Comprehensive roadmap achieving mastery in {learning_goal.title()}.",
            level="Specialization",
            domain=learning_goal.title()
        )
        
        stages = [
            {"level": "Foundation", "title": f"{learning_goal.title()} 101", "desc": "Absolute basics and fundamental concepts.", "pool": beg},
            {"level": "Core", "title": f"Applied {learning_goal.title()}", "desc": "Practical application of introductory material.", "pool": beg.iloc[1:] if len(beg) > 1 else inter},
            {"level": "Intermediate", "title": f"Intermediate {learning_goal.title()}", "desc": "Deeper methodologies and core theory.", "pool": inter},
            {"level": "Advanced", "title": f"Advanced {learning_goal.title()}", "desc": "Complex implementations and high-level use-cases.", "pool": adv},
            {"level": "Specialization", "title": f"{learning_goal.title()} Mastery", "desc": "Niche specialization and expert techniques.", "pool": adv.iloc[1:] if len(adv) > 1 else adv}
        ]
        
        roadmap_stages = []
        used_cids = set()
        
        for stage in stages:
            node = SkillNode(
                node_id=f"dyn_stage_{stage['level'].lower()}",
                title=stage["title"],
                description=stage["desc"],
                level=stage["level"],
                domain=learning_goal.title()
            )
            
            pool = stage["pool"]
            course = None
            
            for _, row in pool.iterrows():
                cid = str(row["course_id"])
                if cid not in used_cids:
                    used_cids.add(cid)
                    course = {
                        "course_id": cid,
                        "title": str(row.get("title", cid)),
                        "difficulty_level": str(row.get("difficulty_level", "Intermediate")),
                        "workload_bucket": str(row.get("workload_bucket", "Medium")),
                        "estimated_duration_hours": float(row.get("estimated_duration_hours", 10)),
                        "popularity_proxy": float(row.get("popularity_proxy", 0.5)),
                        "url": str(row.get("url", "")),
                        "reasons": [
                            f"🎯 Deep semantic match for '{learning_goal}'",
                            f"📈 Difficulty tier map for {stage['level']}",
                            f"⏱️ Workload fit ({row.get('workload_bucket', 'Medium')})"
                        ]
                    }
                    break
                    
            if course:
                roadmap_stages.append({"node": node, "course": course})
                
        return target_node, roadmap_stages

    def build_personalized_roadmap(
        self, 
        learning_goal: str, 
        learner_id: str | None = None,
        dyn_prefs: dict | None = None
    ) -> dict[str, Any]:
        
        target_node, full_path = self.decomposer.decompose(learning_goal)
        
        if not dyn_prefs:
            if learner_id and self.interactions is not None:
                dyn_prefs = compute_dynamic_preferences(learner_id, self.interactions)
            else:
                dyn_prefs = {"preferred_difficulty": 2.0, "pace_preference": 2.0}
        
        if not target_node or not full_path:
            # Fallback to dynamic arbitrary domain builder
            wl_pref = "Medium"
            if dyn_prefs.get("pace_preference", 2.0) > 2.5: wl_pref = "Heavy"
            elif dyn_prefs.get("pace_preference", 2.0) < 1.5: wl_pref = "Light"
                
            dyn_target, dyn_roadmap = self._build_dynamic_fallback(learning_goal, workload_preference=wl_pref)
            if not dyn_target or not dyn_roadmap:
                return {"status": "no_match", "roadmap": [], "skipped": [], "metrics": {}}
                
            return {
                "status": "success",
                "target_node": dyn_target,
                "roadmap": dyn_roadmap,
                "skipped": [],
                "metrics": {"total_duration": sum(s["course"]["estimated_duration_hours"] for s in dyn_roadmap if s["course"])}
            }
            
        mastered = self._get_user_mastered_nodes(learner_id)
        
        roadmap_stages = []
        skipped_stages = []
        used_cids = set()
        total_duration = 0.0
        
        for nid in full_path:
            node = self.kg.get_node(nid)
            if not node: continue
            
            # Prereq lookup dict
            prereqs_nodes = [self.kg.get_node(p) for p in self.kg.get_direct_prerequisites(nid)]
            prereq_titles = [p.title for p in prereqs_nodes if p]
            
            if nid in mastered:
                skipped_stages.append({
                    "node": node,
                    "reason": "Skill Gap Check Passed (Already Mastered)",
                    "prereqs": prereq_titles
                })
                continue
                
            # Highly personal: if user is strictly advanced, skip the 101/Foundation basics
            if dyn_prefs.get('preferred_difficulty', 2.0) >= 2.5 and node.level == "Foundation":
                skipped_stages.append({
                    "node": node,
                    "reason": "Inferred from profile (Advanced Learner)",
                    "prereqs": prereq_titles
                })
                continue
                
            course = self._select_best_course_for_node(nid, dyn_prefs, used_cids)
            if course:
                used_cids.add(course["course_id"])
                total_duration += course.get("estimated_duration_hours", 0)
                
                # Enhanced reasoning logic
                reasons = []
                reasons.append(f"🎯 Core requirement for {target_node.title}")
                
                # Difficulty / Pace reasons
                c_diff = course["difficulty_level"]
                pref_diff = dyn_prefs.get('preferred_difficulty_label', 'Intermediate')
                if c_diff == pref_diff:
                    reasons.append(f"📈 Matches your '{pref_diff}' difficulty preference")
                    
                # Why this stage exists
                if prereq_titles:
                    reasons.append(f"🔗 Builds upon: {', '.join(prereq_titles[:2])}")
                
                course["reasons"] = reasons
                course["prereqs"] = prereq_titles
                
                roadmap_stages.append({
                    "node": node,
                    "course": course,
                    "milestone": f"Unlock {node.title} capability"
                })
            else:
                roadmap_stages.append({
                    "node": node,
                    "course": None,
                    "prereqs": prereq_titles,
                    "milestone": f"Understand {node.title} theory"
                })
                
        return {
            "status": "success",
            "target_node": target_node,
            "roadmap": roadmap_stages,
            "skipped": skipped_stages,
            "metrics": {
                "total_duration": total_duration,
                "total_stages": len(roadmap_stages),
                "skills_acquired": len(full_path)
            }
        }

    def build_llm_branched_roadmaps(
        self, 
        learning_goal: str, 
        learner_id: str | None = None,
        dyn_prefs: dict | None = None
    ) -> dict[str, Any]:
        """
        Dynamically generates exactly 3 distinct branches (e.g. Research, Builder, Full-Stack)
        using Gemini, and maps them directly to the Course Catalog, bypassing heuristics entirely.
        """
        if not dyn_prefs:
            if learner_id and self.interactions is not None:
                dyn_prefs = compute_dynamic_preferences(learner_id, self.interactions)
            else:
                dyn_prefs = {"preferred_difficulty": 2.0, "pace_preference": 2.0}
                
        # Retrieve sparse profile dict to pass to Gemini
        profile_dict = {}
        mastered = self._get_user_mastered_nodes(learner_id)
        if learner_id and self.interactions is not None:
            user_hist = self.interactions[self.interactions["learner_id"] == learner_id]
            if not user_hist.empty:
                profile_dict["courses_completed"] = len(user_hist[user_hist["learning_outcome"] != "dropped"])
                profile_dict["inferred_difficulty"] = dyn_prefs.get('preferred_difficulty_label', 'Intermediate')
        
        # 1. Call LLM for abstract stages
        branches_json = generate_roadmap_branches(learning_goal, learner_profile=profile_dict)
        
        if not isinstance(branches_json, list) or len(branches_json) == 0:
            return {"status": "error", "error_message": "Failed to generate AI branches."}
            
        used_cids = set()
        mastered_lowered = {n.lower() for n in mastered}
        
        final_branches = []
        for branch in branches_json:
            path_name = branch.get("path_name", "Alternative Path")
            desc = branch.get("description", "")
            projects = branch.get("target_projects", [])
            stages = branch.get("stages", [])
            
            roadmap_stages = []
            skipped_stages = []
            total_duration = 0.0
            
            for idx, stage in enumerate(stages):
                node_title = stage.get("title", f"Stage {idx+1}")
                st_desc = stage.get("description", "")
                st_level = stage.get("level", "Intermediate")
                milestone = stage.get("milestone", "Core Concept")
                outcomes = stage.get("outcome_tags", [])
                
                # Check for conceptual overlap with 'mastered' (heuristically using string contains)
                skill_skip_reason = None
                for m in mastered_lowered:
                    if m in node_title.lower() or m.replace('_', ' ') in node_title.lower():
                        skill_skip_reason = "Already Mastered"
                        break
                        
                # Create a temporary SkillNode adapter so the UI logic doesn't break
                dyn_node = SkillNode(
                    node_id=f"llm_{node_title.replace(' ', '_').lower()}",
                    title=node_title,
                    description=st_desc,
                    level=st_level,
                    domain=learning_goal.title(),
                    keywords=outcomes
                )
                
                if skill_skip_reason:
                    skipped_stages.append({
                        "node": dyn_node,
                        "reason": skill_skip_reason,
                        "prereqs": []
                    })
                    continue
                
                # We need to map this 'hallucinated' node to our courses.
                # Strategy: Map using the LLM's title and description string dynamically into the mapper
                search_text = f"{node_title}. {st_desc} {' '.join(outcomes)}"
                course = self._select_best_course_for_synthetic_node(search_text, dyn_prefs, used_cids)
                
                if course:
                    used_cids.add(course["course_id"])
                    total_duration += course.get("estimated_duration_hours", 0)
                    
                    course["reasons"] = [
                        f"🎯 Core requirement for {path_name}",
                        f"📈 Matches your difficulty trajectory",
                        f"🤖 Dynamically AI encoded"
                    ]
                    course["outcome_tags"] = outcomes
                    course["prereqs"] = []
                    
                    roadmap_stages.append({
                        "node": dyn_node,
                        "course": course,
                        "milestone": milestone
                    })
                else:
                    roadmap_stages.append({
                        "node": dyn_node,
                        "course": None,
                        "prereqs": [],
                        "milestone": milestone
                    })
                    
            final_branches.append({
                "path_name": path_name,
                "description": desc,
                "projects": projects,
                "roadmap": roadmap_stages,
                "skipped": skipped_stages,
                "metrics": {
                    "total_duration": total_duration,
                    "total_stages": len(roadmap_stages)
                }
            })
            
        target_mock = SkillNode(node_id="llm_root", title=learning_goal.title(), description="LLM Branched Curriculum", level="Specialization", domain=learning_goal.title())
        return {
            "status": "success",
            "branches": final_branches,
            "target_node": target_mock
        }

    def _select_best_course_for_synthetic_node(self, query: str, dyn_prefs: dict, used_cids: set) -> dict | None:
        """Dynamically maps an LLM-generated stage description to existing courses."""
        # 1. Encode query
        query_emb = self.mapper.encoder.encode([query])
        
        # 2. Get sim scores against ALL courses
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        c_embs = self.mapper.course_embeddings
        if c_embs is None: return None
        
        sims = cosine_similarity(query_emb, c_embs)[0]
        top_indices = np.argsort(sims)[::-1][:20]
        
        matches = [(self.mapper.course_ids[idx], float(sims[idx])) for idx in top_indices if sims[idx] > 0.35]
        
        if not matches: return None
        
        candidate_ids = [cid for cid, score in matches if cid not in used_cids]
        if not candidate_ids: return None
        
        c_df = self.enriched[self.enriched["course_id"].astype(str).isin(candidate_ids)].copy()
        if c_df.empty: return None
        
        scores = []
        wl_map = {"Light": 1, "Medium": 2, "Heavy": 3}
        diff_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3}
        
        pref_pace = float(dyn_prefs.get("pace_preference", 2.0))
        pref_diff = float(dyn_prefs.get("preferred_difficulty", 2.0))
        
        for _, row in c_df.iterrows():
            cid = str(row["course_id"])
            conf = next((s for c, s in matches if c == cid), 0.5)
            
            wl = str(row.get("workload_bucket", "Medium"))
            diff = str(row.get("difficulty_level", "Intermediate"))
            
            wl_val = wl_map.get(wl, 2)
            diff_val = diff_map.get(diff, 2)
            
            wl_delta = abs(pref_pace - wl_val)
            diff_delta = abs(pref_diff - diff_val)
            
            wl_score = 1.0 if wl_delta == 0 else (0.6 if wl_delta == 1 else 0.2)
            diff_score = 1.0 if diff_delta == 0 else (0.6 if diff_delta == 1 else 0.2)
            
            pop = float(row.get("popularity_proxy", 0.5))
            qual = float(row.get("quality_proxy", 0.5))
            
            final = (conf * 0.45) + (qual * 0.15) + (diff_score * 0.15) + (wl_score * 0.15) + (pop * 0.1)
            
            scores.append({
                "course_id": cid,
                "title": str(row.get("title", cid)),
                "difficulty_level": diff,
                "workload_bucket": wl,
                "estimated_duration_hours": float(row.get("estimated_duration_hours", 10)),
                "popularity_proxy": pop,
                "final_score": final,
                "semantic_confidence": conf,
                "readiness_score": min(0.98, final + 0.15),
                "url": str(row.get("url", ""))
            })
            
        scores.sort(key=lambda x: x["final_score"], reverse=True)
        return scores[0]

    def get_prerequisite_graph_for_goal(self, learning_goal: str) -> dict[str, list[str]]:
        """
        Returns a dict of {node_id: [prerequisite_ids]} for the given learning goal.
        Useful for generating graph visualizations in the frontend.
        """
        target_node, _ = self.decomposer.decompose(learning_goal)
        if not target_node:
            return {}
            
        # Standard graph mode: target_node.node_id is part of the actual graph
        if target_node.node_id.startswith("dyn_"):
            # Dynamic graph mode does not have real graph dependencies yet.
            # Return empty or placeholder
            return {}
            
        return self.kg.get_prerequisite_subgraph(target_node.node_id)
        
    def get_course_path_summary(self, roadmap_result: dict[str, Any]) -> str:
        """
        Generates a textual summary of the fully mapped course pathway.
        """
        if roadmap_result.get("status") != "success":
            return "No valid roadmap available."
            
        target = roadmap_result["target_node"]
        stages = roadmap_result["roadmap"]
        skipped = roadmap_result["skipped"]
        
        lines = [f"Roadmap to: {target.title} ({target.domain})", "=" * 40]
        
        if skipped:
            lines.append("\nAlready Mastered:")
            for s in skipped:
                lines.append(f" ✓ {s['node'].title}")
                
        lines.append("\nRecommended Path:")
        for idx, stage in enumerate(stages, 1):
            n = stage["node"]
            c = stage["course"]
            lines.append(f"{idx}. {n.title} [{n.level}]")
            if c:
                lines.append(f"    ↳ Course: {c['title']} (~{c['estimated_duration_hours']:.0f}h)")
            else:
                lines.append(f"    ↳ (No explicit mapped course)")
                
        return "\n".join(lines)

