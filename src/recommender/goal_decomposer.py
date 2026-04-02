"""
goal_decomposer.py
===================
Maps natural language user goals to abstract Skill Nodes and recursively fetches 
ancestor prerequisites from the Knowledge Graph.
"""

from __future__ import annotations
import logging
import numpy as np
from src.recommender.skill_graph import KnowledgeGraph, SkillNode
from src.recommender.course_skill_mapper import CourseSkillMapper

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None

log = logging.getLogger(__name__)

class GoalDecomposer:
    def __init__(self, kg: KnowledgeGraph, mapper: CourseSkillMapper):
        self.kg = kg
        self.mapper = mapper
        
    def map_goal_to_node(self, learning_goal: str) -> SkillNode | None:
        """Find the semantic closest SkillNode to the user's free-text goal."""
        model = self.mapper._get_model()
        if not model or not cosine_similarity:
            return None
            
        goal_emb = model.encode([learning_goal])[0]
        
        nodes = self.kg.get_all_nodes()
        node_texts = [f"Topic: {n.title}. Domain: {n.domain}. Description: {n.description}" for n in nodes]
        node_embs = model.encode(node_texts)
        
        sims = cosine_similarity([goal_emb], node_embs)[0]
        best_idx = int(np.argmax(sims))
        
        if sims[best_idx] > 0.35:  # ensure it's actually somewhat relevant
            return nodes[best_idx]
        return None

    def decompose(self, learning_goal: str) -> tuple[SkillNode | None, list[str]]:
        """
        Returns:
            - target_node: SkillNode representing the final goal
            - required_nodes: Ordered list of node_ids representing the full prerequisite path ending at target.
        """
        target_node = self.map_goal_to_node(learning_goal)
        if not target_node:
            return None, []
            
        prereqs = self.kg.get_prerequisites(target_node.node_id)
        # Prereqs are topologically sorted starting from most basic.
        # Add the target node at the end.
        path = list(prereqs)
        if target_node.node_id not in path:
            path.append(target_node.node_id)
            
        return target_node, path
