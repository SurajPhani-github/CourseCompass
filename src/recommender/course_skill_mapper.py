"""
course_skill_mapper.py
=======================
Maps static courses to abstract Knowledge Graph nodes using Sentence-Transformers.
Implements disk caching to avoid re-embedding 15k courses on every launch.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
import os

import pandas as pd
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    SentenceTransformer = None

from src.recommender.skill_graph import KnowledgeGraph, SkillNode

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

class CourseSkillMapper:
    def __init__(self, enriched_df: pd.DataFrame, kg: KnowledgeGraph):
        self.enriched = enriched_df
        self.kg = kg
        self.model = None
        self.cache_path = CACHE_DIR / "course_skill_mapping.pkl"
        self.mapping = {}  # {course_id: [(node_id, confidence_score), ...]}
        
    def _get_model(self):
        if not self.model:
            if SentenceTransformer is None:
                raise ImportError("Please install `sentence-transformers` for Goal Map functionality.")
            log.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model

    def build_or_load_mapping(self, force_rebuild: bool = False):
        if not force_rebuild and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.mapping = pickle.load(f)
            log.info(f"Loaded {len(self.mapping)} cached course-to-skill mappings.")
            return

        log.info("Building course-to-skill semantic mapping...")
        model = self._get_model()
        
        # 1. Embed node descriptions
        nodes = self.kg.get_all_nodes()
        node_ids = [n.node_id for n in nodes]
        # Boost semantic relevance by repeating title terms and adding domain
        node_texts = [f"Topic: {n.title}. Domain: {n.domain}. Description: {n.description} {n.title}" for n in nodes]
        node_embeddings = model.encode(node_texts, show_progress_bar=False)
        
        # 2. Embed all courses
        course_ids = self.enriched["course_id"].astype(str).tolist()
        course_titles = self.enriched["title"].fillna("").tolist()
        course_doms = self.enriched["inferred_domain"].fillna("").tolist()
        course_skills = self.enriched["skills_tags"].fillna("").tolist()
        
        c_texts = [f"Course: {t}. Domain: {d}. Skills taught: {s}" for t, d, s in zip(course_titles, course_doms, course_skills)]
        
        log.info(f"Encoding {len(course_ids)} courses...")
        c_embeddings = model.encode(c_texts, show_progress_bar=False)
        
        # 3. Compute similarities
        sims = cosine_similarity(c_embeddings, node_embeddings)
        
        # 4. Map top nodes to each course
        for i, cid in enumerate(course_ids):
            scores = sims[i]
            matched_nodes = []
            for j, score in enumerate(scores):
                if score > 0.35: # conservative threshold
                    matched_nodes.append((node_ids[j], float(score)))
            
            # Sort by score descending
            matched_nodes.sort(key=lambda x: x[1], reverse=True)
            self.mapping[cid] = matched_nodes
            
        # Save to cache
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.mapping, f)
            
        log.info("Mapping complete and cached.")
        
    def get_courses_for_node(self, node_id: str, min_confidence: float = 0.40) -> list[tuple[str, float]]:
        """Returns [(course_id, confidence), ...]"""
        matches = []
        for cid, nodes in self.mapping.items():
            for nid, score in nodes:
                if nid == node_id and score >= min_confidence:
                    matches.append((cid, score))
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
