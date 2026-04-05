import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

log = logging.getLogger(__name__)

class ResourceRecommender:
    def __init__(self, encoder_model_name: str = "all-MiniLM-L6-v2"):
        self.encoder_model_name = encoder_model_name
        self._model = None
        
        # Load resources dataset
        target_path = Path(__file__).resolve().parents[1] / "data" / "trusted_resources.json"
        try:
            with open(target_path, "r") as f:
                self.resources = json.load(f)
        except Exception as e:
            log.warning(f"Failed to load trusted_resources.json: {e}")
            self.resources = []
            
        self.resource_embeddings = None
        self._init_embeddings()

    def _get_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.encoder_model_name)
        return self._model

    def _init_embeddings(self):
        if not self.resources:
            return
            
        model = self._get_model()
        texts = []
        for r in self.resources:
            text = f"{r['title']} {r['type']} {r['description']} {' '.join(r['tags'])}"
            texts.append(text)
            
        self.resource_embeddings = model.encode(texts, show_progress_bar=False)

    def recommend_for_skill(self, skill_title: str, skill_desc: str = "", top_k: int = 3):
        """
        Recommends external resources for a given skill based on semantic similarity and weighted heuristics.
        Returns a dict grouped by resource 'type'.
        """
        if not self.resources or self.resource_embeddings is None:
            return {}
            
        model = self._get_model()
        query_text = f"{skill_title} {skill_desc}"
        q_emb = model.encode([query_text], show_progress_bar=False)
        
        sims = cosine_similarity(q_emb, self.resource_embeddings)[0]
        
        # Compute dynamic scores
        # Formula: w1*sim + w2*quality_score + w3*popularity_normalized
        max_pop = max([r["popularity"] for r in self.resources]) if self.resources else 1
        
        scored_resources = []
        for i, r in enumerate(self.resources):
            sim = sims[i]
            q_score = r.get("quality_score", 0.8)
            pop_norm = r.get("popularity", 0) / max_pop
            
            # Weighted ensemble
            final_score = (0.6 * sim) + (0.3 * q_score) + (0.1 * pop_norm)
            
            if final_score > 0.4: # Arbitrary threshold to ensure decent relevance
                scored_resources.append((final_score, r))
                
        scored_resources.sort(key=lambda x: x[0], reverse=True)
        top_matches = [r for _, r in scored_resources[:top_k]]
        
        # Group by type
        grouped = {}
        for r in top_matches:
            rtype = r["type"]
            if rtype not in grouped:
                grouped[rtype] = []
            grouped[rtype].append(r)
            
        return grouped
