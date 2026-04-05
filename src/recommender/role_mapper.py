"""
role_mapper.py
==============
Maps abstract role strings (e.g. "Data Scientist", "ML Engineer") to
ordered skill blocks using semantic similarity against the Knowledge Graph.
Instead of hardcoding role→skill mappings, we retrieve relevant skills
dynamically using SBERT embeddings over the existing KG node descriptions.
"""

import logging
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

log = logging.getLogger(__name__)

# Canonical role definitions — lightweight role descriptions
# that seed the semantic search. NOT a hardcoded skill list.
ROLE_DESCRIPTIONS = {
    "Data Scientist": "Statistics, probability, data analysis, machine learning, deep learning, data visualization, Python, SQL, feature engineering, experiment design, A/B testing",
    "ML Engineer": "Machine learning, model deployment, MLOps, Python, TensorFlow, PyTorch, data pipelines, feature stores, model monitoring, distributed training, cloud ML services",
    "Frontend Developer": "HTML, CSS, JavaScript, React, Vue, Angular, responsive design, web accessibility, UI/UX principles, browser APIs, TypeScript, state management",
    "Backend Developer": "Server-side programming, APIs, REST, databases, SQL, NoSQL, authentication, microservices, Python, Node.js, Java, system design, caching",
    "DevOps Engineer": "CI/CD, Docker, Kubernetes, cloud platforms, AWS, Azure, GCP, infrastructure as code, Terraform, monitoring, Linux, networking, automation",
    "AI Engineer": "Artificial intelligence, deep learning, NLP, computer vision, reinforcement learning, transformer models, LLM fine-tuning, prompt engineering, neural architecture search",
    "Cybersecurity Analyst": "Network security, penetration testing, cryptography, incident response, firewalls, SIEM, threat intelligence, vulnerability assessment, compliance, ethical hacking",
    "Data Analyst": "Data analysis, SQL, Excel, data visualization, Tableau, Power BI, statistics, data cleaning, reporting, business intelligence, dashboards",
    "Product Manager": "Product strategy, agile methodology, user research, roadmap planning, stakeholder management, data-driven decisions, wireframing, market analysis, prioritization",
    "Cloud Architect": "Cloud computing, AWS, Azure, GCP, infrastructure design, serverless, containers, networking, security, cost optimization, high availability, disaster recovery",
    "Full Stack Developer": "Frontend development, backend development, databases, REST APIs, JavaScript, Python, React, Node.js, SQL, deployment, version control, testing",
    "NLP Engineer": "Natural language processing, text classification, named entity recognition, sentiment analysis, transformers, BERT, GPT, tokenization, word embeddings, sequence models",
}


class RoleMapper:
    """
    Dynamically maps a role string to an ordered sequence of skill blocks
    by finding the most semantically relevant KG nodes for that role.
    """

    def __init__(self, kg, mapper):
        """
        Parameters
        ----------
        kg : KnowledgeGraph
            The project's knowledge graph instance.
        mapper : CourseSkillMapper
            The course-skill mapper with a loaded SBERT encoder.
        """
        self.kg = kg
        self.mapper = mapper
        self.encoder = mapper._get_model()

    def get_role_skills(self, role_name: str, top_k: int = 8) -> list:
        """
        Given a role name, returns an ordered list of KG SkillNode objects
        ranked by semantic relevance to the role description.

        Falls back to direct SBERT similarity if the role isn't in
        ROLE_DESCRIPTIONS.
        """
        # Get role description — use canonical if available, else use raw name
        role_desc = ROLE_DESCRIPTIONS.get(role_name, role_name)

        # Embed the role description
        role_vec = self.encoder.encode([role_desc])

        # Embed all KG node descriptions
        all_nodes = self.kg.get_all_nodes()
        if not all_nodes:
            log.warning("Knowledge Graph has no nodes.")
            return []

        node_texts = [f"{n.title}. {n.description}" for n in all_nodes]
        node_vecs = self.encoder.encode(node_texts)

        # Compute similarities
        sims = cosine_similarity(role_vec, node_vecs)[0]

        # Rank and select top-K
        ranked_indices = np.argsort(sims)[::-1][:top_k]

        selected_nodes = []
        for idx in ranked_indices:
            if sims[idx] >= 0.25:  # minimum relevance threshold
                selected_nodes.append(all_nodes[idx])

        # Sort by difficulty level for chronological ordering
        level_order = {"Foundation": 0, "Core": 1, "Intermediate": 2, "Advanced": 3, "Specialization": 4}
        selected_nodes.sort(key=lambda n: level_order.get(n.level, 2))

        return selected_nodes

    def build_role_roadmap(self, role_name: str, enriched_df=None, top_courses_per_block: int = 3) -> dict:
        """
        Builds a complete role-based learning path with skill blocks
        and recommended courses per block.

        Returns
        -------
        dict with keys:
            - role: str
            - skill_blocks: list of dicts, each with:
                - skill: SkillNode
                - courses: list of course dicts
                - difficulty: str
                - reason: str (why this skill is relevant)
        """
        nodes = self.get_role_skills(role_name)

        if not nodes:
            return {"role": role_name, "skill_blocks": [], "status": "no_match"}

        skill_blocks = []
        used_cids = set()

        for node in nodes:
            # Use the mapper to find courses for this skill node
            candidates = self.mapper.get_courses_for_skill(node.node_id, top_k=10)

            # Filter out already-used courses and select top N
            block_courses = []
            for c in candidates:
                cid = str(c.get("course_id", ""))
                if cid not in used_cids:
                    block_courses.append(c)
                    used_cids.add(cid)
                if len(block_courses) >= top_courses_per_block:
                    break

            skill_blocks.append({
                "skill": node,
                "courses": block_courses,
                "difficulty": node.level,
                "reason": f"Essential for {role_name}: {node.description[:80]}…" if len(node.description) > 80 else f"Essential for {role_name}: {node.description}",
            })

        return {
            "role": role_name,
            "skill_blocks": skill_blocks,
            "status": "success",
            "total_blocks": len(skill_blocks),
        }

    @staticmethod
    def get_available_roles() -> list[str]:
        """Returns a sorted list of all pre-defined role names."""
        return sorted(ROLE_DESCRIPTIONS.keys())
