import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
from keybert import KeyBERT

ROOT = Path(__file__).resolve().parents[2]

class DynamicSkillGraphBuilder:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.out_path = ROOT / "data" / "processed" / "dynamic_skill_graph.gpickle"
        # We use a fast, lightweight sentence transformer for KeyBERT extraction
        self.kw_model = KeyBERT('all-MiniLM-L6-v2')
        self.graph = nx.DiGraph()
        
    def load_data(self) -> pd.DataFrame:
        print("Loading Enriched Courses for NLP parsing...")
        df = pd.read_csv(self.data_path)
        return df
        
    def extract_domain_skills(self, df: pd.DataFrame, limit_per_domain=300):
        """
        Parses course descriptions using KeyBERT to extract overarching skill phrases.
        Using a computational subset limit so the MVP can generate locally without crashing.
        """
        domain_skill_metrics = {}
        
        # Difficulty normalization for building topological arrays
        diff_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3, "Mixed": 2}
        
        for domain in df["inferred_domain"].unique():
            if pd.isna(domain): continue
            print(f"Extracting skills for Domain: {domain}")
            
            domain_df = df[df["inferred_domain"] == domain].head(limit_per_domain)
            
            skill_frequencies = {}
            skill_difficulties = {}
            
            for _, row in domain_df.iterrows():
                text = str(row["combined_text"])
                diff = diff_map.get(row.get("difficulty_level", "Mixed"), 2)
                
                # Extract top 3 keywords using KeyBERT with unigrams/bigrams
                keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=3)
                
                for kw, score in keywords:
                    if score < 0.35: continue # Ignore weak statistical matches
                    skill = kw.lower()
                    
                    if skill not in skill_frequencies:
                        skill_frequencies[skill] = 0
                        skill_difficulties[skill] = []
                        
                    skill_frequencies[skill] += 1
                    skill_difficulties[skill].append(diff)
                    
            # Filter and normalize
            filtered_skills = {s: (freq, np.mean(skill_difficulties[s])) 
                               for s, freq in skill_frequencies.items() if freq > 2}
            
            domain_skill_metrics[domain] = filtered_skills
            
        return domain_skill_metrics

    def build_networkx_graph(self, domain_skill_metrics: dict):
        """
        Constructs the DAG.
        Nodes = Skills.
        Edges = Prerequiste dependencies. We infer a prerequisite if an overlap exists
        and the average Difficulty Level scales linearly (e.g., Python (1.2) -> Deep Learning (2.8)).
        """
        print("Constructing NetworkX Dependency Graph...")
        
        for domain, skills in domain_skill_metrics.items():
            # Add implicit Domain Root Node
            domain_node = f"Domain: {domain}"
            self.graph.add_node(domain_node, level="Root", type="Domain")
            
            # Sort skills by difficulty
            sorted_skills = sorted(skills.items(), key=lambda x: x[1][1])
            
            previous_skill = None
            
            for skill, (freq, avg_diff) in sorted_skills:
                # Add node to graph
                level = "Foundation" if avg_diff < 1.5 else "Core" if avg_diff < 2.0 else "Advanced" if avg_diff < 2.6 else "Specialization"
                self.graph.add_node(skill, level=level, frequency=freq, type="Skill")
                
                # Link Domain -> Foundation Skill
                if level == "Foundation":
                    self.graph.add_edge(domain_node, skill, weight=freq)
                
                # Topological link (Prerequisite heuristics)
                # If the previous skill was fundamentally easier, link it.
                if previous_skill and skills[previous_skill][1] < avg_diff:
                    self.graph.add_edge(previous_skill, skill, weight=freq)
                    
                previous_skill = skill
                
    def save_graph(self):
        print(f"Persisting dynamic graph to {self.out_path}")
        with open(self.out_path, 'wb') as f:
            pickle.dump(self.graph, f)
            
    def run_pipeline(self):
        df = self.load_data()
        metrics = self.extract_domain_skills(df)
        self.build_networkx_graph(metrics)
        self.save_graph()
        print(f"Graph Pipeline Complete! Total Nodes: {self.graph.number_of_nodes()}, Total Edges: {self.graph.number_of_edges()}")

if __name__ == "__main__":
    builder = DynamicSkillGraphBuilder(ROOT / "data" / "processed" / "enriched_courses.csv")
    builder.run_pipeline()
