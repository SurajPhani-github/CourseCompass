import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity

# Ensure we're in the right directory contexts
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommender.skill_graph import KnowledgeGraph
from src.recommender.course_skill_mapper import CourseSkillMapper
from src.recommender.goal_decomposer import GoalDecomposer
from src.recommender.roadmap_engine import RoadmapEngine

def setup_evaluation():
    print("Loading data for Evaluation Suite...")
    enriched = pd.read_csv(ROOT / "data" / "processed" / "enriched_courses.csv")
    interactions = pd.read_csv(ROOT / "data" / "processed" / "synthetic_learner_interactions.csv")
    
    kg = KnowledgeGraph()
    mapper = CourseSkillMapper(enriched, kg)
    mapper.build_or_load_mapping(force_rebuild=False)
    
    decomposer = GoalDecomposer(kg, mapper)
    engine = RoadmapEngine(kg, mapper, decomposer, enriched, interactions)
    
    return engine, mapper, kg

def evaluate_semantic_coherence(engine, mapper, test_goals):
    """
    Evaluates how closely linked (semantically) successive stages in a roadmap are.
    Since learning is progressive, Stage N and Stage N+1 should share thematic embedding similarities.
    """
    print("Evaluating Semantic Coherence...")
    coherence_scores = []
    
    for goal in test_goals:
        res = engine.build_personalized_roadmap(goal)
        if res["status"] != "success": continue
        
        roadmap = res.get("roadmap", [])
        if len(roadmap) < 2: continue
        
        # Get course texts to encode
        c_texts = []
        for stage in roadmap:
            cid = str(stage["course"]["course_id"])
            mask = engine.enriched["course_id"].astype(str) == cid
            if mask.any():
                text = engine.enriched[mask].iloc[0]["combined_text"]
                c_texts.append(text)
                
        if len(c_texts) < 2: continue
        
        # Embed and measure steps
        embs = mapper.encoder.encode(c_texts)
        for i in range(len(embs) - 1):
            sim = cosine_similarity([embs[i]], [embs[i+1]])[0][0]
            coherence_scores.append(sim)
            
    return coherence_scores

def evaluate_topological_accuracy(engine, test_goals):
    """
    Checks if a roadmap strictly adheres to increasing difficulty.
    Valid sequences: [Beginner, Beginner, Intermediate, Advanced]
    Invalid: [Advanced, Beginner, ...]
    """
    print("Evaluating Topological Progression Accuracy...")
    
    diff_map = {"Beginner": 1, "Intermediate": 2, "Advanced": 3, "Mixed": 2}
    valid_paths = 0
    total_paths = 0
    
    plot_data = [] # For graphing
    
    for goal in test_goals:
        res = engine.build_personalized_roadmap(goal)
        if res["status"] != "success": continue
        
        roadmap = res.get("roadmap", [])
        if len(roadmap) < 3: continue
        
        diff_array = [diff_map.get(stage["course"]["difficulty_level"], 2) for stage in roadmap]
        plot_data.append((goal, diff_array))
        
        # Check monotonicity
        is_monotonic = all(diff_array[i] <= diff_array[i+1] for i in range(len(diff_array)-1))
        
        if is_monotonic:
            valid_paths += 1
        total_paths += 1
        
    acc = valid_paths / total_paths if total_paths > 0 else 0
    return acc, plot_data

def evaluate_workload_penalty(engine):
    """
    Tests if injecting hard Workload Preferences actively filters candidate results.
    """
    print("Evaluating Workload Penalty Enforcement...")
    goal = "Deep Learning"
    
    # Force heavy workload preference
    res_heavy = engine.build_personalized_roadmap(goal, dyn_prefs={"pace_preference": 3.0, "preferred_difficulty": 2.0})
    heavy_durations = [float(s["course"]["estimated_duration_hours"]) for s in res_heavy.get("roadmap", [])]
    avg_heavy = np.mean(heavy_durations) if heavy_durations else 0
    
    # Force light workload preference
    res_light = engine.build_personalized_roadmap(goal, dyn_prefs={"pace_preference": 1.0, "preferred_difficulty": 2.0})
    light_durations = [float(s["course"]["estimated_duration_hours"]) for s in res_light.get("roadmap", [])]
    avg_light = np.mean(light_durations) if light_durations else 0
    
    return avg_light, avg_heavy

def run_evaluation_suite():
    engine, mapper, kg = setup_evaluation()
    
    # Define a robust test suite of abstract goals
    test_goals = [
        "Data Science", "Machine Learning", "Deep Learning", "Artificial Intelligence",
        "Frontend Development", "Backend Architecture", "Cybersecurity Analyst",
        "DevOps Engineer", "Cloud Native Architect", "Data Analysis", "Python Programming",
        "Business Intelligence", "Natural Language Processing", "Generative AI Systems"
    ]
    
    # 1. Semantic Coherence
    coherences = evaluate_semantic_coherence(engine, mapper, test_goals)
    avg_coh = np.mean(coherences) if coherences else 0
    
    # 2. Topological Progression
    topo_acc, topo_curves = evaluate_topological_accuracy(engine, test_goals)
    
    # 3. Workload Penalty Accuracy
    wk_light, wk_heavy = evaluate_workload_penalty(engine)
    
    print("\n--- COURSECOMPASS EVALUATION REPORT ---")
    print(f"1. Semantic Coherence (Step-wise cosine similarity): {avg_coh:.4f} (Target > 0.40)")
    print(f"2. Topological Monotonicity (Difficulty scaling accuracy): {topo_acc:.2%} (Target = 100%)")
    print(f"3. Workload Precision: Avg Light pacing={wk_light:.1f}h  vs  Avg Heavy pacing={wk_heavy:.1f}h")
    print("---------------------------------------\n")
    
    # ==========================
    # PLOTTING AND VISUALIZATION
    # ==========================
    out_dir = ROOT / "graphs"
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="darkgrid")
    
    # Plot 1: Semantic Coherence Distribution
    plt.figure(figsize=(8, 5))
    sns.kdeplot(coherences, fill=True, color="#6366f1", alpha=0.6)
    plt.axvline(avg_coh, color="red", linestyle="--", label=f"Mean: {avg_coh:.3f}")
    plt.title("Semantic Coherence Distribution (Cosine Similarity)", fontsize=14, fontweight="bold")
    plt.xlabel("Step-wise Cosine Similarity")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "semantic_coherence_distributions.png", dpi=200)
    plt.close()
    
    # Plot 2: Topological Hierarchy Scaling
    plt.figure(figsize=(10, 6))
    for goal, curve in topo_curves[:6]:  # Ploting first 6 sequences uniquely
        sns.lineplot(x=range(1, len(curve)+1), y=curve, marker="o", label=goal, linewidth=2.5)
        
    plt.yticks([1, 2, 3], ["Beginner", "Intermediate", "Advanced"])
    plt.xticks(range(1, 6))
    plt.title("Curriculum Roadmaps: Monotonic Topological Difficulty Scaling", fontsize=14, fontweight="bold")
    plt.xlabel("Roadmap Stage")
    plt.ylabel("Difficulty Stratification")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_progression_curve.png", dpi=200)
    plt.close()
    
    # Plot 3: Penalty Deviations Bar Chart
    plt.figure(figsize=(6, 5))
    bar_data = pd.DataFrame({"Preference": ["Light Pace", "Heavy Pace"], "Average Stage Hours": [wk_light, wk_heavy]})
    sns.barplot(data=bar_data, x="Preference", y="Average Stage Hours", palette=["#4ade80", "#f87171"])
    plt.title("Effect of Workload Penalties on Reranker Distances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "workload_penalty_scatter.png", dpi=200)
    plt.close()

    print(f"Graphs successfully exported to {out_dir}/")
    print("Execution complete.")

if __name__ == "__main__":
    run_evaluation_suite()
