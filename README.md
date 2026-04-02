# 🧭 CourseCompass

CourseCompass is a next-generation **Hybrid Course Recommendation and Curriculum Planning Engine** designed to solve the structural failure of traditional E-learning platforms. 

By unifying **LightGBM Machine Learning Ranking**, **Semantic Sentence-Transformers**, and **Directed Acyclic Knowledge Graphs (DAGs)**, the engine dynamically personalizes educational pathing based on a learner's progressing proficiency, workload limits, and domain affinity.

---

## 🚀 Key Features

### 1. Hybrid Learner Mode (Discovery)
Learner Mode recommends the next logical steps for a student by combining two ranking algorithms:
* **Heuristic Scoring (35% Weight)** 
  Constructs a continuous fitness score evaluating Content Similarity (TF-IDF), Domain Affinity, Transition Flow Mapping, Difficulty Deltas, and explicit Workload pacing.
* **ML Ranker Pipeline (65% Weight)** 
  An offline binary classifier (built on `lightgbm`) trained on exactly 28,000 synthetic interaction events (completion outcomes, dropout risks, sequential engagement). It evaluates a 25-feature vector per-course at inference time to yield an aggressive `🤖 Predicted Success Probability`.

Instead of static tags, learners have **Dynamic Preferences** — as a User begins passing harder courses (e.g. Intermediate -> Advanced), `ranker_features.py` tracks the progression via exponentially-decayed recency weighting and pushes harder, more advanced content without the user having to touch a setting.

### 2. Generative Curriculum Branching (API Mode)
For users with an explicitly targeted competency metric (e.g. "I want to master MLOps" or "LLMs"), the engine pivots away from static heuristics and queries the **Google Gemini API** (`gemini-1.5-pro-latest`).

The Recommender prompts the AI to hallucinate exactly 3 diverging educational paths based on the student's profile (e.g., *Path A: Deep Research Theory*, *Path B: Applied Engineering*, *Path C: Fast-track Developer*). 
Once the LLM returns the abstract framework (complete with custom **Projects** and **Milestones**), the backend intercepts the text and maps it onto the hard 15,000+ course subset using Semantic Dense Encodings (`all-MiniLM-L6-v2`) via Vector Cosine Similarity. 

If the learner has already mastered the concepts implicitly via prior interactions, it automatically triggers a "Skill Skip" across the AI roadmap.

### 3. Localized Roadmap Chat Assistant
Integrated into the newly branched dashboards is a persistent Chatbot explicitly initialized with the context of the user's generated AI paths. Students can interact directly with the roadmap (via `st.chat_input`) to ask for study strategy refinements or deeper explanations of *why* the AI assigned a specific course sequence for their targeted branch.

### 4. Dynamic UI Session Overrides
Users can interrupt the engine's inferred behavioral trajectory. The Streamlit dashboard features explicit **Session Preference Override sliders**, permitting an ad-hoc restructuring of ML outputs (e.g., forcing the model to generate light-workload Beginner material for this single session, while preserving their Advanced historical trajectory).

---

## 🛠️ Architecture Components

| Module | Description | Core Dependencies |
| :--- | :--- | :--- |
| `learner_ranker.py` | Standalone Singleton wrapping the Joblib ML Classifier | `lightgbm`, `joblib` |
| `learner_recommender.py` | Orchestrates the top-N merge between the heuristic array and ML inferences | `pandas`, `numpy` |
| `llm_branching.py` | Bridges Google Gemini API to construct localized roadmap theories and chat completions | `google-generativeai` |
| `course_skill_mapper.py` | Caches mapping vectors intersecting the 15k courses and LLM abstract nodes | `sentence-transformers` |
| `roadmap_engine.py` | Computes cumulative time-to-goal metrics, readiness probabilities, and node routing | `scikit-learn` |
| `streamlit_app.py` | Polished UI Dashboard natively supporting `st.tabs` branch execution and chat UI | `streamlit` |

## 🚦 Quick Deployment 

Ensure dependencies are resolved via `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### 1. (Optional) Force Re-Embedding 
If new domains or nodes are added to the Graph, rebuild the `.pkl` map locally:
```python
from src.recommender.course_skill_mapper import CourseSkillMapper
# Pass force_rebuild=True to overwrite the semantic node embeddings!
mapper.build_or_load_mapping(force_rebuild=True)
```

#### 2. Re-Train the Classification Pipeline
```bash
python train_ranker.py
```
This extracts interactions, constructs dynamic preferences, creates labeling functions, balances classes, and exports `models/lgbm_ranker.pkl`.

#### 3. Launch App Dashboard
```bash
streamlit run src/app/streamlit_app.py
```
