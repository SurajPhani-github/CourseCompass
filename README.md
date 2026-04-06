# CourseCompass: Adaptive Learning Ecosystem
aaa
CourseCompass is a next-generation, AI-driven adaptive learning platform that generates dynamic, prerequisite-aware learning roadmaps. It transitions from traditional heuristic-based recommendations to a fully ML-backed infrastructure with semantic retrieval, collaborative filtering, and generative AI.

## 🧠 Core Architecture & Models Used

The platform utilizes a hybrid recommendation and generation pipeline powered by multiple specialized Machine Learning models:

### 1. NLP Skill Extraction & Semantic Mapping
- **Model:** `all-MiniLM-L6-v2` (via SentenceTransformers & KeyBERT)
- **Role:** Analyzes over 15,000 raw course descriptions to extract overarching skills, cluster them hierarchically, and infer topological difficulty edges. 
- **Application:** Builds the dynamic `NetworkX` Knowledge Graph (DAG) and powers the semantic search for the **Role-Based Career Mapper** (e.g., retrieving required nodes for a "Data Scientist" input).

### 2. Generative AI Tutor & Content Engine
- **Model:** `Llama-3.3-70B-versatile` (via Groq Inference API)
- **Role:** Handles complex semantic reasoning, pedagogical explanation, and content generation.
- **Applications:**
  - **AI Tutor (Tab 6):** Context-aware chatbot that attaches the user's active technical roadmap and provides deep dive explanations.
  - **Dynamic Assessments (Tab 4):** Procedurally generates topic-specific MCQs and scenario-based coding questions tuned to user-selected difficulty.
  - **Project Generator (Tab 5):** Recommends tailored capstone project ideas complete with required tech stacks and learning outcomes based on the user's proficiency level.

### 3. Collaborative Filtering (Peer Recommendations)
- **Model:** Alternating Least Squares (ALS) Matrix Factorization
- **Library:** `implicit` (with NumPy SVD fallback)
- **Role:** Processes synthetic learner engagement matrices (course completion rates, scores) to find peer-based learning patterns.
- **Application:** Used in the "Recommend for Learner" engine to weight pathways other students successfully took, solving for the "cold start" heuristic problem.

### 4. Adaptive Re-Planing Engine
- **Mechanism:** Statistical threshold evaluation and historic trend analysis.
- **Role:** Tracks continuous assessment performance (stored in SQLite/SQLAlchemy) to automatically suggest difficulty adjustments. If a learner scores <40%, it triggers a "Reinforce" action; if >85%, it triggers an "Accelerate" action, dynamically re-routing the knowledge graph.

## 🛠️ Tech Stack
- **Frontend / UI:** Streamlit
- **Persistence Layer:** SQLAlchemy ORM (SQLite backend, MySQL ready)
- **Data Manipulation:** Pandas, NumPy
- **Graph Mathematics:** NetworkX

## 🚀 How to Run

1. **Setup Environment:**
   Install the required dependencies via `pip install -r requirements.txt`. Export your `GROQ_API_KEY`.

2. **Generate the Knowledge Graph:**
   To parse the ~15,000 courses into the semantic graph:
   ```bash
   python src/recommender/skill_extractor.py
   ```

3. **Launch the Application:**
   Run the Streamlit dashboard:
   ```bash
   streamlit run src/app/streamlit_app.py
   ```

## 🗺️ UI Tabs Overview
1. **Recommend for Learner:** View ML-ranked and collaborative peer course suggestions.
2. **Explore by Goal:** Generate topological graph roadmaps or Role-based career paths based on target skills.
3. **Course Catalog:** Raw semantic search over the 15k+ ingested datasets.
4. **Assessments:** Test skills with Llama-generated dynamic quizzes.
5. **Projects:** Generate domain-aligned portfolio projects.
6. **AI Tutor:** Interactive pedagogical chat engine contextualized to your generated roadmaps.
