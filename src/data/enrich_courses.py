"""
enrich_courses.py
=================
Step 1 of the data pipeline.

Reads  : data/raw/coursera_courses.csv
Writes : data/processed/enriched_courses.csv

All enrichment is deterministic and rule-based — no external APIs.
"""

from __future__ import annotations

import re
import math
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "coursera_courses.csv"
OUT_PATH = ROOT / "data" / "processed" / "enriched_courses.csv"

# ──────────────────────────────────────────────────────────────────────────────
# Domain taxonomy & keyword dictionary
# ──────────────────────────────────────────────────────────────────────────────
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    # ── Tech / Engineering ────────────────────────────────────────────────────
    "Machine Learning": [
        "machine learning", "ml", "supervised learning", "unsupervised learning",
        "regression", "classification", "random forest", "gradient boosting",
        "xgboost", "lightgbm", "feature engineering", "model training",
    ],
    "Deep Learning": [
        "deep learning", "neural network", "cnn", "rnn", "lstm", "transformer",
        "bert", "gpt", "attention mechanism", "backpropagation", "pytorch", "tensorflow",
        "keras", "autoencoder", "generative adversarial", "gan",
    ],
    "Artificial Intelligence": [
        "artificial intelligence", "ai", "intelligent systems", "knowledge representation",
        "search algorithm", "planning", "game theory", "reinforcement learning",
        "multi-agent", "natural language", "nlp", "computer vision",
    ],
    "Data Science": [
        "data science", "analytics", "pandas", "numpy", "data wrangling", "eda",
        "exploratory data", "data pipeline", "feature selection", "data cleaning",
        "jupyter", "data analyst", "data engineer",
    ],
    "Statistics": [
        "statistics", "statistical", "hypothesis testing", "probability",
        "bayesian", "inference", "sampling", "distribution", "t-test", "anova",
        "regression analysis", "correlation",
    ],
    "Mathematics": [
        "mathematics", "math", "algebra", "calculus", "linear algebra",
        "differential equations", "discrete math", "number theory",
        "combinatorics", "optimization", "mathematical",
    ],
    "Data Analysis": [
        "data analysis", "excel", "spreadsheet", "pivot table", "tableau",
        "power bi", "looker", "business intelligence", "kpi", "reporting",
        "dashboard", "metrics",
    ],
    "Software Development": [
        "software development", "software engineering", "object-oriented",
        "oop", "design patterns", "clean code", "refactoring", "unit testing",
        "tdd", "api design", "microservices", "system design",
    ],
    "Web Development": [
        "html", "css", "javascript", "react", "angular", "vue", "svelte",
        "frontend", "backend", "full stack", "node.js", "nodejs", "express",
        "django", "flask", "fastapi", "rest api", "graphql", "typescript",
        "next.js", "web development", "web design", "responsive",
    ],
    "Mobile Development": [
        "android", "ios", "swift", "kotlin", "flutter", "react native",
        "mobile app", "mobile development", "xcode",
    ],
    "Cloud Computing": [
        "aws", "amazon web services", "azure", "google cloud", "gcp",
        "cloud computing", "cloud", "kubernetes", "docker", "serverless",
        "lambda", "s3", "ec2", "cloud architecture", "devops", "ci/cd",
        "terraform", "infrastructure as code",
    ],
    "Cybersecurity": [
        "cybersecurity", "security", "ethical hacking", "penetration testing",
        "network security", "cryptography", "vulnerability", "malware",
        "firewalls", "siem", "incident response", "soc", "comptia",
        "certified ethical hacker", "ceh", "cissp",
    ],
    "DevOps": [
        "devops", "continuous integration", "continuous deployment", "ci/cd",
        "docker", "kubernetes", "ansible", "jenkins", "git", "version control",
        "monitoring", "observability", "site reliability",
    ],
    "Databases": [
        "sql", "nosql", "mysql", "postgresql", "mongodb", "redis", "database",
        "relational database", "schema design", "data modeling", "query",
        "indexing", "cassandra", "dynamodb",
    ],
    "Networking": [
        "networking", "tcp/ip", "dns", "http", "network protocol",
        "routing", "switching", "vpn", "cisco", "ccna", "subnetting",
        "wireless", "lan", "wan",
    ],
    "Programming Languages": [
        "python", "java", "c++", "c#", "go", "rust", "scala", "ruby",
        "perl", "r programming", "matlab", "shell scripting", "bash",
        "programming fundamentals",
    ],
    "Computer Science": [
        "computer science", "algorithms", "data structures", "computational",
        "operating systems", "computer architecture", "compiler", "automata",
        "theory of computation", "concurrency", "parallel computing",
    ],
    "Blockchain": [
        "blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart contract",
        "nft", "defi", "web3", "solidity", "distributed ledger",
    ],
    "UI/UX Design": [
        "ux", "user experience", "ui design", "user interface", "usability",
        "wireframe", "prototype", "figma", "sketch", "adobe xd",
        "interaction design", "design thinking", "information architecture",
    ],
    "Robotics": [
        "robotics", "robot", "ros", "autonomous systems", "mechatronics",
        "servo", "actuator", "computer vision for robots",
    ],
    "Game Development": [
        "game development", "unity", "unreal engine", "game design",
        "game programming", "video game", "3d game",
    ],
    # ── Business / Management ─────────────────────────────────────────────────
    "Business": [
        "business", "entrepreneurship", "startup", "business strategy",
        "innovation", "corporate", "management", "operations", "business model",
    ],
    "Marketing": [
        "marketing", "branding", "seo", "sem", "social media", "content marketing",
        "email marketing", "digital marketing", "campaign", "advertising",
        "market research", "consumer behavior",
    ],
    "Finance": [
        "finance", "financial", "investing", "investment", "stock market",
        "valuation", "portfolio", "trading", "asset management", "equity",
        "derivatives", "options", "bonds", "financial modeling",
    ],
    "Accounting": [
        "accounting", "bookkeeping", "financial accounting", "managerial accounting",
        "auditing", "gaap", "balance sheet", "income statement", "cpa",
    ],
    "Economics": [
        "economics", "microeconomics", "macroeconomics", "econometrics",
        "game theory", "behavioral economics", "supply and demand",
    ],
    "Project Management": [
        "project management", "agile", "scrum", "kanban", "pmp", "prince2",
        "sprint", "stakeholder management", "risk management", "milestone",
        "waterfall", "lean",
    ],
    "Human Resources": [
        "human resources", "hr", "recruitment", "talent management",
        "performance management", "employee engagement", "compensation",
        "organizational behavior",
    ],
    "Sales": [
        "sales", "crm", "salesforce", "lead generation", "b2b", "b2c",
        "negotiation", "closing deals", "sales funnel",
    ],
    "Leadership": [
        "leadership", "management skills", "team management", "communication",
        "decision making", "coaching", "mentoring", "executive",
    ],
    "Supply Chain": [
        "supply chain", "logistics", "procurement", "inventory management",
        "operations management", "warehousing", "distribution",
    ],
    "Business Analytics": [
        "business analytics", "tableau", "power bi", "dashboard",
        "business intelligence", "data-driven decision", "kpi", "reporting",
    ],
    # ── Science ───────────────────────────────────────────────────────────────
    "Physics": [
        "physics", "quantum mechanics", "thermodynamics", "mechanics",
        "electromagnetism", "optics", "astrophysics", "nuclear physics",
    ],
    "Chemistry": [
        "chemistry", "organic chemistry", "biochemistry", "chemical",
        "laboratory", "spectroscopy", "reaction",
    ],
    "Biology": [
        "biology", "genetics", "molecular biology", "cell biology",
        "evolution", "ecology", "microbiology",
    ],
    "Biotechnology": [
        "biotechnology", "bioinformatics", "genomics", "proteomics",
        "crispr", "biomedical", "pharmaceutical",
    ],
    "Environmental Science": [
        "environmental", "climate change", "sustainability", "ecology",
        "pollution", "green energy", "renewable energy",
    ],
    # ── Health / Medicine ─────────────────────────────────────────────────────
    "Health & Wellness": [
        "health", "wellness", "nutrition", "fitness", "exercise",
        "lifestyle", "weight loss", "mental wellness",
    ],
    "Public Health": [
        "public health", "epidemiology", "global health", "health policy",
        "disease prevention", "community health",
    ],
    "Psychology": [
        "psychology", "cognition", "behavior", "emotional intelligence",
        "mental health", "cognitive psychology", "behavioral analysis",
        "personality", "social psychology",
    ],
    "Nursing Foundations": [
        "nursing", "clinical", "patient care", "anatomy", "physiology",
        "pharmacology", "healthcare",
    ],
    # ── Humanities / Social Sciences ──────────────────────────────────────────
    "Sociology": [
        "sociology", "social", "society", "culture", "inequality",
        "social movements", "demographics",
    ],
    "Philosophy": [
        "philosophy", "ethics", "logic", "critical thinking", "epistemology",
        "metaphysics", "moral philosophy",
    ],
    "History": [
        "history", "historical", "civilization", "world war", "ancient",
        "medieval", "modern history",
    ],
    "Political Science": [
        "political science", "government", "democracy", "policy",
        "international relations", "geopolitics", "political theory",
    ],
    # ── Creative / Media ──────────────────────────────────────────────────────
    "Graphic Design": [
        "graphic design", "photoshop", "illustrator", "figma", "canva",
        "visual design", "typography", "color theory", "logo design",
        "adobe", "branding design",
    ],
    "Photography": [
        "photography", "photo editing", "lightroom", "camera", "portrait",
        "landscape photography",
    ],
    "Video & Film": [
        "video editing", "premiere pro", "after effects", "final cut",
        "filmmaking", "cinematography", "storytelling", "screenwriting",
    ],
    "Music": [
        "music", "piano", "guitar", "music theory", "songwriting",
        "music production", "ableton", "fl studio", "mixing", "mastering",
    ],
    "Creative Writing": [
        "creative writing", "fiction", "nonfiction", "storytelling",
        "novel writing", "blog writing", "copywriting",
    ],
    "Animation": [
        "animation", "3d animation", "blender", "maya", "motion graphics",
        "character design",
    ],
    # ── Education / Career ────────────────────────────────────────────────────
    "Teaching & Education": [
        "teaching", "education", "curriculum", "pedagogy", "e-learning",
        "instructional design", "classroom management",
    ],
    "Language Learning": [
        "english", "spanish", "french", "german", "mandarin", "japanese",
        "grammar", "vocabulary", "language learning", "ielts", "toefl",
    ],
    "Communication Skills": [
        "communication", "public speaking", "presentation skills",
        "storytelling", "persuasion", "professional writing",
    ],
    "Career Development": [
        "career", "resume", "interview", "job search", "linkedin",
        "personal branding", "networking", "professional development",
    ],
    # ── Legal / Policy / Misc ─────────────────────────────────────────────────
    "Law": [
        "law", "legal", "contract", "intellectual property", "compliance",
        "corporate law", "criminal law",
    ],
    "Sustainability": [
        "sustainability", "csr", "esg", "green", "circular economy",
        "carbon footprint",
    ],
    "Real Estate": [
        "real estate", "property", "mortgage", "investment property",
        "real estate finance",
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# Skills dictionary (domain → list of skills)
# ──────────────────────────────────────────────────────────────────────────────
DOMAIN_SKILLS: dict[str, list[str]] = {
    "Machine Learning": ["Scikit-learn", "Feature Engineering", "Model Evaluation", "Cross-Validation", "Hyperparameter Tuning", "Gradient Boosting", "Ensemble Methods"],
    "Deep Learning": ["PyTorch", "TensorFlow", "CNN", "RNN", "Transfer Learning", "Model Optimization", "GPU Training"],
    "Artificial Intelligence": ["Search Algorithms", "Knowledge Representation", "Reinforcement Learning", "NLP", "Computer Vision", "Planning"],
    "Data Science": ["Pandas", "NumPy", "EDA", "Data Visualization", "Data Cleaning", "Statistical Analysis", "Jupyter Notebooks"],
    "Statistics": ["Hypothesis Testing", "Probability Theory", "Regression Analysis", "ANOVA", "Bayesian Inference", "Sampling Methods"],
    "Mathematics": ["Linear Algebra", "Calculus", "Discrete Math", "Optimization", "Mathematical Proof", "Number Theory"],
    "Data Analysis": ["Excel", "Tableau", "Power BI", "SQL", "Data Storytelling", "Dashboard Design", "KPI Tracking"],
    "Software Development": ["OOP", "Design Patterns", "Unit Testing", "Code Review", "API Design", "Documentation"],
    "Web Development": ["HTML/CSS", "JavaScript", "React", "REST APIs", "Node.js", "Responsive Design", "TypeScript"],
    "Mobile Development": ["Swift", "Kotlin", "Flutter", "Mobile UI", "App Store Deployment", "API Integration"],
    "Cloud Computing": ["AWS", "Azure", "Docker", "Kubernetes", "IaC", "Serverless", "Cloud Architecture"],
    "Cybersecurity": ["Penetration Testing", "Network Security", "Cryptography", "Threat Analysis", "Incident Response", "SIEM"],
    "DevOps": ["CI/CD", "Docker", "Kubernetes", "Ansible", "Jenkins", "Monitoring", "Git"],
    "Databases": ["SQL", "NoSQL", "Schema Design", "Query Optimization", "Indexing", "MongoDB", "PostgreSQL"],
    "Networking": ["TCP/IP", "DNS", "Routing", "Switching", "VPN", "Network Troubleshooting", "CCNA"],
    "Programming Languages": ["Python", "Java", "C++", "Algorithms", "Data Structures", "Debugging"],
    "Computer Science": ["Algorithms", "Data Structures", "Operating Systems", "Computational Theory"],
    "Blockchain": ["Smart Contracts", "Solidity", "Web3", "DeFi", "Cryptography", "Distributed Systems"],
    "UI/UX Design": ["Figma", "User Research", "Wireframing", "Prototyping", "Usability Testing", "Information Architecture"],
    "Robotics": ["ROS", "Mechatronics", "Control Systems", "Sensors", "Computer Vision"],
    "Game Development": ["Unity", "C# Scripting", "Game Design", "3D Modeling", "Physics Engines"],
    "Business": ["Strategic Planning", "Business Models", "Entrepreneurship", "Market Analysis", "Operations"],
    "Marketing": ["SEO", "SEM", "Content Strategy", "Campaign Analytics", "Branding", "Social Media"],
    "Finance": ["Financial Modeling", "Valuation", "Portfolio Management", "Risk Analysis", "Derivatives"],
    "Accounting": ["Financial Statements", "GAAP", "Bookkeeping", "Auditing", "Managerial Accounting"],
    "Economics": ["Microeconomics", "Macroeconomics", "Econometrics", "Market Dynamics"],
    "Project Management": ["Agile", "Scrum", "Sprint Planning", "Stakeholder Management", "Risk Management"],
    "Human Resources": ["Recruitment", "Talent Management", "Performance Reviews", "HR Analytics"],
    "Sales": ["CRM", "Lead Generation", "Negotiation", "Sales Strategy", "Pipeline Management"],
    "Leadership": ["Team Management", "Decision Making", "Conflict Resolution", "Coaching", "Executive Communication"],
    "Supply Chain": ["Procurement", "Logistics", "Inventory Management", "Supply Chain Analytics"],
    "Business Analytics": ["Tableau", "Power BI", "SQL Analytics", "KPI Design", "Dashboarding"],
    "Physics": ["Classical Mechanics", "Quantum Physics", "Electromagnetism", "Thermodynamics"],
    "Chemistry": ["Organic Chemistry", "Reaction Mechanisms", "Lab Techniques", "Spectroscopy"],
    "Biology": ["Cell Biology", "Genetics", "Molecular Biology", "Ecology"],
    "Biotechnology": ["Bioinformatics", "Genomics", "CRISPR", "Protein Analysis"],
    "Environmental Science": ["Climate Modeling", "Sustainability Analysis", "Ecology", "GIS"],
    "Health & Wellness": ["Nutrition", "Fitness Planning", "Mindfulness", "Lifestyle Design"],
    "Public Health": ["Epidemiology", "Health Policy", "Disease Prevention", "Global Health"],
    "Psychology": ["Cognitive Psychology", "Behavioral Analysis", "Emotional Intelligence", "Research Methods"],
    "Nursing Foundations": ["Anatomy", "Physiology", "Patient Care", "Clinical Skills"],
    "Sociology": ["Research Methods", "Social Theory", "Cultural Analysis", "Inequality Studies"],
    "Philosophy": ["Logic", "Ethics", "Critical Thinking", "Philosophical Argumentation"],
    "History": ["Historical Analysis", "Primary Sources", "World History", "Research Writing"],
    "Political Science": ["Comparative Politics", "Policy Analysis", "International Relations"],
    "Graphic Design": ["Photoshop", "Illustrator", "Typography", "Color Theory", "Logo Design"],
    "Photography": ["Camera Techniques", "Lighting", "Photo Editing", "Lightroom"],
    "Video & Film": ["Video Editing", "Premiere Pro", "Cinematography", "Color Grading"],
    "Music": ["Music Theory", "Ear Training", "Instrument Technique", "Music Production", "Mixing"],
    "Creative Writing": ["Storytelling", "Character Development", "Editing", "Genre Writing"],
    "Animation": ["3D Modeling", "Rigging", "Animation Principles", "Motion Graphics"],
    "Teaching & Education": ["Curriculum Design", "Instructional Strategies", "Assessment", "E-Learning Tools"],
    "Language Learning": ["Grammar", "Vocabulary", "Listening Comprehension", "Speaking Practice"],
    "Communication Skills": ["Public Speaking", "Presentation Design", "Active Listening", "Persuasion"],
    "Career Development": ["Resume Writing", "Interview Skills", "Personal Branding", "LinkedIn Optimization"],
    "Law": ["Contract Law", "Legal Research", "Compliance", "Intellectual Property"],
    "Sustainability": ["ESG Frameworks", "Carbon Accounting", "Circular Economy", "Impact Assessment"],
    "Real Estate": ["Property Valuation", "Real Estate Finance", "Market Analysis", "Investment Analysis"],
    "General Studies": ["Critical Thinking", "Research Skills", "Communication", "Problem Solving"],
}

# High-demand domains for popularity proxy
HIGH_DEMAND_DOMAINS = {
    "Machine Learning", "Data Science", "Artificial Intelligence", "Deep Learning",
    "Web Development", "Cloud Computing", "Cybersecurity", "Business", "Finance",
    "Marketing", "Project Management", "Programming Languages", "Software Development",
}

# ──────────────────────────────────────────────────────────────────────────────
# Difficulty keyword sets
# ──────────────────────────────────────────────────────────────────────────────
BEGINNER_KW = {
    "beginner", "introduction", "intro", "basics", "fundamentals",
    "foundations", "getting started", "for beginners", "essential",
    "101", "starter", "crash course", "absolute beginner", "zero to hero",
    "what is", "overview", "primer",
}
INTERMEDIATE_KW = {
    "intermediate", "applied", "practical", "professional", "implementation",
    "techniques", "case study", "hands-on", "guided project", "in practice",
    "with python", "with javascript",
}
ADVANCED_KW = {
    "advanced", "expert", "masterclass", "specialization", "professional certificate",
    "capstone", "system design", "optimization", "research", "architect",
    "deep dive", "mastery", "production-grade",
}

# ──────────────────────────────────────────────────────────────────────────────
# Workload keywords
# ──────────────────────────────────────────────────────────────────────────────
LIGHT_KW = {
    "intro", "overview", "basics", "crash course", "quick start", "quick guide",
    "fundamentals", "starter", "short", "brief", "one hour",
}
HEAVY_KW = {
    "specialization", "professional certificate", "capstone", "bootcamp",
    "masterclass", "nanodegree", "full course", "complete course",
    "ultimate", "comprehensive",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str | float | None) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^\w\s\-,./&]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    """Lower-case word token set, removing very short tokens."""
    tokens = set(re.findall(r"[a-z][a-z0-9\-]+", text.lower()))
    return tokens


def infer_domain(title: str, slug: str, description: str, raw_domain: str) -> str:
    """
    Rule-based domain inference. Priority:
    1. Exact keyword phrase match in combined text
    2. Multi-keyword confidence: domain with most keyword hits
    3. Raw domain normalization (weak signal)
    4. Fallback: General Studies
    """
    combined = f"{title} {slug} {description} {raw_domain}".lower()

    best_domain = "General Studies"
    best_score = 0

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in combined:
                # Longer, more specific keywords get higher weight
                score += 1 + 0.3 * (len(kw.split()) - 1)
        if score > best_score:
            best_score = score
            best_domain = domain

    # Raw domain nudge: if raw_domain is set and we have a low score, try to
    # match raw_domain text to a domain name
    if best_score < 1.5 and raw_domain:
        rd_lower = raw_domain.lower()
        for domain in DOMAIN_KEYWORDS:
            if domain.lower() in rd_lower or rd_lower in domain.lower():
                best_domain = domain
                break

    return best_domain


def infer_subdomain(title: str, domain: str) -> str:
    """Return a simple subdomain string based on title tokens."""
    title_lower = title.lower()
    # very lightweight: just return domain for now; could be extended
    for kw in ["specialization", "professional certificate", "capstone"]:
        if kw in title_lower:
            return f"{domain} — Specialization"
    for kw in ["project", "hands-on", "lab"]:
        if kw in title_lower:
            return f"{domain} — Projects"
    for kw in ["introduction", "foundations", "basics", "fundamentals"]:
        if kw in title_lower:
            return f"{domain} — Foundations"
    return domain


def infer_difficulty(title: str, slug: str, description: str) -> tuple[str, int]:
    combined = f"{title} {slug} {description}".lower()
    tokens = _tokenize(combined)

    adv_score = sum(1 for kw in ADVANCED_KW if kw in combined)
    int_score = sum(1 for kw in INTERMEDIATE_KW if kw in combined)
    beg_score = sum(1 for kw in BEGINNER_KW if kw in combined)

    if adv_score >= int_score and adv_score >= beg_score and adv_score > 0:
        return "Advanced", 3
    if beg_score >= int_score and beg_score > 0:
        return "Beginner", 1
    if int_score > 0:
        return "Intermediate", 2
    return "Intermediate", 2  # default


def parse_workload_raw(raw: str) -> float | None:
    """Try to extract hours from raw workload text."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.lower()
    # "X hours a week" × "Y weeks"  → total
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:[a-z\s]*)\s*(?:per|a|each)?\s*week", raw)
    m2 = re.search(r"(\d+)\s*weeks?", raw)
    if m and m2:
        return float(m.group(1)) * int(m2.group(1))
    # plain "X hours"
    m3 = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", raw)
    if m3:
        return float(m3.group(1))
    return None


def infer_workload(title: str, slug: str, description: str, raw_workload: str) -> tuple[float, str, int]:
    """
    Returns (estimated_duration_hours, workload_bucket, workload_score).
    """
    hrs = parse_workload_raw(raw_workload)
    if hrs is not None:
        if hrs <= 8:
            return hrs, "Light", 1
        elif hrs <= 25:
            return hrs, "Medium", 2
        else:
            return min(hrs, 80), "Heavy", 3

    combined = f"{title} {slug} {description}".lower()

    if any(kw in combined for kw in HEAVY_KW):
        return 40.0, "Heavy", 3
    if any(kw in combined for kw in LIGHT_KW):
        return 5.0, "Light", 1
    # check for project/applied signals → medium
    if any(kw in combined for kw in INTERMEDIATE_KW):
        return 15.0, "Medium", 2
    return 12.0, "Medium", 2  # default


def generate_skills(domain: str, title: str, n_min: int = 3, n_max: int = 8) -> list[str]:
    skills_pool = DOMAIN_SKILLS.get(domain, DOMAIN_SKILLS["General Studies"])
    # Deterministic but varied per title: hash title to pick subset
    h = int(hashlib.md5(title.encode()).hexdigest(), 16)
    n = n_min + (h % (n_max - n_min + 1))
    indices = [(h >> i) % len(skills_pool) for i in range(0, n * 4, 4)]
    seen = []
    for idx in indices:
        s = skills_pool[idx % len(skills_pool)]
        if s not in seen:
            seen.append(s)
        if len(seen) >= n:
            break
    # fill if needed
    for s in skills_pool:
        if len(seen) >= n:
            break
        if s not in seen:
            seen.append(s)
    return seen[:n]


def compute_popularity_proxy(row: dict) -> float:
    score = 0.0
    if row["difficulty_level"] == "Beginner":
        score += 0.30
    elif row["difficulty_level"] == "Intermediate":
        score += 0.20
    if row["inferred_domain"] in HIGH_DEMAND_DOMAINS:
        score += 0.30
    if row["workload_bucket"] == "Light":
        score += 0.15
    elif row["workload_bucket"] == "Medium":
        score += 0.10
    title_len = len(str(row.get("cleaned_title", "")))
    if 20 <= title_len <= 70:
        score += 0.10
    else:
        score += 0.05
    # Add deterministic noise based on uid to vary within same bucket
    uid_hash = int(hashlib.md5(str(row["uid"]).encode()).hexdigest()[:6], 16)
    score += (uid_hash % 100) / 1000.0  # 0–0.1
    return min(1.0, round(score, 4))


def compute_quality_proxy(row: dict) -> float:
    score = 0.0
    if row["has_description"]:
        score += 0.30
    if row["has_domain_raw"]:
        score += 0.10
    if row["inferred_domain"] != "General Studies":
        score += 0.20
    title_len = len(str(row.get("cleaned_title", "")))
    if title_len >= 15:
        score += 0.15
    n_kw = len(str(row.get("content_keywords", "")).split(","))
    score += min(0.15, n_kw * 0.02)
    uid_hash = int(hashlib.md5(str(row["uid"]).encode()).hexdigest()[6:12], 16)
    score += (uid_hash % 100) / 1000.0
    return min(1.0, round(score, 4))


def compute_progression_rank(df: pd.DataFrame) -> pd.Series:
    """
    Within each domain, rank courses by difficulty_score then workload_score.
    Returns integer rank 1…N per domain.
    """
    df = df.copy()
    df["_sort_key"] = df["difficulty_score"] * 10 + df["workload_score"]
    df["progression_rank"] = (
        df.groupby("inferred_domain")["_sort_key"]
        .rank(method="first", ascending=True)
        .astype(int)
    )
    return df["progression_rank"]


# ──────────────────────────────────────────────────────────────────────────────
# Special flags
# ──────────────────────────────────────────────────────────────────────────────
FOUNDATIONAL_KW = {"introduction", "intro", "foundations", "basics", "fundamentals", "101", "getting started"}
PROJECT_KW = {"project", "hands-on", "lab", "workshop", "case study", "guided"}
CERT_KW = {"professional certificate", "certification", "cert prep", "exam", "comptia", "pmp", "aws certified"}


def is_foundational(title: str, slug: str) -> int:
    combined = f"{title} {slug}".lower()
    return int(any(kw in combined for kw in FOUNDATIONAL_KW))


def is_project_based(title: str, slug: str, description: str) -> int:
    combined = f"{title} {slug} {description}".lower()
    return int(any(kw in combined for kw in PROJECT_KW))


def is_cert_prep(title: str, slug: str) -> int:
    combined = f"{title} {slug}".lower()
    return int(any(kw in combined for kw in CERT_KW))


# ──────────────────────────────────────────────────────────────────────────────
# Main enrichment
# ──────────────────────────────────────────────────────────────────────────────

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"Enriching {len(df):,} courses …")

    rows = []
    for _, row in df.iterrows():
        uid_val = row.get("uid")
        if pd.isna(uid_val):
            uid_val = row.get("id")
        uid = str(uid_val) if pd.notna(uid_val) else ""
        title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
        slug = str(row.get("slug", "")) if pd.notna(row.get("slug")) else ""
        description = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
        url = str(row.get("url", "")) if pd.notna(row.get("url")) else ""
        raw_domain = str(row.get("domain", "")) if pd.notna(row.get("domain")) else ""
        raw_workload = str(row.get("workload", "")) if pd.notna(row.get("workload")) else ""

        cleaned_title = _clean_text(title)
        cleaned_slug = _clean_text(slug).replace("-", " ")

        combined_text = " ".join(filter(None, [cleaned_title, cleaned_slug, _clean_text(description)]))

        domain = infer_domain(cleaned_title, cleaned_slug, _clean_text(description), _clean_text(raw_domain))
        subdomain = infer_subdomain(cleaned_title, domain)

        diff_level, diff_score = infer_difficulty(cleaned_title, cleaned_slug, description)
        dur_hrs, wl_bucket, wl_score = infer_workload(cleaned_title, cleaned_slug, description, raw_workload)

        skills = generate_skills(domain, cleaned_title)
        kw_tokens = list(_tokenize(combined_text))[:20]

        has_desc = int(bool(description.strip()))
        has_domain = int(bool(raw_domain.strip()))
        has_workload = int(bool(raw_workload.strip()))

        r = {
            "course_id": uid,
            "uid": uid,
            "title": title,
            "slug": slug,
            "url": url,
            "raw_description": description,
            "raw_domain": raw_domain,
            "raw_workload": raw_workload,
            "cleaned_title": cleaned_title,
            "cleaned_slug": cleaned_slug,
            "combined_text": combined_text,
            "inferred_domain": domain,
            "subdomain": subdomain,
            "difficulty_level": diff_level,
            "difficulty_score": diff_score,
            "estimated_duration_hours": round(dur_hrs, 1),
            "workload_bucket": wl_bucket,
            "workload_score": wl_score,
            "skills_tags": ", ".join(skills),
            "content_keywords": ", ".join(kw_tokens),
            "has_description": has_desc,
            "has_domain_raw": has_domain,
            "has_workload_raw": has_workload,
            "popularity_proxy": 0.0,  # filled below
            "quality_proxy": 0.0,
            "progression_rank": 0,
            "is_foundational": is_foundational(cleaned_title, cleaned_slug),
            "is_project_based": is_project_based(cleaned_title, cleaned_slug, description),
            "is_cert_prep": is_cert_prep(cleaned_title, slug),
        }
        r["popularity_proxy"] = compute_popularity_proxy(r)
        r["quality_proxy"] = compute_quality_proxy(r)
        rows.append(r)

    enriched = pd.DataFrame(rows)
    enriched["progression_rank"] = compute_progression_rank(enriched)

    log.info(f"  Domain distribution:\n{enriched['inferred_domain'].value_counts().head(15).to_string()}")
    log.info(f"  Difficulty distribution:\n{enriched['difficulty_level'].value_counts().to_string()}")
    log.info(f"  Workload distribution:\n{enriched['workload_bucket'].value_counts().to_string()}")

    return enriched


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading raw courses from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    log.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    enriched = enrich(df)

    enriched.to_csv(OUT_PATH, index=False)
    log.info(f"Saved enriched courses → {OUT_PATH}  ({len(enriched):,} rows)")


if __name__ == "__main__":
    main()
