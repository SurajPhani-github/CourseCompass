"""
skill_graph.py
===============
Defines the backend Knowledge Graph (DAG) for prerequisite mappings.

Covers 5 domains:
  - AI / ML / GenAI
  - Web Development
  - Data Engineering
  - Cybersecurity
  - Business Analytics

Each domain has Foundation → Core → Intermediate → Advanced → Specialization nodes.
"""

from __future__ import annotations

import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SkillNode:
    node_id:     str
    title:       str
    description: str
    level:       str    # Foundation | Core | Intermediate | Advanced | Specialization
    domain:      str
    keywords:    list[str] = field(default_factory=list)


# ── Node definitions ──────────────────────────────────────────────────────────

_NODES: list[SkillNode] = [

    # ═══════════════════════════════════════
    # DOMAIN: AI / ML / GenAI
    # ═══════════════════════════════════════
    SkillNode("python",
              "Python Programming",
              "Syntax, data structures, OOP, modules, and functional programming in Python.",
              "Foundation", "AI/ML",
              ["python", "programming", "basics", "functions", "loops", "classes"]),
    SkillNode("math_stats",
              "Probability & Statistics",
              "Distributions, hypothesis testing, A/B testing, linear algebra basics, calculus for ML.",
              "Foundation", "AI/ML",
              ["statistics", "probability", "linear algebra", "calculus", "math"]),
    SkillNode("data_analysis",
              "Data Analysis & Manipulation",
              "Pandas, NumPy, SQL, EDA, data cleaning, visualisation with Matplotlib/Seaborn.",
              "Foundation", "AI/ML",
              ["pandas", "numpy", "eda", "data analysis", "visualization", "matplotlib"]),
    SkillNode("machine_learning",
              "Machine Learning Fundamentals",
              "Supervised/unsupervised learning, scikit-learn, regression, classification, forests, boosting.",
              "Core", "AI/ML",
              ["machine learning", "scikit-learn", "classification", "regression", "random forest", "xgboost"]),
    SkillNode("deep_learning",
              "Deep Learning & Neural Networks",
              "Perceptrons, backpropagation, PyTorch, TensorFlow, Keras, dense networks.",
              "Intermediate", "AI/ML",
              ["deep learning", "neural network", "pytorch", "tensorflow", "keras", "backpropagation"]),
    SkillNode("nlp",
              "Natural Language Processing",
              "Text embeddings, word2vec, sequence models, transformers introduction, BERT.",
              "Advanced", "AI/ML",
              ["nlp", "text classification", "transformers", "bert", "word2vec", "sentiment"]),
    SkillNode("cv",
              "Computer Vision",
              "Image classification, object detection, CNN architectures (ResNet, YOLO), OpenCV.",
              "Advanced", "AI/ML",
              ["computer vision", "image classification", "object detection", "cnn", "yolo", "opencv"]),
    SkillNode("reinforcement_learning",
              "Reinforcement Learning",
              "Q-learning, policy gradients, agent-based learning, MDP, OpenAI Gym.",
              "Advanced", "AI/ML",
              ["reinforcement learning", "q-learning", "policy gradient", "mdp", "reward"]),
    SkillNode("llm",
              "Large Language Models & GenAI",
              "Transformers, attention, fine-tuning, RAG, prompt engineering, agentic AI, LangChain.",
              "Specialization", "GenAI",
              ["llm", "gpt", "large language model", "genai", "generative ai", "langchain", "rag",
               "prompt engineering", "fine-tuning", "chatgpt"]),
    SkillNode("mlops",
              "MLOps & Production ML",
              "Model deployment, scaling, Kubernetes, Docker, ML pipelines, monitoring, MLflow.",
              "Specialization", "Engineering",
              ["mlops", "deployment", "docker", "kubernetes", "mlflow", "model serving", "pipeline"]),

    # ═══════════════════════════════════════
    # DOMAIN: Web Development
    # ═══════════════════════════════════════
    SkillNode("html_css",
              "HTML & CSS",
              "Semantic HTML5, CSS3, flexbox, grid, responsive design, accessibility.",
              "Foundation", "Web Development",
              ["html", "css", "flexbox", "grid", "responsive", "web design", "frontend basics"]),
    SkillNode("javascript",
              "JavaScript Fundamentals",
              "ES6+, DOM manipulation, async/await, fetch API, event handling.",
              "Foundation", "Web Development",
              ["javascript", "js", "es6", "dom", "async", "fetch", "events"]),
    SkillNode("react",
              "React & Modern Frontend",
              "Component model, hooks, state management (Redux/Zustand), React Router, Next.js.",
              "Core", "Web Development",
              ["react", "nextjs", "hooks", "state management", "redux", "frontend framework"]),
    SkillNode("nodejs_backend",
              "Node.js & Backend Development",
              "Express.js, REST APIs, authentication, middleware, server-side JavaScript.",
              "Intermediate", "Web Development",
              ["node.js", "express", "rest api", "backend", "server", "nodejs"]),
    SkillNode("databases_web",
              "Databases & ORMs",
              "SQL (PostgreSQL/MySQL), NoSQL (MongoDB), ORMs (Prisma/Sequelize), database design.",
              "Intermediate", "Web Development",
              ["database", "sql", "mongodb", "postgresql", "orm", "nosql", "schema"]),
    SkillNode("fullstack",
              "Full-Stack Development",
              "End-to-end web apps, deployment (Vercel/Railway), CI/CD, Docker, TypeScript.",
              "Advanced", "Web Development",
              ["fullstack", "full stack", "deployment", "typescript", "ci/cd", "devops"]),
    SkillNode("web_performance",
              "Web Performance & Security",
              "Core Web Vitals, caching, CDN, HTTPS, OWASP Top 10, auth patterns.",
              "Specialization", "Web Development",
              ["web performance", "caching", "security", "owasp", "web vitals", "cdn"]),

    # ═══════════════════════════════════════
    # DOMAIN: Data Engineering
    # ═══════════════════════════════════════
    SkillNode("sql_de",
              "SQL & Relational Databases",
              "Advanced SQL, query optimisation, indexing, window functions, stored procedures.",
              "Foundation", "Data Engineering",
              ["sql", "database", "query", "relational", "postgres", "mysql", "joins", "indexes"]),
    SkillNode("python_de",
              "Python for Data Engineering",
              "File I/O, data serialisation, requests, CLI scripting, virtual environments.",
              "Foundation", "Data Engineering",
              ["python", "scripting", "automation", "etl", "data processing"]),
    SkillNode("data_warehousing",
              "Data Warehousing & Modelling",
              "OLAP vs OLTP, star schema, Snowflake/BigQuery/Redshift, dbt.",
              "Core", "Data Engineering",
              ["data warehouse", "dbt", "bigquery", "redshift", "snowflake", "star schema", "olap"]),
    SkillNode("batch_pipelines",
              "Batch Pipelines & Workflow Orchestration",
              "Apache Airflow, Luigi, Prefect; building DAG-based ETL/ELT workflows.",
              "Intermediate", "Data Engineering",
              ["airflow", "etl", "elt", "pipeline", "orchestration", "prefect", "dag"]),
    SkillNode("spark",
              "Apache Spark & Big Data",
              "Spark DataFrame API, PySpark, partitioning, Spark SQL, distributed computing.",
              "Advanced", "Data Engineering",
              ["spark", "pyspark", "big data", "distributed", "hadoop", "hive"]),
    SkillNode("streaming",
              "Streaming & Real-Time Data",
              "Apache Kafka, Flink, Kinesis, stream processing patterns, exactly-once semantics.",
              "Specialization", "Data Engineering",
              ["kafka", "streaming", "real-time", "flink", "kinesis", "event streaming"]),

    # ═══════════════════════════════════════
    # DOMAIN: Cybersecurity
    # ═══════════════════════════════════════
    SkillNode("networking",
              "Networking Fundamentals",
              "OSI model, TCP/IP, DNS, HTTP/S, firewalls, VPN, subnetting.",
              "Foundation", "Cybersecurity",
              ["networking", "tcp/ip", "dns", "http", "firewall", "vpn", "osi"]),
    SkillNode("linux_security",
              "Linux & System Administration",
              "Shell scripting, permissions, process management, log analysis, hardening.",
              "Foundation", "Cybersecurity",
              ["linux", "bash", "shell", "permissions", "system admin", "hardening"]),
    SkillNode("cryptography",
              "Cryptography & PKI",
              "Symmetric/asymmetric encryption, hashing, TLS/SSL, certificates, key management.",
              "Core", "Cybersecurity",
              ["cryptography", "encryption", "tls", "ssl", "hashing", "rsa", "aes", "certificates"]),
    SkillNode("ethical_hacking",
              "Ethical Hacking & Penetration Testing",
              "Reconnaissance, scanning (Nmap), exploitation (Metasploit), OWASP Top 10, CTF.",
              "Intermediate", "Cybersecurity",
              ["ethical hacking", "penetration testing", "pentesting", "metasploit",
               "nmap", "owasp", "ctf", "vulnerability"]),
    SkillNode("cloud_security",
              "Cloud Security",
              "IAM, zero trust, cloud-native security (AWS/GCP/Azure), SIEM, compliance.",
              "Advanced", "Cybersecurity",
              ["cloud security", "iam", "zero trust", "aws security", "siem", "compliance", "gdpr"]),
    SkillNode("incident_response",
              "Incident Response & Forensics",
              "Threat hunting, DFIR, log forensics, malware analysis, SOC workflows.",
              "Specialization", "Cybersecurity",
              ["incident response", "forensics", "threat hunting", "dfir", "soc", "malware"]),

    # ═══════════════════════════════════════
    # DOMAIN: Business Analytics
    # ═══════════════════════════════════════
    SkillNode("excel_basics",
              "Excel & Spreadsheet Analytics",
              "Formulas, pivot tables, VLOOKUP, data validation, charts, Power Query.",
              "Foundation", "Business Analytics",
              ["excel", "spreadsheet", "pivot table", "vlookup", "power query", "formulas"]),
    SkillNode("sql_analytics",
              "SQL for Analytics",
              "Aggregations, window functions, CTEs, analytical queries, reporting.",
              "Foundation", "Business Analytics",
              ["sql", "analytics", "aggregation", "window functions", "bi", "reporting"]),
    SkillNode("bi_tools",
              "BI Tools & Data Visualisation",
              "Tableau, Power BI, Looker; dashboard design, KPIs, storytelling with data.",
              "Core", "Business Analytics",
              ["tableau", "power bi", "looker", "dashboard", "kpi", "data visualisation", "bi"]),
    SkillNode("business_stats",
              "Business Statistics & Experimentation",
              "Descriptive stats, A/B testing, hypothesis testing, regression analysis for business.",
              "Intermediate", "Business Analytics",
              ["a/b testing", "hypothesis testing", "business statistics", "experimentation",
               "regression", "analysis"]),
    SkillNode("product_analytics",
              "Product & Marketing Analytics",
              "Funnel analysis, cohort analysis, LTV, churn, attribution modelling.",
              "Advanced", "Business Analytics",
              ["product analytics", "cohort", "funnel", "ltv", "churn", "attribution", "marketing"]),
    SkillNode("advanced_analytics",
              "Advanced Analytics & Forecasting",
              "Time series forecasting, causal inference, predictive modelling for business, Python/R.",
              "Specialization", "Business Analytics",
              ["forecasting", "time series", "causal inference", "predictive", "python analytics"]),
]


# ── Edge definitions (prerequisite → target) ──────────────────────────────────

_EDGES: list[tuple[str, str]] = [
    # AI/ML chain
    ("python",           "data_analysis"),
    ("python",           "machine_learning"),
    ("math_stats",       "machine_learning"),
    ("data_analysis",    "machine_learning"),
    ("machine_learning", "deep_learning"),
    ("machine_learning", "mlops"),
    ("deep_learning",    "mlops"),
    ("deep_learning",    "nlp"),
    ("deep_learning",    "cv"),
    ("deep_learning",    "reinforcement_learning"),
    ("nlp",              "llm"),

    # Web Dev chain
    ("html_css",         "javascript"),
    ("javascript",       "react"),
    ("javascript",       "nodejs_backend"),
    ("react",            "fullstack"),
    ("nodejs_backend",   "fullstack"),
    ("databases_web",    "fullstack"),
    ("fullstack",        "web_performance"),

    # Data Engineering chain
    ("sql_de",           "data_warehousing"),
    ("python_de",        "batch_pipelines"),
    ("data_warehousing", "batch_pipelines"),
    ("batch_pipelines",  "spark"),
    ("spark",            "streaming"),

    # Cybersecurity chain
    ("networking",       "cryptography"),
    ("linux_security",   "cryptography"),
    ("cryptography",     "ethical_hacking"),
    ("ethical_hacking",  "cloud_security"),
    ("cloud_security",   "incident_response"),

    # Business Analytics chain
    ("excel_basics",     "bi_tools"),
    ("sql_analytics",    "bi_tools"),
    ("bi_tools",         "business_stats"),
    ("business_stats",   "product_analytics"),
    ("product_analytics","advanced_analytics"),

    # Cross-domain bridges
    ("python",           "python_de"),      # Python is foundation for DE too
    ("math_stats",       "business_stats"), # Stats underpins BA
    ("data_analysis",    "sql_analytics"),  # DA → SQL analytics natural
    ("machine_learning", "advanced_analytics"),  # ML bridges into advanced BA
]


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_full_knowledge_graph() -> nx.DiGraph:
    """Build the full multi-domain prerequisite DAG."""
    G = nx.DiGraph()
    for n in _NODES:
        G.add_node(n.node_id, data=n)
    G.add_edges_from(_EDGES)
    return G


# ── KnowledgeGraph class ──────────────────────────────────────────────────────

class KnowledgeGraph:
    def __init__(self):
        self.G: nx.DiGraph = build_full_knowledge_graph()

    # ── Node access ──────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> Optional[SkillNode]:
        if self.G.has_node(node_id):
            return self.G.nodes[node_id]["data"]
        return None

    def get_all_nodes(self) -> list[SkillNode]:
        return [d["data"] for _, d in self.G.nodes(data=True)]

    def get_domain_nodes(self, domain: str) -> list[SkillNode]:
        return [n for n in self.get_all_nodes() if n.domain == domain]

    def get_all_domains(self) -> list[str]:
        return sorted({n.domain for n in self.get_all_nodes()})

    # ── Prerequisite traversal ────────────────────────────────────────────────

    def get_prerequisites(self, node_id: str) -> list[str]:
        """Returns all ancestor prerequisite node IDs in topological order."""
        if not self.G.has_node(node_id):
            return []
        ancestors = nx.ancestors(self.G, node_id)
        subgraph  = self.G.subgraph(ancestors)
        return list(nx.topological_sort(subgraph))

    def get_full_path(self, node_id: str) -> list[str]:
        """Returns prerequisites + the node itself, topologically sorted."""
        path = self.get_prerequisites(node_id)
        path.append(node_id)
        return path

    def get_direct_prerequisites(self, node_id: str) -> list[str]:
        """Returns only the immediate parent nodes."""
        return list(self.G.predecessors(node_id))

    def get_dependents(self, node_id: str) -> list[str]:
        """Returns nodes that depend on this one (successors)."""
        return list(self.G.successors(node_id))

    # ── Graph-level helpers ───────────────────────────────────────────────────

    def get_all_edges(self) -> list[tuple[str, str]]:
        return list(self.G.edges())

    def get_prerequisite_subgraph(self, node_id: str) -> dict[str, list[str]]:
        """
        Returns a dict {node_id: [prereq_ids]} for all nodes on the path to
        `node_id` — useful for rendering a prerequisite graph in the UI.
        """
        if not self.G.has_node(node_id):
            return {}
        path_nodes = set(self.get_prerequisites(node_id)) | {node_id}
        subgraph = self.G.subgraph(path_nodes)
        return {n: list(subgraph.predecessors(n)) for n in subgraph.nodes()}

    def search_nodes(self, query: str) -> list[tuple[str, SkillNode]]:
        """
        Fuzzy keyword search over node titles, descriptions, and keywords.
        Returns (node_id, node) pairs sorted by match strength (descending).
        """
        query_lower = query.lower()
        scored: list[tuple[int, str, SkillNode]] = []
        for nid, data in self.G.nodes(data=True):
            node: SkillNode = data["data"]
            score = 0
            if query_lower in node.title.lower():
                score += 10
            for kw in node.keywords:
                if query_lower in kw.lower() or kw.lower() in query_lower:
                    score += 3
            if query_lower in node.description.lower():
                score += 1
            if score > 0:
                scored.append((score, nid, node))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(nid, node) for _, nid, node in scored]
