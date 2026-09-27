"""
config.py
=========
Central configuration for the NLP-Based Intelligent Resume Screening,
Skill Extraction and Context-Aware Job Matching System.

Holds:
    - the technical / soft-skill taxonomy used for skill extraction
    - education & degree keyword lists used for education extraction
    - regex helpers for experience extraction
    - the weights used to combine similarity signals into one score
"""

# ---------------------------------------------------------------------------
# SKILL TAXONOMY
# Skills are grouped by category. Grouping is used for the "Skill Coverage"
# breakdown shown in the dashboard and for category-aware reporting.
# ---------------------------------------------------------------------------
SKILL_TAXONOMY = {
    "Programming Languages": [
        "python", "java", "c", "c++", "c#", "javascript", "typescript",
        "go", "golang", "rust", "kotlin", "scala", "r", "php", "ruby", "swift",
    ],
    "Web / Backend": [
        "spring", "spring boot", "hibernate", "node", "node.js", "express",
        "express.js", "django", "flask", "fastapi", "rest api", "restful",
        "graphql", "grpc", "microservices", "asp.net", ".net",
    ],
    "Frontend": [
        "react", "react.js", "angular", "vue", "vue.js", "html", "css",
        "bootstrap", "tailwind", "redux", "next.js",
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "postgres", "mongodb", "mongo",
        "redis", "oracle", "elasticsearch", "sqlite", "cassandra", "dynamodb",
    ],
    "DevOps / Cloud": [
        "docker", "kubernetes", "aws", "azure", "gcp", "jenkins", "ci/cd",
        "terraform", "ansible", "linux", "nginx",
    ],
    "Version Control": ["git", "github", "gitlab", "bitbucket"],
    "AI / Machine Learning / NLP": [
        "machine learning", "deep learning", "natural language processing",
        "nlp", "computer vision", "tensorflow", "pytorch", "scikit-learn",
        "sklearn", "keras", "pandas", "numpy", "transformers", "bert",
        "word2vec", "tf-idf", "named entity recognition", "ner", "spacy",
        "opencv", "llm", "generative ai", "prompt engineering",
    ],
    "Computer Science Fundamentals": [
        "data structures", "algorithms", "data structures and algorithms",
        "oop", "object oriented programming", "operating systems",
        "computer networks", "dbms", "database management", "system design",
    ],
    "Testing": ["unit testing", "junit", "pytest", "selenium", "mocha"],
    "Security": ["jwt", "oauth", "oauth2", "authentication", "authorization"],
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "time management", "collaboration", "adaptability", "critical thinking",
    ],
}

# Flat lookup set of every known skill string -> category, longest-first for
# safe substring matching (e.g. "spring boot" must be matched before "spring").
FLAT_SKILLS_TO_CATEGORY = {}
for _category, _skills in SKILL_TAXONOMY.items():
    for _skill in _skills:
        FLAT_SKILLS_TO_CATEGORY[_skill] = _category

ALL_SKILLS_SORTED = sorted(FLAT_SKILLS_TO_CATEGORY.keys(), key=len, reverse=True)

# ---------------------------------------------------------------------------
# EDUCATION
# ---------------------------------------------------------------------------
DEGREE_KEYWORDS = {
    "phd": "PhD",
    "ph.d": "PhD",
    "doctorate": "PhD",
    "m.tech": "M.Tech",
    "mtech": "M.Tech",
    "master of technology": "M.Tech",
    "m.e": "M.E",
    "me ": "M.E",
    "mba": "MBA",
    "m.sc": "M.Sc",
    "msc": "M.Sc",
    "master of science": "M.Sc",
    "mca": "MCA",
    "master of computer applications": "MCA",
    "b.tech": "B.Tech",
    "btech": "B.Tech",
    "bachelor of technology": "B.Tech",
    "b.e": "B.E",
    "be ": "B.E",
    "bachelor of engineering": "B.E",
    "bca": "BCA",
    "bachelor of computer applications": "BCA",
    "b.sc": "B.Sc",
    "bsc": "B.Sc",
    "bachelor of science": "B.Sc",
    "b.com": "B.Com",
    "12th": "12th / Higher Secondary",
    "higher secondary": "12th / Higher Secondary",
    "10th": "10th / Secondary",
    "secondary school": "10th / Secondary",
}

# ---------------------------------------------------------------------------
# SCORING WEIGHTS
# Final score = weighted sum of lexical (TF-IDF), distributional (Word2Vec)
# and contextual (Sentence-Transformer) similarity + skill coverage.
# ---------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "tfidf": 0.25,
    "word2vec": 0.20,
    "semantic": 0.25,
    "skill_coverage": 0.30,
}

SKILL_MATCH_FUZZY_THRESHOLD = 0.80  # for fuzzy / semantic skill matching
