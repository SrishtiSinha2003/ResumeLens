"""
similarity.py
=============
Objective covered: "TF-IDF/Word2Vec similarity" (+ an optional contextual
Sentence-Transformer signal used for semantic skill matching).

Three independent similarity signals are computed between a resume and a
job description, each capturing something different:

    1. TF-IDF + cosine similarity  -> lexical / keyword overlap
    2. Word2Vec (CBOW) + cosine similarity of averaged word vectors
       -> distributional / co-occurrence based similarity, trained
          on-the-fly on the two documents so the system works fully
          offline without downloading a large pretrained embedding file
    3. Sentence-Transformer embeddings -> contextual / meaning-based
       similarity, and per-skill semantic matching (catches synonyms such
       as "js" <-> "javascript").
"""

from functools import lru_cache

import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import tokenize, remove_stopwords


# ---------------------------------------------------------------------------
# 1. TF-IDF SIMILARITY
# ---------------------------------------------------------------------------
def tfidf_similarity(doc1: str, doc2: str) -> float:
    """Cosine similarity between TF-IDF vectors of two documents."""
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform([doc1, doc2])
        return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# 2. WORD2VEC SIMILARITY
# ---------------------------------------------------------------------------
def _document_vector(tokens, model, vector_size):
    vectors = [model.wv[t] for t in tokens if t in model.wv]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axis=0)


def word2vec_similarity(doc1: str, doc2: str, extra_corpus=None,
                         vector_size=100, window=5, min_count=1, epochs=50) -> float:
    """
    Train a small CBOW Word2Vec model on the supplied documents (plus any
    extra background corpus, e.g. other resumes/JDs in the batch, to give
    the model more context) and compare the mean word-vector of each
    document via cosine similarity.

    Training on-the-fly (rather than requiring a large pretrained model
    download) keeps the system self-contained and reproducible for a
    course/case-study environment.
    """
    tokens1 = remove_stopwords(tokenize(doc1))
    tokens2 = remove_stopwords(tokenize(doc2))

    sentences = [tokens1, tokens2]
    if extra_corpus:
        sentences += [remove_stopwords(tokenize(d)) for d in extra_corpus]

    if not tokens1 or not tokens2:
        return 0.0

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,  # CBOW
        epochs=epochs,
        workers=1,
        seed=42,
    )

    vec1 = _document_vector(tokens1, model, vector_size)
    vec2 = _document_vector(tokens2, model, vector_size)

    if not np.any(vec1) or not np.any(vec2):
        return 0.0

    sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    return float(max(0.0, min(1.0, sim)))


# ---------------------------------------------------------------------------
# 3. SENTENCE-TRANSFORMER (CONTEXTUAL) SIMILARITY - optional, best-effort
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(doc1: str, doc2: str) -> float:
    """
    Contextual similarity using Sentence-BERT embeddings. Returns 0.0 with
    a printed warning (instead of raising) if the model cannot be loaded
    (e.g. no internet access on first run), so the rest of the pipeline
    keeps working using TF-IDF + Word2Vec alone.
    """
    try:
        model = _get_embedding_model()
        embeddings = model.encode([doc1, doc2], normalize_embeddings=True)
        score = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[similarity] semantic_similarity unavailable: {exc}")
        return 0.0


def match_skills_semantically(job_skills, resume_skills, threshold=0.65):
    """
    Best-effort semantic skill matching (catches synonyms / abbreviations).
    Falls back to exact-match only if the embedding model is unavailable.
    """
    if not job_skills:
        return {"matched": [], "missing": [], "details": []}
    if not resume_skills:
        return {"matched": [], "missing": list(job_skills), "details": []}

    try:
        model = _get_embedding_model()
        job_emb = model.encode(job_skills, normalize_embeddings=True)
        res_emb = model.encode(resume_skills, normalize_embeddings=True)
        sim_matrix = cosine_similarity(job_emb, res_emb)
    except Exception:
        sim_matrix = None

    matched, missing, details = [], [], []
    for i, job_skill in enumerate(job_skills):
        if sim_matrix is not None:
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            best_match = resume_skills[best_idx]
        else:
            best_match, best_score = ("", 0.0)
            if job_skill in resume_skills:
                best_match, best_score = job_skill, 1.0

        is_match = job_skill in resume_skills or best_score >= threshold
        (matched if is_match else missing).append(job_skill)
        details.append({
            "job_skill": job_skill,
            "resume_match": best_match,
            "similarity": round(best_score, 3),
        })
    return {"matched": sorted(set(matched)), "missing": sorted(set(missing)), "details": details}
