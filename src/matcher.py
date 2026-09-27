"""
matcher.py
==========
Objective covered: "job-description matching" and "missing-skill
identification". Combines preprocessing, NER, skill extraction and the
three similarity signals into one context-aware compatibility report.
"""

from src.config import SCORE_WEIGHTS
from src.ner_extraction import extract_all as extract_resume_entities
from src.preprocessing import preprocess_pipeline
from src.similarity import (
    match_skills_semantically,
    semantic_similarity,
    tfidf_similarity,
    word2vec_similarity,
)
from src.skill_extractor import extract_skills, extract_skills_by_category


def match_resume_to_job(resume_text: str, job_text: str, weights: dict = None,
                         extra_corpus=None, include_ner: bool = True) -> dict:
    """
    Run the full context-aware matching pipeline for one resume against
    one job description.

    Returns a dict containing every intermediate signal, so the UI/report
    can show a full breakdown rather than a single opaque number.
    """
    weights = weights or SCORE_WEIGHTS

    resume_clean = preprocess_pipeline(resume_text)["filtered_text"]
    job_clean = preprocess_pipeline(job_text)["filtered_text"]

    # ---- similarity signals -------------------------------------------------
    tfidf_score = tfidf_similarity(resume_clean, job_clean)
    w2v_score = word2vec_similarity(resume_clean, job_clean, extra_corpus=extra_corpus)
    semantic_score = semantic_similarity(resume_clean, job_clean)

    # ---- skills ---------------------------------------------------------------
    job_skills = extract_skills(job_text)
    resume_skills = extract_skills(resume_text)
    skill_match = match_skills_semantically(job_skills, resume_skills)
    matched_skills = skill_match["matched"]
    missing_skills = skill_match["missing"]
    skill_coverage = (len(matched_skills) / len(job_skills)) if job_skills else 0.0

    # ---- final weighted score ---------------------------------------------
    final_score = (
        tfidf_score * weights["tfidf"]
        + w2v_score * weights["word2vec"]
        + semantic_score * weights["semantic"]
        + skill_coverage * weights["skill_coverage"]
    ) * 100

    result = {
        "final_score": round(final_score, 2),
        "tfidf_score": round(tfidf_score * 100, 2),
        "word2vec_score": round(w2v_score * 100, 2),
        "semantic_score": round(semantic_score * 100, 2),
        "skill_coverage": round(skill_coverage * 100, 2),
        "job_skills": job_skills,
        "resume_skills": resume_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "skill_match_details": skill_match["details"],
        "resume_skills_by_category": extract_skills_by_category(resume_text),
        "job_skills_by_category": extract_skills_by_category(job_text),
    }

    if include_ner:
        result["candidate_profile"] = extract_resume_entities(resume_text)

    return result
