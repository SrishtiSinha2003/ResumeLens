"""
ranking.py
==========
Objectives covered: "ranking candidates" and "multiple-job matching".

Two symmetric use-cases on top of matcher.match_resume_to_job:

    rank_candidates(resumes, job_text)
        One job description, many resumes -> a ranked shortlist.
        (Recruiter's view: "who is the best fit for this role?")

    rank_jobs_for_resume(resume_text, jobs)
        One resume, many job descriptions -> ranked best-fit roles.
        (Candidate's view: "which of these openings suits me best?")
"""

from src.matcher import match_resume_to_job


def rank_candidates(resumes: list, job_text: str, weights: dict = None) -> list:
    """
    resumes: list of {"name": str, "text": str}
    Returns resumes sorted by final_score (desc), each annotated with its
    full match report and a 1-based `rank`.
    """
    corpus = [r["text"] for r in resumes] + [job_text]
    results = []
    for resume in resumes:
        report = match_resume_to_job(
            resume["text"], job_text, weights=weights, extra_corpus=corpus
        )
        results.append({"name": resume["name"], **report})

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results


def rank_jobs_for_resume(resume_text: str, jobs: list, weights: dict = None) -> list:
    """
    jobs: list of {"title": str, "text": str}
    Returns jobs sorted by final_score (desc) for a single candidate,
    each annotated with its full match report and a 1-based `rank`.
    """
    corpus = [j["text"] for j in jobs] + [resume_text]
    results = []
    for job in jobs:
        report = match_resume_to_job(
            resume_text, job["text"], weights=weights, extra_corpus=corpus,
            include_ner=False,
        )
        results.append({"title": job["title"], **report})

    results.sort(key=lambda r: r["final_score"], reverse=True)
    for i, r in enumerate(results, start=1):
        r["rank"] = i
    return results
