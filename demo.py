"""
demo.py
=======
Command-line demonstration of the full pipeline, useful for quick testing
and for generating console output to paste into a report/notebook.

Run:
    python demo.py
"""

import json
import os

from src.matcher import match_resume_to_job
from src.ranking import rank_candidates, rank_jobs_for_resume

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def load_resumes():
    folder = os.path.join(DATA_DIR, "resumes")
    return [
        {"name": os.path.splitext(f)[0].replace("resume_", "").title(),
         "text": _read(os.path.join(folder, f))}
        for f in sorted(os.listdir(folder)) if f.endswith(".txt")
    ]


def load_jobs():
    folder = os.path.join(DATA_DIR, "jobs")
    return [
        {"title": os.path.splitext(f)[0].replace("job_", "").replace("_", " ").title(),
         "text": _read(os.path.join(folder, f))}
        for f in sorted(os.listdir(folder)) if f.endswith(".txt")
    ]


def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    resumes = load_resumes()
    jobs = load_jobs()

    # 1. Single resume vs single job -----------------------------------------
    section("1) SINGLE RESUME <-> SINGLE JOB DESCRIPTION")
    r, j = resumes[0], jobs[0]
    report = match_resume_to_job(r["text"], j["text"])
    print(f"Resume: {r['name']}  |  Job: {j['title']}")
    print(json.dumps({k: v for k, v in report.items() if k != "skill_match_details"},
                      indent=2, default=str))

    # 2. Candidate ranking: many resumes vs one job --------------------------
    section("2) CANDIDATE RANKING (many resumes -> one job)")
    ranked = rank_candidates(resumes, jobs[0]["text"])
    for c in ranked:
        print(f"#{c['rank']}  {c['name']:<10}  score={c['final_score']:>6}  "
              f"skills matched={len(c['matched_skills'])}/{len(c['job_skills'])}")

    # 3. Multiple-job matching: one resume vs many jobs ----------------------
    section("3) MULTIPLE-JOB MATCHING (one resume -> many jobs)")
    ranked_jobs = rank_jobs_for_resume(resumes[1]["text"], jobs)
    print(f"Candidate: {resumes[1]['name']}")
    for jr in ranked_jobs:
        print(f"#{jr['rank']}  {jr['title']:<25}  score={jr['final_score']:>6}")


if __name__ == "__main__":
    main()
