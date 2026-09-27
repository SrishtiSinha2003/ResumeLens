"""
app.py
======
ResumeLens Pro - NLP-Based Intelligent Resume Screening, Skill Extraction
and Context-Aware Job Matching System.

Three modes, each mapped to a case-study objective:

  1. Resume <-> Job Match   : preprocessing, NER, education/experience,
                               skill extraction, TF-IDF/Word2Vec/semantic
                               similarity, missing-skill identification.
  2. Rank Candidates        : multiple resumes vs one job -> ranked shortlist.
  3. Rank Jobs for Resume   : one resume vs multiple job descriptions
                               (multiple-job matching) -> best-fit roles.
"""

import pandas as pd
import streamlit as st

from src.matcher import match_resume_to_job
from src.preprocessing import extract_text
from src.ranking import rank_candidates, rank_jobs_for_resume

st.set_page_config(
    page_title="ResumeLens Pro",
    page_icon="\u25c9",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# STYLE (kept close to the original ResumeLens dark theme)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }
.stApp { background: #08111f; color: #f1f5f9; }
[data-testid="stDecoration"] {
  display: none;
}
[data-testid="stHeader"] {
  background: transparent;
  height: 0;
}
.block-container { max-width: 1400px; padding: 2rem 2rem 3rem; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:2px; }
.brand-mark { width:38px; height:38px; border:1px solid #38bdf8; border-radius:10px;
  display:flex; align-items:center; justify-content:center; color:#38bdf8;
  font-size:22px; font-weight:800; background:rgba(56,189,248,.08); }
.brand-name { font-size:30px; font-weight:800; color:#f8fafc; letter-spacing:-1px; }
.subtitle { color:#91a3b8; font-size:14px; margin:0 0 20px 50px; }
.metric { background:#0f1b2d; border:1px solid #243650; border-radius:14px;
  padding:18px 20px; min-height:120px; }
.metric-label { color:#91a3b8; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.8px; }
.metric-value { font-size:34px; font-weight:800; color:#f1f5f9; margin:8px 0 4px; }
.skill-item { border:1px solid #1e3a4d; background:rgba(52,211,153,.10); color:#a7f3d0;
  border-radius:9px; padding:8px 10px; font-size:13px; font-weight:600; margin:3px; display:inline-block;}
.missing-item { border:1px solid #59431a; background:rgba(251,191,36,.10); color:#fde68a;
  border-radius:9px; padding:8px 10px; font-size:13px; font-weight:600; margin:3px; display:inline-block;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">
  <div class="brand-mark">R</div>
  <br>
  <div class="brand-name">ResumeLens Pro</div>
</div>
<div class="subtitle">
NLP-based intelligent resume screening, skill extraction & context-aware job matching
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "A hybrid NLP pipeline combining preprocessing, Named Entity "
        "Recognition, taxonomy-based skill extraction, TF-IDF, Word2Vec "
        "and Sentence-Transformer similarity to screen resumes against "
        "job descriptions."
    )
    st.markdown("### Modes")
    st.markdown(
        "- **Resume ↔ Job Match** — single detailed report\n"
        "- **Rank Candidates** — many resumes vs one job\n"
        "- **Rank Jobs** — one resume vs many jobs"
    )
    st.caption("Supported files: PDF, DOCX, TXT")


def _read_upload(file):
    return extract_text(file)


def _score_metrics(report):
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, key, suffix in [
        (c1, "Final score", "final_score", ""),
        (c2, "TF-IDF", "tfidf_score", "%"),
        (c3, "Word2Vec", "word2vec_score", "%"),
        (c4, "Semantic", "semantic_score", "%"),
        (c5, "Skill coverage", "skill_coverage", "%"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{report[key]}{suffix}</div></div>',
                unsafe_allow_html=True,
            )


def _skill_pills(skills, css_class, empty_msg):
    if not skills:
        st.info(empty_msg)
        return
    html = "".join(f'<span class="{css_class}">{s.title()}</span>' for s in skills)
    st.markdown(html, unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(
    ["🔍 Resume ↔ Job Match", "🏆 Rank Candidates", "🎯 Rank Jobs for a Resume"]
)

# ===========================================================================
# TAB 1 — SINGLE RESUME vs SINGLE JOB
# ===========================================================================
with tab1:
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Resume")
        resume_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"], key="single_resume")
    with right:
        st.markdown("#### Job description")
        job_text_input = st.text_area("Paste job description", height=200, key="single_job")

    if st.button("Analyze compatibility", type="primary", key="btn_single"):
        if not resume_file or not job_text_input.strip():
            st.error("Please provide both a resume and a job description.")
        else:
            with st.spinner("Running preprocessing, NER, skill extraction and similarity scoring..."):
                resume_text = _read_upload(resume_file)
                report = match_resume_to_job(resume_text, job_text_input)

            st.success("Analysis complete.")
            _score_metrics(report)
            st.progress(max(0.0, min(report["final_score"] / 100, 1.0)))

            st.markdown("---")
            profile_col, skills_col = st.columns(2, gap="large")

            with profile_col:
                st.markdown("### Candidate profile (NER)")
                profile = report["candidate_profile"]
                st.write(f"**Name:** {profile['name']}")
                st.write(f"**Email:** {profile['email'] or '—'}")
                st.write(f"**Phone:** {profile['phone'] or '—'}")
                st.write(f"**Education:** {', '.join(profile['education']) or '—'}")
                st.write(f"**Estimated experience:** {profile['experience_years']} years")
                st.write(f"**Organizations mentioned:** {', '.join(profile['organizations']) or '—'}")

            with skills_col:
                st.markdown("### Matched skills")
                _skill_pills(report["matched_skills"], "skill-item", "No matching skills detected.")
                st.markdown("### Missing skills")
                _skill_pills(report["missing_skills"], "missing-item", "No missing skills — full coverage!")

            st.markdown("---")
            st.markdown("### Skill breakdown by category")
            cat_col1, cat_col2 = st.columns(2)
            with cat_col1:
                st.caption("Job description")
                st.json(report["job_skills_by_category"])
            with cat_col2:
                st.caption("Resume")
                st.json(report["resume_skills_by_category"])

            with st.expander("Semantic skill-match details"):
                st.dataframe(pd.DataFrame(report["skill_match_details"]), use_container_width=True)

# ===========================================================================
# TAB 2 — RANK CANDIDATES (many resumes -> one job)
# ===========================================================================
with tab2:
    st.markdown("#### Upload multiple resumes")
    resume_files = st.file_uploader(
        "Resumes (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"],
        accept_multiple_files=True, key="multi_resumes",
    )
    job_text_rank = st.text_area("Paste the target job description", height=180, key="rank_job")

    if st.button("Rank candidates", type="primary", key="btn_rank_candidates"):
        if not resume_files or not job_text_rank.strip():
            st.error("Please upload at least one resume and a job description.")
        else:
            with st.spinner("Scoring every candidate against the job description..."):
                resumes = [
                    {"name": f.name, "text": _read_upload(f)} for f in resume_files
                ]
                ranked = rank_candidates(resumes, job_text_rank)

            st.success(f"Ranked {len(ranked)} candidate(s).")
            table = pd.DataFrame([{
                "Rank": r["rank"],
                "Candidate": r["name"],
                "Final score": r["final_score"],
                "TF-IDF": r["tfidf_score"],
                "Word2Vec": r["word2vec_score"],
                "Semantic": r["semantic_score"],
                "Skill coverage %": r["skill_coverage"],
                "Matched skills": len(r["matched_skills"]),
                "Missing skills": len(r["missing_skills"]),
            } for r in ranked])
            st.dataframe(table, use_container_width=True, hide_index=True)

            st.download_button(
                "Download ranking as CSV",
                data=table.to_csv(index=False),
                file_name="candidate_ranking.csv",
                mime="text/csv",
            )

            for r in ranked:
                with st.expander(f"#{r['rank']} — {r['name']} (score {r['final_score']})"):
                    st.write(f"**Matched:** {', '.join(r['matched_skills']) or '—'}")
                    st.write(f"**Missing:** {', '.join(r['missing_skills']) or '—'}")
                    st.write(f"**Estimated experience:** {r['candidate_profile']['experience_years']} years")
                    st.write(f"**Education:** {', '.join(r['candidate_profile']['education']) or '—'}")

# ===========================================================================
# TAB 3 — RANK JOBS FOR ONE RESUME (multiple-job matching)
# ===========================================================================
with tab3:
    st.markdown("#### Upload the candidate's resume")
    resume_file_multi = st.file_uploader("Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="mj_resume")

    st.markdown("#### Add job descriptions")
    st.caption("Separate each job with a line containing only `---`.")
    jobs_raw = st.text_area(
        "Job descriptions",
        height=220,
        key="mj_jobs",
        placeholder="Job Title: Backend Developer\n...description...\n---\nJob Title: Data Scientist\n...description...",
    )

    if st.button("Find best-fit jobs", type="primary", key="btn_rank_jobs"):
        if not resume_file_multi or not jobs_raw.strip():
            st.error("Please provide a resume and at least one job description.")
        else:
            blocks = [b.strip() for b in jobs_raw.split("\n---\n") if b.strip()]
            jobs = []
            for i, block in enumerate(blocks, start=1):
                first_line = block.splitlines()[0]
                title = first_line.replace("Job Title:", "").strip() if "Job Title:" in first_line else f"Job {i}"
                jobs.append({"title": title, "text": block})

            with st.spinner("Scoring the resume against every job description..."):
                resume_text = _read_upload(resume_file_multi)
                ranked_jobs = rank_jobs_for_resume(resume_text, jobs)

            st.success(f"Compared against {len(ranked_jobs)} job description(s).")
            table = pd.DataFrame([{
                "Rank": r["rank"],
                "Job title": r["title"],
                "Final score": r["final_score"],
                "TF-IDF": r["tfidf_score"],
                "Word2Vec": r["word2vec_score"],
                "Semantic": r["semantic_score"],
                "Skill coverage %": r["skill_coverage"],
            } for r in ranked_jobs])
            st.dataframe(table, use_container_width=True, hide_index=True)

            for r in ranked_jobs:
                with st.expander(f"#{r['rank']} — {r['title']} (score {r['final_score']})"):
                    st.write(f"**Matched:** {', '.join(r['matched_skills']) or '—'}")
                    st.write(f"**Missing:** {', '.join(r['missing_skills']) or '—'}")

st.markdown("---")
st.caption(
    "ResumeLens Pro • Python • Streamlit • spaCy • Gensim (Word2Vec) • "
    "Scikit-learn (TF-IDF) • Sentence-Transformers • Educational tool — "
    "scores are an analytical aid, not a hiring decision."
)
