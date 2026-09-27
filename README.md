# ResumeLens Pro
### NLP-Based Intelligent Resume Screening, Skill Extraction and Context-Aware Job Matching System
---

## 1. Problem Statement

Manually screening resumes against job descriptions is slow and inconsistent.
**ResumeLens Pro** automates this by extracting structured information from
resumes and job descriptions and computing a context-aware compatibility
score using multiple complementary NLP techniques.

## 2. Objectives → Where they live in the code

| Objective (from case-study brief) | Module | Notes |
|---|---|---|
| Resume preprocessing | `src/preprocessing.py` | text extraction (PDF/DOCX/TXT), cleaning, tokenization, stop-word removal |
| Skill / education / experience extraction | `src/skill_extractor.py`, `src/ner_extraction.py` | taxonomy matching + spaCy noun-chunk mining; degree keyword scan; regex + date-range experience estimation |
| NER | `src/ner_extraction.py` | spaCy `en_core_web_sm` — PERSON, ORG, GPE, DATE, etc.; used for candidate name, organizations, and to strip noise from skill mining |
| Job-description matching | `src/matcher.py` | combines every signal below into one compatibility report |
| TF-IDF / Word2Vec similarity | `src/similarity.py` | `TfidfVectorizer` + cosine similarity; a CBOW `gensim.models.Word2Vec` trained on-the-fly + mean-pooled document vectors + cosine similarity |
| Ranking candidates | `src/ranking.py` → `rank_candidates()` | many resumes vs one job → sorted shortlist |
| Missing-skill identification | `src/similarity.py` → `match_skills_semantically()` | exact + embedding-based fuzzy matching, reports unmatched job skills |
| Multiple-job matching | `src/ranking.py` → `rank_jobs_for_resume()` | one resume vs many job descriptions → ranked best-fit roles |

A fourth, optional signal — Sentence-BERT contextual similarity
(`all-MiniLM-L6-v2`) — is also computed for semantic skill matching
(catching synonyms like "js" ↔ "javascript"). If the model can't be
downloaded (no internet), the system automatically falls back to
TF-IDF + Word2Vec + exact skill matching so the pipeline still runs.

## 3. Pipeline / Architecture

```
        ┌───────────────┐        ┌───────────────┐
        │    RESUME     │        │      JOB      │
        │ (PDF/DOCX/TXT)│        │  DESCRIPTION  │
        └───────┬───────┘        └───────┬───────┘
                │                        │
        Text Extraction            Text Extraction
                │                        │
                ▼                        ▼
        ┌────────────────────────────────────┐
        │   Preprocessing (clean, tokenize,   │
        │        stop-word removal)           │
        └───────────────┬────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        ▼                                  ▼
┌───────────────┐                 ┌──────────────────┐
│      NER      │                 │ Skill Extraction  │
│ name / org /   │                 │ taxonomy + noun   │
│ education /    │                 │ chunk mining       │
│ experience     │                 └─────────┬──────────┘
└───────┬───────┘                           │
        │                                   ▼
        │                    ┌────────────────────────────┐
        │                    │  Similarity Engine           │
        │                    │  TF-IDF · Word2Vec · Semantic│
        │                    └───────────────┬──────────────┘
        │                                    │
        ▼                                    ▼
┌───────────────────────────────────────────────────┐
│         Context-Aware Matching & Scoring            │
│  matched skills · missing skills · weighted score   │
└───────────────────────┬────────────────────────────┘
                         │
        ┌────────────────┴─────────────────┐
        ▼                                   ▼
 Rank Candidates                     Rank Jobs for Resume
 (many resumes → 1 job)             (1 resume → many jobs)
```

## 4. Mathematical Formulation

**TF-IDF weighting** for term *t* in document *d* of corpus *D*:

```
tf-idf(t, d) = tf(t, d) × log( N / df(t) )
```

**Cosine similarity** between vectors A and B:

```
cos(A, B) = (A · B) / (‖A‖ ‖B‖)
```

**Word2Vec (CBOW)** predicts a target word from its context window;
each document is represented as the mean of its constituent word vectors,
and the two document vectors are compared with cosine similarity.

**Final compatibility score** (weighted linear combination, configurable
in `src/config.py::SCORE_WEIGHTS`):

```
score = 0.25·TFIDF + 0.20·Word2Vec + 0.25·Semantic + 0.30·SkillCoverage
SkillCoverage = |matched_skills| / |job_skills|
```

## 5. Tech Stack

- **Application:** Python, Streamlit
- **NLP / ML:** spaCy, scikit-learn (TF-IDF, cosine similarity), Gensim (Word2Vec), Sentence-Transformers
- **Document processing:** PyPDF2, python-docx
- **Testing:** pytest

## 6. Project Structure

```
resume-screening-system/
├── app.py                      # Streamlit UI (3 modes)
├── demo.py                     # CLI demo over the sample dataset
├── requirements.txt
├── README.md
├── src/
│   ├── config.py                # skill taxonomy, degree keywords, score weights
│   ├── preprocessing.py         # text extraction + cleaning
│   ├── ner_extraction.py        # NER, contact info, education, experience
│   ├── skill_extractor.py       # taxonomy + noun-chunk skill extraction
│   ├── similarity.py            # TF-IDF, Word2Vec, semantic similarity
│   ├── matcher.py                # combines everything into one report
│   └── ranking.py                # candidate ranking + multi-job matching
├── data/
│   ├── resumes/                  # 3 sample resumes (backend / data / frontend)
│   └── jobs/                     # 3 sample job descriptions
└── tests/
    └── test_core.py              # unit tests (no model download required)
```

## 7. Corpus / Dataset

This repo ships a small **synthetic sample dataset** (`data/`) — 3 resumes
and 3 job descriptions across Backend, Data Science and Frontend roles —
so the ranking/multi-job features can be demonstrated end-to-end without
needing a real, private dataset. Swap in real (anonymized) resumes/JDs by
dropping `.pdf`/`.docx`/`.txt` files into the same folders.

## 8. Setup & Run

```bash
# 1. Clone / unzip and enter the project
cd resume-screening-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Run the CLI demo (uses the bundled sample data)
python demo.py

# 5. Run unit tests
pytest -q

# 6. Launch the web app
streamlit run app.py
```

The Streamlit app has three tabs:
1. **Resume ↔ Job Match** — upload one resume, paste one job description, get a full breakdown.
2. **Rank Candidates** — upload several resumes, paste one job description, get a ranked shortlist + CSV export.
3. **Rank Jobs for a Resume** — upload one resume, paste several job descriptions (separated by a line with `---`), get the best-fit roles ranked.

## 9. Existing Literature (summary for the presentation)

The hybrid approach here follows the common pattern in resume-screening
literature: combine a **lexical** signal (TF-IDF / BM25), a
**distributional** signal (Word2Vec / GloVe), and a **contextual**
signal (Transformer-based sentence embeddings), rather than relying on
any single similarity measure — lexical methods miss synonyms, while
embedding-only methods can miss exact keyword requirements that
ATS systems rely on.

## 10. Limitations & Future Work

- Skill taxonomy is curated and finite — an emerging-skill discovery
  step (noun-chunk mining) is included but not yet auto-merged into the
  taxonomy.
- Small-model NER (`en_core_web_sm`) can misclassify some technical
  abbreviations as organizations — a larger model or a fine-tuned
  resume-NER model would improve precision.
- Word2Vec is trained on-the-fly on a small corpus per request; using a
  larger pretrained embedding (e.g. GloVe/FastText) would improve
  robustness for very short documents.
- Future: section-wise scoring, downloadable PDF reports, multilingual
  support, persistent storage of past analyses.

## Disclaimer

This is an educational/analytical tool built for a university case study.
Scores are an analytical aid and do not represent a real ATS score or a
hiring decision.
