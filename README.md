# ResumeLens Pro

### NLP-Based Intelligent Resume Screening, Skill Extraction and Context-Aware Job Matching System

ResumeLens Pro is an educational NLP application that analyzes resumes and job descriptions, extracts structured candidate information, identifies skills and skill gaps, computes multiple similarity signals, and ranks candidates or jobs based on a configurable compatibility score.

> **Disclaimer:** ResumeLens Pro is an educational/analytical tool developed for a university NLP case study. Its scores are analytical aids and do not represent a real ATS score, hiring decision, or recommendation.

---

## 1. Problem Statement

Manually screening resumes against job descriptions is time-consuming and can be inconsistent.

**ResumeLens Pro** automates this process by extracting structured information from resumes and job descriptions and computing a context-aware compatibility score using multiple complementary NLP techniques.

---

## 2. Objectives and Code Mapping

| Objective | Module | Implementation |
|---|---|---|
| Resume preprocessing | `src/preprocessing.py` | PDF/DOCX/TXT text extraction, cleaning, tokenization and stop-word removal |
| Skill extraction | `src/skill_extractor.py` | Curated skill taxonomy + phrase/noun-chunk based mining |
| Education extraction | `src/ner_extraction.py` | Degree/education keyword-based extraction |
| Experience extraction | `src/ner_extraction.py` | Regex/date-range based experience estimation |
| Named Entity Recognition | `src/ner_extraction.py` | spaCy `en_core_web_sm` for PERSON, ORG, GPE, DATE, etc. |
| Job-description matching | `src/matcher.py` | Combines lexical, distributional, semantic and skill-coverage signals |
| TF-IDF similarity | `src/similarity.py` | `TfidfVectorizer` + cosine similarity |
| Word2Vec similarity | `src/similarity.py` | CBOW Word2Vec trained on the document pair + mean-pooled vectors |
| Semantic similarity | `src/similarity.py` | Sentence-BERT (`all-MiniLM-L6-v2`) embeddings + cosine similarity |
| Missing-skill identification | `src/similarity.py` | Exact + embedding-based semantic skill matching |
| Candidate ranking | `src/ranking.py` | Many resumes → one job → ranked shortlist |
| Multiple-job matching | `src/ranking.py` | One resume → many jobs → ranked job list |

---

## 3. Main Features

### Resume ↔ Job Match

Upload a resume and provide a job description to obtain:

- Final compatibility score
- TF-IDF similarity
- Word2Vec similarity
- Sentence-BERT semantic similarity
- Skill coverage
- Matched skills
- Missing skills
- Extracted candidate information
- Education and experience information
- Organizations and other NER entities

### Rank Candidates

Provide one job description and multiple resumes to generate a ranked candidate shortlist.

The application also supports CSV export of ranking results.

### Rank Jobs for a Resume

Provide one resume and multiple job descriptions separated by `---` to rank the available roles according to the resume's compatibility.

---

## 4. Pipeline / Architecture

```text
        ┌────────────────┐          ┌────────────────┐
        │     RESUME     │          │      JOB       │
        │  PDF/DOCX/TXT  │          │  DESCRIPTION   │
        └───────┬────────┘          └───────┬────────┘
                │                           │
                ▼                           ▼
        ┌─────────────────────────────────────────┐
        │            Text Extraction              │
        └────────────────────┬────────────────────┘
                             ▼
        ┌─────────────────────────────────────────┐
        │ Preprocessing: cleaning, tokenization,  │
        │             stop-word removal           │
        └────────────────────┬────────────────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
        ┌───────────────┐       ┌──────────────────┐
        │      NER      │       │ Skill Extraction │
        │ PERSON / ORG  │       │ taxonomy + phrase│
        │ GPE / DATE    │       │ / noun-chunk     │
        └───────┬───────┘       └────────┬─────────┘
                │                        │
                └────────────┬───────────┘
                             ▼
                 ┌──────────────────────────┐
                 │    Similarity Engine     │
                 │                          │
                 │ TF-IDF · Word2Vec       │
                 │ Sentence-BERT           │
                 └────────────┬─────────────┘
                              ▼
                 ┌──────────────────────────┐
                 │ Context-Aware Matching   │
                 │ matched/missing skills   │
                 │ + weighted compatibility │
                 │          score            │
                 └────────────┬─────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Rank Candidates            Rank Jobs
          many → one job             one → many jobs
```

---

## 5. Mathematical Formulation

### TF-IDF

For term *t* in document *d*:

```text
tf-idf(t,d) = tf(t,d) × log(N / df(t))
```

where:

- `tf(t,d)` = term frequency
- `df(t)` = number of documents containing the term
- `N` = number of documents in the corpus

### Cosine Similarity

For vectors `A` and `B`:

```text
cos(A,B) = (A · B) / (||A|| ||B||)
```

### Word2Vec

ResumeLens uses CBOW Word2Vec. A target word is learned from its surrounding context. A document is represented by the mean of the word vectors available in the trained model, and the resulting document vectors are compared using cosine similarity.

### Skill Coverage

```text
SkillCoverage =
    |Matched Skills| / |Job Skills|
```

### Final Compatibility Score

The final score is a configurable weighted combination defined in:

```text
src/config.py
```

Current weights:

```text
Score =
    0.25 × TF-IDF
  + 0.20 × Word2Vec
  + 0.25 × Semantic
  + 0.30 × SkillCoverage
```

The displayed final score is scaled to `0–100`.

---

## 6. Technology Stack

| Category | Technologies |
|---|---|
| Application | Python, Streamlit |
| NLP / ML | spaCy, scikit-learn, Gensim, Sentence-Transformers |
| Similarity | TF-IDF, cosine similarity, Word2Vec, Sentence-BERT |
| Document Processing | PyPDF2, python-docx |
| Testing | pytest |
| Data | CSV, TXT, synthetic evaluation corpus |

---

## 7. Project Structure

```text
resume-screening-system/
│
├── app.py                         # Streamlit application
├── demo.py                        # CLI demonstration
├── requirements.txt
├── README.md
│
├── data/
│   ├── resumes/                   # bundled sample resumes
│   ├── jobs/                      # bundled sample job descriptions
│   │
│   └── evaluation/                # controlled evaluation corpus
│       ├── evaluation_pairs.csv
│       ├── resumes/
│       └── jobs/
│
├── scripts/
│   └── evaluate.py                # quantitative evaluation + ablation
│
├── results/
│   ├── evaluation_results.csv
│   ├── ablation_results.csv
│   └── figures/
│       ├── ablation_f1.png
│       ├── ranking_ndcg.png
│       └── ranking_mrr.png
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # taxonomy, keywords, score weights
│   ├── preprocessing.py           # extraction + preprocessing
│   ├── ner_extraction.py          # NER + education + experience
│   ├── skill_extractor.py         # skill extraction
│   ├── similarity.py              # similarity functions
│   ├── matcher.py                 # integrated matching
│   └── ranking.py                 # candidate/job ranking
│
└── tests/
    └── test_core.py               # unit tests
```

---

## 8. Corpus and Dataset

### Bundled Demonstration Dataset

The repository contains a small synthetic sample dataset:

- 3 resumes
- 3 job descriptions
- Backend, Data Science and Frontend role examples

This dataset is intended for demonstrating the complete application pipeline without requiring private candidate data.

### Evaluation Dataset

A separate controlled evaluation corpus is provided under:

```text
data/evaluation/
```

The current evaluation contains:

- 10 synthetic resumes
- 10 synthetic job descriptions
- 50 labeled resume–job pairs

Each pair has a graded relevance label:

```text
0 = Not relevant
1 = Partially relevant
2 = Highly relevant
```

The evaluation corpus is synthetic and controlled for a university case study. It should not be interpreted as representative of real-world hiring data.

---

## 9. Quantitative Evaluation

The evaluation script can be run using:

```bash
python scripts/evaluate.py
```

It evaluates four configurations:

### A. TF-IDF

```text
TF-IDF
```

### B. TF-IDF + Word2Vec

```text
TF-IDF + Word2Vec
```

### C. TF-IDF + Word2Vec + Sentence-BERT

```text
TF-IDF + Word2Vec + SBERT
```

### D. Full ResumeLens

```text
TF-IDF + Word2Vec + SBERT + Skill Coverage
```

Binary classification metrics use a fixed compatibility-score threshold of `50/100`.

Ranking evaluation uses:

- Precision@1
- Precision@3
- Mean Reciprocal Rank (MRR)
- NDCG@5

For ranking metrics, relevance labels remain graded for NDCG, while relevance `>= 1` is treated as relevant for MRR and Precision@K.

### Evaluation Results

| Configuration | Precision | Recall | F1 | P@1 | P@3 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | 1.000 | 0.097 | 0.176 | 1.000 | 0.933 | 1.000 | 0.996 |
| TF-IDF + Word2Vec | 1.000 | 0.258 | 0.410 | 1.000 | 0.900 | 1.000 | 0.992 |
| TF-IDF + Word2Vec + SBERT | 1.000 | 0.355 | 0.524 | 1.000 | 0.933 | 1.000 | 0.996 |
| **Full ResumeLens** | **1.000** | **0.452** | **0.622** | **1.000** | **0.933** | **1.000** | **0.996** |

### Ablation Observation

The F1 score increases progressively as additional signals are introduced:

```text
TF-IDF                         0.176
        ↓
TF-IDF + Word2Vec             0.410
        ↓
+ Sentence-BERT               0.524
        ↓
Full ResumeLens               0.622
```

This controlled experiment indicates that distributional similarity, contextual semantic similarity and explicit skill coverage provide complementary information for resume–job relevance classification.

The very high ranking metrics should be interpreted in the context of the controlled synthetic corpus; evaluation on a larger human-labeled real-world dataset would be needed to assess generalizability.

---

## 10. Running the Project

### 1. Clone / unzip the project

```bash
cd resume-screening-system
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install the spaCy English model if it is not already installed:

```bash
python -m spacy download en_core_web_sm
```

### 4. Run the CLI demo

```bash
python demo.py
```

### 5. Run unit tests

```bash
pytest -q
```

### 6. Run quantitative evaluation

```bash
python scripts/evaluate.py
```

Results are written to:

```text
results/evaluation_results.csv
results/ablation_results.csv
```

### 7. Launch the Streamlit application

```bash
streamlit run app.py
```

---

## 11. Streamlit Application Modes

### 1. Resume ↔ Job Match

Upload one resume and provide one job description.

The application returns:

- Overall compatibility score
- Component similarity scores
- Skill coverage
- Matched skills
- Missing skills
- Candidate profile
- Education
- Experience
- Organizations and other extracted entities

### 2. Rank Candidates

Upload multiple resumes and provide one job description.

The system calculates compatibility scores and returns a ranked shortlist. Results can also be exported as CSV.

### 3. Rank Jobs for a Resume

Upload one resume and provide multiple job descriptions separated by:

```text
---
```

The system returns the jobs ranked according to the computed compatibility score.

---

## 12. Semantic Matching and Fallback Behavior

ResumeLens optionally uses:

```text
all-MiniLM-L6-v2
```

for contextual semantic similarity and semantic skill matching.

The semantic component can help capture related expressions such as:

```text
JS ↔ JavaScript
```

When the Sentence-BERT model is unavailable, the similarity module handles the failure gracefully and can fall back to the available lexical/distributional signals and exact skill matching.

For the quantitative evaluation reported in this README, Sentence-BERT was successfully loaded and the full hybrid configuration was evaluated with the semantic component enabled.

---

## 13. Existing Literature and Research Motivation

ResumeLens follows a hybrid NLP approach that combines different representations rather than relying on a single similarity measure.

The motivation is:

- **Lexical methods** such as TF-IDF capture exact terms and keyword overlap.
- **Distributional representations** such as Word2Vec capture relationships learned from word co-occurrence.
- **Contextual representations** such as Sentence-BERT capture semantic similarity beyond exact lexical overlap.
- **Explicit skill coverage** provides an interpretable signal for job-specific requirements.

Relevant research directions include:

- Word2Vec-based distributed representations
- Sentence-BERT contextual sentence embeddings
- Transformer-based resume and job-description matching
- Skill extraction from recruitment text
- Resume–job matching benchmarks
- Learning-based resume ranking

Selected references are provided in the project presentation/report.

---

## 14. Limitations

### Curated Skill Taxonomy

The skill taxonomy is finite and manually curated. Phrase/noun-chunk mining can identify additional candidate phrases, but newly discovered skills are not automatically merged into the taxonomy.

### NER Model

The small spaCy model:

```text
en_core_web_sm
```

may misclassify some technical abbreviations or domain-specific terms as organizations or other entities.

### Word2Vec Training

Word2Vec is trained on-the-fly on a small document corpus for each comparison. This makes the implementation self-contained but can be less robust than using a large pretrained embedding model.

### Controlled Evaluation Corpus

The quantitative evaluation uses a small synthetic, controlled dataset rather than a large human-labeled recruitment dataset. The reported metrics therefore demonstrate the behavior of the implemented system on this corpus rather than real-world hiring performance.

### Threshold Sensitivity

Precision, recall and F1 depend on the selected compatibility-score threshold. The reported classification results use a fixed threshold of `50/100`.

---

## 15. Future Scope

Potential extensions include:

- Larger human-labeled resume–JD evaluation datasets
- Automatic emerging-skill discovery and taxonomy expansion
- Section-wise resume/JD scoring
- Larger or pretrained embedding models such as FastText/GloVe
- Fine-tuned resume-specific NER
- Multilingual resume and job-description support
- Downloadable PDF analysis reports
- Persistent storage of previous analyses
- More extensive ranking evaluation
- Learning-to-rank or supervised matching models
- Explainable feature-level comparison dashboards

---

## 16. Testing

The project includes unit tests under:

```text
tests/test_core.py
```

Run:

```bash
pytest -q
```

The tests cover core functionality without requiring the complete evaluation pipeline or a model download.

---

## 17. Academic Use

ResumeLens Pro was developed as an NLP case-study project demonstrating practical applications of:

- Text preprocessing
- Named Entity Recognition
- Information extraction
- Skill extraction
- TF-IDF
- Cosine similarity
- Word2Vec
- Sentence-BERT
- Semantic matching
- Ranking
- Quantitative evaluation
- Ablation analysis

The system is intended to demonstrate NLP methodology and experimental analysis rather than replace human recruitment decisions.

---

## 18. Author / Project

**Project:** ResumeLens Pro  
**Domain:** Natural Language Processing  
**Application:** Intelligent Resume Screening and Job Matching  
**Interface:** Streamlit  
**Language:** Python
