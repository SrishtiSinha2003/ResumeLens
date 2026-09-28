"""
evaluate.py
-----------
python scripts/evaluate.py  
Quantitative evaluation + ablation study for ResumeLens.

Run from the project root:
    python scripts/evaluate.py

Expected evaluation file:
    data/evaluation/evaluation_pairs.csv

Expected CSV columns:
    resume_id, job_id, relevance
    Optional: pair_id, label

Resume and job filenames are inferred from IDs:
    resume_id -> data/evaluation/resumes/<resume_id>.txt
    job_id    -> data/evaluation/jobs/<job_id>.txt

Relevance labels:
    0 = Not relevant
    1 = Partially relevant
    2 = Highly relevant

The script evaluates:
    A. TF-IDF
    B. TF-IDF + Word2Vec
    C. TF-IDF + Word2Vec + Sentence-BERT
    D. Full ResumeLens (original hybrid score + skill coverage)

For binary Precision/Recall/F1:
    relevance >= 1 is treated as relevant.

For ranking:
    labels remain graded (0/1/2) for NDCG.
    MRR and Precision@K use relevance >= 1.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    ndcg_score,
)

# ---------------------------------------------------------------------
# Project root / imports
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SCORE_WEIGHTS
from src.preprocessing import preprocess_pipeline
from src.similarity import (
    _get_embedding_model,
    tfidf_similarity,
    word2vec_similarity,
    semantic_similarity,
    match_skills_semantically,
)
from src.skill_extractor import extract_skills


EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation" / "evaluation_pairs.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


def resolve_data_file(filename: str) -> Path:
    """
    Resolve an evaluation file.

    First checks data/evaluation/.
    Then checks the normal data/resumes or data/jobs directories.
    """
    candidates = [
        PROJECT_ROOT / "data" / "evaluation" / filename,
        PROJECT_ROOT / "data" / "resumes" / filename,
        PROJECT_ROOT / "data" / "jobs" / filename,
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find evaluation file '{filename}'. Checked:\n"
        + "\n".join(str(p) for p in candidates)
    )


def validate_dataset(df: pd.DataFrame) -> None:
    required = {
        "resume_id",
        "job_id",
        "relevance",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"evaluation_pairs.csv is missing columns: {sorted(missing)}"
        )

    allowed_labels = {0, 1, 2}
    actual_labels = set(df["relevance"].dropna().astype(int).unique())

    if not actual_labels.issubset(allowed_labels):
        raise ValueError(
            f"Relevance labels must be 0, 1, or 2. Found: {sorted(actual_labels)}"
        )


def check_sbert() -> None:
    """
    Force-load Sentence-BERT before evaluation.

    This prevents a model-loading failure from silently becoming
    a semantic score of 0.0 inside semantic_similarity().
    """
    print("Loading Sentence-BERT: all-MiniLM-L6-v2 ...")
    try:
        _get_embedding_model()
    except Exception as exc:
        raise RuntimeError(
            "\nSentence-BERT could not be loaded.\n"
            "Please make sure sentence-transformers is installed and "
            "the all-MiniLM-L6-v2 model can be downloaded/loaded.\n\n"
            f"Original error: {exc}"
        ) from exc

    print("Sentence-BERT loaded successfully.\n")


def normalized_ablation_scores(tfidf, w2v, semantic, skill_coverage):
    """
    Compute ablation scores while preserving the original SCORE_WEIGHTS.

    Partial models renormalize only the weights of the components included
    in that model. The full model uses the original ResumeLens formula.
    """

    wt = SCORE_WEIGHTS["tfidf"]
    ww = SCORE_WEIGHTS["word2vec"]
    ws = SCORE_WEIGHTS["semantic"]
    wk = SCORE_WEIGHTS["skill_coverage"]

    # A: TF-IDF only
    score_a = tfidf

    # B: TF-IDF + Word2Vec
    score_b = (
        tfidf * wt + w2v * ww
    ) / (wt + ww)

    # C: TF-IDF + Word2Vec + Sentence-BERT
    score_c = (
        tfidf * wt + w2v * ww + semantic * ws
    ) / (wt + ww + ws)

    # D: exact full ResumeLens formula
    score_d = (
        tfidf * wt
        + w2v * ww
        + semantic * ws
        + skill_coverage * wk
    )

    return {
        "tfidf_only": score_a * 100,
        "tfidf_word2vec": score_b * 100,
        "tfidf_word2vec_sbert": score_c * 100,
        "full_resumelens": score_d * 100,
    }


def precision_at_k(labels, scores, k):
    """
    Binary Precision@K.
    relevance >= 1 is considered relevant.
    """
    order = np.argsort(-np.asarray(scores))
    top_k = order[:k]

    if len(top_k) == 0:
        return 0.0

    relevant = sum(labels[i] >= 1 for i in top_k)
    return relevant / len(top_k)


def reciprocal_rank(labels, scores):
    """
    Reciprocal rank of the first relevant job.
    relevance >= 1 is considered relevant.
    """
    order = np.argsort(-np.asarray(scores))

    for rank, idx in enumerate(order, start=1):
        if labels[idx] >= 1:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(labels, scores, k=5):
    """
    Graded NDCG@K using the original 0/1/2 relevance labels.
    """
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)

    if np.sum(labels) == 0:
        return 0.0

    return float(ndcg_score([labels], [scores], k=k))


def classification_metrics(y_true, y_score, threshold):
    """
    Convert continuous score to binary prediction using a fixed threshold.
    """
    y_true_binary = np.asarray(y_true) >= 1
    y_pred_binary = np.asarray(y_score) >= threshold

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary,
        y_pred_binary,
        average="binary",
        zero_division=0,
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


def ranking_metrics(df, score_column):
    """
    Resume-centric ranking evaluation.

    Each resume_id is treated as one query and its jobs are ranked by
    the model score.
    """
    p1_values = []
    p3_values = []
    mrr_values = []
    ndcg_values = []

    for _, group in df.groupby("resume_id"):
        labels = group["relevance"].astype(int).tolist()
        scores = group[score_column].astype(float).tolist()

        p1_values.append(precision_at_k(labels, scores, 1))
        p3_values.append(precision_at_k(labels, scores, 3))
        mrr_values.append(reciprocal_rank(labels, scores))
        ndcg_values.append(ndcg_at_k(labels, scores, 5))

    return {
        "P@1": float(np.mean(p1_values)),
        "P@3": float(np.mean(p3_values)),
        "MRR": float(np.mean(mrr_values)),
        "NDCG@5": float(np.mean(ndcg_values)),
    }


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main(threshold=50.0):
    print("=" * 70)
    print("ResumeLens - Quantitative Evaluation & Ablation Study")
    print("=" * 70)

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"\nEvaluation dataset not found:\n{EVALUATION_FILE}\n\n"
            "Create the folder data/evaluation and place "
            "evaluation_pairs.csv there first."
        )

    df = pd.read_csv(EVALUATION_FILE)
    validate_dataset(df)

    print(f"Evaluation pairs: {len(df)}")
    print("Relevance distribution:")
    print(df["relevance"].value_counts().sort_index().to_string())
    print()

    # Full model depends on Sentence-BERT.
    check_sbert()

    rows = []

    for index, row in df.iterrows():
        # The evaluation CSV identifies documents by resume_id/job_id.
        # The actual files are stored as <id>.txt in the evaluation
        # resumes/ and jobs/ folders.
        resume_filename = f"{row['resume_id']}.txt"
        job_filename = f"{row['job_id']}.txt"

        resume_path = (
            PROJECT_ROOT / "data" / "evaluation" / "resumes" / resume_filename
        )
        job_path = (
            PROJECT_ROOT / "data" / "evaluation" / "jobs" / job_filename
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume file not found for {row['resume_id']}: {resume_path}"
            )

        if not job_path.exists():
            raise FileNotFoundError(
                f"Job file not found for {row['job_id']}: {job_path}"
            )

        resume_text = read_text_file(resume_path)
        job_text = read_text_file(job_path)

        resume_clean = preprocess_pipeline(resume_text)["filtered_text"]
        job_clean = preprocess_pipeline(job_text)["filtered_text"]

        # -------------------------------------------------------------
        # Three similarity signals
        # -------------------------------------------------------------

        tfidf = tfidf_similarity(resume_clean, job_clean)

        # This intentionally uses the same default behavior as the
        # production matcher: no extra unlabeled corpus.
        w2v = word2vec_similarity(
            resume_clean,
            job_clean,
            extra_corpus=None,
        )

        semantic = semantic_similarity(
            resume_clean,
            job_clean,
        )

        # -------------------------------------------------------------
        # Skill coverage
        # -------------------------------------------------------------

        job_skills = extract_skills(job_text)
        resume_skills = extract_skills(resume_text)

        skill_match = match_skills_semantically(
            job_skills,
            resume_skills,
        )

        matched_skills = skill_match["matched"]
        skill_coverage = (
            len(matched_skills) / len(job_skills)
            if job_skills
            else 0.0
        )

        # -------------------------------------------------------------
        # Ablation scores
        # -------------------------------------------------------------

        scores = normalized_ablation_scores(
            tfidf=tfidf,
            w2v=w2v,
            semantic=semantic,
            skill_coverage=skill_coverage,
        )

        rows.append(
            {
                "resume_id": row["resume_id"],
                "job_id": row["job_id"],
                "resume_file": resume_filename,
                "job_file": job_filename,
                "relevance": int(row["relevance"]),

                "tfidf_score": round(tfidf * 100, 4),
                "word2vec_score": round(w2v * 100, 4),
                "semantic_score": round(semantic * 100, 4),
                "skill_coverage": round(skill_coverage * 100, 4),

                "matched_skills": ", ".join(matched_skills),
                "missing_skills": ", ".join(skill_match["missing"]),

                "A_tfidf_only": round(scores["tfidf_only"], 4),
                "B_tfidf_word2vec": round(scores["tfidf_word2vec"], 4),
                "C_tfidf_word2vec_sbert": round(
                    scores["tfidf_word2vec_sbert"], 4
                ),
                "D_full_resumelens": round(
                    scores["full_resumelens"], 4
                ),
            }
        )

        print(
            f"[{index + 1:02d}/{len(df)}] "
            f"{row['resume_id']} × {row['job_id']} | "
            f"label={int(row['relevance'])} | "
            f"full={scores['full_resumelens']:.2f}"
        )

    results_df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_path = RESULTS_DIR / "evaluation_results.csv"
    results_df.to_csv(pairwise_path, index=False)

    # -------------------------------------------------------------
    # Aggregate evaluation
    # -------------------------------------------------------------

    model_columns = {
        "A - TF-IDF": "A_tfidf_only",
        "B - TF-IDF + Word2Vec": "B_tfidf_word2vec",
        "C - TF-IDF + Word2Vec + SBERT": "C_tfidf_word2vec_sbert",
        "D - Full ResumeLens": "D_full_resumelens",
    }

    summary_rows = []

    for model_name, column in model_columns.items():
        y_true = results_df["relevance"].astype(int).tolist()
        y_score = results_df[column].astype(float).tolist()

        cls = classification_metrics(
            y_true=y_true,
            y_score=y_score,
            threshold=threshold,
        )

        rank = ranking_metrics(
            results_df,
            score_column=column,
        )

        summary_rows.append(
            {
                "Model": model_name,
                "Threshold": threshold,
                "Precision": round(cls["Precision"], 4),
                "Recall": round(cls["Recall"], 4),
                "F1": round(cls["F1"], 4),
                "P@1": round(rank["P@1"], 4),
                "P@3": round(rank["P@3"], 4),
                "MRR": round(rank["MRR"], 4),
                "NDCG@5": round(rank["NDCG@5"], 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = RESULTS_DIR / "ablation_results.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print("\nFiles generated:")
    print(f"  {pairwise_path}")
    print(f"  {summary_path}")

    print("\nEvaluation definitions:")
    print("  Relevant for Precision/Recall/F1/MRR/P@K: relevance >= 1")
    print("  NDCG uses graded relevance: 0 = not, 1 = partial, 2 = high")
    print(f"  Classification threshold: {threshold}/100")
    print("  Ranking query: each resume ranks its available job descriptions")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ResumeLens and run ablation study."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=50.0,
        help="Score threshold (0-100) for binary Precision/Recall/F1. "
             "Default: 50.",
    )

    args = parser.parse_args()

    if not 0 <= args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100.")

    main(threshold=args.threshold)
