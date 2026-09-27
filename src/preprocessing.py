"""
preprocessing.py
================
Resume / job-description preprocessing pipeline.

Objective covered: "Resume preprocessing"

Pipeline stages:
    1. Text extraction (PDF / DOCX / TXT -> raw text)
    2. Normalization  (lowercasing, whitespace, punctuation clean-up)
    3. Tokenization
    4. Stop-word removal
"""

import io
import re

import PyPDF2
from docx import Document

# A small, dependency-free stopword list so the module works even if the
# spaCy model has not been downloaded yet.
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to",
    "in", "on", "at", "for", "with", "by", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it",
    "its", "we", "our", "you", "your", "i", "he", "she", "they", "them",
    "his", "her", "their", "will", "would", "can", "could", "should",
    "has", "have", "had", "do", "does", "did", "not", "no", "yes", "into",
    "about", "than", "such", "over", "under", "up", "down", "out",
}


# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_obj) -> str:
    """Extract raw text from a PDF file-like object or path."""
    reader = PyPDF2.PdfReader(file_obj)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_text_from_docx(file_obj) -> str:
    """Extract raw text from a DOCX file-like object or path."""
    document = Document(file_obj)
    return "\n".join(p.text for p in document.paragraphs)


def extract_text_from_txt(file_obj) -> str:
    if hasattr(file_obj, "read"):
        raw = file_obj.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return raw
    with open(file_obj, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def extract_text(uploaded_file, filename: str = None) -> str:
    """
    Dispatch to the correct extractor based on file extension.
    Works with Streamlit's UploadedFile, a plain path string, or a
    file-like object (in which case `filename` must be supplied).
    """
    name = (filename or getattr(uploaded_file, "name", "")).lower()

    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    if name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)

    raise ValueError(f"Unsupported file type for '{name}'. Use PDF, DOCX or TXT.")


# ---------------------------------------------------------------------------
# NORMALIZATION / CLEANING
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase + collapse whitespace, keep symbols used in tech skills."""
    text = text.lower()
    # Keep letters, numbers, and symbols commonly found in skills (c++, c#, .net)
    text = re.sub(r"[^a-z0-9+#./\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    """Simple whitespace / punctuation tokenizer."""
    text = clean_text(text)
    return [tok for tok in text.split(" ") if tok]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def preprocess_pipeline(raw_text: str) -> dict:
    """
    Run the full preprocessing pipeline and return every intermediate
    artifact so callers (or the UI) can display/inspect each stage.
    """
    cleaned = clean_text(raw_text)
    tokens = tokenize(raw_text)
    filtered_tokens = remove_stopwords(tokens)
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "filtered_text": " ".join(filtered_tokens),
    }
