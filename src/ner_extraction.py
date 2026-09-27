"""
ner_extraction.py
==================
Named-Entity Recognition and structured information extraction.

Objectives covered: "NER", "skill/education/experience extraction"
(the contact-info / education / experience parts of it; skills live in
skill_extractor.py).

Uses spaCy's pretrained `en_core_web_sm` pipeline for entity recognition
(PERSON, ORG, GPE, DATE ...) and combines it with regex + a curated degree
keyword list for robust education and experience-year extraction, since
generic NER models are unreliable at pulling structured resume fields on
their own.
"""

import re
from functools import lru_cache

import spacy

from src.config import DEGREE_KEYWORDS

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(\+?\d{1,3}[-.\s]?)?(\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})")
_YEARS_EXP_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)?",
    re.IGNORECASE,
)
_DATE_RANGE_RE = re.compile(
    r"(20\d{2}|19\d{2})\s*(?:-|–|to)\s*(20\d{2}|19\d{2}|present|current)",
    re.IGNORECASE,
)

_NOISE_ENTITY_LABELS = {
    "GPE", "LOC", "DATE", "TIME", "CARDINAL", "ORDINAL", "MONEY", "PERCENT",
}


@lru_cache(maxsize=1)
def _get_nlp():
    """Lazily load the spaCy model once per process."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:  # model not downloaded
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' is not installed. Run:\n"
            "    python -m spacy download en_core_web_sm"
        ) from exc


# ---------------------------------------------------------------------------
# CONTACT INFO
# ---------------------------------------------------------------------------
def extract_email(text: str) -> str:
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = _PHONE_RE.search(text)
    return match.group(0).strip() if match else ""


def extract_name(text: str, doc=None) -> str:
    """
    Heuristic: the first PERSON entity spaCy finds is usually the
    candidate's name, since resumes typically open with it.
    Falls back to the first non-empty line.
    """
    doc = doc or _get_nlp()(text[:1000])  # name appears near the top
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line[:60]
    return "Unknown"


# ---------------------------------------------------------------------------
# NAMED ENTITIES (generic)
# ---------------------------------------------------------------------------
def extract_entities(text: str) -> dict:
    """Return every entity spaCy recognizes, grouped by label."""
    doc = _get_nlp()(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, [])
        if ent.text.strip() not in entities[ent.label_]:
            entities[ent.label_].append(ent.text.strip())
    return entities


def extract_organizations(text: str, doc=None) -> list:
    doc = doc or _get_nlp()(text)
    return sorted({ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"})


# ---------------------------------------------------------------------------
# EDUCATION EXTRACTION
# ---------------------------------------------------------------------------
def extract_education(text: str) -> list:
    """
    Scan the text for known degree keywords and report the normalized
    degree names found, in the order first encountered.
    """
    lowered = text.lower()
    found = []
    for keyword, normalized in DEGREE_KEYWORDS.items():
        if keyword in lowered and normalized not in found:
            found.append(normalized)
    return found


# ---------------------------------------------------------------------------
# EXPERIENCE EXTRACTION
# ---------------------------------------------------------------------------
def extract_experience_years(text: str) -> float:
    """
    Estimate total years of experience.
    Strategy:
      1. Look for explicit phrases like "3+ years of experience" and take
         the maximum such figure (most resumes state this once, near the top).
      2. If none found, fall back to summing year-ranges found in an
         "experience" section (e.g. 2019-2021, 2021-present).
    """
    explicit_matches = [float(m.group(1)) for m in _YEARS_EXP_RE.finditer(text)]
    if explicit_matches:
        return max(explicit_matches)

    total_months = 0
    for match in _DATE_RANGE_RE.finditer(text):
        start = int(match.group(1))
        end_raw = match.group(2).lower()
        end = 2026 if end_raw in ("present", "current") else int(end_raw)
        if end >= start:
            total_months += (end - start) * 12
    return round(total_months / 12, 1)


# ---------------------------------------------------------------------------
# COMBINED EXTRACTION
# ---------------------------------------------------------------------------
def extract_all(text: str) -> dict:
    """Run every extractor once and return a single structured record."""
    doc = _get_nlp()(text)
    return {
        "name": extract_name(text, doc=doc),
        "email": extract_email(text),
        "phone": extract_phone(text),
        "organizations": extract_organizations(text, doc=doc),
        "education": extract_education(text),
        "experience_years": extract_experience_years(text),
        "entities": extract_entities(text),
    }
