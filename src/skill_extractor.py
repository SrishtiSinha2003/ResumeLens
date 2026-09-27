"""
skill_extractor.py
===================
Objective covered: "skill extraction"

Two complementary strategies:
    1. Taxonomy matching - fast, precise, whole-word regex match against a
       curated skill taxonomy (src/config.SKILL_TAXONOMY). This is the
       primary signal used everywhere else in the pipeline.
    2. Noun-chunk mining (spaCy) - a secondary, best-effort pass that
       surfaces additional candidate phrases which are not yet in the
       taxonomy, useful for spotting emerging tools/skills in a resume or
       job description.
"""

import re
from functools import lru_cache

import spacy

from src.config import ALL_SKILLS_SORTED, FLAT_SKILLS_TO_CATEGORY

_GENERIC_NOISE = {
    "experience", "candidate", "candidates", "team", "responsibility",
    "responsibilities", "requirement", "requirements", "role", "position",
    "company", "organization", "work", "working", "ability", "knowledge",
    "understanding", "degree", "bachelor", "master", "education", "project",
    "projects", "skill", "skills", "developer", "engineer", "application",
    "environment", "job", "description",
}


@lru_cache(maxsize=1)
def _get_nlp():
    return spacy.load("en_core_web_sm")


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 1. TAXONOMY MATCHING
# ---------------------------------------------------------------------------
def extract_skills(text: str) -> list:
    """Return every taxonomy skill found in `text` (longest match wins)."""
    normalized = _normalize(text)
    found = []
    for skill in ALL_SKILLS_SORTED:
        if len(skill) <= 1:
            # Single-letter skills (e.g. "c", "r") need a stricter boundary
            # so they don't match initials like "C." in "C. V. Raman University".
            pattern = r"(?<![a-z0-9.])" + re.escape(skill) + r"(?![a-z0-9.])"
        else:
            pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return found


def extract_skills_by_category(text: str) -> dict:
    """Same as extract_skills but grouped by taxonomy category."""
    grouped = {}
    for skill in extract_skills(text):
        category = FLAT_SKILLS_TO_CATEGORY.get(skill, "Other")
        grouped.setdefault(category, []).append(skill)
    return grouped


# ---------------------------------------------------------------------------
# 2. NOUN-CHUNK MINING (candidate phrases not yet in the taxonomy)
# ---------------------------------------------------------------------------
def extract_candidate_phrases(text: str, max_phrases: int = 15) -> list:
    doc = _get_nlp()(text)
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = _normalize(chunk.text)
        words = phrase.split()
        if not (1 <= len(words) <= 3):
            continue
        if phrase in _GENERIC_NOISE or not phrase:
            continue
        if any(w in _GENERIC_NOISE for w in words) and len(words) == 1:
            continue
        if phrase in FLAT_SKILLS_TO_CATEGORY:
            continue  # already captured by the taxonomy pass
        if phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= max_phrases:
            break
    return phrases
