"""
tests/test_core.py
===================
Unit tests for the parts of the pipeline that do not require downloading
the spaCy / sentence-transformer models, so they run quickly in any
environment (including CI without internet access).

Run with:  pytest -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, tokenize, remove_stopwords, preprocess_pipeline
from src.skill_extractor import extract_skills, extract_skills_by_category
from src.similarity import tfidf_similarity, word2vec_similarity


def test_clean_text_lowercases_and_strips_symbols():
    assert clean_text("Hello, World!!  NLP@2026") == "hello world nlp 2026"


def test_tokenize_and_stopword_removal():
    tokens = tokenize("The quick brown fox jumps over the lazy dog")
    filtered = remove_stopwords(tokens)
    assert "the" not in filtered
    assert "over" not in filtered
    assert "quick" in filtered


def test_preprocess_pipeline_shape():
    result = preprocess_pipeline("I have 5 years of experience with Python and Java.")
    assert set(result.keys()) == {
        "raw_text", "cleaned_text", "tokens", "filtered_tokens", "filtered_text",
    }
    assert "python" in result["filtered_text"]


def test_extract_skills_finds_known_technologies():
    text = "Experienced with Java, Spring Boot, MySQL and Docker on AWS."
    skills = extract_skills(text)
    assert "java" in skills
    assert "spring boot" in skills
    assert "mysql" in skills
    assert "docker" in skills
    assert "aws" in skills


def test_extract_skills_prefers_longest_match():
    # "spring boot" should be matched, not just "spring"
    skills = extract_skills("I use spring boot for backend services")
    assert "spring boot" in skills


def test_extract_skills_by_category_groups_correctly():
    grouped = extract_skills_by_category("Java, MySQL, Docker, Git")
    assert "java" in grouped.get("Programming Languages", [])
    assert "mysql" in grouped.get("Databases", [])
    assert "docker" in grouped.get("DevOps / Cloud", [])
    assert "git" in grouped.get("Version Control", [])


def test_tfidf_similarity_identical_documents():
    text = "python developer with machine learning experience"
    assert tfidf_similarity(text, text) > 0.99


def test_tfidf_similarity_unrelated_documents_is_low():
    a = "python machine learning data science"
    b = "gardening cooking recipes travel"
    assert tfidf_similarity(a, b) < 0.3


def test_word2vec_similarity_runs_and_returns_bounded_score():
    a = "python developer with machine learning and nlp experience"
    b = "looking for a python engineer skilled in nlp and machine learning"
    score = word2vec_similarity(a, b)
    assert 0.0 <= score <= 1.0
