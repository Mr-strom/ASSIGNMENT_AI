"""
Unit tests for the FAQ Assistant Retrieval module.
"""

import pytest
from faq_assistant.retrieval import FAQRetriever, default_retriever


def test_faq_loading():
    """Verify that all FAQs are loaded and indexed."""
    retriever = default_retriever
    assert len(retriever.faqs) >= 12
    assert retriever.tfidf_matrix is not None


def test_in_scope_retrieval():
    """Verify that in-scope queries retrieve the correct FAQ with high confidence."""
    retriever = default_retriever

    # Attendance query
    matches, is_reliable, score = retriever.retrieve("What is the minimum attendance requirement?")
    assert is_reliable is True
    assert score > retriever.threshold
    assert matches[0]["category"] == "Attendance"
    assert "75%" in matches[0]["answer"]

    # Assignment penalty query
    matches, is_reliable, score = retriever.retrieve("How much penalty for late assignment?")
    assert is_reliable is True
    assert matches[0]["category"] == "Assignment"
    assert "10%" in matches[0]["answer"]


def test_out_of_scope_rejection():
    """Verify that irrelevant questions get flagged as unreliable."""
    retriever = default_retriever
    matches, is_reliable, score = retriever.retrieve("What is the recipe for chocolate chip cookies in the dining hall?")
    assert is_reliable is False
    assert score < retriever.threshold


def test_context_formatting():
    """Verify that formatted context adheres to the expected prompt structure."""
    retriever = default_retriever
    matches, is_reliable, _ = retriever.retrieve("What is the minimum attendance?")
    context = retriever.format_context(matches, is_reliable)
    assert "OFFICIAL EVERGREEN UNIVERSITY FAQ POLICY" in context
    assert "Attendance" in context
