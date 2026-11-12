"""
Unit tests for the Query Understanding & Intent Classification module.
"""

import pytest
from faq_assistant.intent import classify_intent_rule_based, VALID_CATEGORIES


def test_intent_categories():
    """Verify that all 5 required categories are recognized."""
    cases = [
        ("What is the minimum attendance to avoid repeating a course?", "Attendance"),
        ("What can I bring into the midterm exam room?", "Examination"),
        ("What is the deadline penalty for late homework submissions?", "Assignment"),
        ("What are the prerequisites and textbook for Data Structures?", "Course Content"),
        ("Where can I meet the professor for office hours?", "General"),
        ("How do I borrow books from the campus library?", "General"),
        ("What are the lab safety rules for closed-toe shoes?", "Course Content"),
        ("How can I request a grade re-evaluation for my test?", "Examination")
    ]

    for query, expected_cat in cases:
        res = classify_intent_rule_based(query)
        assert res.category == expected_cat, f"Query '{query}' expected {expected_cat}, got {res.category}"
        assert res.category in VALID_CATEGORIES
