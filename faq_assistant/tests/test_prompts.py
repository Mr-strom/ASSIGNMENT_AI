"""
Unit tests for Prompt construction and strategy templates.
"""

import pytest
from faq_assistant.prompts import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_role_prompt,
    SYSTEM_PROMPT,
    ROLE_SYSTEM_PROMPT
)


def test_zero_shot_prompt():
    """Verify zero shot prompt contains instructions, context, and query."""
    p = build_zero_shot_prompt(
        query="What is the attendance policy?",
        context="Min 75% attendance.",
        history="Student: Hi\nAssistant: Hello"
    )
    assert "CONVERSATION HISTORY" in p
    assert "OFFICIAL FAQ POLICY CONTEXT" in p
    assert "Min 75% attendance" in p
    assert "What is the attendance policy?" in p


def test_few_shot_prompt():
    """Verify few shot prompt includes demonstrations."""
    p = build_few_shot_prompt(
        query="When is the midterm?",
        context="Midterms take place Week 8."
    )
    assert "FEW-SHOT DEMONSTRATIONS" in p
    assert "Demonstration 1" in p
    assert "Demonstration 2" in p
    assert "Midterms take place Week 8" in p


def test_role_prompt():
    """Verify role prompt invokes Dr. Vance advisor persona."""
    p = build_role_prompt(
        query="Can I drop a course?",
        context="Drop allowed till Week 7."
    )
    assert "ADVISOR REFERENCE HANDBOOK" in p
    assert "DR. VANCE" in p
    assert "Dr. Elena Vance" in ROLE_SYSTEM_PROMPT
