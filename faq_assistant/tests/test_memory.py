"""
Unit tests for Conversation Memory and Follow-Up Resolution.
"""

import pytest
from faq_assistant.memory import ConversationMemory


def test_memory_add_and_limit():
    """Verify turns are properly appended and capped at MAX_TURNS."""
    mem = ConversationMemory(max_turns=3)
    mem.add_turn("Q1", "A1", "Attendance")
    mem.add_turn("Q2", "A2", "Examination")
    mem.add_turn("Q3", "A3", "Assignment")
    mem.add_turn("Q4", "A4", "General")

    turns = mem.get_turns()
    assert len(turns) == 3
    assert turns[0].query == "Q2"
    assert turns[2].query == "Q4"


def test_followup_resolution():
    """Verify follow-up inquiries inherit previous context."""
    mem = ConversationMemory()
    mem.add_turn(
        query="What are the computer laboratory rules?",
        response="No food or drinks are allowed in the computer lab.",
        intent="Course Content"
    )

    enriched = mem.resolve_followup_query("what about for labs?")
    assert "computer laboratory rules" in enriched
    assert "what about for labs?" in enriched


def test_memory_clear():
    """Verify memory clearing."""
    mem = ConversationMemory()
    mem.add_turn("Q", "A", "General")
    assert len(mem.get_turns()) == 1
    mem.clear()
    assert len(mem.get_turns()) == 0
