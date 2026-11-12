"""
Conversation Memory module for maintaining recent conversation context.
Allows multi-turn dialogue, follow-up query contextualization, and history formatting.
"""

import time
from typing import List, Dict, Any, Optional
from faq_assistant.config import MAX_HISTORY_TURNS


class Turn:
    """Represents a single conversational exchange."""
    def __init__(
        self,
        query: str,
        response: str,
        intent: str,
        retrieved_faq_ids: Optional[List[str]] = None,
        strategy: str = "zero-shot"
    ):
        self.query = query
        self.response = response
        self.intent = intent
        self.retrieved_faq_ids = retrieved_faq_ids or []
        self.strategy = strategy
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "intent": self.intent,
            "retrieved_faq_ids": self.retrieved_faq_ids,
            "strategy": self.strategy,
            "timestamp": self.timestamp
        }


class ConversationMemory:
    """
    Tracks recent dialogue history and resolves context-dependent follow-up questions.
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.max_turns = max_turns
        self.turns: List[Turn] = []

    def add_turn(
        self,
        query: str,
        response: str,
        intent: str,
        retrieved_faq_ids: Optional[List[str]] = None,
        strategy: str = "zero-shot"
    ):
        turn = Turn(
            query=query,
            response=response,
            intent=intent,
            retrieved_faq_ids=retrieved_faq_ids,
            strategy=strategy
        )
        self.turns.append(turn)
        # Keep only the last N turns
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_formatted_history(self, max_turns: Optional[int] = None) -> str:
        """
        Formats recent turns into a structured text dialogue block for prompt injection.
        """
        limit = max_turns or self.max_turns
        recent = self.turns[-limit:] if self.turns else []
        if not recent:
            return ""

        formatted_lines = []
        for i, turn in enumerate(recent, 1):
            formatted_lines.append(f"Turn {i}:")
            formatted_lines.append(f"Student: {turn.query}")
            formatted_lines.append(f"Assistant: {turn.response}")

        return "\n".join(formatted_lines)

    def resolve_followup_query(self, query: str) -> str:
        """
        Detects if a query is a short follow-up (e.g. 'what about for labs?', 'what is the penalty?')
        and enriches it with the prior turn's topic to ensure reliable retrieval.
        """
        q_clean = query.strip().lower()

        # Follow-up triggers
        followup_starters = [
            "what about", "how about", "and for", "what if", "can i", "is there",
            "what is the penalty", "how much", "why", "where do i go", "what about for"
        ]

        is_short = len(q_clean.split()) <= 6
        has_trigger = any(q_clean.startswith(trig) for trig in followup_starters)

        if (is_short or has_trigger) and self.turns:
            last_turn = self.turns[-1]
            enriched = f"{last_turn.query} -> {query}"
            return enriched

        return query

    def clear(self):
        """Reset conversation history."""
        self.turns.clear()

    def get_turns(self) -> List[Turn]:
        return list(self.turns)


# Default shared memory instance
default_memory = ConversationMemory()
