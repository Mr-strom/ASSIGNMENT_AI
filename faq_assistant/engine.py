"""
Core FAQ Assistant Engine orchestrating retrieval, intent detection,
prompt construction (zero-shot, few-shot, role), LLM invocation, and memory.
"""

from typing import Dict, Any, Optional, Tuple
from faq_assistant.config import DEFAULT_TEMPERATURE
from faq_assistant.llm import OllamaClient, LLMResult, default_client
from faq_assistant.retrieval import FAQRetriever, default_retriever
from faq_assistant.intent import classify_intent, IntentResult
from faq_assistant.memory import ConversationMemory, default_memory
from faq_assistant.prompts import (
    SYSTEM_PROMPT,
    ROLE_SYSTEM_PROMPT,
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_role_prompt
)


class FAQAssistantEngine:
    """
    Unified engine powering both the Streamlit UI and CLI interfaces.
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        retriever: Optional[FAQRetriever] = None,
        memory: Optional[ConversationMemory] = None
    ):
        self.client = client or default_client
        self.retriever = retriever or default_retriever
        self.memory = memory or default_memory

    def ask(
        self,
        query: str,
        strategy: str = "zero-shot",
        temperature: float = DEFAULT_TEMPERATURE,
        use_memory: bool = True,
        max_tokens: int = 140
    ) -> Dict[str, Any]:
        """
        Process a student question end-to-end:
        1. Query understanding & Intent classification
        2. Contextual follow-up enrichment
        3. FAQ retrieval & reliability check
        4. Prompt assembly according to chosen strategy
        5. LLM inference with timeout/retry and early stop sequences
        6. Unknown question safety verification
        7. Memory update
        """
        # 1. Intent Classification
        intent_res = classify_intent(query, client=self.client)

        # 2. Contextual follow-up resolution
        enriched_query = self.memory.resolve_followup_query(query) if use_memory else query

        # 3. FAQ Retrieval
        matches, is_reliable, max_score = self.retriever.retrieve(enriched_query)
        context_str = self.retriever.format_context(matches, is_reliable)

        # 4. Prompt Assembly
        history_str = self.memory.get_formatted_history() if use_memory else ""
        strategy_lower = strategy.lower().replace(" ", "-")

        if strategy_lower in ["few-shot", "fewshot"]:
            prompt = build_few_shot_prompt(query, context_str, history_str)
            active_system = SYSTEM_PROMPT
        elif strategy_lower in ["role", "role-prompting", "advisor"]:
            prompt = build_role_prompt(query, context_str, history_str)
            active_system = ROLE_SYSTEM_PROMPT
        else:  # default zero-shot
            strategy_lower = "zero-shot"
            prompt = build_zero_shot_prompt(query, context_str, history_str)
            active_system = SYSTEM_PROMPT

        # 5. LLM Generation with concise max_tokens and stop sequences
        stop_seqs = ["\nStudent:", "\n### Student", "\n### TASK", "\n--- Demonstration", "\n### CONVERSATION"]
        llm_res = self.client.generate(
            prompt=prompt,
            system_prompt=active_system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_seqs
        )

        response_text = llm_res.content

        # Fallback guard when retrieval finds no reliable policy and the model does not decline clearly
        official_decline_phrases = [
            "do not have reliable", "don't have reliable",
            "not available in the", "no direct official",
            "unable to find reliable", "official faq policy does not contain",
            "no direct official faq policies"
        ]
        has_official_decline = any(phrase in response_text.lower() for phrase in official_decline_phrases)

        if not is_reliable and not has_official_decline:
            response_text = (
                "I do not have reliable official information regarding this matter in the academic FAQ. "
                "Please contact your course instructor, the CS Department office, or Academic Advising (advising@evergreen.edu) for official assistance."
            )

        # Update conversation memory
        matched_ids = [m["id"] for m in matches] if is_reliable else []
        if use_memory and llm_res.success:
            self.memory.add_turn(
                query=query,
                response=response_text,
                intent=intent_res.category,
                retrieved_faq_ids=matched_ids,
                strategy=strategy_lower
            )

        return {
            "query": query,
            "response": response_text,
            "intent": intent_res.category,
            "intent_confidence": intent_res.confidence,
            "intent_method": intent_res.method,
            "strategy": strategy_lower,
            "is_reliable": is_reliable,
            "similarity_score": round(max_score, 4),
            "retrieved_faqs": matches,
            "latency_seconds": round(llm_res.latency_seconds, 3),
            "model_name": llm_res.model_name,
            "success": llm_res.success,
            "error_message": llm_res.error_message
        }


# Default shared engine instance
default_engine = FAQAssistantEngine()
