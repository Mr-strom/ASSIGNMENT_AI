"""
Query Understanding & Intent Classification module.
Combines LLM-based classification with a high-precision, weighted rule-based fallback
to categorize student queries into:
- Attendance
- Examination
- Assignment
- Course Content
- General
"""

import re
from typing import Optional, Dict, Any, List
from faq_assistant.llm import default_client, OllamaClient
from faq_assistant.prompts import INTENT_CLASSIFICATION_PROMPT

VALID_CATEGORIES = [
    "Attendance",
    "Examination",
    "Assignment",
    "Course Content",
    "General"
]

# Weighted regex patterns: (pattern, weight)
# High-specificity keywords have higher weight to ensure primary topic takes precedence
INTENT_WEIGHTED_PATTERNS = {
    "Attendance": [
        (r"\b(minimum\s+)?attendance\b", 5.0),
        (r"\battend(ance|ing)?\b", 4.5),
        (r"\babsent\b", 4.0),
        (r"\babsence(s)?\b", 4.0),
        (r"\bmiss(ed)?\s+([a-zA-Z0-9%]+\s+)*class(es)?\b", 4.5),
        (r"\bmedical leave\b", 4.5),
        (r"\bcondon(e|ation)\b", 4.5),
        (r"\b75%\b", 3.0),
        (r"\b65%\b", 3.0),
        (r"\bshortage\b", 3.0),
        (r"\bbunk\b", 3.0),
        (r"\bsick\b", 2.5)
    ],
    "Assignment": [
        (r"\bassign(ment|ments)?\b", 5.0),
        (r"\bhomework\b", 4.5),
        (r"\blate penalty\b", 4.5),
        (r"\bsubmi(t|ssion|ssions)\b", 4.0),
        (r"\bdeadline\b", 3.5),
        (r"\bdue date\b", 3.5),
        (r"\bplagiarism\b", 4.5),
        (r"\bcheating\b", 3.5),
        (r"\bmoss\b", 3.5),
        (r"\bcollaboration\b", 3.0),
        (r"\bturnin\b", 3.5),
        (r"\bmidnight\b", 2.5),
        (r"\bleniency\b", 3.0)
    ],
    "Course Content": [
        (r"\b(letter\s+)?grade(s)?\s+(is\s+|are\s+)?(determined|calculated)\b", 6.0),
        (r"\bgrading\s+(breakdown|weight|distribution)\b", 5.0),
        (r"\b(course\s+)?grade\b", 3.5),
        (r"\bsyllabus\b", 5.0),
        (r"\bprereq(uisite|uisites)?\b", 5.0),
        (r"\btextbook(s)?\b", 5.0),
        (r"\bdata structures\b", 4.0),
        (r"\bcs101\b", 4.0),
        (r"\bcs102\b", 4.0),
        (r"\blab(s|oratory|oratories)?\b", 3.5),
        (r"\bembedded\b", 3.5),
        (r"\brobotics\b", 3.5),
        (r"\bsafety glasses\b", 4.0),
        (r"\bclosed-toe shoes\b", 4.0),
        (r"\blecture notes\b", 3.5)
    ],
    "Examination": [
        (r"\bexam\s+hall\b", 4.5),
        (r"\badmit card\b", 5.0),
        (r"\bhall ticket\b", 5.0),
        (r"\bcalculator(s)?\b", 4.0),
        (r"\bmake-?up exam\b", 5.0),
        (r"\bre-?eval(uation)?\b", 5.0),
        (r"\bregrade\b", 5.0),
        (r"\bgrievance\b", 4.0),
        (r"\btest date\b", 4.0),
        (r"\bmidterm(s)?\b", 3.0),
        (r"\b(final\s+)?exam(s|ination|inations)?\b", 2.5)
    ],
    "General": [
        (r"\boffice hour(s)?\b", 5.0),
        (r"\bteaching assistant\b", 4.0),
        (r"\badd drop\b", 5.0),
        (r"\bwithdraw(al)?\b", 4.0),
        (r"\btuition\b", 4.0),
        (r"\bfee(s)?\b", 3.5),
        (r"\blibrary\b", 5.0),
        (r"\bborrow books\b", 5.0),
        (r"\boverdue\b", 4.0),
        (r"\btutor(ing)?\b", 4.0),
        (r"\bacquiring\b", 2.0),
        (r"\bdorm(itory|s)?\b", 4.0),
        (r"\bpet\b", 4.0),
        (r"\brecipe\b", 4.0),
        (r"\bfirewall\b", 4.0),
        (r"\bvideo games\b", 4.0),
        (r"\bprofessor\b", 2.5),
        (r"\binstructor\b", 2.5),
        (r"\bta\b", 2.5)
    ]
}


class IntentResult:
    def __init__(self, category: str, method: str, confidence: float = 1.0, raw_llm_response: Optional[str] = None):
        self.category = category
        self.method = method
        self.confidence = confidence
        self.raw_llm_response = raw_llm_response

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "method": self.method,
            "confidence": self.confidence,
            "raw_llm_response": self.raw_llm_response
        }


def classify_intent_rule_based(query: str) -> IntentResult:
    """Classifies query using weighted keyword and regex heuristics."""
    q_lower = query.lower()
    scores = {cat: 0.0 for cat in VALID_CATEGORIES}

    for cat, weighted_patterns in INTENT_WEIGHTED_PATTERNS.items():
        for pat, weight in weighted_patterns:
            if re.search(pat, q_lower):
                scores[cat] += weight

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    if best_score > 0.0:
        conf = min(0.65 + (best_score * 0.06), 0.98)
        return IntentResult(category=best_cat, method="rule-based", confidence=round(conf, 2))

    return IntentResult(category="General", method="rule-based-default", confidence=0.50)


def classify_intent(
    query: str,
    client: Optional[OllamaClient] = None,
    use_llm: bool = True
) -> IntentResult:
    """
    Classifies the user query's intent. First checks rule-based with high confidence,
    or queries the LLM and cross-references.
    """
    rule_res = classify_intent_rule_based(query)

    # If rule-based has high confidence, fast-track it!
    if rule_res.confidence >= 0.80:
        return rule_res

    if not use_llm:
        return rule_res

    llm = client or default_client

    try:
        prompt = INTENT_CLASSIFICATION_PROMPT.format(query=query)
        result = llm.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=12
        )

        if result.success and result.content:
            raw_text = result.content.strip()
            for cat in VALID_CATEGORIES:
                if re.search(rf"\b{re.escape(cat)}\b", raw_text, re.IGNORECASE):
                    return IntentResult(
                        category=cat,
                        method="llm",
                        confidence=0.92,
                        raw_llm_response=raw_text
                    )

        # Fallback to rule-based
        rule_res.raw_llm_response = result.content if result.success else result.error_message
        return rule_res

    except Exception:
        return rule_res
