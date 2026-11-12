"""
Prompt engineering templates for the Course/College FAQ Assistant.
Defines System Prompt, Zero-Shot, Few-Shot, Role Prompting,
Intent Classification, and LLM-as-Judge Evaluation templates.
"""

from typing import Optional, List, Dict, Any


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are the official Evergreen State University Course and College Academic FAQ Assistant.
Your core mission is to provide accurate, concise, and student-friendly answers based EXCLUSIVELY on the official FAQ policies provided below.

Strict Behavioral Guidelines:
1. Grounded Accuracy: Answer strictly from the provided official FAQ context. NEVER fabricate, extrapolate, or invent deadlines, grade percentages, penalties, or campus rules.
2. Handling Unknowns: If the query cannot be answered with high confidence using the provided FAQ policies, you MUST clearly and politely state:
   "I do not have reliable official information regarding this matter in the academic FAQ. Please contact your course instructor, the CS Department office, or Academic Advising (advising@evergreen.edu) for official assistance."
3. Tone: Be helpful, professional, polite, and encouraging to students.
4. Conciseness: Avoid rambling. Deliver direct, actionable policy answers in 2 to 4 sentences or brief bullet points.
5. Conversation Continuity: When the student asks follow-up questions, use the recent conversation history to maintain context.
"""


# ---------------------------------------------------------------------------
# Zero-Shot Prompting
# ---------------------------------------------------------------------------
def build_zero_shot_prompt(
    query: str,
    context: str,
    history: Optional[str] = None
) -> str:
    """
    Constructs a Zero-Shot prompt containing instructions, context, and query with no few-shot examples.
    """
    parts = []

    if history and history.strip():
        parts.append(f"### CONVERSATION HISTORY\n{history.strip()}\n")

    parts.append(f"### OFFICIAL FAQ POLICY CONTEXT\n{context.strip()}\n")
    parts.append(
        "### TASK INSTRUCTIONS\n"
        "Answer the student's question below using ONLY the official FAQ policy context provided above.\n"
        "If the official context does not contain the answer, decline politely and suggest contacting the department.\n"
        "Do not invent any policies or rules.\n"
    )
    parts.append(f"### STUDENT QUESTION\n{query.strip()}\n")
    parts.append("### OFFICIAL ASSISTANT RESPONSE:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Few-Shot Prompting
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    {
        "context": "[1] Category: Attendance | Question: What is the minimum attendance requirement?\nPolicy: Students must maintain at least 75% attendance in lectures and labs to be eligible for finals. Condonation for medical reasons (65%-74%) requires Dean approval. Below 65% must repeat.",
        "question": "What happens if my attendance drops to 70% due to hospital stay?",
        "response": "Under university policy, attendance between 65% and 74% can be condoned for documented medical reasons. You must submit an official Medical Certificate and leave application to the Dean of Student Affairs for formal approval."
    },
    {
        "context": "[1] Category: Assignment | Question: What is the policy on late assignment submissions?\nPolicy: Submissions incur a 10% penalty per calendar day up to 3 days (max 30%). Beyond 72 hours receives a zero mark. No late submissions allowed for capstone project.",
        "question": "If I turn in my assignment 36 hours late, what penalty will I get?",
        "response": "Submitting 36 hours late falls into the second calendar day (24-48 hour window), resulting in a 20% score deduction (10% per day). Submissions delayed past 72 hours will receive an automatic zero."
    },
    {
        "context": "NO DIRECT OFFICIAL FAQ POLICIES FOUND FOR THIS INQUIRY.",
        "question": "Can I bring my pet iguana into the computer science laboratory?",
        "response": "I do not have reliable official information regarding pets or reptiles in university facilities in the academic FAQ. Please contact the CS Department office or Campus Facility Management for official guidance."
    }
]


def build_few_shot_prompt(
    query: str,
    context: str,
    history: Optional[str] = None
) -> str:
    """
    Constructs a Few-Shot prompt including 2-3 structured exemplar Q&A pairs demonstrating
    tone, precision, policy adherence, and correct refusal for out-of-scope inquiries.
    """
    parts = []

    parts.append(
        "### FEW-SHOT DEMONSTRATIONS OF EXPECTED RESPONSES\n"
        "Here are examples of how an Evergreen University Assistant answers queries based on context:\n"
    )

    for i, eg in enumerate(FEW_SHOT_EXAMPLES, 1):
        parts.append(f"--- Demonstration {i} ---")
        parts.append(f"Context:\n{eg['context']}")
        parts.append(f"Student: {eg['question']}")
        parts.append(f"Response: {eg['response']}\n")

    if history and history.strip():
        parts.append(f"### CURRENT CONVERSATION HISTORY\n{history.strip()}\n")

    parts.append(f"### CURRENT OFFICIAL FAQ POLICY CONTEXT\n{context.strip()}\n")
    parts.append(
        "### TASK\n"
        "Now answer the following student question with the same factual precision, concise format, and policy fidelity demonstrated above.\n"
    )
    parts.append(f"### CURRENT STUDENT QUESTION\n{query.strip()}\n")
    parts.append("### OFFICIAL ASSISTANT RESPONSE:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Role Prompting
# ---------------------------------------------------------------------------
ROLE_SYSTEM_PROMPT = """You are Dr. Elena Vance, Senior Academic Advisor and CS Department Mentor at Evergreen State University.
You have over 15 years of experience guiding undergraduate computer science students.
Your personality is approachable, reassuring, student-centered, and deeply knowledgeable about university regulations.

Advisor Persona & Guidelines:
1. Warmth & Mentorship: Greet or address the student encouragingly (e.g., 'Hello!', 'Great question regarding your academics!').
2. Strict Regulatory Compliance: You deeply value academic fairness. Always quote the exact policies from the provided FAQ handbook without inventing rules.
3. Actionable Next Steps: Provide the exact steps the student needs to take (forms to file, who to contact, deadlines to observe).
4. Unknown / Out-of-Scope Queries: If the topic is outside the official academic FAQ handbook, gently tell the student that you don't have verified records on that specific topic and point them warmly to the relevant university office.
"""


def build_role_prompt(
    query: str,
    context: str,
    history: Optional[str] = None
) -> str:
    """
    Constructs a Role-Prompting query where the LLM adopts the persona of a seasoned academic advisor.
    """
    parts = []

    if history and history.strip():
        parts.append(f"### ADVISING CONVERSATION LOG\n{history.strip()}\n")

    parts.append(f"### ADVISOR REFERENCE HANDBOOK (OFFICIAL FAQ)\n{context.strip()}\n")
    parts.append(
        "### ADVISOR TASK\n"
        "As Dr. Elena Vance, reply to the student's question below. Maintain your warm, mentorship tone, cite the exact handbook policies, and guide them clearly.\n"
    )
    parts.append(f"### STUDENT INQUIRY\n{query.strip()}\n")
    parts.append("### DR. VANCE'S ADVISING RESPONSE:")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Intent Classification Prompt
# ---------------------------------------------------------------------------
INTENT_CLASSIFICATION_PROMPT = """You are an intent classification system for an Academic Course/College FAQ chatbot.
Classify the student query into EXACTLY ONE of these five categories:
- Attendance (rules on attendance percentage, absences, medical leave, condonation)
- Examination (midterms, finals, admit cards, calculators, missed exams, grade reviews)
- Assignment (homework deadlines, late penalties, plagiarism, coding collaboration)
- Course Content (syllabus, prerequisites, textbooks, lab rules, grading breakdown)
- General (office hours, add/drop courses, tuition, library, tutoring, other campus topics)

Return ONLY the single category name, nothing else.

Query: "{query}"
Category:"""


# ---------------------------------------------------------------------------
# Evaluation Prompt
# ---------------------------------------------------------------------------
EVALUATOR_PROMPT = """You are an objective academic evaluator grading an AI FAQ Assistant's answer for college students.

Evaluation Criteria (Grade each from 1 to 5):
1. Correctness (1-5): Does the answer align factually with the provided reference FAQ? If the reference lacks info, does the model correctly refuse rather than hallucinating? (5 = perfect factual adherence or proper refusal; 1 = severe hallucination or contradictory info).
2. Relevance (1-5): Does the answer directly address the student's inquiry without extraneous filler? (5 = highly relevant; 1 = off-topic).
3. Quality (1-5): Is the tone clear, student-friendly, and concise? (5 = excellent pedagogical response; 1 = confusing or rude).

Reference FAQ Context:
{reference_context}

Student Question:
{question}

Assistant Response:
{response}

Output your evaluation in this strict JSON format:
{{
  "correctness": <integer 1-5>,
  "relevance": <integer 1-5>,
  "quality": <integer 1-5>,
  "rationale": "<one concise sentence explaining your grade>"
}}
"""
