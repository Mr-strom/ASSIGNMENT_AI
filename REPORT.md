# Project Report: Course & College FAQ Assistant

**Author:** Mr-strom  
**Repository:** `Mr-strom/ASSIGNMENT_AI`

---

## 1. Objective

The goal of this project is to build a local LLM-based Course and College FAQ Assistant using Python, Streamlit, and Ollama, and to compare how different **prompt engineering strategies** (**Zero-Shot**, **Few-Shot**, and **Role Prompting**) affect factual accuracy, response quality, consistency, and refusal behavior on unknown questions.

Instead of relying on cloud APIs, the assistant runs locally on CPU hardware using `qwen2.5:0.5b` via Ollama, backed by TF-IDF retrieval over a structured JSON FAQ knowledge base (`faq_assistant/faq_data.json`).

---

## 2. System Pipeline

```text
Student Query
     │
     ▼
Intent Classifier (faq_assistant/intent.py)
  - Weighted regex rules across 5 categories, with LLM fallback
     │
     ▼
Conversation Memory (faq_assistant/memory.py)
  - Keeps recent N turns and expands short follow-up questions
     │
     ▼
TF-IDF + Keyword Retriever (faq_assistant/retrieval.py)
  - 14 college FAQs, threshold = 0.26
  - Score >= 0.26 -> inject matching FAQ entries into prompt
  - Score < 0.26  -> pass "no direct official FAQ found" + fallback guard
     │
     ▼
Prompt Builder (faq_assistant/prompts.py)
  - Zero-Shot / Few-Shot / Role (Dr. Elena Vance)
     │
     ▼
Local Ollama Inference (faq_assistant/llm.py)
  - qwen2.5:0.5b, temperature 0.2, max 140 tokens, stop sequences
     │
     ▼
Final Response (Streamlit UI / CLI / Evaluation CSV & Markdown)
```

---

## 3. Prompting Strategies Implemented

### 3.1 Zero-Shot Prompting
Provides the system instructions, conversation history (if any), and retrieved FAQ entries directly without example Q&A pairs. It asks the model to answer concisely using only the provided policy text and decline if the context does not answer the question.

### 3.2 Few-Shot Prompting
Prepends 3 example Q&A demonstrations before the student's actual question:
1. An in-scope attendance condonation example showing exact rule extraction.
2. An assignment late-penalty calculation example (`36 hours -> 20%`).
3. An out-of-scope refusal example (`pet iguana in lab`) showing how to decline politely and refer the student to a campus office.

### 3.3 Role Prompting
Assigns the persona of **Dr. Elena Vance**, a senior Computer Science academic advisor. The system prompt guides the model to reply warmly, explain regulations clearly in steps, and refer students to the right office if a policy is not in the handbook.

---

## 4. Evaluation Setup

The script `faq_assistant/evaluate.py` tests **12 queries** across 4 categories:
- **In-FAQ (Q01–Q04):** Direct questions matching stored policies (attendance, late submission, exam hall rules, CS102 prerequisites).
- **Paraphrased (Q05–Q07):** Indirect wording testing semantic/keyword matching (missing 30% of classes, 2 hours past midnight, letter grade weighting).
- **Follow-Up (Q08–Q09):** Context-dependent questions relying on the previous turn (*"What about for embedded robotics labs?"*, *"What if I return them overdue?"*).
- **Unknown / Out-of-Scope (Q10–Q12):** Questions with no policy basis (pet python in dorms, dining hall recipe, bypassing campus firewall) that should be declined.

Each query is run **3 times** per strategy (108 runs total) and scored on a 1–5 scale for:
- **Correctness:** Presence of expected policy facts (or proper refusal for unknown queries).
- **Relevance:** Directness and reasonable response length.
- **Consistency:** Pairwise Jaccard similarity across the 3 runs after stopword filtering.
- **Quality:** Tone and readability heuristics.

---

## 5. Results Summary

Saved in [results/evaluation_summary.md](results/evaluation_summary.md) and [results/evaluation_results.csv](results/evaluation_results.csv):

| Prompting Strategy | Correctness (1-5) | Relevance (1-5) | Consistency (1-5) | Quality (1-5) | **Overall Score (1-5)** | Intent Accuracy | Unknown Declined | Hallucination Rate | Avg Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 4.79 | 4.93 | 4.44 | 4.60 | **4.69** | 100.0% | 100.0% | 0.0% | 4.91s |
| **Few-Shot** | 4.88 | 4.93 | 4.25 | 4.50 | **4.64** | 100.0% | 100.0% | 0.0% | 4.96s |
| **Role** | 4.54 | 4.77 | 3.72 | 4.75 | **4.44** | 100.0% | 100.0% | 0.0% | 5.36s |

### Per-Query Correctness Breakdown

| Query ID | Type | Query Text | Zero-Shot | Few-Shot | Role | Intent Matched |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Q01 | in-faq | What is the minimum attendance required to be eligible for final exams? | 5.0 | 5.0 | 5.0 | Yes |
| Q02 | in-faq | What is the penalty for submitting an assignment late? | 3.5 | 5.0 | 2.5 | Yes |
| Q03 | in-faq | What items are allowed in the exam hall during midterms? | 5.0 | 5.0 | 2.5 | Yes |
| Q04 | in-faq | What are the prerequisites and recommended textbooks for CS102 Data Structures? | 5.0 | 5.0 | 5.0 | Yes |
| Q05 | paraphrased | If I am sick and miss 30% of classes, can I still take the end semester tests? | 4.5 | 5.0 | 5.0 | Yes |
| Q06 | paraphrased | Is there any leniency if my homework code is uploaded two hours past midnight? | 5.0 | 5.0 | 5.0 | Yes |
| Q07 | paraphrased | How is our final letter grade determined across homework, midterms, and finals? | 5.0 | 5.0 | 5.0 | Yes |
| Q08 | follow-up | What about for embedded robotics labs? | 5.0 | 5.0 | 5.0 | Yes |
| Q09 | follow-up | What if I return them overdue? | 4.5 | 3.5 | 4.5 | Yes |
| Q10 | unknown | Can I bring my pet python to live with me in the freshman dormitory? | 5.0 | 5.0 | 5.0 | Yes |
| Q11 | unknown | What is the secret recipe for the dining hall's chicken parmesan? | 5.0 | 5.0 | 5.0 | Yes |
| Q12 | unknown | How do I bypass the campus firewall to torrent video games in the dorms? | 5.0 | 5.0 | 5.0 | Yes |

---

## 6. Comparison Observations & Limitations

1. **Zero-Shot vs. Few-Shot vs. Role:**
   - **Zero-Shot** had the highest overall score (**4.69**), best consistency (**4.44**), and lowest average latency (**4.91s**), making it the most dependable default for factual policy lookups.
   - **Few-Shot** scored highest on correctness (**4.88**) because the demonstration examples showed the small 0.5B model how to extract concrete numbers (like `10%` and `72 hours` on Q02) and when to refuse.
   - **Role Prompting** scored highest on quality (**4.75**) for tone and helpfulness, but varied more in wording across runs (consistency **3.72**) and occasionally omitted minor details (Q02, Q03) while writing conversational openings.

2. **Unknown Query Refusal:**
   - Combining the TF-IDF similarity threshold (`0.26`) with the fallback refusal guard in `engine.py` led to a **100% refusal rate** across Q10–Q12 with **0% fabricated policies**.

3. **Limitations:**
   - `qwen2.5:0.5b` is small, so complex numerical reasoning or multi-hop synthesis depends heavily on clean context retrieval.
   - TF-IDF + keyword boosting requires relevant synonyms in `faq_data.json` for unusual student phrasing.
   - The evaluation metrics are automated heuristics (keyword presence and token overlap) rather than manual human grading.
