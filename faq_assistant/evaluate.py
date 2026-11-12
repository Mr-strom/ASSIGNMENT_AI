"""
Comprehensive Evaluation Suite for the Course/College FAQ Assistant.
Runs 12 test queries across Zero-Shot, Few-Shot, and Role Prompting.
Calculates Correctness, Relevance, Consistency (3x runs), and Response Quality (1-5).
Saves CSV, Markdown summary, and detailed evaluation artifacts to results/.
"""

import os
import json
import time
import re
import argparse
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

from faq_assistant.config import RESULTS_DIR, DEFAULT_TEMPERATURE
from faq_assistant.engine import FAQAssistantEngine
from faq_assistant.memory import ConversationMemory
from faq_assistant.llm import OllamaClient, default_client
from faq_assistant.prompts import EVALUATOR_PROMPT

# ---------------------------------------------------------------------------
# Test Query Suite (12 Comprehensive Queries)
# ---------------------------------------------------------------------------
TEST_QUERIES = [
    # 1-4: In-FAQ Queries
    {
        "id": "Q01",
        "type": "in-faq",
        "query": "What is the minimum attendance required to be eligible for final exams?",
        "expected_intent": "Attendance",
        "expected_keywords": ["75%", "attendance", "examination", "final"],
        "is_unknown": False,
        "context_turn": None
    },
    {
        "id": "Q02",
        "type": "in-faq",
        "query": "What is the penalty for submitting an assignment late?",
        "expected_intent": "Assignment",
        "expected_keywords": ["10%", "penalty", "day", "deadline", "72 hours"],
        "is_unknown": False,
        "context_turn": None
    },
    {
        "id": "Q03",
        "type": "in-faq",
        "query": "What items are allowed in the exam hall during midterms?",
        "expected_intent": "Examination",
        "expected_keywords": ["calculator", "id card", "admit card", "prohibited"],
        "is_unknown": False,
        "context_turn": None
    },
    {
        "id": "Q04",
        "type": "in-faq",
        "query": "What are the prerequisites and recommended textbooks for CS102 Data Structures?",
        "expected_intent": "Course Content",
        "expected_keywords": ["cs101", "goodrich", "python", "prerequisite"],
        "is_unknown": False,
        "context_turn": None
    },

    # 5-7: Paraphrased Queries
    {
        "id": "Q05",
        "type": "paraphrased",
        "query": "If I am sick and miss 30% of classes, can I still take the end semester tests?",
        "expected_intent": "Attendance",
        "expected_keywords": ["attendance", "75%", "medical", "condonation"],
        "is_unknown": False,
        "context_turn": None
    },
    {
        "id": "Q06",
        "type": "paraphrased",
        "query": "Is there any leniency if my homework code is uploaded two hours past midnight?",
        "expected_intent": "Assignment",
        "expected_keywords": ["10%", "penalty", "day", "calendar"],
        "is_unknown": False,
        "context_turn": None
    },
    {
        "id": "Q07",
        "type": "paraphrased",
        "query": "How is our final letter grade determined across homework, midterms, and finals?",
        "expected_intent": "Course Content",
        "expected_keywords": ["20%", "25%", "35%", "passing", "grade"],
        "is_unknown": False,
        "context_turn": None
    },

    # 8-9: Follow-up Queries (Context Dependent)
    {
        "id": "Q08",
        "type": "follow-up",
        "query": "What about for embedded robotics labs?",
        "expected_intent": "Course Content",
        "expected_keywords": ["shoes", "safety", "closed-toe", "glasses"],
        "is_unknown": False,
        "context_turn": {
            "query": "What are the rules and timings for the computer laboratory?",
            "response": "Computer laboratories are open from 8:00 AM to 9:00 PM on weekdays. Food and drinks are forbidden, and students must log in using university credentials."
        }
    },
    {
        "id": "Q09",
        "type": "follow-up",
        "query": "What if I return them overdue?",
        "expected_intent": "General",
        "expected_keywords": ["fine", "$0.50", "day", "overdue"],
        "is_unknown": False,
        "context_turn": {
            "query": "How many books can undergraduate students borrow from the campus library?",
            "response": "Undergraduates can borrow up to 6 physical books for 21 days with one online renewal."
        }
    },

    # 10-12: Out-of-Scope / Unknown Queries (MUST be declined!)
    {
        "id": "Q10",
        "type": "unknown",
        "query": "Can I bring my pet python to live with me in the freshman dormitory?",
        "expected_intent": "General",
        "expected_keywords": ["do not have reliable", "contact", "official information"],
        "is_unknown": True,
        "context_turn": None
    },
    {
        "id": "Q11",
        "type": "unknown",
        "query": "What is the secret recipe for the dining hall's chicken parmesan?",
        "expected_intent": "General",
        "expected_keywords": ["do not have reliable", "contact", "official information"],
        "is_unknown": True,
        "context_turn": None
    },
    {
        "id": "Q12",
        "type": "unknown",
        "query": "How do I bypass the campus firewall to torrent video games in the dorms?",
        "expected_intent": "General",
        "expected_keywords": ["do not have reliable", "contact", "official information"],
        "is_unknown": True,
        "context_turn": None
    }
]

DECLINE_PATTERNS = [
    r"do not have reliable",
    r"don't have reliable",
    r"no direct official",
    r"not available in the",
    r"please contact",
    r"unable to find",
    r"cannot confirm",
    r"official assistance"
]


def check_refusal(response_text: str) -> bool:
    """Checks whether the response cleanly declines to answer an unknown query."""
    text_lower = response_text.lower()
    return any(re.search(pat, text_lower) for pat in DECLINE_PATTERNS)


def score_response(
    query_item: Dict[str, Any],
    response_text: str,
    latency: float,
    client: Optional[OllamaClient] = None
) -> Dict[str, float]:
    """
    Computes heuristic and rubric scores (1 to 5) for correctness, relevance, and quality.
    """
    is_unknown = query_item["is_unknown"]
    resp_lower = response_text.lower()

    if is_unknown:
        refused = check_refusal(response_text)
        if refused:
            correctness = 5.0
            relevance = 5.0
            quality = 4.8
            hallucinated = 0.0
        else:
            correctness = 1.0
            relevance = 2.0
            quality = 2.0
            hallucinated = 1.0

        return {
            "correctness": correctness,
            "relevance": relevance,
            "quality": quality,
            "hallucinated": hallucinated
        }

    # For known queries: check factual keywords
    kw_hits = 0
    total_kw = len(query_item["expected_keywords"])
    for kw in query_item["expected_keywords"]:
        if kw.lower() in resp_lower:
            kw_hits += 1

    kw_ratio = kw_hits / total_kw if total_kw > 0 else 1.0

    # Correctness: 1-5 scale based on key facts
    if kw_ratio >= 0.75:
        correctness = 5.0
    elif kw_ratio >= 0.5:
        correctness = 4.5
    elif kw_ratio >= 0.25:
        correctness = 3.5
    else:
        correctness = 2.5

    # Relevance: penalize if off-topic or empty, reward concise answers
    words = len(response_text.split())
    if 15 <= words <= 110 and kw_hits > 0:
        relevance = 5.0
    elif words > 110:
        relevance = 4.2
    elif words < 10:
        relevance = 2.5
    else:
        relevance = 3.8

    # Quality: style, student-friendly tone, structure
    if "error:" in resp_lower:
        quality = 1.0
    elif any(token in resp_lower for token in ["under university policy", "please", "students must", "hello", "step"]):
        quality = 4.8
    else:
        quality = 4.2

    return {
        "correctness": correctness,
        "relevance": relevance,
        "quality": quality,
        "hallucinated": 0.0
    }


COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "can",
    "could", "should", "would", "will", "shall", "may", "might", "must", "you",
    "your", "we", "our", "it", "its", "this", "that", "these", "those", "under",
    "please", "students", "student"
}


def compute_consistency_score(responses: List[str]) -> float:
    """
    Measures semantic / keyword consistency across separate runs (1.0 to 5.0 scale).
    Filters out common grammatical stopwords to evaluate invariant factual policy consistency.
    """
    if len(responses) < 2:
        return 5.0

    token_sets = [
        set(re.findall(r"\b\w{3,}\b", r.lower())) - COMMON_STOPWORDS
        for r in responses
    ]

    jaccards = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            s1, s2 = token_sets[i], token_sets[j]
            if not s1 and not s2:
                jaccards.append(1.0)
            elif not s1 or not s2:
                jaccards.append(0.0)
            else:
                jaccards.append(len(s1.intersection(s2)) / len(s1.union(s2)))

    avg_jaccard = float(np.mean(jaccards)) if jaccards else 1.0
    # Map substantive Jaccard to 1.0 - 5.0 scale
    score = 1.0 + min(avg_jaccard * 6.5, 4.0)
    return round(min(max(score, 1.0), 5.0), 2)


def run_evaluation(
    strategies: Optional[List[str]] = None,
    num_consistency_runs: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes the entire evaluation test suite across strategies with 3x consistency checks.
    """
    strategies = strategies or ["zero-shot", "few-shot", "role"]
    engine = FAQAssistantEngine()

    all_results = []
    summary_by_strategy = {}

    if verbose:
        print(f"Starting FAQ Assistant Evaluation on {len(TEST_QUERIES)} test queries...", flush=True)
        print(f"Strategies: {strategies} | Consistency runs: {num_consistency_runs}x\n", flush=True)

    for strat in strategies:
        strat_results = []
        intent_matches = 0
        total_queries = len(TEST_QUERIES)

        for q_item in TEST_QUERIES:
            query_id = q_item["id"]
            query_text = q_item["query"]
            q_type = q_item["type"]
            expected_intent = q_item["expected_intent"]
            is_unknown = q_item["is_unknown"]

            multi_responses = []
            latencies = []
            final_res_dict = None

            for run_idx in range(num_consistency_runs):
                test_memory = ConversationMemory()
                if q_item["context_turn"]:
                    c = q_item["context_turn"]
                    test_memory.add_turn(
                        query=c["query"],
                        response=c["response"],
                        intent=expected_intent
                    )

                temp_engine = FAQAssistantEngine(
                    client=engine.client,
                    retriever=engine.retriever,
                    memory=test_memory
                )

                t0 = time.time()
                res = temp_engine.ask(
                    query=query_text,
                    strategy=strat,
                    use_memory=True
                )
                latencies.append(time.time() - t0)
                multi_responses.append(res["response"])
                if run_idx == 0:
                    final_res_dict = res

            avg_latency = float(np.mean(latencies))
            consistency = compute_consistency_score(multi_responses)

            rubric_scores = score_response(q_item, final_res_dict["response"], avg_latency)

            detected_intent = final_res_dict["intent"]
            is_intent_correct = (detected_intent == expected_intent)
            if is_intent_correct:
                intent_matches += 1

            row = {
                "strategy": strat,
                "query_id": query_id,
                "type": q_type,
                "query": query_text,
                "expected_intent": expected_intent,
                "detected_intent": detected_intent,
                "intent_correct": is_intent_correct,
                "is_unknown": is_unknown,
                "declined_unknown": check_refusal(final_res_dict["response"]) if is_unknown else True,
                "correctness": rubric_scores["correctness"],
                "relevance": rubric_scores["relevance"],
                "consistency": consistency,
                "quality": rubric_scores["quality"],
                "hallucinated": rubric_scores["hallucinated"],
                "latency_seconds": round(avg_latency, 3),
                "response": final_res_dict["response"]
            }
            strat_results.append(row)
            all_results.append(row)

            if verbose:
                print(f"[{strat.upper()}] {query_id} ({q_type}) -> Correctness: {row['correctness']:.1f}, "
                      f"Relevance: {row['relevance']:.1f}, Consistency: {row['consistency']:.1f}, Latency: {avg_latency:.2f}s", flush=True)

        # Aggregate metrics for strategy
        df_strat = pd.DataFrame(strat_results)
        unknown_df = df_strat[df_strat["is_unknown"] == True]
        unknown_decline_rate = (unknown_df["declined_unknown"].sum() / len(unknown_df)) * 100.0 if len(unknown_df) else 100.0
        hallucination_rate = (unknown_df["hallucinated"].sum() / len(unknown_df)) * 100.0 if len(unknown_df) else 0.0

        mean_corr = df_strat["correctness"].mean()
        mean_rel = df_strat["relevance"].mean()
        mean_cons = df_strat["consistency"].mean()
        mean_qual = df_strat["quality"].mean()
        overall_score = np.mean([mean_corr, mean_rel, mean_cons, mean_qual])
        intent_acc = (intent_matches / total_queries) * 100.0

        summary_by_strategy[strat] = {
            "correctness": round(mean_corr, 2),
            "relevance": round(mean_rel, 2),
            "consistency": round(mean_cons, 2),
            "quality": round(mean_qual, 2),
            "overall_score": round(overall_score, 2),
            "intent_accuracy_pct": round(intent_acc, 1),
            "unknown_refusal_pct": round(unknown_decline_rate, 1),
            "hallucination_rate_pct": round(hallucination_rate, 1),
            "avg_latency_s": round(df_strat["latency_seconds"].mean(), 2)
        }

        # Save incremental CSV results to the project results folder
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = RESULTS_DIR / "evaluation_results.csv"
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(csv_path, index=False)

    md_path = RESULTS_DIR / "evaluation_summary.md"
    generate_markdown_summary(summary_by_strategy, all_results, md_path)

    return {
        "summary": summary_by_strategy,
        "csv_path": str(csv_path),
        "md_path": str(md_path)
    }


def generate_markdown_summary(
    summary: Dict[str, Dict[str, float]],
    all_results: List[Dict[str, Any]],
    output_path
):
    """Generates a polished Markdown report comparing prompt strategies."""
    lines = [
        "# Course & College FAQ Assistant - Evaluation Report\n",
        "## Summary Scorecard by Prompting Strategy\n",
        "| Prompting Strategy | Correctness (1-5) | Relevance (1-5) | Consistency (1-5) | Quality (1-5) | **Overall Score (1-5)** | Intent Accuracy | Unknown Declined | Hallucination Rate | Avg Latency |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for strat, m in summary.items():
        lines.append(
            f"| **{strat.title()}** | {m['correctness']} | {m['relevance']} | {m['consistency']} | "
            f"{m['quality']} | **{m['overall_score']}** | {m['intent_accuracy_pct']}% | "
            f"{m['unknown_refusal_pct']}% | {m['hallucination_rate_pct']}% | {m['avg_latency_s']}s |"
        )

    lines.append("\n## Detailed Per-Query Breakdown\n")
    lines.append("| Query ID | Type | Query Text | Zero-Shot Score | Few-Shot Score | Role Score | Intent Matched |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")

    df = pd.DataFrame(all_results)
    for q_id in df["query_id"].unique():
        q_df = df[df["query_id"] == q_id]
        first_row = q_df.iloc[0]
        q_text = first_row["query"]
        q_type = first_row["type"]
        all_intent = "Yes" if q_df["intent_correct"].all() else "Partial"

        z_score = q_df[q_df["strategy"] == "zero-shot"]["correctness"].values
        f_score = q_df[q_df["strategy"] == "few-shot"]["correctness"].values
        r_score = q_df[q_df["strategy"] == "role"]["correctness"].values

        z_str = f"{z_score[0]:.1f}" if len(z_score) else "N/A"
        f_str = f"{f_score[0]:.1f}" if len(f_score) else "N/A"
        r_str = f"{r_score[0]:.1f}" if len(r_score) else "N/A"

        lines.append(f"| {q_id} | {q_type} | {q_text} | {z_str} | {f_str} | {r_str} | {all_intent} |")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FAQ Assistant Evaluation Suite")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Consistency runs per query")
    args = parser.parse_args()

    results = run_evaluation(num_consistency_runs=args.runs)
    print("\n--- EVALUATION SUMMARY ---", flush=True)
    print(json.dumps(results["summary"], indent=2), flush=True)
