"""
Prompt Comparison module.
Executes a single question across Zero-Shot, Few-Shot, and Role Prompting strategies,
displaying outputs side-by-side with latency, word count, and intent telemetry.
"""

import sys
import argparse
from typing import Dict, Any, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from faq_assistant.engine import FAQAssistantEngine
from faq_assistant.intent import classify_intent_rule_based
from faq_assistant.retrieval import default_retriever

console = Console()

STRATEGIES = ["zero-shot", "few-shot", "role"]


def compare_prompting_strategies(query: str) -> Dict[str, Any]:
    """
    Runs the same question through Zero-Shot, Few-Shot, and Role Prompting.
    Returns structured results for each strategy.
    """
    engine = FAQAssistantEngine()
    results = {}

    for strat in STRATEGIES:
        res = engine.ask(query=query, strategy=strat, use_memory=False)
        results[strat] = {
            "strategy": strat,
            "response": res["response"],
            "latency_seconds": res["latency_seconds"],
            "word_count": len(res["response"].split()),
            "intent": res["intent"],
            "similarity_score": res["similarity_score"],
            "is_reliable": res["is_reliable"]
        }

    return {
        "query": query,
        "results": results
    }


def display_comparison_table(comp_data: Dict[str, Any]):
    """
    Render a rich side-by-side terminal comparison table.
    """
    query = comp_data["query"]
    results = comp_data["results"]

    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", title="Prompt Engineering Strategy Comparison", border_style="blue"))

    table = Table(title="Side-by-Side Prompting Strategy Evaluation", show_header=True, header_style="bold magenta")
    table.add_column("Metric / Property", style="dim", width=22)
    table.add_column("Zero-Shot", width=38)
    table.add_column("Few-Shot", width=38)
    table.add_column("Role Prompting (Dr. Vance)", width=38)

    # Latency row
    z_lat = f"{results['zero-shot']['latency_seconds']} s"
    f_lat = f"{results['few-shot']['latency_seconds']} s"
    r_lat = f"{results['role']['latency_seconds']} s"
    table.add_row("Latency", z_lat, f_lat, r_lat)

    # Word count row
    z_wc = f"{results['zero-shot']['word_count']} words"
    f_wc = f"{results['few-shot']['word_count']} words"
    r_wc = f"{results['role']['word_count']} words"
    table.add_row("Response Length", z_wc, f_wc, r_wc)

    # Intent row
    z_int = results['zero-shot']['intent']
    f_int = results['few-shot']['intent']
    r_int = results['role']['intent']
    table.add_row("Detected Intent", z_int, f_int, r_int)

    table.add_section()

    # Response row
    table.add_row(
        "Response Content",
        results['zero-shot']['response'],
        results['few-shot']['response'],
        results['role']['response']
    )

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Compare Prompting Strategies for Course/College FAQ Assistant")
    parser.add_argument("--query", "-q", type=str, default="What is the minimum attendance required for final exams?",
                        help="Student query to evaluate across strategies")
    args = parser.parse_args()

    data = compare_prompting_strategies(args.query)
    display_comparison_table(data)


if __name__ == "__main__":
    main()
