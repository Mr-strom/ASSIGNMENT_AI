"""
Entry point script for the Course & College FAQ Assistant.
Supports launching the Streamlit UI, terminal CLI, evaluation suite, or strategy comparison.
"""

import sys
import subprocess
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def launch_streamlit():
    """Launch the Streamlit web application."""
    app_path = BASE_DIR / "faq_assistant" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"Launching Streamlit Web UI: {' '.join(cmd)}")
    subprocess.run(cmd)


def launch_cli():
    """Launch the interactive terminal chat CLI."""
    from faq_assistant.cli import run_cli
    run_cli()


def launch_eval(runs: int = 3):
    """Run the evaluation benchmark suite."""
    from faq_assistant.evaluate import run_evaluation
    results = run_evaluation(num_consistency_runs=runs)
    print("\nBenchmark completed. Summary saved to results/evaluation_summary.md")


def launch_compare(query: str):
    """Run the side-by-side prompt comparison."""
    from faq_assistant.compare import compare_prompting_strategies, display_comparison_table
    data = compare_prompting_strategies(query)
    display_comparison_table(data)


def main():
    parser = argparse.ArgumentParser(description="Course/College FAQ Assistant Launcher")
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit Web UI")
    parser.add_argument("--cli", action="store_true", help="Launch Interactive CLI")
    parser.add_argument("--eval", action="store_true", help="Run Evaluation Benchmark")
    parser.add_argument("--compare", type=str, metavar="QUERY", help="Compare prompt strategies on a query")
    parser.add_argument("--runs", type=int, default=3, help="Consistency runs for evaluation (default: 3)")

    args = parser.parse_args()

    if args.ui:
        launch_streamlit()
    elif args.eval:
        launch_eval(runs=args.runs)
    elif args.compare:
        launch_compare(args.compare)
    elif args.cli:
        launch_cli()
    else:
        # Default: if no flag given, launch Streamlit
        launch_streamlit()


if __name__ == "__main__":
    main()
