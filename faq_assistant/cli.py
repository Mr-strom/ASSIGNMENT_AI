"""
Interactive Command-Line Interface (CLI) for the Course/College FAQ Assistant.
Features rich terminal styling, prompt-mode switching, multi-turn history, and quick comparisons.
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from faq_assistant.config import SYSTEM_SPECS, OLLAMA_MODEL
from faq_assistant.engine import FAQAssistantEngine
from faq_assistant.compare import compare_prompting_strategies, display_comparison_table
from faq_assistant.intent import classify_intent

console = Console()


def print_welcome_banner(engine: FAQAssistantEngine, current_strategy: str):
    """Display the startup banner with system diagnostics."""
    banner_text = (
        f"[bold cyan]Course & College FAQ Assistant[/bold cyan]\n"
        f"[dim]Local LLM: [bold white]{engine.client.model}[/bold white] | "
        f"RAM: {SYSTEM_SPECS['available_ram_gb']}GB free / {SYSTEM_SPECS['total_ram_gb']}GB total | "
        f"Strategy: [bold green]{current_strategy}[/bold green][/dim]\n\n"
        f"[bold yellow]Commands:[/bold yellow] "
        f"[green]/mode <zero-shot|few-shot|role>[/green]  "
        f"[green]/compare <query>[/green]  "
        f"[green]/history[/green]  "
        f"[green]/clear[/green]  "
        f"[green]/faqs[/green]  "
        f"[green]/help[/green]  "
        f"[green]/exit[/green]"
    )
    console.print(Panel(banner_text, border_style="cyan", title="Academic FAQ CLI"))


def display_faq_catalog(engine: FAQAssistantEngine):
    """Show available FAQs in a clean table."""
    table = Table(title="Official College Knowledge Base (faq_data.json)", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Category", width=16)
    table.add_column("Official Question", width=65)

    for item in engine.retriever.faqs:
        table.add_row(item.id, f"[bold blue]{item.category}[/bold blue]", item.question)

    console.print(table)


def run_cli():
    """Main CLI chat loop."""
    engine = FAQAssistantEngine()
    current_strategy = "zero-shot"

    print_welcome_banner(engine, current_strategy)

    while True:
        try:
            user_input = console.input("\n[bold cyan]Student >[/bold cyan] ").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.startswith("/"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ["/exit", "/quit"]:
                    console.print("[green]Goodbye! Have a great semester at Evergreen.[/green]")
                    break

                elif cmd == "/mode":
                    if arg.lower() in ["zero-shot", "few-shot", "role"]:
                        current_strategy = arg.lower()
                        console.print(f"[bold green]Switched prompt strategy to:[/bold green] {current_strategy}")
                    else:
                        console.print("[bold red]Invalid mode.[/bold red] Choose from: zero-shot, few-shot, role")
                    continue

                elif cmd == "/compare":
                    if not arg:
                        console.print("[yellow]Usage: /compare <question>[/yellow]")
                        continue
                    console.print(f"[dim]Comparing strategies for: '{arg}'...[/dim]")
                    data = compare_prompting_strategies(arg)
                    display_comparison_table(data)
                    continue

                elif cmd == "/history":
                    turns = engine.memory.get_turns()
                    if not turns:
                        console.print("[dim]No previous conversation turns in current session.[/dim]")
                    else:
                        console.print(Panel(engine.memory.get_formatted_history(), title="Conversation History", border_style="blue"))
                    continue

                elif cmd == "/clear":
                    engine.memory.clear()
                    console.print("[green]Conversation history cleared.[/green]")
                    continue

                elif cmd == "/faqs":
                    display_faq_catalog(engine)
                    continue

                elif cmd == "/help":
                    print_welcome_banner(engine, current_strategy)
                    continue

                else:
                    console.print(f"[red]Unknown command:[/red] {cmd}. Type /help for available options.")
                    continue

            # Normal Question Processing
            console.print("[dim]Consulting official college handbook...[/dim]")
            res = engine.ask(query=user_input, strategy=current_strategy, use_memory=True)

            # Metadata header badge
            badge_color = "green" if res["is_reliable"] else "yellow"
            status_text = "Verified Official Policy" if res["is_reliable"] else "Out-of-Scope / General"
            header = (
                f"[bold magenta]Intent:[/bold magenta] [white]{res['intent']}[/white] | "
                f"[bold cyan]Strategy:[/bold cyan] [white]{res['strategy']}[/white] | "
                f"[{badge_color}]{status_text}[/{badge_color}] (Score: {res['similarity_score']}) | "
                f"[dim]Latency: {res['latency_seconds']}s[/dim]"
            )

            console.print(Panel(
                Markdown(res["response"]),
                title=f"Assistant Response ({res['model_name']})",
                subtitle=header,
                border_style="cyan" if res["is_reliable"] else "yellow"
            ))

        except KeyboardInterrupt:
            console.print("\n[green]Session terminated. Goodbye![/green]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    run_cli()
