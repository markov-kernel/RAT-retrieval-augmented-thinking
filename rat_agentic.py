"""
Entry point for the multi-agent research system.
Provides a command-line interface for conducting research using the agent-based approach.
"""

import os
import sys
import json
import logging
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from typing import Dict, Any, Optional

from rat.research.orchestrator import ResearchOrchestrator

def setup_logging():
    log_path = "rat.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.info("Logging setup complete. Logs overwritten each run.")

def create_default_config() -> Dict[str, Any]:
    return {
        "max_iterations": 5,
        "min_new_content": 2,
        "min_confidence": 0.7,
        "search_config": {
            "max_queries": 5,
            "min_priority": 0.3,
            "refinement_threshold": 0.7
        },
        "explore_config": {
            "max_urls": 10,
            "min_priority": 0.3,
            "allowed_domains": []
        },
        "reason_config": {
            "max_parallel_tasks": 3,
            "chunk_size": 30000,
            "min_priority": 0.3
        }
    }

def display_results(results: Dict[str, Any]):
    if "error" in results:
        rprint(f"\n[red]Error during research: {results['error']}[/red]")
        return
    rprint(Panel(
        Markdown(results["paper"]),
        title="[bold green]Research Paper[/bold green]",
        border_style="green"
    ))
    metrics = results.get("metrics", {})
    rprint("\n[bold cyan]Research Metrics:[/bold cyan]")
    rprint(f"Total time: {metrics.get('total_time', 0):.2f} seconds")
    rprint(f"Iterations: {metrics.get('iterations', 0)}")
    rprint(f"Total decisions: {metrics.get('total_decisions', 0)}")
    rprint(f"Total content items: {metrics.get('total_content', 0)}")
    agent_metrics = metrics.get("agent_metrics", {})
    for agent_name, agent_data in agent_metrics.items():
        rprint(f"\n[bold]{agent_name.title()} Agent:[/bold]")
        rprint(f"Decisions made: {agent_data.get('decisions_made', 0)}")
        rprint(f"Successful executions: {agent_data.get('successful_executions', 0)}")
        rprint(f"Failed executions: {agent_data.get('failed_executions', 0)}")
        total_decisions = max(agent_data.get("decisions_made", 1), 1)
        avg_time = agent_data.get("total_execution_time", 0) / total_decisions
        rprint(f"Average execution time: {avg_time:.2f}s")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Launching RAT from rat_agentic.py")

    style = Style.from_dict({'prompt': 'orange bold'})
    session = PromptSession(style=style)
    
    orchestrator = ResearchOrchestrator(create_default_config())
    
    rprint(Panel.fit(
        "[bold cyan]RAT Multi-Agent Research System[/bold cyan]\n"
        "Conduct research using a coordinated team of specialized AI agents",
        title="[bold cyan]🧠 RAT Research[/bold cyan]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/yellow]")
    rprint(" • Type [bold red]'quit'[/bold red] to exit")
    rprint(" • Type [bold magenta]'config'[/bold magenta] to view current configuration")
    rprint(" • Type [bold magenta]'metrics'[/bold magenta] to view latest metrics")
    rprint(" • Enter your research question to begin\n")
    
    latest_results: Optional[Dict[str, Any]] = None
    
    while True:
        try:
            user_input = session.prompt("\nResearch Question: ", style=style).strip()
            if user_input.lower() == 'quit':
                logger.info("User chose to quit from rat_agentic.py")
                rprint("\nGoodbye! 👋")
                break
            elif user_input.lower() == 'config':
                logger.info("Displaying current configuration.")
                config_str = json.dumps(orchestrator.config, indent=2)
                rprint(Panel(
                    Markdown(f"```json\n{config_str}\n```"),
                    title="[bold cyan]Current Configuration[/bold cyan]",
                    border_style="cyan"
                ))
                continue
            elif user_input.lower() == 'metrics':
                if latest_results:
                    logger.info("User requested metrics display for the latest research.")
                    display_results(latest_results)
                else:
                    rprint("[yellow]No research has been conducted yet[/yellow]")
                continue
            if not user_input:
                logger.debug("User provided empty input; ignoring.")
                continue
            logger.info("Starting research for input question: %s", user_input)
            latest_results = orchestrator.start_research(user_input)
            display_results(latest_results)
        except KeyboardInterrupt:
            logger.warning("User interrupted with Ctrl+C. Returning to prompt.")
            continue
        except EOFError:
            logger.info("EOF received, exiting gracefully.")
            break
        except Exception as e:
            logger.exception("Unhandled exception in main loop.")
            rprint(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main()