"""
Entry point for the multi-agent research system.
Provides a command-line interface for conducting research using the agent-based approach.
"""

import sys
import json
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from typing import Dict, Any, Optional
import asyncio
import builtins

from rat.research.manager import ResearchManager

# Ensure that dynamic execution contexts have access to asyncio and rprint
builtins.asyncio = asyncio
builtins.rprint = rprint


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for the research system.
    
    Returns:
        Default configuration dictionary
    """
    return {
        "max_iterations": 5,
        "min_new_content": 2,
        "min_confidence": 0.7,
        "search_config": {
            "max_queries": 5,
            "min_priority": 0.3,
            "refinement_threshold": 0.7,
            "rate_limit": 20      # Updated: Perplexity rate limit set to 20 requests per minute
        },
        "explore_config": {
            "max_urls": 10,
            "min_priority": 0.3,
            "allowed_domains": []
        },
        "reason_config": {
            "max_parallel_tasks": 3,
            "chunk_size": 30000,
            "min_priority": 0.3,
            "rate_limit": 200,     # Updated: O3 mini rate limit set to 200 requests per minute
            "flash_fix_rate_limit": 10
        },
        "execute_config": {
            "model": "claude-3-5-sonnet-20241022",
            "min_priority": 0.3,
            "max_retries": 2
        },
        "max_workers": 20
    }


def display_results(results: Dict[str, Any]):
    """
    Display research results in a formatted way.
    
    Args:
        results: Research results to display
    """
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
        rprint(
            f"Average execution time: "
            f"{agent_data.get('total_execution_time', 0) / max(agent_data.get('decisions_made', 1), 1):.2f}s"
        )


async def main_async():
    """Main entry point for the research system."""
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    manager = ResearchManager(create_default_config())
    
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
    commands = WordCompleter(['quit', 'config', 'metrics'])
    
    while True:
        try:
            user_input = await session.prompt_async("\nResearch Question: ", completer=commands)
            user_input = user_input.strip()
            
            if user_input.lower() == 'quit':
                rprint("\nGoodbye! 👋")
                break
            elif user_input.lower() == 'config':
                rprint(Panel(
                    Markdown(f"```json\n{json.dumps(manager.config, indent=2)}\n```"),
                    title="[bold cyan]Current Configuration[/bold cyan]",
                    border_style="cyan"
                ))
                continue
            elif user_input.lower() == 'metrics':
                if latest_results:
                    display_results(latest_results)
                else:
                    rprint("[yellow]No research has been conducted yet[/yellow]")
                continue
            rprint(f"\n[bold cyan]Starting research on:[/bold cyan] {user_input}")
            latest_results = await manager.start_research(user_input)
            display_results(latest_results)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            rprint(f"[red]Error: {str(e)}[/red]")


def main():
    """Run the async main function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()