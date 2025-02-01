"""
Test script to run the research agent with a sample question.
Now asynchronous.
"""

import asyncio
from rat.research.orchestrator import ResearchOrchestrator
from rich import print as rprint

async def main():
    orchestrator = ResearchOrchestrator()
    question = "What are the main features and pricing of Billit's accounting software, and how does it compare to competitors in Belgium?"
    rprint(f"[bold cyan]Starting research on: {question}[/bold cyan]")
    results = await orchestrator.start_research(question)
    if "error" in results:
        rprint(f"[red]Error: {results['error']}[/red]")
    else:
        rprint("\n[bold green]Research completed![/bold green]")
        rprint("\n[bold]Results:[/bold]")
        print(results["paper"])
        rprint("\n[bold]Sources:[/bold]")
        for source in results.get("sources", []):
            print(f"- {source}")
        rprint("\n[bold]Metrics:[/bold]")
        metrics = results.get("metrics", {})
        print(f"Total time: {metrics.get('total_time', 0):.2f} seconds")
        print(f"Iterations: {metrics.get('iterations', 0)}")
        print(f"Total decisions: {metrics.get('total_decisions', 0)}")
        print(f"Total content items: {metrics.get('total_content', 0)}")

if __name__ == "__main__":
    asyncio.run(main())