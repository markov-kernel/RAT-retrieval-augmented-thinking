"""
Test script to run the research agent with a sample question.
"""

import os
import logging
from rich import print as rprint
from rat.research.orchestrator import ResearchOrchestrator

def setup_logging():
    """
    Configure logging so that logs are overwritten on each new run.
    """
    log_path = "rat.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        filename=log_path,
        filemode="w",  # Overwrite mode
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.info("Logging setup complete (test_agent). Logs overwritten each run.")

def main():
    # Ensure logging is set up
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting test_agent.py main function.")
    
    # Initialize the orchestrator with default or custom config
    orchestrator = ResearchOrchestrator()
    
    # Define a test question
    question = (
        "What are the main features and pricing of Billit's accounting software, "
        "and how does it compare to competitors in Belgium?"
    )
    rprint(f"[bold cyan]Starting research on: {question}[/bold cyan]")
    logger.info("Test question: %s", question)
    
    # Run the research
    results = orchestrator.start_research(question)
    
    # Print results
    if "error" in results:
        rprint(f"[red]Error: {results['error']}[/red]")
        logger.error("Research ended with an error: %s", results['error'])
    else:
        rprint("\n[bold green]Research completed![/bold green]")
        logger.info("Research completed successfully.")
        
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
    main() 