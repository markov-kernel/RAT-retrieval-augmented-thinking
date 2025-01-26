"""
Main entry point for the multi-agent research system.
Provides a command-line interface for conducting research using specialized agents.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import time

from .orchestrator import ResearchOrchestrator
from .output_manager import OutputManager

console = Console()

def setup_logging():
    """
    Configure logging so that logs are overwritten on each run.
    """
    log_path = "rat.log"
    # Remove the log file if it exists, so we overwrite on every new run
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        filename=log_path,
        filemode="w",  # Overwrite mode
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    
    logging.info("Logger initialized. All logs will be written to rat.log and overwritten each run.")

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    load_dotenv()
    
    config = {
        'max_iterations': int(os.getenv('MAX_ITERATIONS', '5')),
        'min_new_content': int(os.getenv('MIN_NEW_CONTENT', '3')),
        'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.7')),
        'search_config': {
            'max_results': int(os.getenv('MAX_SEARCH_RESULTS', '10')),
            'min_relevance': float(os.getenv('MIN_SEARCH_RELEVANCE', '0.6')),
            'api_key': os.getenv('PERPLEXITY_API_KEY')
        },
        'explore_config': {
            'max_urls': int(os.getenv('MAX_URLS', '20')),
            'min_priority': float(os.getenv('MIN_URL_PRIORITY', '0.5')),
            'allowed_domains': json.loads(os.getenv('ALLOWED_DOMAINS', '[]')),
            'api_key': os.getenv('FIRECRAWL_API_KEY')
        },
        'reason_config': {
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '4000')),
            'min_confidence': float(os.getenv('MIN_ANALYSIS_CONFIDENCE', '0.7')),
            'parallel_threads': int(os.getenv('PARALLEL_ANALYSIS_THREADS', '4')),
            'api_key': os.getenv('DEEPSEEK_API_KEY')
        },
        'execute_config': {
            'max_code_length': int(os.getenv('MAX_CODE_LENGTH', '1000')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'timeout': int(os.getenv('TIMEOUT_SECONDS', '30')),
            'api_key': os.getenv('CLAUDE_API_KEY')
        }
    }
    return config

def display_welcome():
    """Display welcome message and system information."""
    welcome_text = """
# RAT - Retrieval Augmented Thinking

Welcome to the multi-agent research system! This tool helps you conduct comprehensive research using:

1. Search Agent (Perplexity) - Intelligent web searching
2. Explore Agent (Firecrawl) - URL content extraction
3. Reasoning Agent (DeepSeek) - Content analysis
4. Execution Agent (Claude) - Code generation and structured output

Enter your research question below, or type 'help' for more information.
"""
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))

def display_help():
    """Display help information."""
    help_text = """
## Available Commands

- `research <question>` - Start a new research session
- `config` - View current configuration
- `metrics` - View research metrics
- `help` - Display this help message
- `exit` - Exit the system

## Tips

- Be specific in your research questions
- Use quotes for exact phrases
- Type 'exit' to quit at any time
"""
    console.print(Panel(Markdown(help_text), title="Help", border_style="green"))

def run_research(question: str, config: Dict[str, Any]) -> None:
    """Run research with the given question."""
    orchestrator = ResearchOrchestrator(config)
    results = orchestrator.start_research(question)
    
    if "error" in results:
        logger.error("Research error: %s", results['error'])
        console.print(f"[red]Research error: {results['error']}[/red]")
    else:
        logger.info("Research completed successfully")
        console.print(Panel(Markdown(results["paper"]), title="Research Results", border_style="green"))

def main():
    """Main entry point for the research system."""
    import argparse
    
    # Setup logging once at startup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RAT main program...")

    parser = argparse.ArgumentParser(description="RAT - Retrieval Augmented Thinking")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start in interactive mode")
    parser.add_argument("question", nargs="?", 
                       help="Research question (if not using interactive mode)")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.interactive:
        display_welcome()
        commands = WordCompleter(['research', 'config', 'metrics', 'help', 'exit'])
        
        orchestrator: Optional[ResearchOrchestrator] = None
        
        while True:
            try:
                user_input = prompt('RAT> ', completer=commands).strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    logger.info("User exited the interactive session")
                    console.print("[yellow]Exiting research system...[/yellow]")
                    break
                    
                if user_input.lower() == 'help':
                    display_help()
                    continue
                    
                if user_input.lower() == 'config':
                    logger.info("Displaying configuration")
                    console.print(Panel(json.dumps(config, indent=2), title="Configuration", border_style="cyan"))
                    continue
                    
                if user_input.lower() == 'metrics' and orchestrator:
                    logger.info("Calculating research metrics")
                    metrics = orchestrator._calculate_metrics(time.time())
                    console.print(Panel(json.dumps(metrics, indent=2), title="Research Metrics", border_style="magenta"))
                    continue
                    
                if user_input.lower().startswith('research '):
                    question = user_input[9:].strip()
                    if not question:
                        logger.warning("Empty research question provided")
                        console.print("[red]Please provide a research question.[/red]")
                        continue
                    logger.info("Starting research with question: %s", question)
                    run_research(question, config)
                    continue
                    
                logger.warning("Unknown command received: %s", user_input)
                console.print("[red]Unknown command. Type 'help' for available commands.[/red]")
                
            except KeyboardInterrupt:
                logger.info("Operation cancelled by user")
                console.print("\n[yellow]Operation cancelled. Type 'exit' to quit.[/yellow]")
                continue
                
            except Exception as e:
                logger.exception("Exception occurred in interactive loop")
                console.print(f"[red]Error: {str(e)}[/red]")
                continue
    else:
        if not args.question:
            parser.error("Research question is required when not in interactive mode")
        logger.info("Starting non-interactive research with question: %s", args.question)
        run_research(args.question, config)

if __name__ == '__main__':
    main()
