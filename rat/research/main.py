"""
Main entry point for the multi-agent research system.
Provides a command-line interface for conducting research using specialized agents.
"""

import os
import sys
import json
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
            'api_key': os.getenv('PERPLEXITY_API_KEY'),
            'max_workers': int(os.getenv('MAX_PARALLEL_SEARCHES', '10')),
            'rate_limit': int(os.getenv('SEARCH_RATE_LIMIT', '100'))
        },
        'explore_config': {
            'max_urls': int(os.getenv('MAX_URLS', '20')),
            'min_priority': float(os.getenv('MIN_URL_PRIORITY', '0.5')),
            'allowed_domains': json.loads(os.getenv('ALLOWED_DOMAINS', '[]')),
            'api_key': os.getenv('FIRECRAWL_API_KEY'),
            'max_workers': int(os.getenv('MAX_PARALLEL_EXPLORES', '10')),
            'rate_limit': int(os.getenv('EXPLORE_RATE_LIMIT', '100'))
        },
        'reason_config': {
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '4000')),
            'min_confidence': float(os.getenv('MIN_ANALYSIS_CONFIDENCE', '0.7')),
            'max_workers': int(os.getenv('MAX_PARALLEL_REASON', '5')),
            'rate_limit': int(os.getenv('REASON_RATE_LIMIT', '10')),  # 10 RPM for Gemini
            'api_key': os.getenv('GEMINI_API_KEY'),
            'gemini_timeout': int(os.getenv('GEMINI_TIMEOUT', '180'))
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
3. Reasoning Agent (Gemini) - Content analysis using Gemini 2.0 Flash Thinking

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
        console.print(f"[red]Research error: {results['error']}[/red]")
    else:
        console.print(Panel(Markdown(results["paper"]), title="Research Results", border_style="green"))

def main():
    """Main entry point for the research system."""
    import argparse
    
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
                    console.print("[yellow]Exiting research system...[/yellow]")
                    break
                    
                if user_input.lower() == 'help':
                    display_help()
                    continue
                    
                if user_input.lower() == 'config':
                    console.print(Panel(json.dumps(config, indent=2), title="Configuration", border_style="cyan"))
                    continue
                    
                if user_input.lower() == 'metrics' and orchestrator:
                    metrics = orchestrator._calculate_metrics(time.time())
                    console.print(Panel(json.dumps(metrics, indent=2), title="Research Metrics", border_style="magenta"))
                    continue
                    
                if user_input.lower().startswith('research '):
                    question = user_input[9:].strip()
                    if not question:
                        console.print("[red]Please provide a research question.[/red]")
                        continue
                    
                    run_research(question, config)
                    continue
                    
                console.print("[red]Unknown command. Type 'help' for available commands.[/red]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled. Type 'exit' to quit.[/yellow]")
                continue
                
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                continue
    else:
        if not args.question:
            parser.error("Research question is required when not in interactive mode")
        run_research(args.question, config)

if __name__ == '__main__':
    main()
