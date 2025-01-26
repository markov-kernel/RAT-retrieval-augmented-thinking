"""
Research session manager.
Coordinates the research process using DeepSeek, Perplexity, and Jina Reader.
"""

import os
from typing import Dict, Any, List
from rich import print as rprint
from pathlib import Path

from .perplexity_client import PerplexityClient
from .jina_client import JinaClient
from .output_manager import OutputManager

class ResearchSession:
    def __init__(self):
        self.perplexity = PerplexityClient()
        self.jina = JinaClient()
        self.output_manager = OutputManager()
        self.current_research_dir = None
        
    def start_research(self, question: str) -> Dict[str, Any]:
        """
        Start a new research session.
        
        Args:
            question: The research question
            
        Returns:
            Research results including paper and metadata
        """
        try:
            rprint("\nStarting research...")
            
            # Create research directory
            self.current_research_dir = self.output_manager.create_research_dir(question)
            
            # Initial research using Perplexity
            rprint("Conducting initial research...")
            search_results = self.perplexity.search(question)
            
            # Extract content from sources
            rprint("Extracting content from sources...")
            source_contents = []
            
            for url in search_results.get("urls", []):
                try:
                    content = self.jina.extract_content(url)
                    if content["text"]:  # Only save if we got content
                        # Save source content
                        self.output_manager.save_source_content(
                            self.current_research_dir, 
                            content
                        )
                        source_contents.append(content)
                except Exception as e:
                    rprint(f"[yellow]Warning: Failed to extract content from {url}: {str(e)}[/yellow]")
            
            # Generate research paper
            rprint("Generating research paper...")
            paper = self._generate_paper(
                question,
                search_results["content"],
                source_contents
            )
            
            # Save the paper
            self.output_manager.save_research_paper(
                self.current_research_dir,
                paper
            )
            
            return paper
            
        except Exception as e:
            rprint(f"[red]Error in research session: {str(e)}[/red]")
            return {
                "paper": "Error occurred during research.",
                "error": str(e)
            }
            
    def _generate_paper(
        self, 
        question: str, 
        initial_research: str,
        source_contents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a research paper from the collected information.
        
        Args:
            question: Original research question
            initial_research: Initial research results
            source_contents: Content extracted from sources
            
        Returns:
            Dictionary containing the paper and metadata
        """
        # Combine all source texts
        all_content = [
            initial_research,
            *[content["text"] for content in source_contents]
        ]
        
        # Extract sources for citation
        sources = []
        for content in source_contents:
            meta = content["metadata"]
            sources.append({
                "title": content["title"],
                "url": meta["url"],
                "author": meta["author"],
                "date": meta["published_date"]
            })
        
        # Generate paper content (for now, just combine with headers)
        paper_content = f"""# Research Report: {question}

## Executive Summary

{initial_research}

## Detailed Analysis

"""
        
        # Add content from each source
        for content in source_contents:
            if content["title"]:
                paper_content += f"\n### {content['title']}\n\n"
            paper_content += f"{content['text']}\n\n"
            
        # Add sources section
        paper_content += "\n## Sources\n\n"
        for source in sources:
            paper_content += (
                f"- [{source['title']}]({source['url']})"
                f"{f' by {source['author']}' if source['author'] else ''}"
                f"{f' ({source['date']})' if source['date'] else ''}\n"
            )
            
        return {
            "paper": paper_content,
            "title": question,
            "sources": sources
        } 