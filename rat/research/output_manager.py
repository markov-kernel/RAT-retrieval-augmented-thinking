"""
Output manager for research results.
Handles saving and organizing research outputs in a structured way.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import re
from rich import print as rprint

class OutputManager:
    def __init__(self, base_dir: str = "research_outputs"):
        """
        Initialize the output manager.
        
        Args:
            base_dir: Base directory for research outputs
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_research_dir(self, research_question: str) -> Path:
        """
        Create a directory for the current research session.
        
        Args:
            research_question: The research question being investigated
            
        Returns:
            Path to the research directory
        """
        # Create a safe directory name from the research question
        safe_name = self._create_safe_dirname(research_question)
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{safe_name}"
        
        # Create the directory
        research_dir = self.base_dir / dir_name
        research_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the original research question
        self._save_research_info(research_dir, research_question)
        
        return research_dir
        
    def save_research_paper(self, research_dir: Path, content: Dict[str, Any]) -> Path:
        """
        Save the research paper and its metadata.
        
        Args:
            research_dir: Directory to save the paper in
            content: Research paper content and metadata
            
        Returns:
            Path to the saved paper
        """
        try:
            # Save the main paper
            paper_path = research_dir / "research_paper.md"
            with open(paper_path, "w", encoding="utf-8") as f:
                f.write(content["paper"])
                
            # Save metadata
            meta_path = research_dir / "metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": content.get("title", ""),
                    "timestamp": datetime.now().isoformat(),
                    "sources": content.get("sources", []),
                    "stats": {
                        "word_count": len(content["paper"].split()),
                        "source_count": len(content.get("sources", [])),
                        "section_count": content["paper"].count("#")
                    }
                }, f, indent=2)
                
            rprint(f"[green]Research paper saved to {paper_path}[/green]")
            return paper_path
            
        except Exception as e:
            rprint(f"[red]Error saving research paper: {str(e)}[/red]")
            return None
            
    def save_source_content(self, research_dir: Path, source_content: Dict[str, Any]) -> Path:
        """
        Save extracted content from a source.
        
        Args:
            research_dir: Directory to save the source in
            source_content: Content and metadata from the source
            
        Returns:
            Path to the saved source
        """
        try:
            # Create sources directory
            sources_dir = research_dir / "sources"
            sources_dir.mkdir(exist_ok=True)
            
            # Create a safe filename from the title or URL
            filename = self._create_safe_filename(
                source_content.get("title") or 
                source_content.get("metadata", {}).get("url", "source")
            )
            
            # Save the source content
            source_path = sources_dir / f"{filename}.md"
            with open(source_path, "w", encoding="utf-8") as f:
                # Write metadata header
                f.write("---\n")
                json.dump(source_content["metadata"], f, indent=2)
                f.write("\n---\n\n")
                
                # Write content
                f.write(source_content["text"])
                
            return source_path
            
        except Exception as e:
            rprint(f"[red]Error saving source content: {str(e)}[/red]")
            return None
            
    def _create_safe_dirname(self, name: str, max_length: int = 50) -> str:
        """Create a safe directory name from a string."""
        # Remove special characters and spaces
        safe = re.sub(r'[^\w\s-]', '', name)
        safe = re.sub(r'[-\s]+', '_', safe).strip('-_')
        
        # Truncate if too long
        if len(safe) > max_length:
            safe = safe[:max_length]
            
        return safe
        
    def _create_safe_filename(self, name: str, max_length: int = 100) -> str:
        """Create a safe filename from a string."""
        return self._create_safe_dirname(name, max_length)
        
    def _save_research_info(self, research_dir: Path, question: str) -> None:
        """Save the original research question and timestamp."""
        info_path = research_dir / "research_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump({
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }, f, indent=2) 