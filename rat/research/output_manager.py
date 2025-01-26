"""
Output manager for research results.
Handles saving research outputs, intermediate results, and performance metrics.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import shutil

class OutputManager:
    """
    Manages research outputs and metrics.
    
    Handles:
    - Creating research directories
    - Saving research papers
    - Tracking intermediate results
    - Recording performance metrics
    """
    
    def __init__(self):
        """Initialize the output manager."""
        self.base_dir = Path("research_outputs")
        
    def create_research_dir(self, question: str) -> Path:
        """
        Create a directory for research outputs.
        
        Args:
            question: Research question
            
        Returns:
            Path to created directory
        """
        # Create timestamped directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{self._sanitize_filename(question[:50])}"
        
        # Create directory
        research_dir = self.base_dir / dir_name
        research_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial metadata
        self.save_metadata(research_dir, {
            "question": question,
            "started_at": timestamp,
            "status": "in_progress"
        })
        
        return research_dir
        
    def save_research_paper(self, research_dir: Path, paper: Dict[str, Any]):
        """
        Save the research paper and update metadata.
        
        Args:
            research_dir: Research output directory
            paper: Paper content and metadata
        """
        # Save paper content
        paper_path = research_dir / "research_paper.md"
        paper_path.write_text(paper["paper"])
        
        # Save paper info
        info_path = research_dir / "research_info.json"
        info = {
            "title": paper["title"],
            "sources": paper["sources"],
            "metrics": paper.get("metrics", {})
        }
        info_path.write_text(json.dumps(info, indent=2))
        
        # Update metadata
        self.save_metadata(research_dir, {
            "status": "completed",
            "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metrics": paper.get("metrics", {})
        })
        
    def save_context_state(self, research_dir: Path, context_data: Dict[str, Any]):
        """
        Save intermediate context state.
        
        Args:
            research_dir: Research output directory
            context_data: Context state to save
        """
        # Create states directory
        states_dir = research_dir / "states"
        states_dir.mkdir(exist_ok=True)
        
        # Save state with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = states_dir / f"context_state_{timestamp}.json"
        state_path.write_text(json.dumps(context_data, indent=2))
        
        # Keep only last 5 states to save space
        self._cleanup_old_states(states_dir)
        
    def save_iteration_metrics(
        self,
        research_dir: Path,
        iterations: List[Dict[str, Any]]
    ):
        """
        Save iteration performance metrics.
        
        Args:
            research_dir: Research output directory
            iterations: List of iteration metrics
        """
        metrics_path = research_dir / "iteration_metrics.json"
        metrics_path.write_text(json.dumps({
            "iterations": iterations,
            "summary": self._calculate_metrics_summary(iterations)
        }, indent=2))
        
    def save_metadata(self, research_dir: Path, updates: Dict[str, Any]):
        """
        Update research session metadata.
        
        Args:
            research_dir: Research output directory
            updates: Metadata updates
        """
        metadata_path = research_dir / "metadata.json"
        
        # Load existing metadata
        if metadata_path.exists():
            current_metadata = json.loads(metadata_path.read_text())
        else:
            current_metadata = {}
            
        # Update metadata
        current_metadata.update(updates)
        
        # Save updated metadata
        metadata_path.write_text(json.dumps(current_metadata, indent=2))
        
    def _sanitize_filename(self, name: str) -> str:
        """
        Create a safe filename from text.
        
        Args:
            name: Text to convert to filename
            
        Returns:
            Safe filename
        """
        # Replace unsafe characters
        safe_chars = "-_"
        filename = "".join(
            c if c.isalnum() or c in safe_chars else "_"
            for c in name
        )
        return filename.strip("_")
        
    def _cleanup_old_states(self, states_dir: Path):
        """
        Keep only the most recent state files.
        
        Args:
            states_dir: Directory containing state files
        """
        state_files = sorted(
            states_dir.glob("context_state_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old files
        for file in state_files[5:]:  # Keep 5 most recent
            file.unlink()
            
    def _calculate_metrics_summary(
        self,
        iterations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics across iterations.
        
        Args:
            iterations: List of iteration metrics
            
        Returns:
            Summary metrics
        """
        if not iterations:
            return {}
            
        return {
            "total_iterations": len(iterations),
            "total_decisions": sum(it["decisions"] for it in iterations),
            "total_new_content": sum(it["new_content"] for it in iterations),
            "total_time": sum(it["time"] for it in iterations),
            "avg_decisions_per_iteration": (
                sum(it["decisions"] for it in iterations) / len(iterations)
            ),
            "avg_new_content_per_iteration": (
                sum(it["new_content"] for it in iterations) / len(iterations)
            ),
            "avg_time_per_iteration": (
                sum(it["time"] for it in iterations) / len(iterations)
            )
        }
