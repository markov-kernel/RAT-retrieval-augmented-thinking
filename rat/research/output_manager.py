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
    def __init__(self):
        self.base_dir = Path("research_outputs")
    
    def create_research_dir(self, question: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{self._sanitize_filename(question[:50])}"
        research_dir = self.base_dir / dir_name
        research_dir.mkdir(parents=True, exist_ok=True)
        self.save_metadata(research_dir, {
            "question": question,
            "started_at": timestamp,
            "status": "in_progress"
        })
        return research_dir
    
    def save_research_paper(self, research_dir: Path, paper: Dict[str, Any]):
        paper_path = research_dir / "research_paper.md"
        paper_path.write_text(paper["paper"])
        info_path = research_dir / "research_info.json"
        info = {
            "title": paper["title"],
            "sources": paper["sources"],
            "metrics": paper.get("metrics", {})
        }
        info_path.write_text(json.dumps(info, indent=2))
        self.save_metadata(research_dir, {
            "status": "completed",
            "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metrics": paper.get("metrics", {})
        })
    
    def save_context_state(self, research_dir: Path, context_data: Dict[str, Any]):
        states_dir = research_dir / "states"
        states_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = states_dir / f"context_state_{timestamp}.json"
        state_path.write_text(json.dumps(context_data, indent=2))
        self._cleanup_old_states(states_dir)
    
    def save_iteration_metrics(self, research_dir: Path, iterations: List[Dict[str, Any]]):
        metrics_path = research_dir / "iteration_metrics.json"
        metrics_path.write_text(json.dumps({
            "iterations": iterations,
            "summary": self._calculate_metrics_summary(iterations)
        }, indent=2))
    
    def save_metadata(self, research_dir: Path, updates: Dict[str, Any]):
        metadata_path = research_dir / "metadata.json"
        if metadata_path.exists():
            current_metadata = json.loads(metadata_path.read_text())
        else:
            current_metadata = {}
        current_metadata.update(updates)
        metadata_path.write_text(json.dumps(current_metadata, indent=2))
    
    def _sanitize_filename(self, name: str) -> str:
        safe_chars = "-_"
        filename = "".join(
            c if c.isalnum() or c in safe_chars else "_"
            for c in name
        )
        return filename.strip("_")
    
    def _cleanup_old_states(self, states_dir: Path):
        state_files = sorted(
            states_dir.glob("context_state_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for file in state_files[5:]:
            file.unlink()
    
    def _calculate_metrics_summary(self, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not iterations:
            return {}
        return {
            "total_iterations": len(iterations),
            "total_decisions": sum(it["decisions"] for it in iterations),
            "total_new_content": sum(it["new_content"] for it in iterations) if "new_content" in iterations[0] else 0,
            "total_time": sum(it["time"] for it in iterations),
            "avg_decisions_per_iteration": (
                sum(it["decisions"] for it in iterations) / len(iterations)
            ),
            "avg_new_content_per_iteration": (
                sum(it["new_content"] for it in iterations) / len(iterations)
            ) if "new_content" in iterations[0] else 0,
            "avg_time_per_iteration": (
                sum(it["time"] for it in iterations) / len(iterations)
            )
        }