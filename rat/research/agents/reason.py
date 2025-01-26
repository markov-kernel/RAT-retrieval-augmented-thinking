"""
Reasoning agent for analyzing research content using DeepSeek.
Handles content analysis, parallel processing, and insight generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
from openai import OpenAI
import os

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class AnalysisTask:
    """
    Represents a content analysis task.
    
    Attributes:
        content: Content to analyze
        priority: Analysis priority (0-1)
        rationale: Why this analysis is needed
        chunk_index: Index if this is part of a chunked analysis
    """
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()

class ReasoningAgent(BaseAgent):
    """
    Agent responsible for analyzing content using the DeepSeek API.
    
    Handles content analysis, parallel processing for large contexts,
    and insight generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning agent.
        
        Args:
            config: Optional configuration parameters
        """
        super().__init__("reason", config)
        
        # Initialize DeepSeek client
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        # Configuration
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.chunk_size = self.config.get("chunk_size", 30000)  # ~30k tokens per chunk
        self.min_priority = self.config.get("min_priority", 0.3)
        
        # Tracking
        self.analysis_tasks: Dict[str, AnalysisTask] = {}
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Analyze the research context and decide on reasoning tasks.
        
        Args:
            context: Current research context
            
        Returns:
            List of reasoning-related decisions
        """
        decisions = []
        
        # Get content needing analysis
        search_results = context.get_content(
            "main",
            ContentType.SEARCH_RESULT
        )
        explored_content = context.get_content(
            "main",
            ContentType.URL_CONTENT
        )
        
        # Analyze search results
        if search_results:
            decisions.extend(
                self._create_analysis_decisions(
                    content_items=search_results,
                    content_type="search_results",
                    base_priority=0.9
                )
            )
            
        # Analyze explored content
        if explored_content:
            decisions.extend(
                self._create_analysis_decisions(
                    content_items=explored_content,
                    content_type="url_content",
                    base_priority=0.8
                )
            )
            
        return decisions
        
    def can_handle(self, decision: ResearchDecision) -> bool:
        """
        Check if this agent can handle a decision.
        
        Args:
            decision: Decision to evaluate
            
        Returns:
            True if this agent can handle the decision
        """
        return decision.decision_type == DecisionType.REASON
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a reasoning decision.
        
        Args:
            decision: Reasoning decision to execute
            
        Returns:
            Analysis results and insights
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            
            # Check if content needs chunking
            if len(content.split()) > self.chunk_size:
                results = self._parallel_analyze_content(content, content_type)
            else:
                results = self._analyze_content_chunk(content, content_type)
                
            success = bool(results.get("analysis"))
            if success:
                rprint(f"[green]Analysis completed for {content_type}[/green]")
            else:
                rprint(f"[yellow]No analysis generated for {content_type}[/yellow]")
                
        except Exception as e:
            rprint(f"[red]Analysis error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "analysis": "",
                "insights": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
        
    def _create_analysis_decisions(
        self,
        content_items: List[ContentItem],
        content_type: str,
        base_priority: float
    ) -> List[ResearchDecision]:
        """
        Create analysis decisions for content items.
        
        Args:
            content_items: Content items to analyze
            content_type: Type of content being analyzed
            base_priority: Base priority for these items
            
        Returns:
            List of analysis decisions
        """
        decisions = []
        
        for item in content_items:
            # Skip if priority too low
            if item.priority * base_priority < self.min_priority:
                continue
                
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.REASON,
                    priority=item.priority * base_priority,
                    context={
                        "content": item.content,
                        "content_type": content_type,
                        "metadata": item.metadata
                    },
                    rationale=f"Analyze {content_type} content"
                )
            )
            
        return decisions
        
    def _parallel_analyze_content(
        self,
        content: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Analyze large content in parallel chunks.
        
        Args:
            content: Content to analyze
            content_type: Type of content being analyzed
            
        Returns:
            Combined analysis results
        """
        # Split content into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # Analyze chunks in parallel
        chunk_results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            future_to_chunk = {
                executor.submit(
                    self._analyze_content_chunk,
                    chunk,
                    f"{content_type}_chunk_{i}"
                ): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    result["chunk_index"] = chunk_index
                    chunk_results.append(result)
                except Exception as e:
                    rprint(f"[red]Error in chunk {chunk_index}: {str(e)}[/red]")
                    
        # Combine chunk results
        return self._combine_chunk_results(chunk_results)
        
    def _analyze_content_chunk(
        self,
        content: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Analyze a single chunk of content using DeepSeek.
        
        Args:
            content: Content chunk to analyze
            content_type: Type of content being analyzed
            
        Returns:
            Analysis results for this chunk
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{
                    "role": "system",
                    "content": (
                        "You are an expert research analyst. Analyze the following "
                        "content and extract key insights, patterns, and implications."
                    )
                }, {
                    "role": "user",
                    "content": f"Content type: {content_type}\n\nContent:\n{content}"
                }],
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "analysis": analysis,
                "insights": self._extract_insights(analysis),
                "content_type": content_type
            }
            
        except Exception as e:
            rprint(f"[red]DeepSeek API error: {str(e)}[/red]")
            return {
                "analysis": "",
                "insights": [],
                "content_type": content_type,
                "error": str(e)
            }
            
    def _combine_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine results from multiple analyzed chunks.
        
        Args:
            chunk_results: List of chunk analysis results
            
        Returns:
            Combined analysis
        """
        # Sort chunks by index
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        
        # Combine analyses
        combined_analysis = "\n\n".join(
            chunk["analysis"] for chunk in sorted_chunks
            if chunk.get("analysis")
        )
        
        # Merge insights
        all_insights = []
        for chunk in sorted_chunks:
            all_insights.extend(chunk.get("insights", []))
            
        # Remove duplicates while preserving order
        unique_insights = list(dict.fromkeys(all_insights))
        
        return {
            "analysis": combined_analysis,
            "insights": unique_insights,
            "chunk_count": len(chunk_results)
        }
        
    def _extract_insights(self, analysis: str) -> List[str]:
        """
        Extract key insights from an analysis.
        
        Args:
            analysis: Analysis text to process
            
        Returns:
            List of extracted insights
        """
        # TODO: Use more sophisticated insight extraction
        # For now, split on newlines and filter
        lines = analysis.split("\n")
        insights = []
        
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered items
            if line.startswith(("-", "*", "•")) or (
                len(line) > 2 and line[0].isdigit() and line[1] == "."
            ):
                insights.append(line.lstrip("- *•").strip())
                
        return insights
