"""
Orchestrator for coordinating the multi-agent research workflow.
Manages agent interactions, research flow, and data persistence.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time
from rich import print as rprint
from pathlib import Path

from .agents.search import SearchAgent
from .agents.explore import ExploreAgent
from .agents.reason import ReasoningAgent
from .agents.execute import ExecutionAgent
from .perplexity_client import PerplexityClient
from .firecrawl_client import FirecrawlClient
from .output_manager import OutputManager
from .agents.base import ResearchDecision, DecisionType
from .agents.context import ResearchContext, ContentType, ContentItem

@dataclass
class ResearchIteration:
    """
    Represents a single iteration of the research process.
    
    Attributes:
        iteration_number: Current iteration number
        decisions_made: List of decisions made
        content_added: New content items added
        metrics: Performance metrics for this iteration
    """
    iteration_number: int
    decisions_made: List[ResearchDecision]
    content_added: List[ContentItem]
    metrics: Dict[str, Any]
    timestamp: float = time.time()

class ResearchOrchestrator:
    """
    Coordinates the multi-agent research workflow.
    
    Manages agent interactions, research flow, and ensures all components
    work together effectively.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the research orchestrator.
        
        Args:
            config: Optional configuration parameters
        """
        self.config = config or {}
        
        # Initialize clients
        self.perplexity = PerplexityClient()
        self.firecrawl = FirecrawlClient()
        
        # Initialize agents
        self.search_agent = SearchAgent(
            self.perplexity,
            self.config.get("search_config")
        )
        self.explore_agent = ExploreAgent(
            self.firecrawl,
            self.config.get("explore_config")
        )
        self.reason_agent = ReasoningAgent(
            self.config.get("reason_config")
        )
        self.execute_agent = ExecutionAgent(
            self.config.get("execute_config")
        )
        
        # Initialize managers
        self.output_manager = OutputManager()
        
        # Configuration
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_new_content = self.config.get("min_new_content", 1)  # Lower threshold since we're getting good content
        self.min_confidence = self.config.get("min_confidence", 0.7)
        
        # State tracking
        self.current_context: Optional[ResearchContext] = None
        self.iterations: List[ResearchIteration] = []
        self.research_dir: Optional[Path] = None
        
    def start_research(self, question: str) -> Dict[str, Any]:
        """
        Start a new research process.
        
        Args:
            question: Research question to investigate
            
        Returns:
            Research results and metrics
        """
        start_time = time.time()
        
        try:
            # Create research directory
            self.research_dir = self.output_manager.create_research_dir(question)
            
            # Initialize research context
            self.current_context = ResearchContext(initial_question=question)
            
            # Main research loop
            iteration = 0
            while iteration < self.max_iterations:
                iteration += 1
                rprint(f"\n[bold cyan]Starting iteration {iteration}[/bold cyan]")
                
                # Run one iteration
                iteration_result = self._run_iteration(iteration)
                self.iterations.append(iteration_result)
                
                # Check if we should continue
                if self._should_terminate(iteration_result):
                    break
                    
            # Generate final output
            results = self._generate_final_output()
            
            # Save results
            if self.research_dir:
                self.output_manager.save_research_paper(self.research_dir, results)
            
            # Calculate overall metrics
            total_time = time.time() - start_time
            results["metrics"] = self._calculate_metrics(total_time)
            
            return results
            
        except Exception as e:
            rprint(f"[red]Research error: {str(e)}[/red]")
            return {
                "error": str(e),
                "paper": "Error occurred during research",
                "metrics": {
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
            }
            
    def _run_iteration(self, iteration_number: int) -> ResearchIteration:
        """
        Run one iteration of the research process.
        
        Args:
            iteration_number: Current iteration number
            
        Returns:
            Results of this iteration
        """
        iteration_start = time.time()
        decisions_made = []
        content_added = []
        
        try:
            # 1. Get decisions from all agents
            all_decisions = self._gather_agent_decisions()
            
            # 2. Sort decisions by priority
            sorted_decisions = sorted(
                all_decisions,
                key=lambda d: d.priority,
                reverse=True
            )
            
            # 3. Execute decisions
            for decision in sorted_decisions:
                agent = self._get_agent_for_decision(decision)
                if agent:
                    try:
                        result = agent.execute_decision(decision)
                        decisions_made.append(decision)
                        
                        # Add results to context
                        if result:
                            # Process result based on decision type
                            if decision.decision_type == DecisionType.SEARCH:
                                # For search results, extract content and urls
                                content_str = result.get('content', '')
                                urls = result.get('urls', [])
                                token_count = self.current_context._estimate_tokens(content_str)
                                
                                content_item = ContentItem(
                                    content_type=self._get_content_type(decision),
                                    content=content_str,
                                    metadata={
                                        "decision_type": decision.decision_type.value,
                                        "iteration": iteration_number,
                                        "urls": urls
                                    },
                                    token_count=token_count,
                                    priority=decision.priority
                                )
                            else:
                                # For other types, handle as before
                                content_str = result if isinstance(result, str) else json.dumps(result)
                                token_count = self.current_context._estimate_tokens(content_str)
                                
                                content_item = ContentItem(
                                    content_type=self._get_content_type(decision),
                                    content=result,
                                    metadata={
                                        "decision_type": decision.decision_type.value,
                                        "iteration": iteration_number
                                    },
                                    token_count=token_count,
                                    priority=decision.priority
                                )
                            self.current_context.add_content("main", content_item=content_item)
                            content_added.append(content_item)
                            
                    except Exception as e:
                        rprint(f"[red]Error executing decision: {str(e)}[/red]")
                        
        except Exception as e:
            rprint(f"[red]Iteration error: {str(e)}[/red]")
            
        # Calculate iteration metrics
        metrics = {
            "iteration_time": time.time() - iteration_start,
            "decisions_made": len(decisions_made),
            "content_added": len(content_added),
            "agent_metrics": self._get_agent_metrics()
        }
        
        return ResearchIteration(
            iteration_number=iteration_number,
            decisions_made=decisions_made,
            content_added=content_added,
            metrics=metrics
        )
        
    def _gather_agent_decisions(self) -> List[ResearchDecision]:
        """
        Gather decisions from all agents.
        
        Returns:
            Combined list of decisions
        """
        all_decisions = []
        
        # Get decisions from each agent
        for agent in [
            self.search_agent,
            self.explore_agent,
            self.reason_agent,
            self.execute_agent
        ]:
            try:
                decisions = agent.analyze(self.current_context)
                all_decisions.extend(decisions)
            except Exception as e:
                rprint(f"[red]Error getting decisions from {agent.name}: {str(e)}[/red]")
                
        return all_decisions
        
    def _get_agent_for_decision(
        self,
        decision: ResearchDecision
    ) -> Optional[Any]:
        """
        Get the appropriate agent for a decision.
        
        Args:
            decision: Decision to handle
            
        Returns:
            Agent that can handle the decision
        """
        agent_map = {
            DecisionType.SEARCH: self.search_agent,
            DecisionType.EXPLORE: self.explore_agent,
            DecisionType.REASON: self.reason_agent,
            DecisionType.EXECUTE: self.execute_agent
        }
        
        return agent_map.get(decision.decision_type)
        
    def _get_content_type(self, decision: ResearchDecision) -> ContentType:
        """
        Map decision type to content type.
        
        Args:
            decision: Decision to map
            
        Returns:
            Appropriate content type
        """
        type_map = {
            DecisionType.SEARCH: ContentType.SEARCH_RESULT,
            DecisionType.EXPLORE: ContentType.EXPLORED_CONTENT,
            DecisionType.REASON: ContentType.ANALYSIS,
            DecisionType.EXECUTE: ContentType.STRUCTURED_OUTPUT
        }
        
        return type_map.get(decision.decision_type, ContentType.OTHER)
        
    def _should_terminate(self, iteration: ResearchIteration) -> bool:
        """
        Check if research should terminate.
        
        Args:
            iteration: Last completed iteration
            
        Returns:
            True if research should stop
        """
        # Check if we found enough content
        if len(iteration.content_added) < self.min_new_content:
            rprint("[yellow]Terminating: Not enough new content found[/yellow]")
            return True
            
        # Check if we have high confidence results
        analysis_content = self.current_context.get_content(
            "main",
            ContentType.ANALYSIS
        )
        if analysis_content:
            latest = analysis_content[-1]
            confidence = latest.content.get("confidence", 0)
            if confidence >= self.min_confidence:
                rprint("[green]Terminating: Reached confidence threshold[/green]")
                return True
                
        return False
        
    def _generate_final_output(self) -> Dict[str, Any]:
        """
        Generate the final research output.
        
        Returns:
            Research paper and metadata
        """
        # Get all content by type
        search_results = self.current_context.get_content(
            "main",
            ContentType.SEARCH_RESULT
        )
        explored_content = self.current_context.get_content(
            "main",
            ContentType.EXPLORED_CONTENT
        )
        analysis = self.current_context.get_content(
            "main",
            ContentType.ANALYSIS
        )
        structured_output = self.current_context.get_content(
            "main",
            ContentType.STRUCTURED_OUTPUT
        )
        
        # Generate paper sections
        sections = []
        
        # Introduction
        sections.append(f"# {self.current_context.initial_question}\n")
        sections.append("## Introduction\n")
        if search_results:
            sections.append(search_results[0].content)
            
        # Main findings
        sections.append("\n## Key Findings\n")
        for result in analysis:
            if isinstance(result.content, dict):
                insights = result.content.get("insights", [])
                for insight in insights:
                    sections.append(f"- {insight}\n")
            else:
                sections.append(f"- {result.content}\n")
                
        # Detailed analysis
        sections.append("\n## Detailed Analysis\n")
        for content in explored_content:
            if isinstance(content.content, dict):
                title = content.content.get("title", "")
                text = content.content.get("text", "")
                if title and text:
                    sections.append(f"\n### {title}\n\n{text}\n")
            else:
                sections.append(f"\n{content.content}\n")
                
        # Technical details
        if structured_output:
            sections.append("\n## Technical Details\n")
            for output in structured_output:
                if isinstance(output.content, dict):
                    if output.content.get("format") == "json":
                        sections.append("```json\n")
                        sections.append(
                            json.dumps(output.content["output"], indent=2)
                        )
                        sections.append("\n```\n")
                    else:
                        sections.append("```\n")
                        sections.append(output.content.get("output", ""))
                        sections.append("\n```\n")
                else:
                    sections.append(f"{output.content}\n")
                    
        # Sources
        sections.append("\n## Sources\n")
        sources = set()
        for content in explored_content:
            url = content.content.get("metadata", {}).get("url")
            if url:
                sources.add(url)
                
        for url in sorted(sources):
            sections.append(f"- {url}\n")
            
        return {
            "paper": "\n".join(sections),
            "title": self.current_context.initial_question,
            "sources": list(sources)
        }
        
    def _calculate_metrics(self, total_time: float) -> Dict[str, Any]:
        """
        Calculate overall research metrics.
        
        Args:
            total_time: Total research time
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "total_time": total_time,
            "iterations": len(self.iterations),
            "total_decisions": sum(
                len(it.decisions_made) for it in self.iterations
            ),
            "total_content": sum(
                len(it.content_added) for it in self.iterations
            ),
            "agent_metrics": self._get_agent_metrics()
        }
        
        # Add per-iteration metrics
        metrics["iterations_data"] = [
            {
                "number": it.iteration_number,
                "time": it.metrics["iteration_time"],
                "decisions": len(it.decisions_made),
                "content": len(it.content_added)
            }
            for it in self.iterations
        ]
        
        return metrics
        
    def _get_agent_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from all agents.
        
        Returns:
            Combined agent metrics
        """
        return {
            "search": self.search_agent.get_metrics(),
            "explore": self.explore_agent.get_metrics(),
            "reason": self.reason_agent.get_metrics(),
            "execute": self.execute_agent.get_metrics()
        }
