"""
Orchestrator for coordinating the multi-agent research workflow.
Manages agent interactions, research flow, and data persistence.
Supports parallel execution of decisions by type.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time
import logging
from rich import print as rprint
from pathlib import Path

from .agents.search import SearchAgent
from .agents.explore import ExploreAgent
from .agents.reason import ReasoningAgent
from .perplexity_client import PerplexityClient
from .firecrawl_client import FirecrawlClient
from .output_manager import OutputManager
from .agents.base import ResearchDecision, DecisionType
from .agents.context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)

@dataclass
class ResearchIteration:
    iteration_number: int
    decisions_made: List[ResearchDecision]
    content_added: List[ContentItem]
    metrics: Dict[str, Any]
    timestamp: float = time.time()

class ResearchOrchestrator:
    """
    Coordinates the multi-agent research workflow.
    Manages agent interactions, research flow, parallel decisions, etc.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.perplexity = PerplexityClient()
        self.firecrawl = FirecrawlClient()
        
        self.search_agent = SearchAgent(
            self.perplexity,
            {
                **(self.config.get("search_config") or {}),
                "max_workers": self.config.get("max_parallel_searches", 10),
                "rate_limit": self.config.get("search_rate_limit", 100)
            }
        )
        self.explore_agent = ExploreAgent(
            self.firecrawl,
            {
                **(self.config.get("explore_config") or {}),
                "max_workers": self.config.get("max_parallel_explores", 10),
                "rate_limit": self.config.get("explore_rate_limit", 100)
            }
        )
        self.reason_agent = ReasoningAgent(
            {
                **(self.config.get("reason_config") or {}),
                "max_workers": self.config.get("max_parallel_reason", 5),
                "rate_limit": self.config.get("reason_rate_limit", 50)
            }
        )
        
        self.output_manager = OutputManager()
        
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_new_content = self.config.get("min_new_content", 1)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        
        self.current_context: Optional[ResearchContext] = None
        self.iterations: List[ResearchIteration] = []
        self.research_dir: Optional[Path] = None

    def start_research(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        try:
            question = " ".join(question.split())
            if not question:
                raise ValueError("Research question cannot be empty")
            rprint(f"\n[bold]Starting research on: {question}[/bold]\n")
            logger.info("Starting research on question='%s'", question)
            
            self.research_dir = self.output_manager.create_research_dir(question)
            self.current_context = ResearchContext(initial_question=question)
            
            iteration = 0
            while iteration < self.max_iterations:
                iteration += 1
                rprint(f"\n[bold cyan]Starting iteration {iteration}[/bold cyan]")
                logger.info("Beginning iteration %d of max %d", iteration, self.max_iterations)
                try:
                    iteration_result = self._run_iteration(iteration)
                    self.iterations.append(iteration_result)
                    if self._should_terminate(iteration_result):
                        break
                except Exception as e:
                    rprint(f"[red]Error in iteration {iteration}: {str(e)}[/red]")
                    logger.exception("Exception in iteration %d", iteration)
                    if iteration == 1:
                        raise
                    break
            results = self._generate_final_output()
            if self.research_dir:
                self.output_manager.save_research_paper(self.research_dir, results)
            total_time = time.time() - start_time
            results["metrics"] = self._calculate_metrics(total_time)
            logger.info("Research completed successfully in %.2f seconds.", total_time)
            return results
        except Exception as e:
            rprint(f"[red]Research error: {str(e)}[/red]")
            logger.exception("Error during the research process.")
            return {
                "error": str(e),
                "paper": "Error occurred during research",
                "metrics": {
                    "error": str(e),
                    "total_time": time.time() - start_time
                }
            }

    def _run_iteration(self, iteration_number: int) -> ResearchIteration:
        iteration_start = time.time()
        all_decisions = []
        content_added = []
        logger.debug("Running iteration %d", iteration_number)
        try:
            reason_decisions = self.reason_agent.analyze(self.current_context)
            if any(d.decision_type == DecisionType.TERMINATE for d in reason_decisions):
                all_decisions.extend(reason_decisions)
                logger.debug("Terminate decision found in reason_decisions, skipping other agents.")
            else:
                search_decisions = self.search_agent.analyze(self.current_context)
                explore_decisions = self.explore_agent.analyze(self.current_context)
                
                all_decisions = reason_decisions + search_decisions + explore_decisions
            
            decisions_by_type = {
                DecisionType.SEARCH: [],
                DecisionType.EXPLORE: [],
                DecisionType.REASON: []
            }
            for d in all_decisions:
                if d.decision_type != DecisionType.TERMINATE:
                    decisions_by_type[d.decision_type].append(d)
            
            # Phase 1: Parallel Search
            if decisions_by_type[DecisionType.SEARCH]:
                logger.debug("Executing %d SEARCH decisions in parallel.", len(decisions_by_type[DecisionType.SEARCH]))
                search_results = self.search_agent.execute_parallel(decisions_by_type[DecisionType.SEARCH])
                for d, res in zip(decisions_by_type[DecisionType.SEARCH], search_results):
                    if res and "error" not in res:
                        item = self._create_content_item(d, res, iteration_number)
                        self.current_context.add_content("main", content_item=item)
                        content_added.append(item)
            
            # Phase 2: Parallel Explore
            if decisions_by_type[DecisionType.EXPLORE]:
                logger.debug("Executing %d EXPLORE decisions in parallel.", len(decisions_by_type[DecisionType.EXPLORE]))
                explore_results = self.explore_agent.execute_parallel(decisions_by_type[DecisionType.EXPLORE])
                for d, res in zip(decisions_by_type[DecisionType.EXPLORE], explore_results):
                    if res and "error" not in res:
                        item = self._create_content_item(d, res, iteration_number)
                        self.current_context.add_content("main", content_item=item)
                        content_added.append(item)
            
            # Phase 3: Parallel Reason
            if decisions_by_type[DecisionType.REASON]:
                logger.debug("Executing %d REASON decisions in parallel.", len(decisions_by_type[DecisionType.REASON]))
                reason_results = self.reason_agent.execute_parallel(decisions_by_type[DecisionType.REASON])
                for d, res in zip(decisions_by_type[DecisionType.REASON], reason_results):
                    if res and "error" not in res:
                        item = self._create_content_item(d, res, iteration_number)
                        self.current_context.add_content("main", content_item=item)
                        content_added.append(item)
            
            # Handle any TERMINATE decisions
            terminate_decisions = [d for d in all_decisions if d.decision_type == DecisionType.TERMINATE]
            if terminate_decisions:
                logger.debug("Handling %d TERMINATE decisions.", len(terminate_decisions))
                for d in terminate_decisions:
                    # The reason agent can handle the actual finalize if needed:
                    result = self.reason_agent.execute_decision(d)
                    if result:
                        item = self._create_content_item(d, result, iteration_number)
                        self.current_context.add_content("main", content_item=item)
                        content_added.append(item)
        
        except Exception as e:
            rprint(f"[red]Iteration error: {str(e)}[/red]")
            logger.exception("Error in iteration %d", iteration_number)
        
        metrics = {
            "iteration_time": time.time() - iteration_start,
            "decisions_made": len(all_decisions),
            "content_added": len(content_added),
            "parallel_metrics": {
                "search_tasks": len(decisions_by_type[DecisionType.SEARCH]),
                "explore_tasks": len(decisions_by_type[DecisionType.EXPLORE]),
                "reason_tasks": len(decisions_by_type[DecisionType.REASON]),
            },
            "agent_metrics": self._get_agent_metrics()
        }
        logger.info(
            "Iteration %d complete. Decisions made=%d, Content added=%d, Duration=%.2fsec",
            iteration_number,
            len(all_decisions),
            len(content_added),
            metrics["iteration_time"]
        )
        return ResearchIteration(iteration_number, all_decisions, content_added, metrics)

    def _create_content_item(self, decision: ResearchDecision,
                             result: Dict[str, Any], iteration_number: int) -> ContentItem:
        if decision.decision_type == DecisionType.SEARCH:
            content_str = result.get('content', '')
            urls = result.get('urls', [])
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.SEARCH_RESULT,
                content=content_str,
                metadata={
                    "decision_type": decision.decision_type.value,
                    "iteration": iteration_number,
                    "urls": urls
                },
                token_count=token_count,
                priority=decision.priority
            )
        elif decision.decision_type == DecisionType.EXPLORE:
            content_str = json.dumps(result)
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.EXPLORED_CONTENT,
                content=result,
                metadata={
                    "decision_type": decision.decision_type.value,
                    "iteration": iteration_number
                },
                token_count=token_count,
                priority=decision.priority
            )
        elif decision.decision_type == DecisionType.REASON:
            content_str = json.dumps(result)
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.ANALYSIS,
                content=result,
                metadata={
                    "decision_type": decision.decision_type.value,
                    "iteration": iteration_number
                },
                token_count=token_count,
                priority=decision.priority
            )
        else:
            # TERMINATE or any fallback
            content_str = json.dumps(result)
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.OTHER,
                content=result,
                metadata={
                    "decision_type": decision.decision_type.value,
                    "iteration": iteration_number
                },
                token_count=token_count,
                priority=decision.priority
            )

    def _should_terminate(self, iteration: ResearchIteration) -> bool:
        logger.debug(
            "Checking if we should terminate at iteration %d. content_added=%d, min_new_content=%d",
            iteration.iteration_number, len(iteration.content_added), self.min_new_content
        )
        # If there's a TERMINATE decision
        terminate_decision = any(d.decision_type == DecisionType.TERMINATE for d in iteration.decisions_made)
        if terminate_decision:
            logger.info("Terminating because ReasoningAgent gave a TERMINATE decision at iteration %d", iteration.iteration_number)
            rprint("[green]Terminating: ReasoningAgent indicated completion.[/green]")
            return True
        # If we got no new content
        if len(iteration.content_added) < self.min_new_content:
            logger.info(
                "Terminating because iteration %d had only %d new items, below min_new_content=%d",
                iteration.iteration_number, len(iteration.content_added), self.min_new_content
            )
            rprint("[yellow]Terminating: No further new content was added.[/yellow]")
            return True
        return False
    
    def _generate_final_output(self) -> Dict[str, Any]:
        search_results = self.current_context.get_content("main", ContentType.SEARCH_RESULT)
        explored_content = self.current_context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis = self.current_context.get_content("main", ContentType.ANALYSIS)
        
        sections = []
        sections.append(f"# {self.current_context.initial_question}\n")
        sections.append("## Introduction\n")
        if search_results:
            sections.append(search_results[0].content)
        sections.append("\n## Key Findings\n")
        for result in analysis:
            if isinstance(result.content, dict):
                insights = result.content.get("insights", [])
                for insight in insights:
                    sections.append(f"- {insight}\n")
            else:
                sections.append(f"- {result.content}\n")
        sections.append("\n## Detailed Analysis\n")
        for content in explored_content:
            if isinstance(content.content, dict):
                title = content.content.get("title", "")
                text = content.content.get("text", "")
                if title and text:
                    sections.append(f"\n### {title}\n\n{text}\n")
            else:
                sections.append(f"\n{content.content}\n")
        sections.append("\n## Sources\n")
        sources = set()
        for c in explored_content:
            url = c.content.get("metadata", {}).get("url")
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
        metrics = {
            "total_time": total_time,
            "iterations": len(self.iterations),
            "total_decisions": sum(len(it.decisions_made) for it in self.iterations),
            "total_content": sum(len(it.content_added) for it in self.iterations),
            "agent_metrics": self._get_agent_metrics()
        }
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
        return {
            "search": self.search_agent.get_metrics(),
            "explore": self.explore_agent.get_metrics(),
            "reason": self.reason_agent.get_metrics()
        }