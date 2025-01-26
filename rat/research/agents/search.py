"""
Search agent for managing Perplexity-based research queries.
Handles query refinement, result tracking, and search history management.
Supports parallel search execution with rate limiting.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
import threading

from ..perplexity_client import PerplexityClient
from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class SearchQuery:
    query: str
    priority: float
    rationale: str
    parent_query_id: Optional[str] = None
    timestamp: float = time.time()
    status: str = "pending"
    results: Optional[Dict[str, Any]] = None

class SearchAgent(BaseAgent):
    """
    Agent responsible for managing search operations using the Perplexity API.
    Supports parallel search execution with rate limiting.
    """
    def __init__(self, perplexity_client: PerplexityClient, config: Optional[Dict[str, Any]] = None):
        super().__init__("search", config)
        self.perplexity = perplexity_client
        self.query_history: Dict[str, SearchQuery] = {}
        self._query_lock = threading.Lock()
        
        self.max_queries = self.config.get("max_queries", 5)
        self.min_priority = self.config.get("min_priority", 0.3)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)
    
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        decisions = []
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results and len(self.query_history) == 0:
            initial_queries = self._generate_query_variations(context.initial_question)
            for query_info in initial_queries:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.SEARCH,
                        priority=1.0,
                        context={
                            "query": query_info["query"],
                            "rationale": query_info["rationale"],
                            "variation_type": query_info["type"]
                        },
                        rationale=f"Parallel initial search: {query_info['rationale']}"
                    )
                )
        return decisions
    
    def can_handle(self, decision: ResearchDecision) -> bool:
        return decision.decision_type == DecisionType.SEARCH
    
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        start_time = time.time()
        success = False
        results = {}
        try:
            query = decision.context["query"]
            if decision.priority < self.min_priority:
                rprint(f"[yellow]SearchAgent: Priority too low for query '{query}'[/yellow]")
                results = {"content": "", "urls": []}
            else:
                for attempt in range(self.max_retries):
                    try:
                        results = self.perplexity.search(query)
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            rprint(f"[yellow]SearchAgent: Retry {attempt + 1} for query '{query}': {e}[/yellow]")
                            time.sleep(self.retry_delay)
                        else:
                            raise
                with self._query_lock:
                    query_id = str(len(self.query_history) + 1)
                    self.query_history[query_id] = SearchQuery(
                        query=query,
                        priority=decision.priority,
                        rationale=decision.context.get("rationale", ""),
                        parent_query_id=decision.context.get("parent_query_id"),
                        status="completed",
                        results=results
                    )
                    results["query_id"] = query_id
            success = True
            rprint(f"[green]SearchAgent: Search completed for query: '{query}'[/green]")
        except Exception as e:
            rprint(f"[red]SearchAgent error: {str(e)}[/red]")
            results = {"error": str(e), "urls": []}
            with self._query_lock:
                query_id = str(len(self.query_history) + 1)
                self.query_history[query_id] = SearchQuery(
                    query=query,
                    priority=decision.priority,
                    rationale=decision.context.get("rationale", ""),
                    parent_query_id=decision.context.get("parent_query_id"),
                    status="failed",
                    results=results
                )
        finally:
            exec_time = time.time() - start_time
            self.log_decision(decision, success, exec_time)
        return results
    
    def _generate_query_variations(self, question: str) -> List[Dict[str, Any]]:
        sanitized_question = " ".join(question.split())
        variations = [
            {
                "query": sanitized_question,
                "rationale": "Direct search using original question",
                "type": "original"
            }
        ]
        if "vs" in sanitized_question.lower() or "compare" in sanitized_question.lower():
            parts = sanitized_question.lower().split(" vs ")
            if len(parts) == 2:
                variations.extend([
                    {
                        "query": f"detailed information about {parts[0].strip()}",
                        "rationale": f"Focus on first comparison target: {parts[0].strip()}",
                        "type": "comparison_first"
                    },
                    {
                        "query": f"detailed information about {parts[1].strip()}",
                        "rationale": f"Focus on second comparison target: {parts[1].strip()}",
                        "type": "comparison_second"
                    }
                ])
        variations.append({
            "query": f"specific details technical features {sanitized_question}",
            "rationale": "Focus on technical details and features",
            "type": "technical"
        })
        variations.append({
            "query": f"overview market context competitors {sanitized_question}",
            "rationale": "Focus on market context and competition",
            "type": "market"
        })
        return variations
    
    def get_query_status(self, query_id: str) -> Optional[Dict[str, Any]]:
        with self._query_lock:
            q = self.query_history.get(query_id)
            if q:
                return {
                    "query": q.query,
                    "status": q.status,
                    "results": q.results,
                    "timestamp": q.timestamp
                }
        return None