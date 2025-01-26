"""
Search agent for managing Perplexity-based research queries.
Handles query refinement, result tracking, and search history management.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
import logging

from ..perplexity_client import PerplexityClient
from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)

@dataclass
class SearchQuery:
    """
    Represents a search query and its context.
    
    Attributes:
        query: The search query text
        priority: Query priority (0-1)
        rationale: Why the query was generated
        parent_query_id: If nested
        timestamp: When created
    """
    query: str
    priority: float
    rationale: str
    parent_query_id: Optional[str] = None
    timestamp: float = time.time()

class SearchAgent(BaseAgent):
    """
    Agent responsible for search operations using Perplexity.
    Dedup logic is now handled in the Orchestrator,
    so we do not repeat queries from here if orchestrator filters them.
    """
    
    def __init__(self, perplexity_client: PerplexityClient, config: Optional[Dict[str, Any]] = None):
        super().__init__("search", config)
        self.perplexity = perplexity_client
        self.query_history: Dict[str, SearchQuery] = {}
        
        self.max_queries = self.config.get("max_queries", 5)
        self.min_priority = self.config.get("min_priority", 0.3)
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        The SearchAgent might propose an initial search if no search results exist,
        but you might also let ReasoningAgent handle it. Minimizing duplication is wise.
        """
        decisions = []
        
        # If no search results at all, we do an initial question
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results and len(self.query_history) == 0:
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.SEARCH,
                    priority=1.0,
                    context={
                        "query": context.initial_question,
                        "rationale": "Initial search for the research question"
                    },
                    rationale="No existing search results found"
                )
            )
        return decisions
        
    def can_handle(self, decision: ResearchDecision) -> bool:
        return decision.decision_type == DecisionType.SEARCH
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        self._enforce_rate_limit()
        
        start_time = time.time()
        success = False
        results = {}
        
        try:
            query = decision.context.get("query", "").strip()
            if not query:
                rprint("[yellow]SearchAgent: Empty query, skipping[/yellow]")
                results = {"content": "", "urls": []}
            else:
                # Execute search
                results = self.perplexity.search(query)
                
                # Add to query history
                query_id = str(len(self.query_history) + 1)
                self.query_history[query_id] = SearchQuery(
                    query=query,
                    priority=decision.priority,
                    rationale=decision.context.get("rationale", ""),
                    parent_query_id=decision.context.get("parent_query_id")
                )
                results["query_id"] = query_id
            
            success = True
            rprint(f"[green]SearchAgent: Search completed for query: '{query}'[/green]")
            
        except Exception as e:
            rprint(f"[red]SearchAgent error: {str(e)}[/red]")
            results = {"error": str(e), "urls": []}
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
        
        return results
