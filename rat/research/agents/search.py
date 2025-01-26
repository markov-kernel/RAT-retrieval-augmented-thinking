"""
Search agent for managing Perplexity-based research queries.
Handles query refinement, result tracking, and search history management.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint

from ..perplexity_client import PerplexityClient
from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class SearchQuery:
    """
    Represents a search query and its context.
    
    Attributes:
        query: The search query text
        priority: Query priority (0-1)
        rationale: Why this query was generated
        parent_query_id: ID of the query that led to this one
        timestamp: When the query was created
    """
    query: str
    priority: float
    rationale: str
    parent_query_id: Optional[str] = None
    timestamp: float = time.time()

class SearchAgent(BaseAgent):
    """
    Agent responsible for managing search operations using the Perplexity API.
    """
    
    def __init__(self, perplexity_client: PerplexityClient, config: Optional[Dict[str, Any]] = None):
        super().__init__("search", config)
        self.perplexity = perplexity_client
        self.query_history: Dict[str, SearchQuery] = {}
        
        # Configuration
        self.max_queries = self.config.get("max_queries", 5)
        self.min_priority = self.config.get("min_priority", 0.3)
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Typically, the search agent doesn't proactively create new queries
        beyond the initial question unless a knowledge gap is found.
        We handle that logic in the ReasoningAgent or Orchestrator.
        """
        decisions = []
        
        # If no search results at all, we do the initial question
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results and len(self.query_history) == 0:
            # Propose an initial search
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
        return (decision.decision_type == DecisionType.SEARCH)
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        start_time = time.time()
        success = False
        results = {}
        
        try:
            query = decision.context["query"]
            
            # Basic safety check on query priority
            if decision.priority < self.min_priority:
                rprint(f"[yellow]SearchAgent: Priority too low for query '{query}'[/yellow]")
                results = {
                    "content": "",
                    "urls": []
                }
            else:
                # Execute search
                results = self.perplexity.search(query)
                
                # Add a query ID
                query_id = str(len(self.query_history) + 1)
                self.query_history[query_id] = SearchQuery(
                    query=query,
                    priority=decision.priority,
                    rationale=decision.context.get("rationale", ""),
                    parent_query_id=decision.context.get("parent_query_id"),
                )
                
                # Insert the query_id for reference
                results["query_id"] = query_id
            
            success = True
            rprint(f"[green]SearchAgent: Search completed for query: '{query}'[/green]")
            
        except Exception as e:
            rprint(f"[red]SearchAgent error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "urls": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
        
        return results
