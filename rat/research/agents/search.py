"""
Search agent for managing Perplexity-based research queries.
Now fully asynchronous.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
import logging
import asyncio

from ..perplexity_client import PerplexityClient
from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)


@dataclass
class SearchQuery:
    query: str
    priority: float
    rationale: str
    parent_query_id: Optional[str] = None
    timestamp: float = time.time()


class SearchAgent(BaseAgent):
    """
    Agent responsible for search operations using Perplexity.
    """

    def __init__(self, perplexity_client: PerplexityClient, config: Optional[Dict[str, Any]] = None):
        super().__init__("search", config)
        self.perplexity = perplexity_client
        self.query_history: Dict[str, SearchQuery] = {}
        self.max_queries = self.config.get("max_queries", 5)
        self.min_priority = self.config.get("min_priority", 0.3)

    async def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        decisions = []
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

    async def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        await self._enforce_rate_limit()
        start_time = asyncio.get_event_loop().time()
        success = False
        results = {}
        try:
            query = decision.context.get("query", "").strip()
            if not query:
                rprint("[yellow]SearchAgent: Empty query, skipping[/yellow]")
                results = {"content": "", "urls": []}
            else:
                results = await self.perplexity.search(query)
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
            execution_time = asyncio.get_event_loop().time() - start_time
            self.log_decision(decision, success, execution_time)
        return results