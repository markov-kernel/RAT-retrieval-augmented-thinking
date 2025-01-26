"""
Search agent for managing Perplexity-based research queries.
Handles query refinement, result tracking, and search history management.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
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
    
    Handles query refinement, result tracking, and integration with the
    research context.
    """
    
    def __init__(self, perplexity_client: PerplexityClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the search agent.
        
        Args:
            perplexity_client: Client for Perplexity API interactions
            config: Optional configuration parameters
        """
        super().__init__("search", config)
        self.perplexity = perplexity_client
        self.query_history: Dict[str, SearchQuery] = {}
        
        # Configuration
        self.max_queries = self.config.get("max_queries", 5)
        self.min_priority = self.config.get("min_priority", 0.3)
        self.refinement_threshold = self.config.get("refinement_threshold", 0.7)
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Analyze the research context and decide on search actions.
        
        Args:
            context: Current research context
            
        Returns:
            List of search-related decisions
        """
        decisions = []
        
        # Get existing search content
        search_results = context.get_content(
            "main",
            ContentType.SEARCH_RESULT
        )
        
        # Check if we've hit our query limit
        if len(self.query_history) >= self.max_queries:
            rprint("[yellow]Search agent: Maximum number of queries reached[/yellow]")
            return decisions
        
        # Initial search if no results yet
        if not search_results:
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.SEARCH,
                    priority=1.0,
                    context={
                        "query": context.initial_question,
                        "rationale": "Initial search to begin research"
                    },
                    rationale="No existing search results found"
                )
            )
            return decisions
            
        # Analyze existing results for gaps
        knowledge_gaps = self._identify_knowledge_gaps(context)
        
        # Generate refined queries
        for gap in knowledge_gaps:
            if gap.get("priority", 0) < self.min_priority:
                continue
                
            query = self._generate_refined_query(gap, context)
            if query:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.SEARCH,
                        priority=gap.get("priority", 0.5),
                        context={
                            "query": query.query,
                            "rationale": query.rationale,
                            "parent_query_id": query.parent_query_id
                        },
                        rationale=f"Addressing knowledge gap: {gap['description']}"
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
        return decision.decision_type == DecisionType.SEARCH
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a search decision.
        
        Args:
            decision: Search decision to execute
            
        Returns:
            Search results and metadata
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            query = decision.context["query"]
            
            # Execute search
            results = self.perplexity.search(query)
            
            # Track the query
            query_id = str(len(self.query_history) + 1)
            self.query_history[query_id] = SearchQuery(
                query=query,
                priority=decision.priority,
                rationale=decision.context["rationale"],
                parent_query_id=decision.context.get("parent_query_id"),
                timestamp=time.time()
            )
            
            results.update({
                "query_id": query_id,
                "urls": results.get("urls", [])
            })
            
            success = True
            rprint(f"[green]Search completed: {query}[/green]")
            
        except Exception as e:
            rprint(f"[red]Search error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "urls": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
        
    def _identify_knowledge_gaps(self, context: ResearchContext) -> List[Dict[str, Any]]:
        """
        Analyze current context to identify knowledge gaps.
        
        Args:
            context: Current research context
            
        Returns:
            List of identified knowledge gaps
        """
        # Get all content for analysis
        all_content = context.get_content("main")
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        
        # Basic gap identification
        gaps = []
        
        # If we have initial search results, generate follow-up queries
        if search_results:
            # Follow-up on Billit details
            if not any("stakeholders" in str(content.content).lower() for content in search_results):
                gaps.append({
                    "description": "Find key stakeholders and management team at Billit",
                    "priority": 0.9
                })
                
            # Follow-up on competitors
            if not any("market share" in str(content.content).lower() for content in search_results):
                gaps.append({
                    "description": "Research market share and competitive positioning of accounting software in Belgium",
                    "priority": 0.8
                })
                
            # Follow-up on pricing details
            if not any("pricing comparison" in str(content.content).lower() for content in search_results):
                gaps.append({
                    "description": "Compare detailed pricing and features of Belgian accounting software platforms",
                    "priority": 0.7
                })
        else:
            # Initial search if no results yet
            gaps.append({
                "description": "Need initial research on Billit and Belgian accounting software",
                "priority": 1.0
            })
            
        return gaps
        
    def _generate_refined_query(
        self,
        gap: Dict[str, Any],
        context: ResearchContext
    ) -> Optional[SearchQuery]:
        """
        Generate a refined search query to address a knowledge gap.
        
        Args:
            gap: Identified knowledge gap
            context: Current research context
            
        Returns:
            Generated search query or None if no refinement needed
        """
        existing_queries = set(q.query for q in self.query_history.values())
        
        # Generate specific queries based on the gap description
        query = None
        if "stakeholders" in gap["description"].lower():
            query = "Who are the key executives, management team, and stakeholders at Billit Belgium? List their roles and backgrounds."
        elif "market share" in gap["description"].lower():
            query = "What is the market share and competitive positioning of accounting software companies in Belgium? Compare Billit with its main competitors."
        elif "pricing comparison" in gap["description"].lower():
            query = "Compare the detailed pricing, features, and plans of major accounting and e-invoicing software platforms in Belgium including Billit."
        else:
            # Fallback to basic refinement
            query = f"{context.initial_question} {gap['description']}"
        
        if query and query not in existing_queries:
            return SearchQuery(
                query=query,
                priority=gap["priority"],
                rationale=f"Addressing gap: {gap['description']}"
            )
            
        return None
