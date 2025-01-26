"""
Explore agent for managing URL content extraction using Firecrawl.
Handles webpage scraping, content cleaning, and metadata extraction.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
from urllib.parse import urlparse

from ..firecrawl_client import FirecrawlClient
from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class ExplorationTarget:
    """
    Represents a URL to explore and its context.
    
    Attributes:
        url: The URL to explore
        priority: Exploration priority (0-1)
        rationale: Why this URL was selected
        source_query_id: ID of the search query that found this URL
    """
    url: str
    priority: float
    rationale: str
    source_query_id: Optional[str] = None
    timestamp: float = time.time()

class ExploreAgent(BaseAgent):
    """
    Agent responsible for managing URL exploration using the Firecrawl API.
    
    Handles webpage scraping, content cleaning, and integration with the
    research context.
    """
    
    def __init__(self, firecrawl_client: FirecrawlClient, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the explore agent.
        
        Args:
            firecrawl_client: Client for Firecrawl API interactions
            config: Optional configuration parameters
        """
        super().__init__("explore", config)
        self.firecrawl = firecrawl_client
        self.explored_urls: Dict[str, ExplorationTarget] = {}
        
        # Configuration
        self.max_urls = self.config.get("max_urls", 10)
        self.min_priority = self.config.get("min_priority", 0.3)
        self.allowed_domains = self.config.get("allowed_domains", [])
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Analyze the research context and decide on URL exploration actions.
        
        Args:
            context: Current research context
            
        Returns:
            List of exploration-related decisions
        """
        decisions = []
        
        # Get search results to find URLs
        search_results = context.get_content(
            "main",
            ContentType.SEARCH_RESULT
        )
        
        # Check if we've hit our URL limit
        if len(self.explored_urls) >= self.max_urls:
            rprint("[yellow]Explore agent: Maximum number of URLs reached[/yellow]")
            return decisions
        
        # Process each search result
        for result in search_results:
            urls = result.content.get("urls", [])
            query_id = result.content.get("query_id")
            
            for url in urls:
                if not self._is_valid_url(url):
                    continue
                    
                # Skip if already explored
                if url in self.explored_urls:
                    continue
                    
                # Calculate priority based on result priority
                priority = result.priority * 0.8  # Slight reduction from search priority
                
                if priority >= self.min_priority:
                    decisions.append(
                        ResearchDecision(
                            decision_type=DecisionType.EXPLORE,
                            priority=priority,
                            context={
                                "url": url,
                                "source_query_id": query_id,
                                "rationale": f"URL found in search results: {url}"
                            },
                            rationale=f"New URL discovered from search"
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
        return decision.decision_type == DecisionType.EXPLORE
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a URL exploration decision.
        
        Args:
            decision: Exploration decision to execute
            
        Returns:
            Extracted content and metadata
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            url = decision.context["url"]
            
            # Extract content
            results = self.firecrawl.extract_content(url)
            
            # Track the exploration
            self.explored_urls[url] = ExplorationTarget(
                url=url,
                priority=decision.priority,
                rationale=decision.context["rationale"],
                source_query_id=decision.context.get("source_query_id"),
                timestamp=time.time()
            )
            
            success = bool(results.get("text"))
            if success:
                rprint(f"[green]Content extracted: {url}[/green]")
            else:
                rprint(f"[yellow]No content extracted from: {url}[/yellow]")
                
        except Exception as e:
            rprint(f"[red]Exploration error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "text": "",
                "metadata": {"url": url}
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
        
    def _is_valid_url(self, url: str) -> bool:
        """
        Validate a URL for exploration.
        
        Args:
            url: URL to validate
            
        Returns:
            True if the URL is valid and allowed
        """
        try:
            parsed = urlparse(url)
            
            # Basic validation
            if not all([parsed.scheme, parsed.netloc]):
                return False
                
            # Check allowed domains if specified
            if self.allowed_domains:
                domain = parsed.netloc.lower()
                if not any(domain.endswith(d.lower()) for d in self.allowed_domains):
                    return False
                    
            return True
            
        except Exception:
            return False
