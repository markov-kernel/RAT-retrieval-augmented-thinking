"""
Explore agent for managing URL content extraction using Firecrawl.
Handles webpage scraping, content cleaning, and metadata extraction.
Supports parallel URL exploration with rate limiting and retries.
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
from rich import print as rprint
from urllib.parse import urlparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        status: Current status of exploration
        results: Exploration results if completed
    """
    url: str
    priority: float
    rationale: str
    source_query_id: Optional[str] = None
    timestamp: float = time.time()
    status: str = "pending"  # pending, running, completed, failed
    results: Optional[Dict[str, Any]] = None

class ExploreAgent(BaseAgent):
    """
    Agent responsible for managing URL exploration using the Firecrawl API.
    
    Handles webpage scraping, content cleaning, and integration with the
    research context. Supports parallel exploration with rate limiting.
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
        self._url_lock = threading.Lock()
        
        # Configuration
        self.max_urls = self.config.get("max_urls", 10)
        self.min_priority = self.config.get("min_priority", 0.3)
        self.allowed_domains = self.config.get("allowed_domains", [])
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay = self.config.get("retry_delay", 1.0)  # seconds
        self.max_parallel_domains = self.config.get("max_parallel_domains", 3)
        
        # Domain rate limiting
        self._domain_requests: Dict[str, List[float]] = {}
        self._domain_lock = threading.Lock()
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Look at search results, generate decisions to explore unvisited URLs.
        Groups URLs by domain for efficient parallel processing.
        """
        decisions = []
        
        # Get search results with potential URLs
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        
        # Check if we've hit our URL limit
        with self._url_lock:
            if len(self.explored_urls) >= self.max_urls:
                rprint("[yellow]ExploreAgent: Reached maximum URL limit[/yellow]")
                return decisions
        
        # Group URLs by domain for parallel processing
        domain_urls: Dict[str, List[Dict[str, Any]]] = {}
        
        # Process each search result
        for result in search_results:
            if not isinstance(result.content, dict):
                continue
                
            urls = result.content.get("urls", [])
            query_id = result.content.get("query_id")
            
            for url in urls:
                # Basic validation
                if not self._is_valid_url(url):
                    continue
                    
                with self._url_lock:
                    if url in self.explored_urls:
                        continue
                
                # Group by domain
                domain = urlparse(url).netloc.lower()
                if domain not in domain_urls:
                    domain_urls[domain] = []
                
                priority = result.priority * 0.8
                if priority >= self.min_priority:
                    domain_urls[domain].append({
                        "url": url,
                        "priority": priority,
                        "query_id": query_id,
                        "rationale": f"URL found in search results: {url}"
                    })
        
        # Create decisions for each domain's URLs
        for domain, urls in domain_urls.items():
            # Sort URLs by priority within each domain
            urls.sort(key=lambda x: x["priority"], reverse=True)
            
            # Take top N URLs per domain based on priority
            for url_info in urls[:self.max_parallel_domains]:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXPLORE,
                        priority=url_info["priority"],
                        context={
                            "url": url_info["url"],
                            "source_query_id": url_info["query_id"],
                            "rationale": url_info["rationale"],
                            "domain": domain
                        },
                        rationale=f"Parallel exploration of domain {domain}"
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
        Execute a URL exploration decision with retries and rate limiting.
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            url = decision.context["url"]
            domain = decision.context["domain"]
            
            # Check domain rate limits
            if not self._check_domain_rate_limit(domain):
                rprint(f"[yellow]ExploreAgent: Rate limit hit for domain {domain}[/yellow]")
                time.sleep(1.0)  # Basic backoff
            
            # Execute with retries
            for attempt in range(self.max_retries):
                try:
                    results = self.firecrawl.extract_content(url)
                    break
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        rprint(f"[yellow]ExploreAgent: Retry {attempt + 1} for {url}: {e}[/yellow]")
                        time.sleep(self.retry_delay)
                    else:
                        raise
            
            # Track exploration with thread safety
            with self._url_lock:
                self.explored_urls[url] = ExplorationTarget(
                    url=url,
                    priority=decision.priority,
                    rationale=decision.context["rationale"],
                    source_query_id=decision.context.get("source_query_id"),
                    status="completed",
                    results=results
                )
            
            success = bool(results.get("text"))
            if success:
                rprint(f"[green]ExploreAgent: Content extracted from {url}[/green]")
            else:
                rprint(f"[yellow]ExploreAgent: No content extracted from {url}[/yellow]")
                
        except Exception as e:
            rprint(f"[red]ExploreAgent error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "text": "",
                "metadata": {"url": decision.context.get("url", "")}
            }
            
            # Log failed exploration
            with self._url_lock:
                self.explored_urls[url] = ExplorationTarget(
                    url=url,
                    priority=decision.priority,
                    rationale=decision.context["rationale"],
                    source_query_id=decision.context.get("source_query_id"),
                    status="failed",
                    results=results
                )
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
    
    def _check_domain_rate_limit(self, domain: str) -> bool:
        """Check if we can make another request to this domain."""
        with self._domain_lock:
            now = time.time()
            if domain not in self._domain_requests:
                self._domain_requests[domain] = []
            
            # Remove old timestamps
            self._domain_requests[domain] = [
                ts for ts in self._domain_requests[domain]
                if now - ts < 60  # Keep last minute
            ]
            
            # Check rate limit (max 10 requests per minute per domain)
            if len(self._domain_requests[domain]) >= 10:
                return False
            
            # Add new timestamp
            self._domain_requests[domain].append(now)
            return True
    
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
    
    def get_exploration_status(self, url: str) -> Optional[Dict[str, Any]]:
        """Get the current status and results of a URL exploration."""
        with self._url_lock:
            target = self.explored_urls.get(url)
            if target:
                return {
                    "url": target.url,
                    "status": target.status,
                    "results": target.results,
                    "timestamp": target.timestamp
                }
        return None
