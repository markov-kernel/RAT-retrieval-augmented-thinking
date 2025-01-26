# Python Files in Repository

## rat/__init__.py

```python
"""
RAT (Retrieval Augmented Thinking) package.
This package provides tools for enhanced AI responses through structured reasoning and research.
"""

from .research.main import main

__all__ = ['main']

__version__ = "0.1.0"
```

## rat/research/__init__.py

```python
"""
RAT Research package initialization.
This module provides research capabilities for the RAT system.
"""

from .perplexity_client import PerplexityClient
from .firecrawl_client import FirecrawlClient
from .orchestrator import ResearchOrchestrator
from .output_manager import OutputManager

__all__ = [
    'PerplexityClient',
    'FirecrawlClient',
    'ResearchOrchestrator',
    'OutputManager'
]
```

## rat/research/__main__.py

```python
"""
Main entry point for executing the research package as a module.
"""

from .main import main

if __name__ == '__main__':
    main()
```

## rat/research/agents/__init__.py

```python
"""
Multi-agent system for research orchestration.
Provides specialized agents for search, exploration, and reasoning.
"""

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContextBranch, ContentType
from .search import SearchAgent
from .explore import ExploreAgent
from .reason import ReasoningAgent

__all__ = [
    'BaseAgent',
    'ResearchContext',
    'ContextBranch',
    'ResearchDecision',
    'DecisionType',
    'ContentType',
    'SearchAgent',
    'ExploreAgent',
    'ReasoningAgent'
]
```

## rat/research/agents/base.py

```python
"""
Base agent interface and core decision-making structures.
Defines the contract that all specialized research agents must implement.
Includes support for parallel processing and concurrency control.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from rich import print as rprint
from concurrent.futures import ThreadPoolExecutor
import threading
import time

class DecisionType(Enum):
    """Types of decisions an agent can make during research."""
    SEARCH = "search"      # New search query needed
    EXPLORE = "explore"    # URL exploration needed
    REASON = "reason"      # Deep analysis needed (deepseek-reasoner)
    TERMINATE = "terminate"# Research complete or no further steps

@dataclass
class ResearchDecision:
    """
    Represents a decision made by an agent during the research process.
    
    Attributes:
        decision_type: The type of action recommended
        priority: Priority level (0-1) for this decision
        context: Additional context or parameters for the decision
        rationale: Explanation for why this decision was made
        id: Optional unique identifier for tracking parallel tasks
    """
    decision_type: DecisionType
    priority: float
    context: Dict[str, Any]
    rationale: str
    id: Optional[str] = None
    
    def __post_init__(self):
        if not 0 <= self.priority <= 1:
            raise ValueError("Priority must be between 0 and 1")
        if not self.id:
            self.id = f"{self.decision_type.value}_{time.time_ns()}"

class BaseAgent(ABC):
    """
    Base class for all research agents.
    
    Each specialized agent (search, explore, reason) must implement
    the analyze method to make decisions based on the current research context.
    Supports parallel execution of decisions with concurrency control.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the agent.
        
        Args:
            name: Unique identifier for this agent instance
            config: Optional configuration parameters
        """
        self.name = name
        self.config = config or {}
        self.decisions_made = []
        self.metrics = {
            "decisions_made": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_execution_time": 0.0,
            "parallel_executions": 0,
            "max_concurrent_tasks": 0,
            "rate_limit_delays": 0,
            "retry_attempts": 0
        }
        
        # Concurrency controls
        self.max_workers = self.config.get("max_workers", 5)
        self.rate_limit = self.config.get("rate_limit", 100)  # requests per minute
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._active_tasks: Set[str] = set()
        self._tasks_lock = threading.Lock()
        self._last_request_time = 0
        self._rate_limit_lock = threading.Lock()
    
    @abstractmethod
    def analyze(self, context: 'ResearchContext') -> List['ResearchDecision']:
        """
        Analyze the current research context and make decisions.
        
        Args:
            context: Current state of the research process
            
        Returns:
            List of decisions recommended by this agent
        """
        pass
    
    def execute_parallel(self, decisions: List['ResearchDecision']) -> List[Dict[str, Any]]:
        """
        Execute multiple decisions in parallel.
        
        Args:
            decisions: List of decisions to execute
            
        Returns:
            List of results from executing the decisions
        """
        if not decisions:
            return []
        
        results = []
        futures = {}
        
        try:
            # Submit all tasks
            for decision in decisions:
                if self._can_start_task():
                    future = self.executor.submit(self._execute_with_rate_limit, decision)
                    futures[future] = decision
                    with self._tasks_lock:
                        self._active_tasks.add(decision.id)
                        self.metrics["max_concurrent_tasks"] = max(
                            self.metrics["max_concurrent_tasks"],
                            len(self._active_tasks)
                        )
            
            # Collect results as they complete
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    rprint(f"[red]Error in parallel execution of {futures[future].id}: {e}[/red]")
                    results.append({"error": str(e), "decision_id": futures[future].id})
                finally:
                    with self._tasks_lock:
                        self._active_tasks.remove(futures[future].id)
            
            self.metrics["parallel_executions"] += len(decisions)
            return results
            
        except Exception as e:
            rprint(f"[red]Error in parallel execution batch: {e}[/red]")
            return [{"error": str(e)}]
    
    def _can_start_task(self) -> bool:
        """Check if we can start a new task based on concurrency limits."""
        with self._tasks_lock:
            return len(self._active_tasks) < self.max_workers
    
    def _execute_with_rate_limit(self, decision: 'ResearchDecision') -> Dict[str, Any]:
        """Execute a decision with rate limiting."""
        with self._rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            if time_since_last < (60 / self.rate_limit):
                sleep_time = (60 / self.rate_limit) - time_since_last
                self.metrics["rate_limit_delays"] += 1
                time.sleep(sleep_time)
            self._last_request_time = time.time()
        
        return self.execute_decision(decision)
    
    def log_decision(self, decision: 'ResearchDecision', success: bool = True, execution_time: float = 0.0):
        """
        Log a decision made by this agent and update metrics.
        
        Args:
            decision: The decision that was made
            success: Whether the decision execution was successful
            execution_time: Time taken to execute the decision
        """
        self.decisions_made.append(decision)
        self.metrics["decisions_made"] += 1
        if success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        self.metrics["total_execution_time"] += execution_time
    
    @abstractmethod
    def can_handle(self, decision: 'ResearchDecision') -> bool:
        """
        Check if this agent can handle a given decision type.
        
        Args:
            decision: Decision to evaluate
            
        Returns:
            True if this agent can handle the decision
        """
        pass
    
    @abstractmethod
    def execute_decision(self, decision: 'ResearchDecision') -> Dict[str, Any]:
        """
        Execute a decision made by this or another agent.
        
        Args:
            decision: Decision to execute
            
        Returns:
            Results of executing the decision
        """
        pass
    
    def get_decision_history(self) -> List['ResearchDecision']:
        """
        Get the history of decisions made by this agent.
        
        Returns:
            List of past decisions
        """
        return self.decisions_made.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics for this agent.
        
        Returns:
            Dictionary of agent metrics
        """
        return self.metrics.copy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
```

## rat/research/agents/context.py

```python
"""
Research context management system.
Handles token counting, content branching, and state management for the research process.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import json
import time
from uuid import uuid4

class ContentType(Enum):
    """Types of content that can be stored in the research context."""
    QUERY = "query"
    SEARCH_RESULT = "search_result"
    URL_CONTENT = "url_content"
    ANALYSIS = "analysis"
    CODE_OUTPUT = "code_output"
    EXPLORED_CONTENT = "explored_content"
    STRUCTURED_OUTPUT = "structured_output"
    OTHER = "other"

@dataclass
class ContentItem:
    """
    A piece of content in the research context.
    
    Attributes:
        content_type: Type of this content
        content: The actual content data
        metadata: Additional information about this content
        token_count: Number of tokens in this content
        priority: Priority of this content (0-1)
        timestamp: When this content was added
        id: Unique identifier for this content item
    """
    content_type: ContentType
    content: Any
    metadata: Dict[str, Any]
    token_count: int
    priority: float = 0.5
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid4()))

@dataclass
class ContextBranch:
    """
    A branch of the research context for parallel processing.
    
    Attributes:
        branch_id: Unique identifier for this branch
        parent_id: ID of the parent branch (if any)
        content_items: Content items in this branch
        token_count: Total tokens in this branch
        created_at: When this branch was created
        metadata: Additional branch-specific metadata
    """
    branch_id: str
    parent_id: Optional[str]
    content_items: List[ContentItem] = field(default_factory=list)
    token_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class ResearchContext:
    """
    Manages the state and evolution of a research session.
    
    Handles token counting, content organization, and parallel processing
    through branching and merging operations.
    """
    
    MAX_TOKENS_PER_BRANCH = 64000  # deepseek-reasoner limit
    
    def __init__(self, initial_question: str):
        """
        Initialize a new research context.
        
        Args:
            initial_question: The research question to investigate
        """
        self.initial_question = initial_question
        self.main_branch = ContextBranch(
            branch_id="main",
            parent_id=None
        )
        self.branches: Dict[str, ContextBranch] = {"main": self.main_branch}
        self.merged_branches: Set[str] = set()
        self.version = 0
    
    def create_branch(self, parent_branch_id: str = "main") -> ContextBranch:
        """
        Create a new branch from an existing one.
        
        Args:
            parent_branch_id: ID of the branch to fork from
            
        Returns:
            The newly created branch
        """
        if parent_branch_id not in self.branches:
            raise ValueError(f"Parent branch {parent_branch_id} not found")
            
        new_branch_id = str(uuid4())
        parent = self.branches[parent_branch_id]
        
        # Create new branch with copied content
        new_branch = ContextBranch(
            branch_id=new_branch_id,
            parent_id=parent_branch_id,
            content_items=parent.content_items.copy(),
            token_count=parent.token_count
        )
        
        self.branches[new_branch_id] = new_branch
        return new_branch
    
    def add_content(self,
                   branch_id: str,
                   content_item: Optional[ContentItem] = None,
                   content_type: Optional[ContentType] = None,
                   content: Optional[Any] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   token_count: Optional[int] = None,
                   priority: float = 0.5) -> ContentItem:
        """
        Add new content to a specific branch.
        
        Can be called with either:
        1. branch_id and content_item
        2. branch_id and individual parameters (content_type, content, metadata, etc.)
        
        Raises ValueError if adding content would exceed the per-branch token limit.
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")
            
        branch = self.branches[branch_id]
        
        if content_item:
            item = content_item
            token_usage = item.token_count
        else:
            if content_type is None or content is None or metadata is None:
                raise ValueError("Must provide either content_item or all of: content_type, content, metadata")
                
            # Estimate tokens if not provided
            if token_count is None:
                token_usage = self._estimate_tokens(str(content))
            else:
                token_usage = token_count
                
            # Create new content item
            item = ContentItem(
                content_type=content_type,
                content=content,
                metadata=metadata,
                token_count=token_usage,
                priority=priority
            )
            
        # Check token limit
        if branch.token_count + token_usage > self.MAX_TOKENS_PER_BRANCH:
            raise ValueError(
                f"Adding this content would exceed the token limit "
                f"({self.MAX_TOKENS_PER_BRANCH}) for branch {branch_id}"
            )
            
        branch.content_items.append(item)
        branch.token_count += token_usage
        
        return item
    
    def merge_branches(self, branch_ids: List[str], target_branch_id: str = "main"):
        """
        Merge multiple branches into a target branch.
        
        Args:
            branch_ids: List of branch IDs to merge
            target_branch_id: Branch to merge into
        """
        if target_branch_id not in self.branches:
            raise ValueError(f"Target branch {target_branch_id} not found")
            
        target = self.branches[target_branch_id]
        merged_content: Dict[str, ContentItem] = {
            item.id: item for item in target.content_items
        }
        
        # Merge each branch
        for branch_id in branch_ids:
            if branch_id not in self.branches:
                raise ValueError(f"Branch {branch_id} not found")
                
            branch = self.branches[branch_id]
            
            # Add unique content items
            for item in branch.content_items:
                if item.id not in merged_content:
                    merged_content[item.id] = item
            
            self.merged_branches.add(branch_id)
            
        # Update target branch
        target.content_items = list(merged_content.values())
        target.token_count = sum(item.token_count for item in target.content_items)
        
        # Increment version after successful merge
        self.version += 1
    
    def get_content(self, 
                   branch_id: str,
                   content_type: Optional[ContentType] = None) -> List[ContentItem]:
        """
        Get content items from a specific branch, optionally filtered by ContentType.
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")
            
        items = self.branches[branch_id].content_items
        
        if content_type:
            items = [item for item in items if item.content_type == content_type]
            
        return items
    
    def _estimate_tokens(self, content: str) -> int:
        """
        Estimate the number of tokens in a piece of content.
        Simple approximation: ~4 characters per token.
        """
        return len(content) // 4
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context to a dictionary for serialization.
        """
        return {
            "initial_question": self.initial_question,
            "version": self.version,
            "branches": {
                bid: {
                    "branch_id": b.branch_id,
                    "parent_id": b.parent_id,
                    "token_count": b.token_count,
                    "created_at": b.created_at,
                    "metadata": b.metadata,
                    "content_items": [
                        {
                            "id": item.id,
                            "content_type": item.content_type.value,
                            "content": item.content,
                            "metadata": item.metadata,
                            "token_count": item.token_count,
                            "priority": item.priority,
                            "timestamp": item.timestamp
                        }
                        for item in b.content_items
                    ]
                }
                for bid, b in self.branches.items()
            },
            "merged_branches": list(self.merged_branches)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchContext':
        """
        Create a context instance from a dictionary.
        """
        context = cls(data["initial_question"])
        context.version = data["version"]
        context.merged_branches = set(data["merged_branches"])
        
        # Reconstruct branches
        context.branches = {}
        for bid, bdata in data["branches"].items():
            branch = ContextBranch(
                branch_id=bdata["branch_id"],
                parent_id=bdata["parent_id"],
                token_count=bdata["token_count"],
                created_at=bdata["created_at"],
                metadata=bdata["metadata"]
            )
            
            # Reconstruct content items
            for idata in bdata["content_items"]:
                item = ContentItem(
                    content_type=ContentType(idata["content_type"]),
                    content=idata["content"],
                    metadata=idata["metadata"],
                    token_count=idata["token_count"],
                    priority=idata.get("priority", 0.5),
                    timestamp=idata["timestamp"],
                    id=idata["id"]
                )
                branch.content_items.append(item)
                
            context.branches[bid] = branch
            
        return context
```

## rat/research/agents/explore.py

```python
"""
Explore agent for managing URL content extraction using Firecrawl.
Handles webpage scraping, content cleaning, and metadata extraction.
Supports parallel URL exploration with rate limiting and retries.

Key Features:
1. Smart URL grouping by domain
2. Batch scraping for multiple URLs from same domain
3. Rate limiting and retry logic
4. Optional structured data extraction
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
        
        # Batch scraping config
        self.use_batch_scrape_threshold = self.config.get("use_batch_scrape_threshold", 3)
        self.max_urls_per_batch = self.config.get("max_urls_per_batch", 10)
        
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
            
            # If we have enough URLs for this domain, use batch scraping
            if len(urls) >= self.use_batch_scrape_threshold:
                # Take top N URLs for batch processing
                batch_urls = urls[:self.max_urls_per_batch]
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXPLORE,
                        priority=max(u["priority"] for u in batch_urls),
                        context={
                            "urls": [u["url"] for u in batch_urls],
                            "source_query_ids": [u["query_id"] for u in batch_urls],
                            "rationale": f"Batch scraping {len(batch_urls)} URLs from domain {domain}",
                            "domain": domain,
                            "is_batch": True
                        },
                        rationale=f"Batch exploration of domain {domain}"
                    )
                )
            else:
                # Otherwise use single-page scraping for each URL
                for url_info in urls[:self.max_parallel_domains]:
                    decisions.append(
                        ResearchDecision(
                            decision_type=DecisionType.EXPLORE,
                            priority=url_info["priority"],
                            context={
                                "url": url_info["url"],
                                "source_query_id": url_info["query_id"],
                                "rationale": url_info["rationale"],
                                "domain": domain,
                                "is_batch": False
                            },
                            rationale=f"Single-page exploration: {url_info['url']}"
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
        Supports both single-page and batch scraping.
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            domain = decision.context["domain"]
            is_batch = decision.context.get("is_batch", False)
            
            # Check domain rate limits
            if not self._check_domain_rate_limit(domain):
                rprint(f"[yellow]ExploreAgent: Rate limit hit for domain {domain}[/yellow]")
                time.sleep(1.0)  # Basic backoff
            
            if is_batch:
                # Batch scraping
                urls = decision.context["urls"]
                source_query_ids = decision.context.get("source_query_ids", [None] * len(urls))
                
                # Execute batch scrape with retries
                for attempt in range(self.max_retries):
                    try:
                        batch_results = self.firecrawl.scrape_urls_batch(urls)
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            rprint(f"[yellow]ExploreAgent: Retry {attempt + 1} for batch: {e}[/yellow]")
                            time.sleep(self.retry_delay)
                        else:
                            raise
                
                # Track each URL's results
                with self._url_lock:
                    for url, result, query_id in zip(urls, batch_results, source_query_ids):
                        self.explored_urls[url] = ExplorationTarget(
                            url=url,
                            priority=decision.priority,
                            rationale=decision.context["rationale"],
                            source_query_id=query_id,
                            status="completed",
                            results=result
                        )
                
                # Return combined results
                results = {
                    "batch_results": batch_results,
                    "urls": urls,
                    "domain": domain
                }
                success = True
                rprint(f"[green]ExploreAgent: Batch-scraped {len(urls)} URLs from {domain}[/green]")
                
            else:
                # Single-page scraping
                url = decision.context["url"]
                
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
                
                # Track exploration
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
            
            if decision.context.get("is_batch"):
                # For batch failures, create error results for each URL
                urls = decision.context["urls"]
                results = {
                    "batch_results": [
                        {
                            "error": str(e),
                            "text": "",
                            "metadata": {"url": url}
                        }
                        for url in urls
                    ],
                    "urls": urls,
                    "domain": domain
                }
                
                # Mark all URLs as failed
                with self._url_lock:
                    for url in urls:
                        self.explored_urls[url] = ExplorationTarget(
                            url=url,
                            priority=decision.priority,
                            rationale=decision.context["rationale"],
                            source_query_id=None,
                            status="failed",
                            results={"error": str(e), "text": "", "metadata": {"url": url}}
                        )
            else:
                # Single URL failure
                url = decision.context["url"]
                results = {
                    "error": str(e),
                    "text": "",
                    "metadata": {"url": url}
                }
                
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
        Validate a URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL is valid
        """
        try:
            if not url:
                return False
                
            # Basic URL parsing
            parsed = urlparse(url)
            if not parsed.netloc:
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
```

## rat/research/agents/reason.py

```python
"""
Reasoning agent for analyzing research content using DeepSeek (deepseek-reasoner).
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
Supports parallel processing of content analysis and decision making.

Enhancement:
1. We still instruct deepseek-reasoner to produce JSON, but it may be malformed.
2. We do a "second pass" with deepseek-chat if parsing fails, passing `response_format={'type':'json_object'}` to obtain valid JSON.
3. This preserves all existing flows while guaranteeing structured JSON where needed.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
from openai import OpenAI
import os
import logging

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)

@dataclass
class AnalysisTask:
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()

class ReasoningAgent(BaseAgent):
    """
    Agent responsible for analyzing content using the DeepSeek API (deepseek-reasoner).
    Splits content if it exceeds 64k tokens, calls the API, merges results, etc.
    Also drives the research forward (search, explore, or terminate).

    - We instruct the deepseek-reasoner to produce JSON, then attempt to parse it.
    - If parsing fails, we call deepseek-chat with response_format={'type':'json_object'}
      to repair/morph the text into valid JSON.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("reason", config)
        
        # Primary client for deepseek-reasoner calls
        self.deepseek_reasoner = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # Secondary client for deepseek-chat calls (to fix malformed JSON)
        # You can reuse the same DEEPSEEK_API_KEY or have a separate one.
        self.deepseek_chat = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.max_tokens_per_call = 64000
        self.max_output_tokens = 8192
        self.request_timeout = self.config.get("deepseek_timeout", 180)
        
        self.min_priority = self.config.get("min_priority", 0.3)

        # For chunk merges
        self.chunked_analyses: Dict[Tuple[str,int], List[Dict[str,Any]]] = {}

        logger.info(
            "ReasoningAgent initialized with max_parallel_tasks=%d, max_tokens_per_call=%d, max_output_tokens=%d",
            self.max_parallel_tasks,
            self.max_tokens_per_call,
            self.max_output_tokens
        )

    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        decisions = []
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results:
            # Generate initial queries if no search results
            initial_queries = self._generate_initial_queries(context.initial_question)
            for query_info in initial_queries:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.SEARCH,
                        priority=1.0,
                        context={
                            "query": query_info["query"],
                            "rationale": query_info["rationale"]
                        },
                        rationale=f"Parallel initial search: {query_info['rationale']}"
                    )
                )
            return decisions
        
        # Identify unprocessed search or explored content
        unprocessed_search = [
            item for item in search_results
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        unprocessed_explored = [
            item for item in explored_content
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        
        for item in unprocessed_search + unprocessed_explored:
            if item.priority < self.min_priority:
                continue
            content_str = str(item.content)
            tokens_estimated = len(content_str) // 4
            if tokens_estimated > self.max_tokens_per_call:
                # Chunk-based approach
                chunks = self._split_content_into_chunks(content_str)
                for idx, chunk in enumerate(chunks):
                    decisions.append(
                        ResearchDecision(
                            decision_type=DecisionType.REASON,
                            priority=0.9,
                            context={
                                "content": chunk,
                                "content_type": item.content_type.value,
                                "item_id": item.id,
                                "chunk_info": {
                                    "index": idx,
                                    "total_chunks": len(chunks)
                                }
                            },
                            rationale=f"Analyze chunk {idx+1}/{len(chunks)} of {item.content_type.value} content"
                        )
                    )
            else:
                # Single chunk
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.REASON,
                        priority=0.9,
                        context={
                            "content": content_str,
                            "content_type": item.content_type.value,
                            "item_id": item.id
                        },
                        rationale=f"Analyze new {item.content_type.value} content"
                    )
                )
        
        # Additional knowledge gap detection
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_by_source = {}
        for aitem in analysis_items:
            sid = aitem.metadata.get("source_content_id")
            if sid:
                analysis_by_source.setdefault(sid, []).append(aitem)
        
        gaps = self._parallel_identify_gaps(context.initial_question, analysis_by_source, context)
        for gap in gaps:
            if gap["type"] == "search":
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.SEARCH,
                        priority=0.8,
                        context={
                            "query": gap["query"],
                            "rationale": gap["rationale"]
                        },
                        rationale=f"Fill knowledge gap: {gap['rationale']}"
                    )
                )
            elif gap["type"] == "explore":
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXPLORE,
                        priority=0.75,
                        context={
                            "url": gap["url"],
                            "rationale": gap["rationale"]
                        },
                        rationale=f"Explore URL for more details: {gap['rationale']}"
                    )
                )
        
        if self._should_terminate(context):
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.TERMINATE,
                    priority=1.0,
                    context={},
                    rationale="Research question appears sufficiently answered"
                )
            )
        
        return decisions
    
    def can_handle(self, decision: ResearchDecision) -> bool:
        return decision.decision_type == DecisionType.REASON
    
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        start_time = time.time()
        success = False
        results = {}
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            item_id = decision.context["item_id"]
            
            tokens_estimated = len(content) // 4
            if tokens_estimated > self.max_tokens_per_call:
                msg = f"Chunk is still too large: {tokens_estimated} tokens > {self.max_tokens_per_call}"
                logger.warning(msg)
                raise ValueError(msg)
            
            # Instruct deepseek-reasoner to produce JSON in the analysis
            # (Might be malformed; we will fix it with deepseek-chat if needed.)
            single_result = self._analyze_content_chunk(content, content_type)
            success = bool(single_result.get("analysis"))

            chunk_info = decision.context.get("chunk_info")
            if chunk_info:
                # If chunk, store partial
                source_key = (item_id, chunk_info["total_chunks"])
                self.chunked_analyses.setdefault(source_key, []).append(single_result)
                
                # If we now have all chunk analyses, unify them
                if len(self.chunked_analyses[source_key]) == chunk_info["total_chunks"]:
                    merged = self._merge_chunk_analyses(self.chunked_analyses[source_key])
                    del self.chunked_analyses[source_key]
                    results = merged
                    success = True
                else:
                    # Partial chunk result
                    results = {
                        "analysis": "",
                        "insights": [],
                        "content_type": content_type,
                        "partial_chunk": True
                    }
            else:
                # Single chunk, final
                results = single_result
            
            decision.context["analyzed_by_reasoner"] = True
            results["analyzed_item_id"] = item_id
            
            if success and not results.get("partial_chunk"):
                rprint(f"[green]ReasoningAgent: Analysis completed for '{content_type}'[/green]")
            elif results.get("partial_chunk"):
                rprint(f"[yellow]ReasoningAgent: Partial chunk analysis stored for '{content_type}'[/yellow]")
            else:
                rprint(f"[yellow]ReasoningAgent: No analysis produced for '{content_type}'[/yellow]")
                
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            results = {"error": str(e), "analysis": "", "insights": []}
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            logger.info("Reasoning decision executed. Success=%s, time=%.2fsec", success, execution_time)
        return results
    
    def _analyze_content_chunk(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Analyze a chunk of content using DeepSeek.
        Handles both single-page and batch exploration results.
        """
        try:
            # Prepare the content for analysis
            if content_type == ContentType.EXPLORED_CONTENT.value:
                # Parse the content if it's a string
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Handle batch exploration results
                if isinstance(content, dict) and content.get("type") == "batch_exploration":
                    # Special handling for batch results
                    domain = content.get("domain", "")
                    urls = content.get("urls", [])
                    batch_results = content.get("results", [])
                    
                    # Analyze each result in the batch
                    combined_analysis = []
                    combined_insights = []
                    
                    for idx, (url, result) in enumerate(zip(urls, batch_results)):
                        # Skip empty or error results
                        if not result or "error" in result:
                            continue
                            
                        # Analyze the individual page
                        page_analysis = self._analyze_single_page(result, url)
                        if page_analysis:
                            combined_analysis.append(
                                f"Page {idx + 1} ({url}):\n{page_analysis['analysis']}"
                            )
                            combined_insights.extend(page_analysis.get("insights", []))
                    
                    # Combine all analyses
                    return {
                        "analysis": "\n\n".join(combined_analysis),
                        "insights": combined_insights,
                        "content_type": content_type,
                        "domain": domain,
                        "url_count": len(urls),
                        "successful_analyses": len(combined_analysis)
                    }
                else:
                    # Single page result
                    return self._analyze_single_page(content, content.get("metadata", {}).get("url", ""))
            else:
                # Handle search results or other content types
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert research analyst. Analyze the following content "
                            "and provide insights. Format your response as JSON with 'analysis' "
                            "and 'insights' fields. Keep insights concise and focused."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Content type: {content_type}\n\nContent to analyze:\n{content}"
                    }
                ]
                
                response = self.deepseek_reasoner.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    temperature=0.3,
                    timeout=self.request_timeout
                )
                
                try:
                    result_text = response.choices[0].message.content
                    result_json = json.loads(result_text)
                    return {
                        **result_json,
                        "content_type": content_type
                    }
                except json.JSONDecodeError:
                    # If JSON is malformed, try to fix it with deepseek-chat
                    fixed_json = self._transform_malformed_json_with_chat(result_text)
                    result_json = json.loads(fixed_json)
                    return {
                        **result_json,
                        "content_type": content_type
                    }
                    
        except Exception as e:
            logger.exception("Error analyzing content chunk")
            return {
                "error": str(e),
                "analysis": "",
                "insights": [],
                "content_type": content_type
            }
            
    def _analyze_single_page(self, content: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Analyze a single webpage's content.
        
        Args:
            content: The page content and metadata
            url: The source URL
            
        Returns:
            Analysis results for the page
        """
        try:
            # Extract the relevant content
            title = content.get("title", "")
            text = content.get("text", "")
            metadata = content.get("metadata", {})
            
            # Prepare the content for analysis
            page_content = f"Title: {title}\n\nURL: {url}\n\nContent:\n{text}"
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert research analyst. Analyze the following webpage "
                        "content and provide insights. Format your response as JSON with "
                        "'analysis' and 'insights' fields. Keep insights concise and focused."
                    )
                },
                {
                    "role": "user",
                    "content": page_content
                }
            ]
            
            response = self.deepseek_reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                max_tokens=self.max_output_tokens,
                temperature=0.3,
                timeout=self.request_timeout
            )
            
            try:
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)
                return {
                    **result_json,
                    "url": url,
                    "metadata": metadata
                }
            except json.JSONDecodeError:
                # If JSON is malformed, try to fix it with deepseek-chat
                fixed_json = self._transform_malformed_json_with_chat(result_text)
                result_json = json.loads(fixed_json)
                return {
                    **result_json,
                    "url": url,
                    "metadata": metadata
                }
                
        except Exception as e:
            logger.exception("Error analyzing single page: %s", url)
            return {
                "error": str(e),
                "analysis": "",
                "insights": [],
                "url": url
            }

    def _transform_malformed_json_with_chat(self, possibly_malformed: str) -> str:
        """
        Call deepseek-chat to fix malformed JSON text.
        Returns the corrected JSON string or "[]" if it cannot repair it.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that strictly outputs valid JSON. "
                        "Take the user's input (which may be malformed JSON or random text) "
                        "and produce well-formed JSON that best represents it. "
                        "If the input appears to be an array, output a JSON array. "
                        "Otherwise output a JSON object. "
                        "If you cannot parse it meaningfully, output an empty array."
                    )
                },
                {
                    "role": "user",
                    "content": possibly_malformed
                }
            ]
            response = self.deepseek_chat.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Unable to fix malformed JSON with deepseek-chat: %s", e)
            return "[]"

    def _merge_chunk_analyses(self, partials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple chunk analyses into a single cohesive analysis, 
        combining their 'analysis' text and 'insights'.
        """
        combined_analysis = []
        combined_insights = []
        
        for result in partials:
            if result.get("analysis"):
                combined_analysis.append(result["analysis"])
            if result.get("insights"):
                combined_insights.extend(result["insights"])
        
        final_analysis = "\n\n".join(combined_analysis)
        return {
            "analysis": final_analysis,
            "insights": combined_insights,
            "content_type": partials[0]["content_type"] if partials else "analysis_chunked"
        }

    def _split_content_into_chunks(self, content: str) -> List[str]:
        """
        Split large content into chunks that are below the max token limit.
        Roughly: 1 token ~ 4 chars. We subtract ~1000 tokens as a safety buffer.
        """
        safe_limit = self.max_tokens_per_call - 1000
        safe_limit_chars = safe_limit * 4
        
        words = content.split()
        chunks = []
        current_chunk_words = []
        current_length = 0
        
        for w in words:
            token_estimate = len(w) + 1  # approximate
            if (current_length + token_estimate) >= safe_limit_chars:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = [w]
                current_length = token_estimate
            else:
                current_chunk_words.append(w)
                current_length += token_estimate
        
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
        
        return chunks

    def _generate_initial_queries(self, question: str) -> List[Dict[str, Any]]:
        """
        We instruct deepseek-reasoner to produce multiple queries in JSON. 
        If it's malformed, we again correct with deepseek-chat for JSON.
        """
        try:
            sanitized_question = " ".join(question.split())
            reasoner_response = self.deepseek_reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Generate exactly 3 to 5 different search queries in valid JSON format. "
                            "Each item in the JSON array must be an object with 'query' and 'rationale' keys. "
                            "No text outside the JSON array."
                        )
                    },
                    {
                        "role": "user",
                        "content": sanitized_question
                    }
                ],
                temperature=0.7,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            raw_text = reasoner_response.choices[0].message.content.strip()
            # Try to parse
            try:
                queries = json.loads(raw_text)
            except json.JSONDecodeError:
                # Fix with deepseek-chat
                corrected = self._transform_malformed_json_with_chat(raw_text)
                try:
                    queries = json.loads(corrected)
                except:
                    # fallback
                    return [{
                        "query": question,
                        "rationale": "Direct fallback search - JSON fix failed"
                    }]
            # Validate minimal structure
            if not isinstance(queries, list):
                raise ValueError("Expected a JSON list of queries.")
            for q in queries:
                if not isinstance(q, dict) or "query" not in q or "rationale" not in q:
                    raise ValueError("Each query must have 'query' and 'rationale'.")
            return queries
        except Exception as e:
            rprint(f"[red]Error generating initial queries: {e}[/red]")
            logger.exception("Error in _generate_initial_queries.")
            return [{
                "query": question,
                "rationale": "Direct fallback search"
            }]

    def _parallel_identify_gaps(self,
                                question: str,
                                analysis_by_source: Dict[str, List[ContentItem]],
                                context: ResearchContext
                               ) -> List[Dict[str, Any]]:
        """
        Attempt to find knowledge gaps in the analysis, possibly creating new search or explore tasks.
        """
        try:
            tasks = []
            for source_id, items in analysis_by_source.items():
                combined_analysis = " ".join(str(item.content) for item in items)
                tasks.append({
                    "question": question,
                    "analysis": combined_analysis,
                    "source_id": source_id
                })
            
            all_gaps = []
            with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
                futures = [
                    executor.submit(
                        self._analyze_single_source_gaps,
                        t["question"],
                        t["analysis"],
                        t["source_id"]
                    )
                    for t in tasks
                ]
                for fut in as_completed(futures):
                    try:
                        all_gaps.extend(fut.result())
                    except Exception as e:
                        rprint(f"[red]Error in gap analysis: {e}[/red]")
                        logger.exception("Error analyzing knowledge gaps.")
            
            return self._deduplicate_gaps(all_gaps)
        except Exception as e:
            rprint(f"[red]Error in parallel gap identification: {e}[/red]")
            logger.exception("Error in _parallel_identify_gaps.")
            return []
    
    def _analyze_single_source_gaps(self, question: str, analysis: str, source_id: str) -> List[Dict[str, Any]]:
        """
        Ask deepseek-reasoner to produce JSON gap info. If malformed, fix with chat.
        """
        try:
            response = self.deepseek_reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Identify knowledge gaps in the current analysis. Output valid JSON array. "
                            "Each item must have: 'type' (search|explore), 'query' or 'url', 'rationale'. "
                            "No text outside the JSON array."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nCurrent Analysis:\n{analysis}"
                    }
                ],
                temperature=0.7,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            raw_text = response.choices[0].message.content.strip()
            
            try:
                gaps = json.loads(raw_text)
            except json.JSONDecodeError:
                # fix with chat
                corrected = self._transform_malformed_json_with_chat(raw_text)
                try:
                    gaps = json.loads(corrected)
                except:
                    # fallback
                    return []
            
            if not isinstance(gaps, list):
                raise ValueError("Expected JSON list.")
            # validate
            for gap in gaps:
                if gap["type"] not in ["search", "explore"]:
                    raise ValueError("type must be search or explore.")
                if gap["type"] == "search" and "query" not in gap:
                    raise ValueError("search item missing query.")
                if gap["type"] == "explore" and "url" not in gap:
                    raise ValueError("explore item missing url.")
                if "rationale" not in gap:
                    raise ValueError("gap missing rationale.")
                gap["source_id"] = source_id
            
            return gaps
        except Exception as e:
            rprint(f"[red]Error analyzing source gaps: {e}[/red]")
            logger.exception("Error analyzing source gaps for source_id=%s", source_id)
            return []

    def _deduplicate_gaps(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique_gaps = {}
        for gap in gaps:
            key = gap["query"] if gap["type"] == "search" else gap["url"]
            if key not in unique_gaps:
                unique_gaps[key] = gap
            else:
                # keep the gap with the longer rationale
                if len(gap["rationale"]) > len(unique_gaps[key]["rationale"]):
                    unique_gaps[key] = gap
        return list(unique_gaps.values())

    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Asks deepseek-reasoner for a JSON {complete: bool, confidence: float, missing: []}.
        If malformed, we fix it with chat. If we get `complete==true and confidence>=0.8`, we return True.
        """
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if not analysis_items:
            return False

        combined_analysis = " ".join(str(a.content) for a in analysis_items)
        try:
            response = self.deepseek_reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research completion analyst. Evaluate if we have enough info. "
                            "Return JSON with {\"complete\": bool, \"confidence\": 0..1, \"missing\": []}."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {context.initial_question}\n\nFindings:\n{combined_analysis}"
                    }
                ],
                temperature=0.7,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            raw_text = response.choices[0].message.content
            # try parse
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                # fix with chat
                corrected = self._transform_malformed_json_with_chat(raw_text)
                try:
                    result = json.loads(corrected)
                except:
                    return False

            complete = bool(result.get("complete", False))
            confidence = float(result.get("confidence", 0))
            return (complete and confidence >= 0.8)
        except Exception as e:
            rprint(f"[red]Error checking termination: {e}[/red]")
            logger.exception("Error in _should_terminate check.")
            return False
```

## rat/research/agents/search.py

```python
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
```

## rat/research/firecrawl_client.py

```python
"""
Firecrawl client for web scraping functionality.
This module handles interactions with the Firecrawl API for extracting content
from web pages and processing the extracted data.

Key Features:
1. Single-page extraction (primary /scrape endpoint)
2. Batch scraping for multiple URLs
3. Optional LLM-based structured data extraction
4. Content cleaning and formatting
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from rich import print as rprint
from firecrawl import FirecrawlApp
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class FirecrawlClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        # Configuration
        self.config = config or {}
        self.request_timeout = self.config.get("request_timeout", 60)  # 60s default
        
        # Initialize client (timeout will be used in requests)
        self.app = FirecrawlApp(api_key=self.api_key)
        
        logger.info(
            "FirecrawlClient initialized with timeout=%d",
            self.request_timeout
        )
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single webpage using Firecrawl's /scrape endpoint.
        This is the primary method for single-page extraction, returning markdown.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            logger.info("Extracting content from URL: %s (single scrape)", url)
            
            # Make the request to scrape the URL with timeout
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                    'request_timeout': self.request_timeout
                }
            )
            
            return self._process_extracted_content(result.get('data', {}), url)
            
        except Exception as e:
            rprint(f"[red]Firecrawl API request failed (single scrape) for {url}: {str(e)}[/red]")
            logger.exception("Error extracting content from URL (single scrape): %s", url)
            return {
                "title": "",
                "text": "",
                "metadata": {
                    "url": url,
                    "error": str(e)
                }
            }

    def scrape_urls_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Batch scrape multiple URLs in one Firecrawl call using /batch/scrape.
        More efficient than multiple single-page scrapes when you have several URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of processed results, one per URL (preserving order)
        """
        if not urls:
            return []
            
        logger.info("Batch-scraping %d URLs", len(urls))
        
        try:
            # Ensure all URLs have protocols
            urls = [
                f"https://{url}" if not url.startswith(('http://', 'https://')) else url
                for url in urls
            ]
            
            # Call Firecrawl batch scrape
            result = self.app.batch_scrape_urls(
                urls,
                params={
                    'formats': ['markdown'],
                    'request_timeout': self.request_timeout
                }
            )
            
            # Process each page in the batch
            batch_data = result.get('data', [])
            processed_results = []
            
            for page_data in batch_data:
                # Each page_data is similar to a single scrape result
                processed = self._process_extracted_content(
                    page_data,
                    page_data.get('metadata', {}).get('sourceURL', '')
                )
                processed_results.append(processed)
                
            return processed_results
            
        except Exception as e:
            rprint(f"[red]Firecrawl batch scrape failed: {str(e)}[/red]")
            logger.exception("Error in batch_scrape_urls for URLs: %s", urls)
            # Return empty results for all URLs
            return [
                {
                    "title": "",
                    "text": "",
                    "metadata": {
                        "url": url,
                        "error": str(e)
                    }
                }
                for url in urls
            ]

    def extract_data(self, url: str, prompt: str) -> Dict[str, Any]:
        """
        Extract structured data from a webpage using Firecrawl's LLM capabilities.
        Uses the /scrape endpoint with formats=['json'] and a custom prompt.
        
        Args:
            url: The URL to extract data from
            prompt: Instructions for the LLM about what data to extract
            
        Returns:
            Dict containing the extracted structured data and metadata
        """
        try:
            logger.info(
                "Extracting structured data from URL: %s with prompt='%s'",
                url, prompt
            )
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['json'],
                    'jsonOptions': {
                        'prompt': prompt
                    },
                    'request_timeout': self.request_timeout
                }
            )
            
            data = result.get('data', {})
            extracted = data.get('json', {})
            meta = data.get('metadata', {})
            
            return {
                'url': meta.get('sourceURL', url),
                'extracted_fields': extracted,
                'metadata': meta
            }
            
        except Exception as e:
            rprint(f"[red]Firecrawl structured extraction failed for {url}: {str(e)}[/red]")
            logger.exception("Error extracting structured data from URL: %s", url)
            return {
                'url': url,
                'extracted_fields': {},
                'metadata': {'error': str(e)}
            }
            
    def _process_extracted_content(self, data: Dict[str, Any], original_url: str) -> Dict[str, Any]:
        """
        Process and clean the extracted content.
        
        Args:
            data: Raw API response data
            original_url: The original URL that was scraped
            
        Returns:
            Processed and cleaned content
        """
        metadata = data.get("metadata", {})
        markdown_content = data.get("markdown", "")
        
        processed = {
            "title": metadata.get("title", metadata.get("ogTitle", "")),
            "text": self._clean_text(markdown_content),
            "metadata": {
                "url": metadata.get("sourceURL", original_url),
                "author": metadata.get("author", ""),
                "published_date": metadata.get("publishedDate", ""),
                "domain": metadata.get("ogSiteName", ""),
                "word_count": len(markdown_content.split()) if markdown_content else 0,
                "language": metadata.get("language", ""),
                "status_code": metadata.get("statusCode", 200)
            }
        }
        
        return processed
        
    def _clean_text(self, text: str) -> str:
        """
        Clean and format extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and formatted text
        """
        if not text:
            return ""
            
        # Remove extra whitespace while preserving markdown formatting
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Preserve markdown headings and lists
            if line.strip().startswith(("#", "-", "*", "1.", ">")):
                cleaned_lines.append(line)
            else:
                # Clean normal text lines
                cleaned = " ".join(line.split())
                if cleaned:
                    cleaned_lines.append(cleaned)
        
        return "\n\n".join(cleaned_lines)
```

## rat/research/main.py

```python
"""
Main entry point for the multi-agent research system.
Provides a command-line interface for conducting research using specialized agents.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import time

from .orchestrator import ResearchOrchestrator
from .output_manager import OutputManager

console = Console()

def setup_logging():
    """
    Configure logging so that logs are overwritten on each run.
    """
    log_path = "rat.log"
    # Remove the log file if it exists, so we overwrite on every new run
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        filename=log_path,
        filemode="w",  # Overwrite mode
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    
    logging.info("Logger initialized. All logs will be written to rat.log and overwritten each run.")

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    load_dotenv()
    
    config = {
        'max_iterations': int(os.getenv('MAX_ITERATIONS', '5')),
        'min_new_content': int(os.getenv('MIN_NEW_CONTENT', '3')),
        'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.7')),
        'search_config': {
            'max_results': int(os.getenv('MAX_SEARCH_RESULTS', '10')),
            'min_relevance': float(os.getenv('MIN_SEARCH_RELEVANCE', '0.6')),
            'api_key': os.getenv('PERPLEXITY_API_KEY')
        },
        'explore_config': {
            'max_urls': int(os.getenv('MAX_URLS', '20')),
            'min_priority': float(os.getenv('MIN_URL_PRIORITY', '0.5')),
            'allowed_domains': json.loads(os.getenv('ALLOWED_DOMAINS', '[]')),
            'api_key': os.getenv('FIRECRAWL_API_KEY')
        },
        'reason_config': {
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '4000')),
            'min_confidence': float(os.getenv('MIN_ANALYSIS_CONFIDENCE', '0.7')),
            'parallel_threads': int(os.getenv('PARALLEL_ANALYSIS_THREADS', '4')),
            'api_key': os.getenv('DEEPSEEK_API_KEY')
        },
        'execute_config': {
            'max_code_length': int(os.getenv('MAX_CODE_LENGTH', '1000')),
            'max_retries': int(os.getenv('MAX_RETRIES', '3')),
            'timeout': int(os.getenv('TIMEOUT_SECONDS', '30')),
            'api_key': os.getenv('CLAUDE_API_KEY')
        }
    }
    return config

def display_welcome():
    """Display welcome message and system information."""
    welcome_text = """
# RAT - Retrieval Augmented Thinking

Welcome to the multi-agent research system! This tool helps you conduct comprehensive research using:

1. Search Agent (Perplexity) - Intelligent web searching
2. Explore Agent (Firecrawl) - URL content extraction
3. Reasoning Agent (DeepSeek) - Content analysis
4. Execution Agent (Claude) - Code generation and structured output

Enter your research question below, or type 'help' for more information.
"""
    console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))

def display_help():
    """Display help information."""
    help_text = """
## Available Commands

- `research <question>` - Start a new research session
- `config` - View current configuration
- `metrics` - View research metrics
- `help` - Display this help message
- `exit` - Exit the system

## Tips

- Be specific in your research questions
- Use quotes for exact phrases
- Type 'exit' to quit at any time
"""
    console.print(Panel(Markdown(help_text), title="Help", border_style="green"))

def run_research(question: str, config: Dict[str, Any]) -> None:
    """Run research with the given question."""
    orchestrator = ResearchOrchestrator(config)
    results = orchestrator.start_research(question)
    
    if "error" in results:
        logger.error("Research error: %s", results['error'])
        console.print(f"[red]Research error: {results['error']}[/red]")
    else:
        logger.info("Research completed successfully")
        console.print(Panel(Markdown(results["paper"]), title="Research Results", border_style="green"))

def main():
    """Main entry point for the research system."""
    import argparse
    
    # Setup logging once at startup
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting RAT main program...")

    parser = argparse.ArgumentParser(description="RAT - Retrieval Augmented Thinking")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Start in interactive mode")
    parser.add_argument("question", nargs="?", 
                       help="Research question (if not using interactive mode)")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.interactive:
        display_welcome()
        commands = WordCompleter(['research', 'config', 'metrics', 'help', 'exit'])
        
        orchestrator: Optional[ResearchOrchestrator] = None
        
        while True:
            try:
                user_input = prompt('RAT> ', completer=commands).strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() == 'exit':
                    logger.info("User exited the interactive session")
                    console.print("[yellow]Exiting research system...[/yellow]")
                    break
                    
                if user_input.lower() == 'help':
                    display_help()
                    continue
                    
                if user_input.lower() == 'config':
                    logger.info("Displaying configuration")
                    console.print(Panel(json.dumps(config, indent=2), title="Configuration", border_style="cyan"))
                    continue
                    
                if user_input.lower() == 'metrics' and orchestrator:
                    logger.info("Calculating research metrics")
                    metrics = orchestrator._calculate_metrics(time.time())
                    console.print(Panel(json.dumps(metrics, indent=2), title="Research Metrics", border_style="magenta"))
                    continue
                    
                if user_input.lower().startswith('research '):
                    question = user_input[9:].strip()
                    if not question:
                        logger.warning("Empty research question provided")
                        console.print("[red]Please provide a research question.[/red]")
                        continue
                    logger.info("Starting research with question: %s", question)
                    run_research(question, config)
                    continue
                    
                logger.warning("Unknown command received: %s", user_input)
                console.print("[red]Unknown command. Type 'help' for available commands.[/red]")
                
            except KeyboardInterrupt:
                logger.info("Operation cancelled by user")
                console.print("\n[yellow]Operation cancelled. Type 'exit' to quit.[/yellow]")
                continue
                
            except Exception as e:
                logger.exception("Exception occurred in interactive loop")
                console.print(f"[red]Error: {str(e)}[/red]")
                continue
    else:
        if not args.question:
            parser.error("Research question is required when not in interactive mode")
        logger.info("Starting non-interactive research with question: %s", args.question)
        run_research(args.question, config)

if __name__ == '__main__':
    main()
```

## rat/research/orchestrator.py

```python
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
            # Handle both single-page and batch exploration results
            if "batch_results" in result:
                # For batch results, create a combined content item
                batch_results = result["batch_results"]
                urls = result.get("urls", [])
                domain = result.get("domain", "")
                
                # Combine the batch results into a single content item
                combined_content = {
                    "type": "batch_exploration",
                    "domain": domain,
                    "urls": urls,
                    "results": batch_results
                }
                
                content_str = json.dumps(combined_content)
                token_count = self.current_context._estimate_tokens(content_str)
                
                return ContentItem(
                    content_type=ContentType.EXPLORED_CONTENT,
                    content=combined_content,
                    metadata={
                        "decision_type": decision.decision_type.value,
                        "iteration": iteration_number,
                        "is_batch": True,
                        "domain": domain,
                        "url_count": len(urls)
                    },
                    token_count=token_count,
                    priority=decision.priority
                )
            else:
                # Single-page exploration result
                content_str = json.dumps(result)
                token_count = self.current_context._estimate_tokens(content_str)
                return ContentItem(
                    content_type=ContentType.EXPLORED_CONTENT,
                    content=result,
                    metadata={
                        "decision_type": decision.decision_type.value,
                        "iteration": iteration_number,
                        "is_batch": False,
                        "url": result.get("metadata", {}).get("url", "")
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
        elif decision.decision_type == DecisionType.TERMINATE:
            content_str = json.dumps(result)
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.FINAL_ANALYSIS,
                content=result,
                metadata={
                    "decision_type": decision.decision_type.value,
                    "iteration": iteration_number
                },
                token_count=token_count,
                priority=decision.priority
            )
        else:
            raise ValueError(f"Unknown decision type: {decision.decision_type}")

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
```

## rat/research/output_manager.py

```python
"""
Output manager for research results.
Handles saving research outputs, intermediate results, and performance metrics.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import shutil

class OutputManager:
    def __init__(self):
        self.base_dir = Path("research_outputs")
    
    def create_research_dir(self, question: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{self._sanitize_filename(question[:50])}"
        research_dir = self.base_dir / dir_name
        research_dir.mkdir(parents=True, exist_ok=True)
        self.save_metadata(research_dir, {
            "question": question,
            "started_at": timestamp,
            "status": "in_progress"
        })
        return research_dir
    
    def save_research_paper(self, research_dir: Path, paper: Dict[str, Any]):
        paper_path = research_dir / "research_paper.md"
        paper_path.write_text(paper["paper"])
        info_path = research_dir / "research_info.json"
        info = {
            "title": paper["title"],
            "sources": paper["sources"],
            "metrics": paper.get("metrics", {})
        }
        info_path.write_text(json.dumps(info, indent=2))
        self.save_metadata(research_dir, {
            "status": "completed",
            "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metrics": paper.get("metrics", {})
        })
    
    def save_context_state(self, research_dir: Path, context_data: Dict[str, Any]):
        states_dir = research_dir / "states"
        states_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = states_dir / f"context_state_{timestamp}.json"
        state_path.write_text(json.dumps(context_data, indent=2))
        self._cleanup_old_states(states_dir)
    
    def save_iteration_metrics(self, research_dir: Path, iterations: List[Dict[str, Any]]):
        metrics_path = research_dir / "iteration_metrics.json"
        metrics_path.write_text(json.dumps({
            "iterations": iterations,
            "summary": self._calculate_metrics_summary(iterations)
        }, indent=2))
    
    def save_metadata(self, research_dir: Path, updates: Dict[str, Any]):
        metadata_path = research_dir / "metadata.json"
        if metadata_path.exists():
            current_metadata = json.loads(metadata_path.read_text())
        else:
            current_metadata = {}
        current_metadata.update(updates)
        metadata_path.write_text(json.dumps(current_metadata, indent=2))
    
    def _sanitize_filename(self, name: str) -> str:
        safe_chars = "-_"
        filename = "".join(
            c if c.isalnum() or c in safe_chars else "_"
            for c in name
        )
        return filename.strip("_")
    
    def _cleanup_old_states(self, states_dir: Path):
        state_files = sorted(
            states_dir.glob("context_state_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        for file in state_files[5:]:
            file.unlink()
    
    def _calculate_metrics_summary(self, iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not iterations:
            return {}
        return {
            "total_iterations": len(iterations),
            "total_decisions": sum(it["decisions"] for it in iterations),
            "total_new_content": sum(it["new_content"] for it in iterations) if "new_content" in iterations[0] else 0,
            "total_time": sum(it["time"] for it in iterations),
            "avg_decisions_per_iteration": (
                sum(it["decisions"] for it in iterations) / len(iterations)
            ),
            "avg_new_content_per_iteration": (
                sum(it["new_content"] for it in iterations) / len(iterations)
            ) if "new_content" in iterations[0] else 0,
            "avg_time_per_iteration": (
                sum(it["time"] for it in iterations) / len(iterations)
            )
        }
```

## rat/research/perplexity_client.py

```python
"""
Perplexity API client for web search functionality.
Uses the Perplexity API to perform intelligent web searches and extract relevant information.
"""

import os
import re
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PerplexityClient:
    """
    A simple client that calls Perplexity's /chat/completions endpoint
    using requests. See https://docs.perplexity.ai/api-reference/chat-completions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set in environment variables.")
        self.base_url = "https://api.perplexity.ai/chat/completions"

        # Configuration
        self.config = config or {}
        self.model = self.config.get("model", "sonar")
        self.request_timeout = self.config.get("request_timeout", 60)  # 60s default
        self.system_message = "You are a research assistant helping to find accurate, up-to-date information."
        
        logger.info(
            "PerplexityClient initialized with model=%s, timeout=%d",
            self.model,
            self.request_timeout
        )

    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Perplexity /chat/completions API
        by passing a short system message and a user message in the
        'messages' array, along with any relevant parameters.
        """

        # Build the messages array with system + user roles
        messages = [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Prepare the JSON data to send
        # (feel free to tweak parameters as needed, or add more)
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            # Example: filter the search domain, or remove it if you want all results
            # "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            # presence_penalty and frequency_penalty can be adjusted as needed
            "presence_penalty": 0,
            "frequency_penalty": 1
        }

        # Log what we're about to send (for debug)
        logger.info("Sending request to Perplexity with query='%s' and model=%s", query, self.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=self.request_timeout  # Add timeout to prevent hanging
            )
            if not response.ok:
                # Log and raise an error if the HTTP status is not 200
                logger.error(
                    "Perplexity API returned an error %d: %s",
                    response.status_code,
                    response.text
                )
                response.raise_for_status()

            # Parse response JSON
            resp_json = response.json()

            # The assistant's text is typically in resp_json["choices"][0]["message"]["content"]
            # We'll combine it in 'content' for consistency.
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                content = resp_json["choices"][0]["message"].get("content", "")
            else:
                content = ""

            # Extract citations or references, if present
            urls = self._extract_urls(content)
            # Some Perplexity responses also have a top-level "citations" array
            # to show references. We could merge them into `urls` if we like:
            citations = resp_json.get("citations", [])
            for citation_url in citations:
                if citation_url not in urls:
                    urls.append(citation_url)

            return {
                "content": content,
                "urls": urls
            }

        except requests.RequestException as e:
            logger.exception("Error in Perplexity search request:")
            return {
                "content": "",
                "urls": [],
                "error": str(e)
            }

    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from any text. We also attempt to parse typical references like:
        [Source: https://example.com]
        """
        citation_pattern = r'\[Source:\s*(https?://[^\]]+)\]'
        citation_urls = re.findall(citation_pattern, text)

        url_pattern = r'https?://\S+'
        raw_urls = re.findall(url_pattern, text)

        all_urls = list(set(citation_urls + raw_urls))
        return all_urls
```

## rat_agentic.py

```python
"""
Entry point for the multi-agent research system.
Provides a command-line interface for conducting research using the agent-based approach.
"""

import os
import sys
import json
import logging
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from typing import Dict, Any, Optional

from rat.research.orchestrator import ResearchOrchestrator

def setup_logging():
    log_path = "rat.log"
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(
        filename=log_path,
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.info("Logging setup complete. Logs overwritten each run.")

def create_default_config() -> Dict[str, Any]:
    return {
        "max_iterations": 5,
        "min_new_content": 2,
        "min_confidence": 0.7,
        "search_config": {
            "max_queries": 5,
            "min_priority": 0.3,
            "refinement_threshold": 0.7
        },
        "explore_config": {
            "max_urls": 10,
            "min_priority": 0.3,
            "allowed_domains": []
        },
        "reason_config": {
            "max_parallel_tasks": 3,
            "chunk_size": 30000,
            "min_priority": 0.3
        }
    }

def display_results(results: Dict[str, Any]):
    if "error" in results:
        rprint(f"\n[red]Error during research: {results['error']}[/red]")
        return
    rprint(Panel(
        Markdown(results["paper"]),
        title="[bold green]Research Paper[/bold green]",
        border_style="green"
    ))
    metrics = results.get("metrics", {})
    rprint("\n[bold cyan]Research Metrics:[/bold cyan]")
    rprint(f"Total time: {metrics.get('total_time', 0):.2f} seconds")
    rprint(f"Iterations: {metrics.get('iterations', 0)}")
    rprint(f"Total decisions: {metrics.get('total_decisions', 0)}")
    rprint(f"Total content items: {metrics.get('total_content', 0)}")
    agent_metrics = metrics.get("agent_metrics", {})
    for agent_name, agent_data in agent_metrics.items():
        rprint(f"\n[bold]{agent_name.title()} Agent:[/bold]")
        rprint(f"Decisions made: {agent_data.get('decisions_made', 0)}")
        rprint(f"Successful executions: {agent_data.get('successful_executions', 0)}")
        rprint(f"Failed executions: {agent_data.get('failed_executions', 0)}")
        total_decisions = max(agent_data.get("decisions_made", 1), 1)
        avg_time = agent_data.get("total_execution_time", 0) / total_decisions
        rprint(f"Average execution time: {avg_time:.2f}s")

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Launching RAT from rat_agentic.py")

    style = Style.from_dict({'prompt': 'orange bold'})
    session = PromptSession(style=style)
    
    orchestrator = ResearchOrchestrator(create_default_config())
    
    rprint(Panel.fit(
        "[bold cyan]RAT Multi-Agent Research System[/bold cyan]\n"
        "Conduct research using a coordinated team of specialized AI agents",
        title="[bold cyan]🧠 RAT Research[/bold cyan]",
        border_style="cyan"
    ))
    rprint("[yellow]Commands:[/yellow]")
    rprint(" • Type [bold red]'quit'[/bold red] to exit")
    rprint(" • Type [bold magenta]'config'[/bold magenta] to view current configuration")
    rprint(" • Type [bold magenta]'metrics'[/bold magenta] to view latest metrics")
    rprint(" • Enter your research question to begin\n")
    
    latest_results: Optional[Dict[str, Any]] = None
    
    while True:
        try:
            user_input = session.prompt("\nResearch Question: ", style=style).strip()
            if user_input.lower() == 'quit':
                logger.info("User chose to quit from rat_agentic.py")
                rprint("\nGoodbye! 👋")
                break
            elif user_input.lower() == 'config':
                logger.info("Displaying current configuration.")
                config_str = json.dumps(orchestrator.config, indent=2)
                rprint(Panel(
                    Markdown(f"```json\n{config_str}\n```"),
                    title="[bold cyan]Current Configuration[/bold cyan]",
                    border_style="cyan"
                ))
                continue
            elif user_input.lower() == 'metrics':
                if latest_results:
                    logger.info("User requested metrics display for the latest research.")
                    display_results(latest_results)
                else:
                    rprint("[yellow]No research has been conducted yet[/yellow]")
                continue
            if not user_input:
                logger.debug("User provided empty input; ignoring.")
                continue
            logger.info("Starting research for input question: %s", user_input)
            latest_results = orchestrator.start_research(user_input)
            display_results(latest_results)
        except KeyboardInterrupt:
            logger.warning("User interrupted with Ctrl+C. Returning to prompt.")
            continue
        except EOFError:
            logger.info("EOF received, exiting gracefully.")
            break
        except Exception as e:
            logger.exception("Unhandled exception in main loop.")
            rprint(f"[red]Error: {str(e)}[/red]")

if __name__ == "__main__":
    main()
```

## setup.py

```python
from setuptools import setup, find_packages

setup(
    name="rat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "python-dotenv",
        "rich",
        "prompt_toolkit",
        "requests",
    ],
    entry_points={
        'console_scripts': [
            'rat-research=rat.rat_research:main',
        ],
    },
    author="Skirano",
    description="Retrieval Augmented Thinking - Enhanced AI responses through structured reasoning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Doriandarko/RAT-retrieval-augmented-thinking",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 
```

## test_agent.py

```python
"""
Test script to run the research agent with a sample question.
"""

import os
import logging
from rich import print as rprint
from rat.research.orchestrator import ResearchOrchestrator

def setup_logging():
    """
    Configure logging so that logs are overwritten on each new run.
    """
    log_path = "rat.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        filename=log_path,
        filemode="w",  # Overwrite mode
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    logging.info("Logging setup complete (test_agent). Logs overwritten each run.")

def main():
    # Ensure logging is set up
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting test_agent.py main function.")
    
    # Initialize the orchestrator with default or custom config
    orchestrator = ResearchOrchestrator()
    
    # Define a test question
    question = (
        "What are the main features and pricing of Billit's accounting software, "
        "and how does it compare to competitors in Belgium?"
    )
    rprint(f"[bold cyan]Starting research on: {question}[/bold cyan]")
    logger.info("Test question: %s", question)
    
    # Run the research
    results = orchestrator.start_research(question)
    
    # Print results
    if "error" in results:
        rprint(f"[red]Error: {results['error']}[/red]")
        logger.error("Research ended with an error: %s", results['error'])
    else:
        rprint("\n[bold green]Research completed![/bold green]")
        logger.info("Research completed successfully.")
        
        rprint("\n[bold]Results:[/bold]")
        print(results["paper"])
        
        rprint("\n[bold]Sources:[/bold]")
        for source in results.get("sources", []):
            print(f"- {source}")
        
        rprint("\n[bold]Metrics:[/bold]")
        metrics = results.get("metrics", {})
        print(f"Total time: {metrics.get('total_time', 0):.2f} seconds")
        print(f"Iterations: {metrics.get('iterations', 0)}")
        print(f"Total decisions: {metrics.get('total_decisions', 0)}")
        print(f"Total content items: {metrics.get('total_content', 0)}")

if __name__ == "__main__":
    main() 
```

