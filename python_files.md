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
import logging

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions an agent can make during research."""
    SEARCH = "search"      # New search query needed
    EXPLORE = "explore"    # URL exploration needed
    REASON = "reason"      # Deep analysis needed (using Gemini now)
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
    """
    decision_type: DecisionType
    priority: float
    context: Dict[str, Any]
    rationale: str
    
    def __post_init__(self):
        if not 0 <= self.priority <= 1:
            raise ValueError("Priority must be between 0 and 1")

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
        # Rate limit: requests per minute
        self.rate_limit = self.config.get("rate_limit", 100)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._active_tasks: Set[str] = set()
        self._tasks_lock = threading.Lock()
        self._last_request_time = 0.0
        self._rate_limit_lock = threading.Lock()
        
        # Initialize logger
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def _enforce_rate_limit(self):
        """
        Ensure we do not exceed self.rate_limit requests per minute.
        We'll do a simple 'sleep' if we haven't waited long enough
        since the last request.
        """
        if self.rate_limit <= 0:
            return  # no limiting
            
        with self._rate_limit_lock:
            current_time = time.time()
            elapsed = current_time - self._last_request_time
            
            # requests-per-minute => min interval
            min_interval = 60.0 / self.rate_limit
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
                self.metrics["rate_limit_delays"] += 1
            
            self._last_request_time = time.time()
            
    @abstractmethod
    def analyze(self, context: 'ResearchContext') -> List[ResearchDecision]:
        """
        Analyze the current research context and make decisions.
        
        Args:
            context: Current state of the research process
            
        Returns:
            List of decisions recommended by this agent
        """
        pass
    
    def log_decision(self, decision: ResearchDecision, success: bool = True, execution_time: float = 0.0):
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
    def can_handle(self, decision: ResearchDecision) -> bool:
        """
        Check if this agent can handle a given decision type.
        
        Args:
            decision: Decision to evaluate
            
        Returns:
            True if this agent can handle the decision
        """
        pass
    
    @abstractmethod
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a decision made by this or another agent.
        
        Args:
            decision: Decision to execute
            
        Returns:
            Results of executing the decision
        """
        pass
    
    def get_decision_history(self) -> List[ResearchDecision]:
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
    EXPLORED_CONTENT = "explored_content"
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
    
    # Increased limit to match large Gemini input allowance.
    # We'll chunk if needed, up to 1,048,576 tokens total
    MAX_TOKENS_PER_BRANCH = 1048576
    
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
Explore agent for extracting content from URLs.
Now acts as a simple executor that processes EXPLORE decisions from the ReasoningAgent.
"""

from typing import List, Dict, Any, Optional
import logging
from rich import print as rprint

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType
from ..firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)

class ExploreAgent(BaseAgent):
    """
    Agent responsible for extracting content from URLs.
    Acts as an executor for EXPLORE decisions made by the ReasoningAgent.
    """
    
    def __init__(self, firecrawl_client: FirecrawlClient, config=None):
        super().__init__("explore", config)
        self.firecrawl = firecrawl_client
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        No longer generates decisions - URL selection is handled by ReasoningAgent.
        """
        return []
        
    def can_handle(self, decision: ResearchDecision) -> bool:
        """Check if this agent can handle a decision."""
        return decision.decision_type == DecisionType.EXPLORE
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute an EXPLORE decision by scraping the URL.
        
        Args:
            decision: The EXPLORE decision containing the URL to scrape
            
        Returns:
            Dict containing the scraped content
        """
        url = decision.context["url"]
        self.logger.info(f"Exploring URL: {url}")
        
        try:
            scrape_result = self.firecrawl.extract_content(url)
            
            if not scrape_result:
                self.logger.warning(f"No content extracted from URL: {url}")
                return {}
                
            return {
                "url": url,
                "title": scrape_result.get("title", ""),
                "text": scrape_result.get("text", ""),
                "metadata": {
                    **scrape_result.get("metadata", {}),
                    "relevance": decision.context.get("relevance", 0.0),
                    "rationale": decision.context.get("rationale", "")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error exploring URL {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e)
            }
```

## rat/research/agents/reason.py

```python
"""
Reasoning agent for analyzing research content using the Gemini 2.0 Flash Thinking model.
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
Supports parallel processing of content analysis and decision making.

Key responsibilities:
1. Analyzing content using Gemini
2. Deciding which URLs to explore
3. Identifying knowledge gaps
4. Determining when to terminate research
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
import os
import json
import logging
import re
from urllib.parse import urlparse
import threading
from time import sleep

import google.generativeai as genai

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

# Get loggers
logger = logging.getLogger(__name__)
api_logger = logging.getLogger('api.gemini')

@dataclass
class AnalysisTask:
    """
    Represents a content analysis task.
    
    Attributes:
        content: The textual content to analyze
        priority: Analysis priority (0-1)
        rationale: Why this analysis is needed
        chunk_index: If chunked, the index of this chunk
    """
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()

class ReasoningAgent(BaseAgent):
    """
    Reasoning agent for analyzing research content using the Gemini 2.0 Flash Thinking model.
    Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
    Supports parallel processing of content analysis and decision making.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reasoning agent."""
        super().__init__("reason", config)
        
        # Configure Gemini model
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 50000,
            "response_mime_type": "text/plain"
        }
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"

        # Initialize the model
        self._model = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Configure the Gemini client
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # We'll chunk input up to a 1,048,576 token estimate
        self.max_tokens_per_call = 1048576  
        self.max_output_tokens = 65536
        self.request_timeout = self.config.get("gemini_timeout", 180)

        # For concurrency chunk splitting
        self.chunk_margin = 5000   # Safety margin
        
        # Add max_parallel_tasks from config
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)

        # Priority thresholds
        self.min_priority = self.config.get("min_priority", 0.3)
        self.min_url_relevance = self.config.get("min_url_relevance", 0.6)

        # URL tracking
        self.explored_urls: Set[str] = set()
        
        # Flash-fix rate limiting
        self.flash_fix_rate_limit = self.config.get("flash_fix_rate_limit", 10)
        self._flash_fix_last_time = 0.0
        self._flash_fix_lock = threading.Lock()
        
        logger.info("ReasoningAgent initialized to use Gemini model: %s", self.model_name)
        
        # Tracking
        self.analysis_tasks: Dict[str, AnalysisTask] = {}
        
    @property
    def model(self):
        """Lazy initialization of the Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
        return self._model

    def _get_model(self) -> genai.GenerativeModel:
        """Get or create the Gemini model instance."""
        return self.model  # Use the property

    def _enforce_flash_fix_limit(self):
        """
        Ensure we do not exceed flash_fix_rate_limit requests per minute.
        """
        if self.flash_fix_rate_limit <= 0:
            return
            
        with self._flash_fix_lock:
            current_time = time.time()
            elapsed = current_time - self._flash_fix_last_time
            min_interval = 60.0 / self.flash_fix_rate_limit
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
                self.metrics["rate_limit_delays"] += 1
            
            self._flash_fix_last_time = time.time()
            
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Primary entry point for making decisions about next research steps.
        Now also responsible for URL exploration decisions.
        """
        decisions = []
        
        # 1. If no search results yet, start with a broad search
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results:
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.SEARCH,
                    priority=1.0,
                    context={
                        "query": context.initial_question,
                        "rationale": "Initial broad search for the research question"
                    },
                    rationale="Starting research with a broad search query"
                )
            )
            return decisions
        
        # 2. Process unvisited URLs from search results
        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        explored_urls = {
            item.content.get("url", "") for item in explored_content
            if isinstance(item.content, dict)
        }
        self.explored_urls.update(explored_urls)
        
        # Collect unvisited URLs from search results
        unvisited_urls = set()
        for result in search_results:
            if isinstance(result.content, dict):
                urls = result.content.get("urls", [])
                unvisited_urls.update(
                    url for url in urls 
                    if url not in self.explored_urls
                )
        
        # Filter and prioritize URLs
        relevant_urls = self._filter_relevant_urls(
            list(unvisited_urls), 
            context
        )
        
        # Create EXPLORE decisions for relevant URLs
        for url, relevance in relevant_urls:
            if relevance >= self.min_url_relevance:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXPLORE,
                        priority=relevance,
                        context={
                            "url": url,
                            "relevance": relevance,
                            "rationale": "URL deemed relevant to research question"
                        },
                        rationale=f"URL relevance score: {relevance:.2f}"
                    )
                )
        
        # 3. Process any unanalyzed content
        unprocessed_search = [
            item for item in search_results
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        unprocessed_explored = [
            item for item in explored_content
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        
        # Create REASON decisions for unprocessed content
        for item in unprocessed_search + unprocessed_explored:
            if item.priority < self.min_priority:
                continue
                
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.REASON,
                    priority=0.9,
                    context={
                        "content": item.content,
                        "content_type": item.content_type.value,
                        "item_id": item.id
                    },
                    rationale=f"Analyze new {item.content_type.value} content"
                )
            )
        
        # 4. Check for knowledge gaps - but only if we have context
        # Build combined text from search results, explored content, and existing analysis
        search_text = "\n".join(
            str(item.content) 
            for item in search_results 
            if isinstance(item.content, str)
        )
        explored_text = "\n".join(
            str(item.content) 
            for item in explored_content 
            if isinstance(item.content, str)
        )
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_text = "\n".join(
            item.content.get("analysis", "")
            if isinstance(item.content, dict) else str(item.content)
            for item in analysis_items
        )

        combined_analysis = f"{search_text}\n\n{explored_text}\n\n{analysis_text}".strip()
        
        if combined_analysis:  # Only check for gaps if we have some real context
            gaps = self._identify_knowledge_gaps(
                context.initial_question,
                combined_analysis
            )
            
            # Filter out any gaps with placeholders
            filtered_gaps = []
            for gap in gaps:
                query_str = gap.get("query", "")
                url_str = gap.get("url", "")
                # Skip if the LLM hallucinated placeholders
                if "[" in query_str or "]" in query_str or "[" in url_str or "]" in url_str:
                    self.logger.warning(f"Skipping gap with placeholders: {gap}")
                    continue
                filtered_gaps.append(gap)
            
            # Create new SEARCH or EXPLORE decisions for filtered gaps
            for gap in filtered_gaps:
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
                    if gap["url"] not in self.explored_urls:
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
        else:
            # No real context yet, skip knowledge gap detection
            self.logger.info("Skipping knowledge gap analysis because we have no contextual text.")
        
        # 5. Check if we should terminate
        if self._should_terminate(context):
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.TERMINATE,
                    priority=1.0,
                    context={},
                    rationale="Research question appears to be sufficiently answered"
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
        return decision.decision_type == DecisionType.REASON
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a reasoning decision using Gemini.
        Potentially split content into multiple chunks and run them in parallel, then merge.
        Only stores the actual analysis text, not any suggestions or placeholders.
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            item_id = decision.context["item_id"]
            
            # Convert content to string for token counting
            content_str = str(content)
            tokens_estimated = len(content_str) // 4
            
            if tokens_estimated > self.max_tokens_per_call:
                # Chunked parallel analysis
                chunk_results = self._parallel_analyze_content(content_str, content_type)
                combined = self._combine_chunk_results(chunk_results)
                results = combined
            else:
                # Single chunk
                single_result = self._analyze_content_chunk(content_str, content_type)
                results = single_result
            
            # We only consider it successful if we got actual analysis text
            success = bool(results.get("analysis", "").strip())
            
            # Tag the original content item as "analyzed_by_reasoner"
            decision.context["analyzed_by_reasoner"] = True
            
            # Package only the analysis-related content, no suggestions or placeholders
            final_results = {
                "analysis": results.get("analysis", ""),
                "insights": results.get("insights", []),
                "analyzed_item_id": item_id
            }
            
            if success:
                rprint(f"[green]ReasoningAgent: Analysis completed for content type '{content_type}'[/green]")
            else:
                rprint(f"[yellow]ReasoningAgent: No analysis produced for '{content_type}'[/yellow]")
                
            return final_results
            
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            return {
                "error": str(e),
                "analysis": "",
                "insights": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
        
    def _parallel_analyze_content(self, content: str, content_type: str) -> List[Dict[str, Any]]:
        """
        Splits large text into ~64k token chunks, then spawns parallel requests to Gemini.
        """
        words = content.split()
        chunk_size_words = self.max_tokens_per_call * 4  # approximate word limit
        chunks = []
        
        idx = 0
        while idx < len(words):
            chunk = words[idx: idx + chunk_size_words]
            chunks.append(" ".join(chunk))
            idx += chunk_size_words
        
        chunk_results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            future_map = {
                executor.submit(self._analyze_content_chunk, chunk, f"{content_type}_chunk_{i}"): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_map):
                chunk_index = future_map[future]
                try:
                    res = future.result()
                    res["chunk_index"] = chunk_index
                    chunk_results.append(res)
                except Exception as e:
                    rprint(f"[red]Error in chunk {chunk_index}: {e}[/red]")
        
        return chunk_results
        
    def _analyze_content_chunk(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Calls Gemini with a single-turn prompt to analyze the content.
        We'll store only the final 'analysis' portion, ignoring any next-step JSON the LLM appends.
        """
        # Enforce main rate limit
        self._enforce_rate_limit()
        
        # Create a chat session
        model = self._get_model()
        chat_session = model.start_chat(history=[])

        # We explicitly instruct the model to avoid placeholders and next steps
        prompt = (
            "You are an advanced reasoning model (Gemini 2.0 Flash Thinking). "
            "Analyze the following text for key insights, patterns, or relevant facts. "
            "Provide ONLY factual analysis and insights. "
            "DO NOT include any placeholders (like [company name] or [person]).\n"
            "DO NOT suggest next steps or additional searches.\n"
            "DO NOT output JSON or structured data.\n\n"
            f"CONTENT:\n{content}\n\n"
            "Please provide your analysis below (plain text only):"
        )
        
        response = chat_session.send_message(prompt)
        analysis_text = response.text.strip()
        
        # Extract insights from the analysis text
        insights = self._extract_insights(analysis_text)
        
        return {
            "analysis": analysis_text,
            "insights": insights
        }

    def _extract_insights(self, analysis_text: str) -> List[str]:
        """
        Basic extraction of bullet points or lines from the analysis text.
        """
        lines = analysis_text.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if (
                line.startswith("-") or line.startswith("*") or 
                line.startswith("•") or (len(line) > 2 and line[:2].isdigit())
            ):
                insights.append(line.lstrip("-*•").strip())

        return insights
        
    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merges analysis from multiple chunk results into a single structure.
        Only combines actual analysis text and insights, not suggestions or placeholders.
        """
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        
        # Only combine actual analysis text, not any suggestions
        combined_analysis = "\n\n".join([
            res["analysis"] for res in sorted_chunks 
            if res.get("analysis", "").strip()
        ])
        
        # Combine insights, removing duplicates
        combined_insights = []
        for res in sorted_chunks:
            insights = res.get("insights", [])
            combined_insights.extend(insight for insight in insights if insight.strip())
        
        # Remove duplicates while preserving order
        unique_insights = list(dict.fromkeys(combined_insights))
        
        return {
            "analysis": combined_analysis,
            "insights": unique_insights,
            "chunk_count": len(chunk_results)
        }

    def _fix_json_with_gemini_exp(self, malformed_json_str: str) -> str:
        """
        Attempts to fix malformed JSON by calling a second Gemini model
        (gemini-2.0-flash-exp) that is instructed to output valid JSON only.
        """
        # Enforce flash-fix rate limit
        self._enforce_flash_fix_limit()
        
        # We can define a separate model name for the "JSON-fixer" step.
        fix_model_name = "gemini-2.0-flash-exp"
        
        # Provide generation settings suitable for a JSON-fixing prompt.
        # Typically you want a low creativity / temperature so it just
        # cleans up the JSON and doesn't hallucinate extra structure.
        fix_generation_config = {
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 1,
            "max_output_tokens": 1024,
        }
        
        fix_model = genai.GenerativeModel(
            model_name=fix_model_name,
            generation_config=fix_generation_config
        )
        
        chat_session = fix_model.start_chat(history=[])
        
        # In the prompt, we strongly instruct that it must return valid JSON only
        # with no extra commentary or text.
        prompt = (
            "You are an expert at transforming malformed JSON into correct JSON.\n\n"
            "Your job is to fix any invalid or malformed JSON so it can be parsed.\n"
            "Output *only* the corrected JSON—no extra commentary or text.\n\n"
            "Here is the malformed JSON:\n"
            f"{malformed_json_str}\n\n"
            "Now return only valid JSON."
        )
        
        response = chat_session.send_message(prompt)
        return response.text

    def _call_gemini(self, prompt: str, context: str = "") -> str:
        """Helper method to call Gemini API with logging."""
        api_logger.info(f"Gemini API Request - Context length: {len(context)}")
        api_logger.debug(f"Prompt: {prompt}")
        
        try:
            model = self._get_model()
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            api_logger.debug(f"Response: {response.text}")
            return response.text.strip()
            
        except Exception as e:
            api_logger.error(f"Gemini API error: {str(e)}")
            raise

    def _identify_knowledge_gaps(self, question: str, current_analysis: str) -> List[Dict[str, Any]]:
        """
        Identify missing information and suggest next steps.
        Skip any suggestions containing placeholder text in brackets.
        
        Args:
            question: The research question being investigated
            current_analysis: Combined text from search results, explored content, and analysis
            
        Returns:
            List of gap dictionaries, each containing type (search/explore), query/url, and rationale
        """
        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "identify specific missing information and suggest concrete next steps.\n\n"
            "IMPORTANT RULES:\n"
            "1. DO NOT use placeholders like [company name] or [person] - only suggest specific, concrete queries/URLs\n"
            "2. Base your suggestions ONLY on the actual content provided\n"
            "3. If you can't identify specific gaps, return an empty array\n"
            "4. Each suggestion must be actionable and clearly related to the research question\n\n"
            f"RESEARCH QUESTION: {question}\n\n"
            f"CURRENT ANALYSIS:\n{current_analysis}\n\n"
            "Respond with a JSON array of gaps in this format:\n"
            '[{"type": "search"|"explore", "query"|"url": "specific text", "rationale": "why needed"}]\n'
            "Return ONLY the JSON array, no other text."
        )
        
        try:
            content_str = self._call_gemini(prompt, current_analysis)
            if not content_str:
                return []
            
            # Clean up the response
            content_str = self._clean_json_response(content_str)
            
            try:
                gaps = json.loads(content_str)
                if not isinstance(gaps, list):
                    return []
                
                # Filter out any gaps with placeholders or invalid structure
                filtered_gaps = []
                for gap in gaps:
                    if not isinstance(gap, dict):
                        continue
                        
                    # Validate required fields
                    if "type" not in gap or gap["type"] not in ["search", "explore"]:
                        continue
                    
                    # Get the relevant field based on type
                    content_field = "query" if gap["type"] == "search" else "url"
                    if content_field not in gap or "rationale" not in gap:
                        continue
                    
                    content_str = gap[content_field]
                    rationale_str = gap["rationale"]
                    
                    # Skip if any field contains placeholders
                    if (
                        "[" in content_str or "]" in content_str or 
                        "[" in rationale_str or "]" in rationale_str
                    ):
                        self.logger.warning(
                            f"Skipping gap with placeholders: {content_str}"
                        )
                        continue
                    
                    filtered_gaps.append(gap)
                
                return filtered_gaps
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse knowledge gaps JSON response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error identifying knowledge gaps: {str(e)}")
            return []

    def _clean_json_response(self, content: str) -> str:
        """Helper to clean up JSON responses from the model."""
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            start_idx = content.find("\n", content.find("```")) + 1
            end_idx = content.rfind("```")
            if end_idx > start_idx:
                content = content[start_idx:end_idx].strip()
            else:
                content = content.replace("```", "").strip()
            
        # Remove any "json" language identifier
        content = content.replace("json", "").strip()
        
        return content
    
    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Use Gemini to decide if the research question has been sufficiently answered.
        """
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if len(analysis_items) < 3:  # Need some minimum analysis
            return False

        combined_analysis = "\n".join(
            str(item.content.get("analysis", "")) for item in analysis_items
        )

        prompt = (
            "You are an advanced research assistant. Given a research question and the current analysis, "
            "determine if the question has been sufficiently answered.\n\n"
            f"QUESTION: {context.initial_question}\n\n"
            f"CURRENT ANALYSIS:\n{combined_analysis}\n\n"
            "Respond with a single word: YES if the question is sufficiently answered, NO if not."
        )
        
        try:
            answer = self._call_gemini(prompt, combined_analysis)
            return answer.strip().upper() == "YES"
        except Exception:
            return False

    def _filter_relevant_urls(
        self, 
        urls: List[str], 
        context: ResearchContext
    ) -> List[tuple[str, float]]:
        """
        Filter and score URLs based on relevance to the research question.
        Returns: List of (url, relevance_score) tuples.
        """
        if not urls:
            return []
            
        # Batch URLs for efficient processing
        batch_size = 5
        url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
        relevant_urls = []
        
        for batch in url_batches:
            prompt = (
                "You are an expert at determining URL relevance for research questions.\n"
                "For each URL, analyze its potential relevance to the research question "
                "and provide a relevance score between 0.0 and 1.0.\n\n"
                f"RESEARCH QUESTION: {context.initial_question}\n\n"
                "URLs to evaluate:\n"
            )
            
            for url in batch:
                domain = urlparse(url).netloc
                path = urlparse(url).path
                prompt += f"- {domain}{path}\n"
                
            prompt += (
                "\nRespond with a JSON array of scores in this format:\n"
                "[{\"url\": \"...\", \"score\": 0.X, \"reason\": \"...\"}]\n"
                "ONLY return the JSON array, no other text."
            )
            
            try:
                content = self._call_gemini(prompt)
                
                # Clean markdown formatting if present
                if content.startswith("```"):
                    content = content[content.find("\n")+1:content.rfind("```")].strip()
                content = content.replace("json", "").strip()
                
                scores = json.loads(content)
                
                for score_obj in scores:
                    url = score_obj["url"]
                    score = float(score_obj["score"])
                    relevant_urls.append((url, score))
                    
            except Exception as e:
                logger.error(f"Error scoring URLs: {str(e)}")
                # Fall back to basic keyword matching for this batch
                for url in batch:
                    relevance = self._basic_url_relevance(url, context.initial_question)
                    relevant_urls.append((url, relevance))
                
        return sorted(relevant_urls, key=lambda x: x[1], reverse=True)

    def _basic_url_relevance(self, url: str, question: str) -> float:
        """
        Basic fallback method for URL relevance when LLM scoring fails.
        Returns a score between 0.0 and 1.0.
        """
        # Extract keywords from question
        keywords = set(re.findall(r'\w+', question.lower()))
        
        # Parse URL components
        parsed = urlparse(url)
        domain_parts = parsed.netloc.lower().split('.')
        path_parts = parsed.path.lower().split('/')
        
        # Count keyword matches in domain and path
        domain_matches = sum(1 for part in domain_parts if part in keywords)
        path_matches = sum(1 for part in path_parts if part in keywords)
        
        # Weight domain matches more heavily than path matches
        score = (domain_matches * 0.6 + path_matches * 0.4) / max(len(keywords), 1)
        return min(max(score, 0.0), 1.0)
```

## rat/research/agents/search.py

```python
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
```

## rat/research/firecrawl_client.py

```python
"""
Firecrawl client for web scraping functionality.
This module handles interactions with the Firecrawl API for extracting content
from web pages and processing the extracted data.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from rich import print as rprint
from firecrawl import FirecrawlApp
import logging

load_dotenv()

# Get API logger
api_logger = logging.getLogger('api.firecrawl')

class FirecrawlClient:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a webpage using Firecrawl API.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        api_logger.info(f"Firecrawl API Request - URL: {url}")
        
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                api_logger.debug(f"Added https:// protocol to URL: {url}")
            
            # Make the request to scrape the URL
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                }
            )
            
            processed_result = self._process_extracted_content(result.get('data', {}), url)
            api_logger.debug(f"Processed content from {url}: {len(processed_result.get('text', ''))} chars")
            return processed_result
            
        except Exception as e:
            api_logger.error(f"Firecrawl API request failed for {url}: {str(e)}")
            return {
                "title": "",
                "text": "",
                "metadata": {
                    "url": url,
                    "error": str(e)
                }
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
            "text": markdown_content,
            "metadata": {
                "url": metadata.get("sourceURL", original_url),
                "author": metadata.get("author", ""),
                "published_date": "",  # Firecrawl doesn't provide this directly
                "domain": metadata.get("ogSiteName", ""),
                "word_count": len(markdown_content.split()) if markdown_content else 0,
                "language": metadata.get("language", ""),
                "status_code": metadata.get("statusCode", 200)
            }
        }
        
        # Clean and format the text if needed
        if processed["text"]:
            processed["text"] = self._clean_text(processed["text"])
            api_logger.debug(f"Cleaned text for {original_url}: {len(processed['text'])} chars")
        else:
            api_logger.warning(f"No text content extracted from {original_url}")
        
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

# Clear existing handlers to avoid duplicate logs
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Configure main application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rat.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure API logging
api_logger = logging.getLogger('api')
api_logger.setLevel(logging.DEBUG)
api_handler = logging.FileHandler('rat_api.log', mode='w')
api_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
api_logger.addHandler(api_handler)

# Configure Firecrawl API logging
firecrawl_logger = logging.getLogger('api.firecrawl')
firecrawl_logger.setLevel(logging.DEBUG)
firecrawl_logger.addHandler(api_handler)
firecrawl_logger.propagate = False

# Ensure API logger doesn't propagate to root logger
api_logger.propagate = False

logger = logging.getLogger(__name__)
console = Console()

def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    load_dotenv()
    
    config = {
        'max_iterations': int(os.getenv('MAX_ITERATIONS', '5')),
        'min_new_content': int(os.getenv('MIN_NEW_CONTENT', '3')),
        'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.7')),
        
        # Limit search to 100 requests/min
        'search_config': {
            'max_results': int(os.getenv('MAX_SEARCH_RESULTS', '10')),
            'min_relevance': float(os.getenv('MIN_SEARCH_RELEVANCE', '0.6')),
            'api_key': os.getenv('PERPLEXITY_API_KEY'),
            'max_workers': int(os.getenv('MAX_PARALLEL_SEARCHES', '10')),
            'rate_limit': int(os.getenv('SEARCH_RATE_LIMIT', '100'))  # 100 requests/min for Perplexity
        },
        # Limit Firecrawl to 50 requests/min
        'explore_config': {
            'max_urls': int(os.getenv('MAX_URLS', '20')),
            'min_priority': float(os.getenv('MIN_URL_PRIORITY', '0.5')),
            'allowed_domains': json.loads(os.getenv('ALLOWED_DOMAINS', '[]')),
            'api_key': os.getenv('FIRECRAWL_API_KEY'),
            'max_workers': int(os.getenv('MAX_PARALLEL_EXPLORES', '10')),
            'rate_limit': int(os.getenv('EXPLORE_RATE_LIMIT', '50'))  # 50 requests/min for Firecrawl
        },
        # Limit Gemini "thinking" to 10 requests/min
        # and "flash" JSON-fixer also 10/min
        'reason_config': {
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '4000')),
            'min_confidence': float(os.getenv('MIN_ANALYSIS_CONFIDENCE', '0.7')),
            'max_workers': int(os.getenv('MAX_PARALLEL_REASON', '5')),
            'rate_limit': int(os.getenv('REASON_RATE_LIMIT', '10')),  # 10 requests/min for main Gemini
            'flash_fix_rate_limit': int(os.getenv('FLASH_FIX_RATE_LIMIT', '10')),  # 10/min for JSON-fixing
            'api_key': os.getenv('GEMINI_API_KEY'),
            'gemini_timeout': int(os.getenv('GEMINI_TIMEOUT', '180'))
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
3. Reasoning Agent (Gemini) - Content analysis using Gemini 2.0 Flash Thinking

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
        console.print(f"[red]Research error: {results['error']}[/red]")
    else:
        console.print(Panel(Markdown(results["paper"]), title="Research Results", border_style="green"))

def main():
    """Main entry point for the research system."""
    import argparse
    
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
                    console.print("[yellow]Exiting research system...[/yellow]")
                    break
                    
                if user_input.lower() == 'help':
                    display_help()
                    continue
                    
                if user_input.lower() == 'config':
                    console.print(Panel(json.dumps(config, indent=2), title="Configuration", border_style="cyan"))
                    continue
                    
                if user_input.lower() == 'metrics' and orchestrator:
                    metrics = orchestrator._calculate_metrics(time.time())
                    console.print(Panel(json.dumps(metrics, indent=2), title="Research Metrics", border_style="magenta"))
                    continue
                    
                if user_input.lower().startswith('research '):
                    question = user_input[9:].strip()
                    if not question:
                        console.print("[red]Please provide a research question.[/red]")
                        continue
                    
                    run_research(question, config)
                    continue
                    
                console.print("[red]Unknown command. Type 'help' for available commands.[/red]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Operation cancelled. Type 'exit' to quit.[/yellow]")
                continue
                
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]")
                continue
    else:
        if not args.question:
            parser.error("Research question is required when not in interactive mode")
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
import textwrap
from weasyprint import HTML
import markdown

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
    """
    Represents a single iteration of the research process.
    
    Attributes:
        iteration_number: Current iteration number
        decisions_made: List of decisions made
        content_added: New content items added
        metrics: Performance metrics for this iteration
        timestamp: float - when the iteration occurred
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
    work together effectively. Supports parallel execution of decisions.
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
        
        # Initialize agents with concurrency + rate limits
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
                "rate_limit": self.config.get("explore_rate_limit", 50)
            }
        )
        self.reason_agent = ReasoningAgent(
            {
                **(self.config.get("reason_config") or {}),
                "max_workers": self.config.get("max_parallel_reason", 5),
                "rate_limit": self.config.get("reason_rate_limit", 10),
                "flash_fix_rate_limit": self.config.get("flash_fix_rate_limit", 10)
            }
        )
        
        # Initialize output manager
        self.output_manager = OutputManager()
        
        # High-level orchestrator config
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_new_content = self.config.get("min_new_content", 1)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        
        # State tracking
        self.current_context: Optional[ResearchContext] = None
        self.iterations: List[ResearchIteration] = []
        self.research_dir: Optional[Path] = None
        
        # Keep track of previously executed search queries to avoid duplicates
        self.previous_searches = set()
        
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
        The ReasoningAgent is the main driver, deciding next steps.
        Other agents can also propose decisions.
        """
        iteration_start = time.time()
        all_decisions = []
        content_added = []
        
        try:
            # 1) ReasoningAgent - the main driver
            reason_decisions = self.reason_agent.analyze(self.current_context)
            
            # If the ReasoningAgent says "terminate", gather that decision and skip everything
            if any(d.decision_type == DecisionType.TERMINATE for d in reason_decisions):
                all_decisions.extend(reason_decisions)
            else:
                # 2) Also gather decisions from other agents
                search_decisions = self.search_agent.analyze(self.current_context)
                explore_decisions = self.explore_agent.analyze(self.current_context)
                
                # Combine them all
                all_decisions = reason_decisions + search_decisions + explore_decisions
            
            # 3) Sort decisions by priority
            sorted_decisions = sorted(all_decisions, key=lambda d: d.priority, reverse=True)
            
            # 4) Execute decisions, skipping duplicates for SEARCH
            for decision in sorted_decisions:
                if decision.decision_type == DecisionType.TERMINATE:
                    # If we see a TERMINATE decision, do not continue
                    break
                
                agent = self._get_agent_for_decision(decision)
                if not agent:
                    continue
                
                if decision.decision_type == DecisionType.SEARCH:
                    # Check for duplicates
                    query_str = decision.context.get("query", "").strip()
                    if not query_str:
                        continue
                    
                    if query_str in self.previous_searches:
                        rprint(f"[yellow]Skipping duplicate search: '{query_str}'[/yellow]")
                        continue
                    else:
                        self.previous_searches.add(query_str)
                
                # Now actually execute
                try:
                    result = agent.execute_decision(decision)
                    
                    # Add results to context if we got any
                    if result:
                        content_item = self._create_content_item(
                            decision=decision,
                            result=result,
                            iteration_number=iteration_number
                        )
                        self.current_context.add_content("main", content_item=content_item)
                        content_added.append(content_item)
                        
                except Exception as e:
                    rprint(f"[red]Error executing decision: {str(e)}[/red]")
                        
        except Exception as e:
            rprint(f"[red]Iteration error: {str(e)}[/red]")
            
        metrics = {
            "iteration_time": time.time() - iteration_start,
            "decisions_made": len(all_decisions),
            "content_added": len(content_added),
            "agent_metrics": self._get_agent_metrics()
        }
        
        return ResearchIteration(
            iteration_number=iteration_number,
            decisions_made=all_decisions,
            content_added=content_added,
            metrics=metrics
        )

    def _create_content_item(
        self,
        decision: ResearchDecision,
        result: Dict[str, Any],
        iteration_number: int
    ) -> ContentItem:
        """Helper to create a ContentItem from a decision result."""
        if decision.decision_type == DecisionType.SEARCH:
            # For search results, extract content and urls
            content_str = result.get('content', '')
            urls = result.get('urls', [])
            token_count = self.current_context._estimate_tokens(content_str)
            
            return ContentItem(
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
            
            return ContentItem(
                content_type=self._get_content_type(decision),
                content=result,
                metadata={
                    "decision_type": decision.decision_type.value,
                    "iteration": iteration_number
                },
                token_count=token_count,
                priority=decision.priority
            )

    def _should_terminate(self, iteration: ResearchIteration) -> bool:
        """
        Decide if we should break out of the loop.
        Now primarily driven by the ReasoningAgent's TERMINATE decision.
        """
        # 1) If there's a TERMINATE decision from the ReasoningAgent
        terminate_decision = any(
            d.decision_type == DecisionType.TERMINATE for d in iteration.decisions_made
        )
        if terminate_decision:
            rprint("[green]Terminating: ReasoningAgent indicated completion.[/green]")
            return True
        
        # 2) Get fresh decisions from all agents to check if there's more work to do
        try:
            # Get new decisions from each agent
            reason_decisions = self.reason_agent.analyze(self.current_context)
            search_decisions = self.search_agent.analyze(self.current_context)
            explore_decisions = self.explore_agent.analyze(self.current_context)
            
            # Filter out duplicate searches
            valid_decisions = [
                d for d in (reason_decisions + search_decisions + explore_decisions)
                if (d.decision_type != DecisionType.SEARCH or 
                    d.context.get("query", "").strip() not in self.previous_searches)
            ]
            
            if not valid_decisions:
                rprint("[yellow]Terminating: No further valid decisions from any agent.[/yellow]")
                return True
                
        except Exception as e:
            rprint(f"[red]Error checking for new decisions: {str(e)}[/red]")
            # On error, continue the research to be safe
            return False
        
        return False
        
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
            self.reason_agent
        ]:
            try:
                decisions = agent.analyze(self.current_context)
                all_decisions.extend(decisions)
            except Exception as e:
                rprint(f"[red]Error getting decisions from {agent.name}: {str(e)}[/red]")
                
        return all_decisions
        
    def _get_agent_for_decision(self, decision: ResearchDecision) -> Optional[Any]:
        """Get the appropriate agent for a given decision type."""
        agent_map = {
            DecisionType.SEARCH: self.search_agent,
            DecisionType.EXPLORE: self.explore_agent,
            DecisionType.REASON: self.reason_agent
        }
        return agent_map.get(decision.decision_type)
        
    def _get_content_type(self, decision: ResearchDecision) -> ContentType:
        """Map decision types to content types."""
        type_map = {
            DecisionType.SEARCH: ContentType.SEARCH_RESULT,
            DecisionType.EXPLORE: ContentType.EXPLORED_CONTENT,
            DecisionType.REASON: ContentType.ANALYSIS,
            DecisionType.TERMINATE: ContentType.OTHER
        }
        return type_map.get(decision.decision_type, ContentType.OTHER)
        
    def _generate_comprehensive_paper_markdown(self, context: ResearchContext) -> str:
        """
        Make a second call to Gemini to produce a very thorough Markdown report,
        using all discovered search results, explored text, and analyses.
        """
        # 1) Gather up all final relevant data
        search_items = context.get_content("main", ContentType.SEARCH_RESULT)
        explored_items = context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis_items = context.get_content("main", ContentType.ANALYSIS)

        # Combine them into one big text corpus
        combined_corpus = []

        combined_corpus.append("### Final Consolidated Research\n")
        combined_corpus.append("[SEARCH RESULTS]\n")
        for s in search_items:
            combined_corpus.append(str(s.content))

        combined_corpus.append("\n[EXPLORED CONTENT]\n")
        for e in explored_items:
            combined_corpus.append(str(e.content))

        combined_corpus.append("\n[ANALYSIS TEXT]\n")
        for a in analysis_items:
            if isinstance(a.content, dict):
                combined_corpus.append(a.content.get("analysis", ""))
            else:
                combined_corpus.append(str(a.content))

        big_text = "\n\n".join(combined_corpus)

        # 2) Build a specialized prompt for the final deep-dive Markdown
        prompt = textwrap.dedent(f"""
            You are an advanced AI that just completed a comprehensive multi-step research.
            Now produce a SINGLE, richly detailed research paper in valid Markdown.
            Incorporate all relevant facts, context, analysis, and insights from the text below.
            
            Provide a thorough, well-structured breakdown:
            - Large headings
            - Subheadings
            - Bullet points
            - Tables if relevant
            - Detailed comparisons and references

            Return ONLY Markdown. No extra JSON or placeholders.

            RESEARCH CORPUS:
            {big_text}

            Please produce the final research paper in Markdown now:
        """).strip()

        # 3) Make the LLM call using the ReasoningAgent's model
        final_markdown = self._call_gemini_for_report(prompt)
        return final_markdown

    def _call_gemini_for_report(self, prompt: str) -> str:
        """
        Minimal method to do a single-turn call to the same Gemini model used by ReasoningAgent.
        Return the raw text (which should be valid Markdown).
        """
        try:
            # Reuse the ReasoningAgent's model for consistency
            response = self.reason_agent.model.generate_content(prompt)
            text = response.text.strip()
            
            # Clean markdown code blocks if present
            if text.startswith("```"):
                # Find the first newline after the opening ```
                start_idx = text.find("\n") + 1
                # Find the last ``` and exclude it
                end_idx = text.rfind("```")
                if end_idx > start_idx:
                    text = text[start_idx:end_idx].strip()
                else:
                    # If no proper end block found, just remove all ``` markers
                    text = text.replace("```", "").strip()
            
            return text
        except Exception as e:
            logger.error(f"Error in final paper LLM call: {e}")
            return "## Error generating comprehensive paper"

    def _convert_markdown_to_pdf(self, markdown_text: str, out_path: Path):
        """
        Convert Markdown to PDF using WeasyPrint.
        """
        # Convert MD -> HTML with basic styling
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2em; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; }}
                h2 {{ color: #34495e; margin-top: 2em; }}
                h3 {{ color: #7f8c8d; }}
                code {{ background: #f8f9fa; padding: 0.2em 0.4em; border-radius: 3px; }}
                pre {{ background: #f8f9fa; padding: 1em; border-radius: 5px; overflow-x: auto; }}
                blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 1em; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            {markdown.markdown(markdown_text, extensions=['fenced_code', 'tables'])}
        </body>
        </html>
        """
        
        # Write to PDF
        HTML(string=html_content).write_pdf(str(out_path))

    def _generate_final_output(self) -> Dict[str, Any]:
        """
        Generate the final research output with both quick summary and comprehensive deep-dive,
        plus PDF export.
        """
        # Get all content by type
        search_results = self.current_context.get_content("main", ContentType.SEARCH_RESULT)
        explored_content = self.current_context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis = self.current_context.get_content("main", ContentType.ANALYSIS)

        # Generate the comprehensive markdown
        comprehensive_md = self._generate_comprehensive_paper_markdown(self.current_context)

        # Convert to PDF if we have a research directory
        if self.research_dir:
            pdf_path = self.research_dir / "research_paper.pdf"
            self._convert_markdown_to_pdf(comprehensive_md, pdf_path)

        # Return the final dict with the comprehensive markdown
        return {
            "paper": comprehensive_md,
            "title": self.current_context.initial_question,
            "sources": []  # TODO: Extract sources from search results and explored content
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
        Gather metrics from each agent.
        """
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
    """
    Manages research outputs and metrics.
    
    Handles:
    - Creating research directories
    - Saving research papers
    - Tracking intermediate results
    - Recording performance metrics
    """
    
    def __init__(self):
        """Initialize the output manager."""
        self.base_dir = Path("research_outputs")
        
    def create_research_dir(self, question: str) -> Path:
        """
        Create a directory for research outputs.
        
        Args:
            question: Research question
            
        Returns:
            Path to created directory
        """
        # Create timestamped directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{timestamp}_{self._sanitize_filename(question[:50])}"
        
        # Create directory
        research_dir = self.base_dir / dir_name
        research_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial metadata
        self.save_metadata(research_dir, {
            "question": question,
            "started_at": timestamp,
            "status": "in_progress"
        })
        
        return research_dir
        
    def save_research_paper(self, research_dir: Path, paper: Dict[str, Any]):
        """
        Save the research paper and update metadata.
        
        Args:
            research_dir: Research output directory
            paper: Paper content and metadata
        """
        # Save paper content
        paper_path = research_dir / "research_paper.md"
        paper_path.write_text(paper["paper"])
        
        # Save paper info
        info_path = research_dir / "research_info.json"
        info = {
            "title": paper["title"],
            "sources": paper["sources"],
            "metrics": paper.get("metrics", {})
        }
        info_path.write_text(json.dumps(info, indent=2))
        
        # Update metadata
        self.save_metadata(research_dir, {
            "status": "completed",
            "completed_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "metrics": paper.get("metrics", {})
        })
        
    def save_context_state(self, research_dir: Path, context_data: Dict[str, Any]):
        """
        Save intermediate context state.
        
        Args:
            research_dir: Research output directory
            context_data: Context state to save
        """
        # Create states directory
        states_dir = research_dir / "states"
        states_dir.mkdir(exist_ok=True)
        
        # Save state with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_path = states_dir / f"context_state_{timestamp}.json"
        state_path.write_text(json.dumps(context_data, indent=2))
        
        # Keep only last 5 states to save space
        self._cleanup_old_states(states_dir)
        
    def save_iteration_metrics(
        self,
        research_dir: Path,
        iterations: List[Dict[str, Any]]
    ):
        """
        Save iteration performance metrics.
        
        Args:
            research_dir: Research output directory
            iterations: List of iteration metrics
        """
        metrics_path = research_dir / "iteration_metrics.json"
        metrics_path.write_text(json.dumps({
            "iterations": iterations,
            "summary": self._calculate_metrics_summary(iterations)
        }, indent=2))
        
    def save_metadata(self, research_dir: Path, updates: Dict[str, Any]):
        """
        Update research session metadata.
        
        Args:
            research_dir: Research output directory
            updates: Metadata updates
        """
        metadata_path = research_dir / "metadata.json"
        
        # Load existing metadata
        if metadata_path.exists():
            current_metadata = json.loads(metadata_path.read_text())
        else:
            current_metadata = {}
            
        # Update metadata
        current_metadata.update(updates)
        
        # Save updated metadata
        metadata_path.write_text(json.dumps(current_metadata, indent=2))
        
    def _sanitize_filename(self, name: str) -> str:
        """
        Create a safe filename from text.
        
        Args:
            name: Text to convert to filename
            
        Returns:
            Safe filename
        """
        # Replace unsafe characters
        safe_chars = "-_"
        filename = "".join(
            c if c.isalnum() or c in safe_chars else "_"
            for c in name
        )
        return filename.strip("_")
        
    def _cleanup_old_states(self, states_dir: Path):
        """
        Keep only the most recent state files.
        
        Args:
            states_dir: Directory containing state files
        """
        state_files = sorted(
            states_dir.glob("context_state_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old files
        for file in state_files[5:]:  # Keep 5 most recent
            file.unlink()
            
    def _calculate_metrics_summary(
        self,
        iterations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate summary metrics across iterations.
        
        Args:
            iterations: List of iteration metrics
            
        Returns:
            Summary metrics
        """
        if not iterations:
            return {}
            
        return {
            "total_iterations": len(iterations),
            "total_decisions": sum(it["decisions"] for it in iterations),
            "total_new_content": sum(it["new_content"] for it in iterations),
            "total_time": sum(it["time"] for it in iterations),
            "avg_decisions_per_iteration": (
                sum(it["decisions"] for it in iterations) / len(iterations)
            ),
            "avg_new_content_per_iteration": (
                sum(it["new_content"] for it in iterations) / len(iterations)
            ),
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
import json
import logging
from openai import OpenAI
from typing import List, Dict, Any
from rich import print as rprint
from dotenv import load_dotenv
import httpx

load_dotenv()

# Get API logger
api_logger = logging.getLogger('api.perplexity')

class PerplexityClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        
        self.model = "sonar-pro"
        self.system_message = (
            "You are a research assistant helping to find accurate and up-to-date information. "
            "When providing information, always cite your sources in the format [Source: URL]. "
            "Focus on finding specific, factual information and avoid speculation."
        )
        
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Perplexity API.
        
        Args:
            query: The search query
            
        Returns:
            Dict containing search results and extracted URLs
        """
        api_logger.info(f"Perplexity API Request - Query: {query}")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                stream=False
            )
            
            content = response.choices[0].message.content
            urls = self._extract_urls(content)
            
            api_logger.debug(f"Response data: {json.dumps({'content': content, 'urls': urls}, indent=2)}")
            
            return {
                "content": content,
                "urls": urls,
                "query": query,
                "metadata": {
                    "model": self.model,
                    "usage": response.usage
                }
            }
            
        except Exception as e:
            api_logger.error(f"Perplexity API error: {str(e)}")
            return {
                "content": "",
                "urls": [],
                "query": query,
                "metadata": {}
            }
            
    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text, including those in citation format.
        
        Args:
            text: Text to extract URLs from
            
        Returns:
            List of extracted URLs
        """
        # Look for URLs in citation format [Source: URL]
        citation_pattern = r'\[Source: (https?://[^\]]+)\]'
        citation_urls = re.findall(citation_pattern, text)
        
        # Also look for raw URLs
        url_pattern = r'https?://\S+'
        raw_urls = re.findall(url_pattern, text)
        
        # Combine and deduplicate URLs
        all_urls = list(set(citation_urls + raw_urls))
        return all_urls

    def validate_url(self, url: str) -> bool:
        """
        Validate if a URL is accessible and safe to scrape.
        """
        import requests
        from urllib.parse import urlparse
        
        try:
            # Parse URL
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
                
            # Check if URL is accessible
            response = requests.head(url, timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False 
```

## rat_agentic.py

```python
"""
Entry point for the multi-agent research system.
Provides a command-line interface for conducting research using the agent-based approach.
"""

import sys
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from typing import Dict, Any, Optional
import json

from rat.research.orchestrator import ResearchOrchestrator

def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for the research system.
    
    Returns:
        Default configuration dictionary
    """
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
        },
        
        "execute_config": {
            "model": "claude-3-5-sonnet-20241022",
            "min_priority": 0.3,
            "max_retries": 2
        }
    }

def display_results(results: Dict[str, Any]):
    """
    Display research results in a formatted way.
    
    Args:
        results: Research results to display
    """
    if "error" in results:
        rprint(f"\n[red]Error during research: {results['error']}[/red]")
        return
        
    # Display paper
    rprint(Panel(
        Markdown(results["paper"]),
        title="[bold green]Research Paper[/bold green]",
        border_style="green"
    ))
    
    # Display metrics
    metrics = results.get("metrics", {})
    
    rprint("\n[bold cyan]Research Metrics:[/bold cyan]")
    rprint(f"Total time: {metrics.get('total_time', 0):.2f} seconds")
    rprint(f"Iterations: {metrics.get('iterations', 0)}")
    rprint(f"Total decisions: {metrics.get('total_decisions', 0)}")
    rprint(f"Total content items: {metrics.get('total_content', 0)}")
    
    # Display agent metrics
    agent_metrics = metrics.get("agent_metrics", {})
    for agent_name, agent_data in agent_metrics.items():
        rprint(f"\n[bold]{agent_name.title()} Agent:[/bold]")
        rprint(f"Decisions made: {agent_data.get('decisions_made', 0)}")
        rprint(f"Successful executions: {agent_data.get('successful_executions', 0)}")
        rprint(f"Failed executions: {agent_data.get('failed_executions', 0)}")
        rprint(
            f"Average execution time: "
            f"{agent_data.get('total_execution_time', 0) / max(agent_data.get('decisions_made', 1), 1):.2f}s"
        )

def main():
    """Main entry point for the research system."""
    # Initialize style
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    # Create orchestrator with default config
    orchestrator = ResearchOrchestrator(create_default_config())
    
    # Display welcome message
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
                rprint("\nGoodbye! 👋")
                break
                
            elif user_input.lower() == 'config':
                rprint(Panel(
                    Markdown(f"```json\n{json.dumps(orchestrator.config, indent=2)}\n```"),
                    title="[bold cyan]Current Configuration[/bold cyan]",
                    border_style="cyan"
                ))
                continue
                
            elif user_input.lower() == 'metrics':
                if latest_results:
                    display_results(latest_results)
                else:
                    rprint("[yellow]No research has been conducted yet[/yellow]")
                continue
                
            # Conduct research
            rprint(f"\n[bold cyan]Starting research on:[/bold cyan] {user_input}")
            latest_results = orchestrator.start_research(user_input)
            
            # Display results
            display_results(latest_results)
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
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

from rat.research.orchestrator import ResearchOrchestrator
from rich import print as rprint

def main():
    # Initialize the orchestrator
    orchestrator = ResearchOrchestrator()
    
    # Define a test question
    question = "What are the main features and pricing of Billit's accounting software, and how does it compare to competitors in Belgium?"
    
    # Run the research
    rprint(f"[bold cyan]Starting research on: {question}[/bold cyan]")
    results = orchestrator.start_research(question)
    
    # Print results
    if "error" in results:
        rprint(f"[red]Error: {results['error']}[/red]")
    else:
        rprint("\n[bold green]Research completed![/bold green]")
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

