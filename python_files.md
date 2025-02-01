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
This async version uses asyncio locks and awaits where appropriate.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from rich import print as rprint
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Types of decisions an agent can make during research."""
    SEARCH = "search"      # New search query needed
    EXPLORE = "explore"    # URL exploration needed
    REASON = "reason"      # Deep analysis needed (using Gemini now)
    TERMINATE = "terminate"  # Research complete or no further steps


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
    All methods are now asynchronous.
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
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
        self.max_workers = self.config.get("max_workers", 5)
        self.rate_limit = self.config.get("rate_limit", 100)
        self._active_tasks: Set[str] = set()
        self._tasks_lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._rate_limit_lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{name}")

    async def _enforce_rate_limit(self):
        """
        Ensure we do not exceed self.rate_limit requests per minute.
        Uses an asynchronous lock and sleep.
        """
        if self.rate_limit <= 0:
            return  # no limiting
        async with self._rate_limit_lock:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self._last_request_time
            min_interval = 60.0 / self.rate_limit
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                await asyncio.sleep(sleep_time)
                self.metrics["rate_limit_delays"] += 1
            self._last_request_time = asyncio.get_event_loop().time()

    @abstractmethod
    async def analyze(self, context: 'ResearchContext') -> List[ResearchDecision]:
        """
        Analyze the current research context and make decisions.
        """
        pass

    def log_decision(self, decision: ResearchDecision, success: bool = True, execution_time: float = 0.0):
        """
        Log a decision made by this agent and update metrics.
        """
        self.decisions_made.append(decision)
        self.metrics["decisions_made"] += 1
        if success:
            self.metrics["successful_executions"] += 1
        else:
            self.metrics["failed_executions"] += 1
        self.metrics["total_execution_time"] += execution_time

    @abstractmethod
    async def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a decision made by this or another agent.
        """
        pass

    def get_decision_history(self) -> List[ResearchDecision]:
        """
        Get the history of decisions made by this agent.
        """
        return self.decisions_made.copy()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the current metrics for this agent.
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

    def save_to_file(self, file_path: str):
        """
        Persist the research context to a JSON file.
        
        Args:
            file_path: Path where the context JSON should be saved.
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, file_path: str) -> 'ResearchContext':
        """
        Load a research context from a JSON file.
        
        Args:
            file_path: Path to the JSON file.
            
        Returns:
            An instance of ResearchContext.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
```

## rat/research/agents/explore.py

```python
"""
Explore agent for extracting content from URLs.
Now acts as a simple executor that processes EXPLORE decisions.
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
    """

    def __init__(self, firecrawl_client: FirecrawlClient, config=None):
        super().__init__("explore", config)
        self.firecrawl = firecrawl_client
        self.logger = logging.getLogger(__name__)

    async def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        # URL selection is handled by ReasoningAgent.
        return []

    def can_handle(self, decision: ResearchDecision) -> bool:
        return decision.decision_type == DecisionType.EXPLORE

    async def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute an EXPLORE decision by scraping the URL.
        """
        url = decision.context["url"]
        self.logger.info(f"Exploring URL: {url}")
        try:
            scrape_result = await self.firecrawl.extract_content(url)
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
            return {"url": url, "error": str(e)}
```

## rat/research/agents/reason.py

```python
"""
Reasoning agent for analyzing research content using the o3-mini model with high reasoning effort.
Now acts as the "lead agent" that decides next steps.
All methods are now asynchronous.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
from rich import print as rprint
import logging
import json
import re
from urllib.parse import urlparse
from openai import OpenAI

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)
api_logger = logging.getLogger('api.o3mini')


@dataclass
class AnalysisTask:
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()


class ReasoningAgent(BaseAgent):
    """
    Reasoning agent for analyzing research content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("reason", config)
        self.model_name = "o3-mini"
        self.max_output_tokens = self.config.get("max_output_tokens", 50000)
        self.request_timeout = self.config.get("o3_mini_timeout", 180)
        self.reasoning_effort = "high"
        self.chunk_margin = 5000
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.min_priority = self.config.get("min_priority", 0.3)
        self.min_url_relevance = self.config.get("min_url_relevance", 0.6)
        self.explored_urls: Set[str] = set()
        self.flash_fix_rate_limit = self.config.get("flash_fix_rate_limit", 10)
        self._flash_fix_last_time = 0.0
        self._flash_fix_lock = asyncio.Lock()
        logger.info("ReasoningAgent initialized to use o3-mini model: %s", self.model_name)
        self.analysis_tasks: Dict[str, AnalysisTask] = {}

    async def _enforce_flash_fix_limit(self):
        if self.flash_fix_rate_limit <= 0:
            return
        async with self._flash_fix_lock:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self._flash_fix_last_time
            min_interval = 60.0 / self.flash_fix_rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
                self.metrics["rate_limit_delays"] += 1
            self._flash_fix_last_time = asyncio.get_event_loop().time()

    async def _call_o3_mini(self, prompt: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({"role": "assistant", "content": context})
        messages.append({"role": "user", "content": prompt})
        api_logger.info(f"o3-mini API Request - Prompt length: {len(prompt)}")
        try:
            # Use the new OpenAI API format
            client = OpenAI()
            params = {
                "model": self.model_name,
                "messages": messages,
                "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_output_tokens,
            }
            if "JSON" in prompt or "json" in prompt:
                params["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**params)
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            api_logger.error(f"o3-mini API error: {str(e)}")
            raise

    async def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        decisions = []
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

        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        explored_urls = { 
            item.content.get("url", "") for item in explored_content
            if isinstance(item.content, dict)
        }
        self.explored_urls.update(explored_urls)
        unvisited_urls = set()
        for result in search_results:
            if isinstance(result.content, dict):
                urls = result.content.get("urls", [])
                unvisited_urls.update(url for url in urls if url not in self.explored_urls)
        relevant_urls = await self._filter_relevant_urls(list(unvisited_urls), context)
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
        unprocessed_search = [item for item in search_results if not item.metadata.get("analyzed_by_reasoner")]
        unprocessed_explored = [item for item in explored_content if not item.metadata.get("analyzed_by_reasoner")]
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
        search_text = "\n".join(str(item.content) for item in search_results if isinstance(item.content, str))
        explored_text = "\n".join(str(item.content) for item in explored_content if isinstance(item.content, str))
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_text = "\n".join(
            (item.content.get("analysis", "") if isinstance(item.content, dict) else str(item.content))
            for item in analysis_items
        )
        combined_analysis = f"{search_text}\n\n{explored_text}\n\n{analysis_text}".strip()
        if combined_analysis:
            gaps = await self._identify_knowledge_gaps(context.initial_question, combined_analysis)
            filtered_gaps = []
            for gap in gaps:
                query_str = gap.get("query", "")
                url_str = gap.get("url", "")
                if any(x in query_str or x in url_str for x in ("[", "]")):
                    self.logger.warning(f"Skipping gap with placeholders: {gap}")
                    continue
                filtered_gaps.append(gap)
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
            self.logger.info("Skipping knowledge gap analysis due to lack of context.")
        if await self._should_terminate(context):
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
        return decision.decision_type == DecisionType.REASON

    async def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        success = False
        results = {}
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            item_id = decision.context["item_id"]
            content_str = str(content)
            tokens_estimated = len(content_str) // 4
            if tokens_estimated > self.max_output_tokens:
                chunk_results = await self._parallel_analyze_content(content_str, content_type)
                results = self._combine_chunk_results(chunk_results)
            else:
                results = await self._analyze_content_chunk(content_str, content_type)
            success = bool(results.get("analysis", "").strip())
            decision.context["analyzed_by_reasoner"] = True
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
            return {"error": str(e), "analysis": "", "insights": []}
        finally:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.log_decision(decision, success, execution_time)

    async def _parallel_analyze_content(self, content: str, content_type: str) -> List[Dict[str, Any]]:
        words = content.split()
        chunk_size_words = self.max_output_tokens * 4
        chunks = []
        idx = 0
        while idx < len(words):
            chunk = words[idx: idx + chunk_size_words]
            chunks.append(" ".join(chunk))
            idx += chunk_size_words
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(asyncio.create_task(self._analyze_content_chunk(chunk, f"{content_type}_chunk_{i}")))
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)
        for i, res in enumerate(chunk_results):
            res["chunk_index"] = i
        return chunk_results

    async def _analyze_content_chunk(self, content: str, content_type: str) -> Dict[str, Any]:
        await self._enforce_flash_fix_limit()
        prompt = (
            "You are an advanced reasoning model (o3-mini) with high reasoning effort. "
            "Analyze the following text for key insights, patterns, or relevant facts. "
            "Provide ONLY factual analysis and insights without placeholders or next-step suggestions.\n\n"
            f"CONTENT:\n{content}\n\n"
            "Please provide your analysis below (plain text only):"
        )
        response_text = await self._call_o3_mini(prompt)
        analysis_text = response_text.strip()
        insights = self._extract_insights(analysis_text)
        return {"analysis": analysis_text, "insights": insights}

    def _extract_insights(self, analysis_text: str) -> List[str]:
        lines = analysis_text.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if (line.startswith("-") or line.startswith("*") or line.startswith("•") or 
                (len(line) > 2 and line[:2].isdigit())):
                insights.append(line.lstrip("-*•").strip())
        return insights

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        combined_analysis = "\n\n".join(res["analysis"] for res in sorted_chunks if res.get("analysis", "").strip())
        combined_insights = []
        for res in sorted_chunks:
            combined_insights.extend(insight for insight in res.get("insights", []) if insight.strip())
        unique_insights = list(dict.fromkeys(combined_insights))
        return {"analysis": combined_analysis, "insights": unique_insights, "chunk_count": len(chunk_results)}

    async def _identify_knowledge_gaps(self, question: str, current_analysis: str) -> List[Dict[str, Any]]:
        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "identify specific missing information and suggest concrete next steps.\n\n"
            "IMPORTANT RULES:\n"
            "1. DO NOT use placeholders like [company name] or [person].\n"
            "2. Base suggestions solely on the provided content.\n"
            "3. If no specific gaps can be identified, return an empty array.\n"
            "4. Each suggestion must be actionable and clearly linked to the research question.\n\n"
            f"RESEARCH QUESTION: {question}\n\n"
            f"CURRENT ANALYSIS:\n{current_analysis}\n\n"
            "Return a JSON object with a 'gaps' array in this format:\n"
            "{\"gaps\": [{\"type\": \"search\"|\"explore\", \"query\"|\"url\": \"specific text\", \"rationale\": \"why needed\"}]}"
        )
        try:
            response = await self._call_o3_mini(prompt)
            content_str = response.strip()
            if not content_str:
                return []
            try:
                result = json.loads(content_str)
                gaps = result.get("gaps", [])
                filtered_gaps = []
                for gap in gaps:
                    if not isinstance(gap, dict):
                        continue
                    if "type" not in gap or gap["type"] not in ["search", "explore"]:
                        continue
                    content_field = "query" if gap["type"] == "search" else "url"
                    if content_field not in gap or "rationale" not in gap:
                        continue
                    if any(x in gap[content_field] or x in gap["rationale"] for x in ("[", "]")):
                        self.logger.warning(f"Skipping gap with placeholders: {gap}")
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
        content = content.strip()
        if content.startswith("```"):
            start_idx = content.find("\n") + 1
            end_idx = content.rfind("```")
            if end_idx > start_idx:
                content = content[start_idx:end_idx].strip()
            else:
                content = content.replace("```", "").strip()
        content = content.replace("json", "").strip()
        return content

    async def _should_terminate(self, context: ResearchContext) -> bool:
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if len(analysis_items) < 3:
            return False
        combined_analysis = "\n".join(
            str(item.content.get("analysis", "")) for item in analysis_items if isinstance(item.content, dict)
        )
        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "determine if the question has been sufficiently answered.\n\n"
            f"QUESTION: {context.initial_question}\n\n"
            f"CURRENT ANALYSIS:\n{combined_analysis}\n\n"
            "Respond with a single word: YES if answered, NO if not."
        )
        try:
            answer = await self._call_o3_mini(prompt, combined_analysis)
            return answer.strip().upper() == "YES"
        except Exception:
            return False

    async def _filter_relevant_urls(self, urls: List[str], context: ResearchContext) -> List[tuple]:
        if not urls:
            return []
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
                content = await self._call_o3_mini(prompt)
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
                for url in batch:
                    relevance = self._basic_url_relevance(url, context.initial_question)
                    relevant_urls.append((url, relevance))
        return sorted(relevant_urls, key=lambda x: x[1], reverse=True)

    def _basic_url_relevance(self, url: str, question: str) -> float:
        import re
        from urllib.parse import urlparse
        keywords = set(re.findall(r'\w+', question.lower()))
        parsed = urlparse(url)
        domain_parts = parsed.netloc.lower().split('.')
        path_parts = parsed.path.lower().split('/')
        domain_matches = sum(1 for part in domain_parts if part in keywords)
        path_matches = sum(1 for part in path_parts if part in keywords)
        score = (domain_matches * 0.6 + path_matches * 0.4) / max(len(keywords), 1)
        return min(max(score, 0.0), 1.0)
```

## rat/research/agents/search.py

```python
"""
Search agent for managing Perplexity-based research queries.
Now fully asynchronous.
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
```

## rat/research/firecrawl_client.py

```python
"""
Firecrawl client for web scraping functionality.
Now uses asyncio.to_thread to wrap blocking calls.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from rich import print as rprint
from firecrawl import FirecrawlApp
import logging
import asyncio

load_dotenv()

api_logger = logging.getLogger('api.firecrawl')


class FirecrawlClient:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        self.app = FirecrawlApp(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    async def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Asynchronously extract content from a webpage.
        """
        api_logger.info(f"Firecrawl API Request - URL: {url}")
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                api_logger.debug(f"Added https:// protocol to URL: {url}")
            result = await asyncio.to_thread(self.app.scrape_url, url, params={'formats': ['markdown']})
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
        metadata = data.get("metadata", {})
        markdown_content = data.get("markdown", "")
        processed = {
            "title": metadata.get("title", metadata.get("ogTitle", "")),
            "text": markdown_content,
            "metadata": {
                "url": metadata.get("sourceURL", original_url),
                "author": metadata.get("author", ""),
                "published_date": "",
                "domain": metadata.get("ogSiteName", ""),
                "word_count": len(markdown_content.split()) if markdown_content else 0,
                "language": metadata.get("language", ""),
                "status_code": metadata.get("statusCode", 200)
            }
        }
        if processed["text"]:
            processed["text"] = self._clean_text(processed["text"])
            api_logger.debug(f"Cleaned text for {original_url}: {len(processed['text'])} chars")
        else:
            api_logger.warning(f"No text content extracted from {original_url}")
        return processed

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith(("#", "-", "*", "1.", ">")):
                cleaned_lines.append(line)
            else:
                cleaned = " ".join(line.split())
                if cleaned:
                    cleaned_lines.append(cleaned)
        return "\n\n".join(cleaned_lines)
```

## rat/research/main.py

```python
"""
Main entry point for the multi-agent research system.
Now uses an async main loop.
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
import asyncio

from .orchestrator import ResearchOrchestrator
from .output_manager import OutputManager

# Logging configuration
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rat.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
api_logger = logging.getLogger('api')
api_logger.setLevel(logging.DEBUG)
api_handler = logging.FileHandler('rat_api.log', mode='w')
api_handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
)
api_logger.addHandler(api_handler)
firecrawl_logger = logging.getLogger('api.firecrawl')
firecrawl_logger.setLevel(logging.DEBUG)
firecrawl_logger.addHandler(api_handler)
firecrawl_logger.propagate = False
api_logger.propagate = False

logger = logging.getLogger(__name__)
console = Console()


def load_config() -> Dict[str, Any]:
    load_dotenv()
    config = {
        'max_iterations': int(os.getenv('MAX_ITERATIONS', '5')),
        'min_new_content': int(os.getenv('MIN_NEW_CONTENT', '3')),
        'min_confidence': float(os.getenv('MIN_CONFIDENCE', '0.7')),
        'search_config': {
            'max_results': int(os.getenv('MAX_SEARCH_RESULTS', '10')),
            'min_relevance': float(os.getenv('MIN_SEARCH_RELEVANCE', '0.6')),
            'api_key': os.getenv('PERPLEXITY_API_KEY'),
            'max_workers': int(os.getenv('MAX_PARALLEL_SEARCHES', '10')),
            'rate_limit': int(os.getenv('SEARCH_RATE_LIMIT', '100'))
        },
        'explore_config': {
            'max_urls': int(os.getenv('MAX_URLS', '20')),
            'min_priority': float(os.getenv('MIN_URL_PRIORITY', '0.5')),
            'allowed_domains': json.loads(os.getenv('ALLOWED_DOMAINS', '[]')),
            'api_key': os.getenv('FIRECRAWL_API_KEY'),
            'max_workers': int(os.getenv('MAX_PARALLEL_EXPLORES', '10')),
            'rate_limit': int(os.getenv('EXPLORE_RATE_LIMIT', '50'))
        },
        'reason_config': {
            'max_chunk_size': int(os.getenv('MAX_CHUNK_SIZE', '4000')),
            'min_confidence': float(os.getenv('MIN_ANALYSIS_CONFIDENCE', '0.7')),
            'max_workers': int(os.getenv('MAX_PARALLEL_REASON', '5')),
            'rate_limit': int(os.getenv('REASON_RATE_LIMIT', '10')),
            'flash_fix_rate_limit': int(os.getenv('FLASH_FIX_RATE_LIMIT', '10')),
            'api_key': os.getenv('GEMINI_API_KEY'),
            'gemini_timeout': int(os.getenv('GEMINI_TIMEOUT', '180'))
        }
    }
    return config


def display_welcome():
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


async def run_research(question: str, config: Dict[str, Any]) -> None:
    orchestrator = ResearchOrchestrator(config)
    results = await orchestrator.start_research(question)
    if "error" in results:
        console.print(f"[red]Research error: {results['error']}[/red]")
    else:
        console.print(Panel(Markdown(results["paper"]), title="Research Results", border_style="green"))


async def main_async():
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
                    await run_research(question, config)
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
        await run_research(args.question, config)


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()
```

## rat/research/manager.py

```python
"""
Manager for coordinating the multi-agent research workflow.
Now fully asynchronous.
"""

import json
import time
import logging
from asyncio import gather, create_task
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from openai import OpenAI

from rat.research.agents.search import SearchAgent
from rat.research.agents.explore import ExploreAgent
from rat.research.agents.reason import ReasoningAgent
from rat.research.agents.base import ResearchDecision, DecisionType
from rat.research.agents.context import ResearchContext, ContentType, ContentItem
from rat.research.perplexity_client import PerplexityClient
from rat.research.firecrawl_client import FirecrawlClient
from rat.research.output_manager import OutputManager

logger = logging.getLogger(__name__)


class AgentTask:
    """
    Wrapper for an agent decision execution.
    """
    def __init__(self, decision: ResearchDecision, agent, callback):
        self.decision = decision
        self.agent = agent
        self.callback = callback

    async def run(self):
        result = await self.agent.execute_decision(self.decision)
        self.callback(self.decision, result)
        return result


class ResearchManager:
    """
    Central manager for the multi-agent research process.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.perplexity = PerplexityClient()
        self.firecrawl = FirecrawlClient()
        self.search_agent = SearchAgent(self.perplexity, self.config.get("search_config", {}))
        self.explore_agent = ExploreAgent(self.firecrawl, self.config.get("explore_config", {}))
        self.reason_agent = ReasoningAgent(self.config.get("reason_config", {}))
        self.output_manager = OutputManager()
        self.max_iterations = self.config.get("max_iterations", 5)
        self.current_context: Optional[ResearchContext] = None
        self.research_dir: Optional[Path] = None
        self.previous_searches = set()

    async def start_research(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        self.research_dir = self.output_manager.create_research_dir(question)
        self.current_context = ResearchContext(initial_question=question)
        self.persist_context()
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")
            decisions = self.collect_decisions()
            if not decisions:
                logger.info("No new decisions, terminating research")
                break
            if any(d.decision_type == DecisionType.TERMINATE for d in decisions):
                logger.info("Terminate decision received")
                break
            await self.dispatch_decisions(decisions, iteration)
            self.persist_context()
            if self._should_terminate():
                logger.info("Termination condition met based on context")
                break
        final_output = await self._generate_final_output()
        total_time = time.time() - start_time
        final_output["metrics"] = {
            "total_time": total_time,
            "iterations": iteration
        }
        if self.research_dir:
            self.output_manager.save_research_paper(self.research_dir, final_output)
        return final_output

    def collect_decisions(self) -> List[ResearchDecision]:
        decisions = []
        try:
            reason_decisions = asyncio.run(self.reason_agent.analyze(self.current_context))
        except Exception as e:
            logger.error(f"Error in reason agent analysis: {e}")
            reason_decisions = []
        try:
            search_decisions = asyncio.run(self.search_agent.analyze(self.current_context))
        except Exception as e:
            logger.error(f"Error in search agent analysis: {e}")
            search_decisions = []
        try:
            explore_decisions = asyncio.run(self.explore_agent.analyze(self.current_context))
        except Exception as e:
            logger.error(f"Error in explore agent analysis: {e}")
            explore_decisions = []
        decisions.extend(reason_decisions)
        decisions.extend(search_decisions)
        decisions.extend(explore_decisions)
        filtered = []
        for d in decisions:
            if d.decision_type == DecisionType.SEARCH:
                query = d.context.get("query", "").strip()
                if query in self.previous_searches:
                    logger.info(f"Skipping duplicate search: {query}")
                    continue
                else:
                    self.previous_searches.add(query)
            filtered.append(d)
        filtered.sort(key=lambda d: d.priority, reverse=True)
        return filtered

    async def dispatch_decisions(self, decisions: List[ResearchDecision], iteration: int):
        tasks = []
        for decision in decisions:
            if decision.decision_type == DecisionType.TERMINATE:
                continue
            agent = self._select_agent(decision)
            if not agent:
                continue
            task = AgentTask(decision, agent, self.update_context_with_result)
            tasks.append(create_task(task.run()))
        if tasks:
            await gather(*tasks)

    def update_context_with_result(self, decision: ResearchDecision, result: Dict[str, Any]):
        content_item = self._create_content_item(decision, result)
        if content_item and self.current_context:
            try:
                self.current_context.add_content("main", content_item=content_item)
            except Exception as e:
                logger.error(f"Error updating context: {e}")

    def _create_content_item(self, decision: ResearchDecision, result: Dict[str, Any]) -> ContentItem:
        if decision.decision_type == DecisionType.SEARCH:
            content_str = result.get("content", "")
            urls = result.get("urls", [])
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.SEARCH_RESULT,
                content=content_str,
                metadata={"decision_type": decision.decision_type.value, "urls": urls},
                token_count=token_count,
                priority=decision.priority
            )
        else:
            if isinstance(result, dict):
                content_str = result.get("analysis", json.dumps(result))
            else:
                content_str = str(result)
            token_count = self.current_context._estimate_tokens(content_str)
            content_type = (ContentType.EXPLORED_CONTENT if decision.decision_type == DecisionType.EXPLORE 
                            else ContentType.ANALYSIS)
            return ContentItem(
                content_type=content_type,
                content=result,
                metadata={"decision_type": decision.decision_type.value},
                token_count=token_count,
                priority=decision.priority
            )

    def _select_agent(self, decision: ResearchDecision):
        if decision.decision_type == DecisionType.SEARCH:
            return self.search_agent
        elif decision.decision_type == DecisionType.EXPLORE:
            return self.explore_agent
        elif decision.decision_type == DecisionType.REASON:
            return self.reason_agent
        else:
            return None

    def _should_terminate(self) -> bool:
        if self.current_context:
            contents = self.current_context.get_content("main")
            return len(contents) >= self.config.get("min_new_content", 3)
        return False

    def persist_context(self):
        if self.current_context and self.research_dir:
            context_file = self.research_dir / "research_context.json"
            self.current_context.save_to_file(str(context_file))
            logger.info(f"Context persisted to {context_file}")

    async def _generate_comprehensive_markdown(self) -> str:
        search_items = self.current_context.get_content("main", ContentType.SEARCH_RESULT)
        explored_items = self.current_context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis_items = self.current_context.get_content("main", ContentType.ANALYSIS)
        corpus = []
        corpus.append("### Consolidated Research\n")
        corpus.append("#### Search Results\n")
        for item in search_items:
            corpus.append(str(item.content))
        corpus.append("\n#### Explored Content\n")
        for item in explored_items:
            corpus.append(str(item.content))
        corpus.append("\n#### Analysis\n")
        for item in analysis_items:
            if isinstance(item.content, dict):
                corpus.append(item.content.get("analysis", ""))
            else:
                corpus.append(str(item.content))
        big_text = "\n\n".join(corpus)
        prompt = (
            "You are an advanced AI tasked with generating a comprehensive research paper in Markdown. "
            "Using the following research corpus, produce a detailed, well-structured paper with headings, subheadings, bullet points, and tables if necessary.\n\n"
            "RESEARCH CORPUS:\n"
            f"{big_text}\n\n"
            "Please produce the final research paper in Markdown:"
        )
        
        # Use the new OpenAI API format
        client = OpenAI()
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="high",
            max_completion_tokens=50000
        )
        final_markdown = response.choices[0].message.content.strip()
        return final_markdown

    async def _generate_final_output(self) -> Dict[str, Any]:
        comprehensive_md = await self._generate_comprehensive_markdown()
        if self.research_dir:
            pdf_path = self.research_dir / "research_paper.pdf"
            await self._convert_markdown_to_pdf(comprehensive_md, pdf_path)
        return {
            "paper": comprehensive_md,
            "title": self.current_context.initial_question,
            "sources": []
        }

    async def _convert_markdown_to_pdf(self, markdown_text: str, out_path: Path):
        import markdown
        from weasyprint import HTML
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
        HTML(string=html_content).write_pdf(str(out_path))

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

## rat/research/orchestrator.py

```python
"""
Orchestrator for coordinating the multi-agent research workflow.
Now fully asynchronous.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from rich import print as rprint
from pathlib import Path
import markdown
from weasyprint import HTML
import openai

from rat.research.agents.search import SearchAgent
from rat.research.agents.explore import ExploreAgent
from rat.research.agents.reason import ReasoningAgent
from rat.research.perplexity_client import PerplexityClient
from rat.research.firecrawl_client import FirecrawlClient
from rat.research.output_manager import OutputManager
from rat.research.agents.base import ResearchDecision, DecisionType
from rat.research.agents.context import ResearchContext, ContentType, ContentItem

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
        self.output_manager = OutputManager()
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_new_content = self.config.get("min_new_content", 1)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.current_context: Optional[ResearchContext] = None
        self.iterations: List[ResearchIteration] = []
        self.research_dir: Optional[Path] = None
        self.previous_searches = set()

    async def start_research(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        self.research_dir = self.output_manager.create_research_dir(question)
        self.current_context = ResearchContext(initial_question=question)
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            rprint(f"\n[bold cyan]Starting iteration {iteration}[/bold cyan]")
            iteration_result = await self._run_iteration(iteration)
            self.iterations.append(iteration_result)
            if self._should_terminate(iteration_result):
                break
        results = await self._generate_final_output()
        total_time = time.time() - start_time
        results["metrics"] = self._calculate_metrics(total_time)
        if self.research_dir:
            self.output_manager.save_research_paper(self.research_dir, results)
        return results

    async def _run_iteration(self, iteration_number: int) -> ResearchIteration:
        iteration_start = time.time()
        all_decisions = []
        content_added = []
        try:
            reason_decisions = await self.reason_agent.analyze(self.current_context)
            if any(d.decision_type == DecisionType.TERMINATE for d in reason_decisions):
                all_decisions.extend(reason_decisions)
            else:
                search_decisions = await self.search_agent.analyze(self.current_context)
                explore_decisions = await self.explore_agent.analyze(self.current_context)
                all_decisions = reason_decisions + search_decisions + explore_decisions
            sorted_decisions = sorted(all_decisions, key=lambda d: d.priority, reverse=True)
            for decision in sorted_decisions:
                if decision.decision_type == DecisionType.TERMINATE:
                    break
                agent = self._get_agent_for_decision(decision)
                if not agent:
                    continue
                if decision.decision_type == DecisionType.SEARCH:
                    query_str = decision.context.get("query", "").strip()
                    if not query_str:
                        continue
                    if query_str in self.previous_searches:
                        rprint(f"[yellow]Skipping duplicate search: '{query_str}'[/yellow]")
                        continue
                    else:
                        self.previous_searches.add(query_str)
                try:
                    result = await agent.execute_decision(decision)
                    if result:
                        content_item = self._create_content_item(decision, result, iteration_number)
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

    def _create_content_item(self, decision: ResearchDecision, result: Dict[str, Any], iteration_number: int) -> ContentItem:
        if decision.decision_type == DecisionType.SEARCH:
            content_str = result.get('content', '')
            urls = result.get('urls', [])
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=self._get_content_type(decision),
                content=content_str,
                metadata={"decision_type": decision.decision_type.value, "iteration": iteration_number, "urls": urls},
                token_count=token_count,
                priority=decision.priority
            )
        else:
            content_str = result if isinstance(result, str) else json.dumps(result)
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=self._get_content_type(decision),
                content=result,
                metadata={"decision_type": decision.decision_type.value, "iteration": iteration_number},
                token_count=token_count,
                priority=decision.priority
            )

    def _should_terminate(self, iteration: ResearchIteration) -> bool:
        terminate_decision = any(d.decision_type == DecisionType.TERMINATE for d in iteration.decisions_made)
        if terminate_decision:
            rprint("[green]Terminating: ReasoningAgent indicated completion.[/green]")
            return True
        try:
            reason_decisions = asyncio.run(self.reason_agent.analyze(self.current_context))
            search_decisions = asyncio.run(self.search_agent.analyze(self.current_context))
            explore_decisions = asyncio.run(self.explore_agent.analyze(self.current_context))
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
            return False
        return False

    def _get_agent_for_decision(self, decision: ResearchDecision) -> Optional[Any]:
        agent_map = {
            DecisionType.SEARCH: self.search_agent,
            DecisionType.EXPLORE: self.explore_agent,
            DecisionType.REASON: self.reason_agent
        }
        return agent_map.get(decision.decision_type)

    def _get_content_type(self, decision: ResearchDecision) -> ContentType:
        type_map = {
            DecisionType.SEARCH: ContentType.SEARCH_RESULT,
            DecisionType.EXPLORE: ContentType.EXPLORED_CONTENT,
            DecisionType.REASON: ContentType.ANALYSIS,
            DecisionType.TERMINATE: ContentType.OTHER
        }
        return type_map.get(decision.decision_type, ContentType.OTHER)

    async def _call_o3_mini_for_report(self, prompt: str) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high",
                max_completion_tokens=self.reason_agent.max_output_tokens
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                start_idx = text.find("\n") + 1
                end_idx = text.rfind("```")
                if end_idx > start_idx:
                    text = text[start_idx:end_idx].strip()
                else:
                    text = text.replace("```", "").strip()
            return text
        except Exception as e:
            logger.error(f"Error in final paper LLM call: {e}")
            return "## Error generating comprehensive paper"

    async def _generate_comprehensive_paper_markdown(self, context: ResearchContext) -> str:
        search_items = context.get_content("main", ContentType.SEARCH_RESULT)
        explored_items = context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
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
        prompt = (
            "You are an advanced AI that just completed a comprehensive multi-step research.\n"
            "Now produce a SINGLE, richly detailed research paper in valid Markdown.\n"
            "Incorporate all relevant facts, context, analysis, and insights from the text below.\n\n"
            "Provide a thorough, well-structured breakdown:\n"
            "- Large headings\n"
            "- Subheadings\n"
            "- Bullet points\n"
            "- Tables if relevant\n"
            "- Detailed comparisons and references\n\n"
            "Return ONLY Markdown. RULE: ensure that all tables are valid Markdown tables. No extra JSON or placeholders.\n\n"
            "RESEARCH CORPUS:\n"
            f"{big_text}\n\n"
            "Please produce the final research paper in Markdown now:"
        ).strip()
        final_markdown = await self._call_o3_mini_for_report(prompt)
        return final_markdown

    async def _convert_markdown_to_pdf(self, markdown_text: str, out_path: Path):
        import markdown
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
        HTML(string=html_content).write_pdf(str(out_path))

    async def _generate_final_output(self) -> Dict[str, Any]:
        comprehensive_md = await self._generate_comprehensive_paper_markdown(self.current_context)
        if self.research_dir:
            pdf_path = self.research_dir / "research_paper.pdf"
            await self._convert_markdown_to_pdf(comprehensive_md, pdf_path)
        return {
            "paper": comprehensive_md,
            "title": self.current_context.initial_question,
            "sources": []
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
Now uses async OpenAI API calls.
"""

import os
import re
import json
import logging
import openai
from typing import List, Dict, Any
from rich import print as rprint
from dotenv import load_dotenv
import asyncio

load_dotenv()
api_logger = logging.getLogger('api.perplexity')


class PerplexityClient:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.client = openai
        self.client.api_key = self.api_key
        self.client.api_base = "https://api.perplexity.ai"
        self.model = "sonar-reasoning"
        self.system_message = (
            "You are a research assistant helping to find accurate and up-to-date information. "
            "When providing information, always cite your sources in the format [Source: URL]. "
            "Focus on finding specific, factual information and avoid speculation."
        )

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Perform an asynchronous web search.
        """
        api_logger.info(f"Perplexity API Request - Query: {query}")
        try:
            params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7,
                "stream": False
            }
            response = await openai.ChatCompletion.acreate(**params)
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
        citation_pattern = r'$begin:math:display$Source: (https?://[^$end:math:display$]+)\]'
        citation_urls = re.findall(citation_pattern, text)
        url_pattern = r'https?://\S+'
        raw_urls = re.findall(url_pattern, text)
        all_urls = list(set(citation_urls + raw_urls))
        return all_urls

    async def validate_url(self, url: str) -> bool:
        import requests
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            return await asyncio.to_thread(lambda: requests.head(url, timeout=5).status_code == 200)
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
import json
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import WordCompleter
from typing import Dict, Any, Optional
import asyncio

from rat.research.manager import ResearchManager


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
            "refinement_threshold": 0.7,
            "rate_limit": 20      # Updated: Perplexity rate limit set to 20 requests per minute
        },
        "explore_config": {
            "max_urls": 10,
            "min_priority": 0.3,
            "allowed_domains": []
        },
        "reason_config": {
            "max_parallel_tasks": 3,
            "chunk_size": 30000,
            "min_priority": 0.3,
            "rate_limit": 200,     # Updated: O3 mini rate limit set to 200 requests per minute
            "flash_fix_rate_limit": 10
        },
        "execute_config": {
            "model": "claude-3-5-sonnet-20241022",
            "min_priority": 0.3,
            "max_retries": 2
        },
        "max_workers": 20
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
        rprint(
            f"Average execution time: "
            f"{agent_data.get('total_execution_time', 0) / max(agent_data.get('decisions_made', 1), 1):.2f}s"
        )


async def main_async():
    """Main entry point for the research system."""
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    manager = ResearchManager(create_default_config())
    
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
    commands = WordCompleter(['quit', 'config', 'metrics'])
    
    while True:
        try:
            user_input = await session.prompt_async("\nResearch Question: ", completer=commands)
            user_input = user_input.strip()
            
            if user_input.lower() == 'quit':
                rprint("\nGoodbye! 👋")
                break
            elif user_input.lower() == 'config':
                rprint(Panel(
                    Markdown(f"```json\n{json.dumps(manager.config, indent=2)}\n```"),
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
            rprint(f"\n[bold cyan]Starting research on:[/bold cyan] {user_input}")
            latest_results = await manager.start_research(user_input)
            display_results(latest_results)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            rprint(f"[red]Error: {str(e)}[/red]")


def main():
    """Run the async main function."""
    asyncio.run(main_async())


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
Now asynchronous.
"""

import asyncio
from rat.research.orchestrator import ResearchOrchestrator
from rich import print as rprint

async def main():
    orchestrator = ResearchOrchestrator()
    question = "What are the main features and pricing of Billit's accounting software, and how does it compare to competitors in Belgium?"
    rprint(f"[bold cyan]Starting research on: {question}[/bold cyan]")
    results = await orchestrator.start_research(question)
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
    asyncio.run(main())
```

