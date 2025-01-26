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
Provides specialized agents for search, exploration, reasoning, and code execution.
"""

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContextBranch, ContentType
from .search import SearchAgent
from .explore import ExploreAgent
from .reason import ReasoningAgent
from .execute import ExecutionAgent

__all__ = [
    'BaseAgent',
    'ResearchContext',
    'ContextBranch',
    'ResearchDecision',
    'DecisionType',
    'ContentType',
    'SearchAgent',
    'ExploreAgent',
    'ReasoningAgent',
    'ExecutionAgent'
]
```

## rat/research/agents/base.py

```python
"""
Base agent interface and core decision-making structures.
Defines the contract that all specialized research agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from rich import print as rprint

class DecisionType(Enum):
    """Types of decisions an agent can make during research."""
    SEARCH = "search"  # New search query needed
    EXPLORE = "explore"  # URL exploration needed
    REASON = "reason"  # Deep analysis needed
    EXECUTE = "execute"  # Code execution needed
    TERMINATE = "terminate"  # Research complete

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
    
    Each specialized agent (search, explore, reason, execute) must implement
    the analyze method to make decisions based on the current research context.
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
            "total_execution_time": 0.0
        }
    
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
    content: str
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
                   content: Optional[str] = None,
                   metadata: Optional[Dict[str, Any]] = None,
                   token_count: Optional[int] = None,
                   priority: float = 0.5) -> ContentItem:
        """
        Add new content to a specific branch.
        
        Can be called with either:
        1. branch_id and content_item
        2. branch_id and individual parameters (content_type, content, metadata, etc.)
        
        Args:
            branch_id: Branch to add content to
            content_item: Pre-constructed ContentItem (if provided, other params are ignored)
            content_type: Type of content being added
            content: The content text/data
            metadata: Additional content metadata
            token_count: Pre-computed token count (if available)
            priority: Priority of this content (0-1), defaults to 0.5
            
        Returns:
            The created/added content item
        """
        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")
            
        branch = self.branches[branch_id]
        
        if content_item:
            item = content_item
            token_count = item.token_count
        else:
            if content_type is None or content is None or metadata is None:
                raise ValueError("Must provide either content_item or all of: content_type, content, metadata")
                
            # Estimate tokens if not provided
            if token_count is None:
                token_count = self._estimate_tokens(content)
                
            # Create new content item
            item = ContentItem(
                content_type=content_type,
                content=content,
                metadata=metadata,
                token_count=token_count,
                priority=priority
            )
            
        # Check token limit
        if branch.token_count + token_count > self.MAX_TOKENS_PER_BRANCH:
            raise ValueError(
                f"Adding this content would exceed the token limit "
                f"({self.MAX_TOKENS_PER_BRANCH}) for branch {branch_id}"
            )
            
        branch.content_items.append(item)
        branch.token_count += token_count
        
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
        Get content items from a specific branch.
        
        Args:
            branch_id: Branch to get content from
            content_type: Optional filter by content type
            
        Returns:
            List of matching content items
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
        
        Args:
            content: Text content to analyze
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(content) // 4
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the context to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the context
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
        
        Args:
            data: Dictionary representation of a context
            
        Returns:
            New ResearchContext instance
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

## rat/research/agents/execute.py

```python
"""
Execution agent for generating code and structured output using Claude.
Handles code generation, data formatting, and output validation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
import anthropic
import os
import json

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class ExecutionTask:
    """
    Represents a code or structured output task.
    
    Attributes:
        task_type: Type of task (code/json/etc)
        content: Content to process
        priority: Task priority (0-1)
        rationale: Why this task is needed
    """
    task_type: str
    content: str
    priority: float
    rationale: str
    timestamp: float = time.time()

class ExecutionAgent(BaseAgent):
    """
    Agent responsible for generating code and structured output using Claude.
    
    Handles code generation, data formatting, and output validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution agent.
        
        Args:
            config: Optional configuration parameters
        """
        super().__init__("execute", config)
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Configuration
        self.model = self.config.get("model", "claude-3-5-sonnet-20241022")
        self.min_priority = self.config.get("min_priority", 0.3)
        self.max_retries = self.config.get("max_retries", 2)
        
        # Tracking
        self.execution_tasks: Dict[str, ExecutionTask] = {}
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Analyze the research context and decide on execution tasks.
        
        Args:
            context: Current research context
            
        Returns:
            List of execution-related decisions
        """
        decisions = []
        
        # Get analyzed content that might need structured output
        analyzed_content = context.get_content(
            "main",
            ContentType.ANALYSIS
        )
        
        for content in analyzed_content:
            # Check if content needs code generation
            if self._needs_code_generation(content):
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXECUTE,
                        priority=content.priority * 0.9,
                        context={
                            "task_type": "code",
                            "content": content.content,
                            "metadata": content.metadata
                        },
                        rationale="Generate code implementation"
                    )
                )
                
            # Check if content needs JSON formatting
            if self._needs_json_formatting(content):
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXECUTE,
                        priority=content.priority * 0.8,
                        context={
                            "task_type": "json",
                            "content": content.content,
                            "metadata": content.metadata
                        },
                        rationale="Convert analysis to structured JSON"
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
        return decision.decision_type == DecisionType.EXECUTE
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a code or structured output decision.
        
        Args:
            decision: Execution decision to execute
            
        Returns:
            Generated output and metadata
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            task_type = decision.context["task_type"]
            content = decision.context["content"]
            metadata = decision.context.get("metadata", {})
            
            # Track the task
            task_id = str(len(self.execution_tasks) + 1)
            self.execution_tasks[task_id] = ExecutionTask(
                task_type=task_type,
                content=content,
                priority=decision.priority,
                rationale=decision.rationale
            )
            
            # Execute with retries
            for attempt in range(self.max_retries + 1):
                try:
                    if task_type == "code":
                        results = self._generate_code(content, metadata)
                    elif task_type == "json":
                        results = self._format_json(content, metadata)
                    else:
                        raise ValueError(f"Unknown task type: {task_type}")
                        
                    success = True
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    rprint(f"[yellow]Attempt {attempt + 1} failed: {str(e)}[/yellow]")
                    time.sleep(1)  # Brief delay before retry
                    
            if success:
                rprint(f"[green]{task_type.title()} generation completed[/green]")
            else:
                rprint(f"[yellow]No output generated for {task_type}[/yellow]")
                
        except Exception as e:
            rprint(f"[red]Execution error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "output": "",
                "metadata": {
                    "task_type": decision.context.get("task_type", "unknown")
                }
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
        
    def _needs_code_generation(self, content: ContentItem) -> bool:
        """
        Check if content needs code generation.
        
        Args:
            content: Content item to check
            
        Returns:
            True if code generation is needed
        """
        # Look for code-related keywords
        code_indicators = [
            "implementation",
            "code",
            "function",
            "class",
            "algorithm",
            "script"
        ]
        
        text = content.content.lower()
        return any(indicator in text for indicator in code_indicators)
        
    def _needs_json_formatting(self, content: ContentItem) -> bool:
        """
        Check if content needs JSON formatting.
        
        Args:
            content: Content item to check
            
        Returns:
            True if JSON formatting is needed
        """
        # Look for structured data indicators
        json_indicators = [
            "structured",
            "json",
            "data format",
            "schema",
            "key-value",
            "mapping"
        ]
        
        text = content.content.lower()
        return any(indicator in text for indicator in json_indicators)
        
    def _generate_code(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate code using Claude.
        
        Args:
            content: Content to generate code from
            metadata: Additional context
            
        Returns:
            Generated code and metadata
        """
        messages = [{
            "role": "system",
            "content": (
                "You are an expert code generator. Generate clean, efficient, "
                "and well-documented code based on the provided description."
            )
        }, {
            "role": "user",
            "content": f"Generate code for:\n\n{content}"
        }]
        
        response = self.claude_client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=4000
        )
        
        generated_code = response.content[0].text
        
        return {
            "output": generated_code,
            "language": self._detect_language(generated_code),
            "metadata": metadata
        }
        
    def _format_json(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert content to structured JSON using Claude.
        
        Args:
            content: Content to convert to JSON
            metadata: Additional context
            
        Returns:
            Formatted JSON and metadata
        """
        messages = [{
            "role": "system",
            "content": (
                "You are an expert at converting unstructured text into clean, "
                "well-structured JSON format."
            )
        }, {
            "role": "user",
            "content": f"Convert to JSON:\n\n{content}"
        }]
        
        response = self.claude_client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=4000
        )
        
        json_str = response.content[0].text
        
        # Validate JSON
        try:
            json_data = json.loads(json_str)
            return {
                "output": json_data,
                "format": "json",
                "metadata": metadata
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON generated: {str(e)}")
            
    def _detect_language(self, code: str) -> str:
        """
        Detect the programming language of generated code.
        
        Args:
            code: Generated code to analyze
            
        Returns:
            Detected language name
        """
        # TODO: Implement smarter language detection
        # For now, use simple keyword matching
        language_indicators = {
            "python": ["def ", "import ", "class ", "print("],
            "javascript": ["function", "const ", "let ", "var "],
            "java": ["public class", "private ", "void ", "String"],
            "cpp": ["#include", "int main", "std::", "void"]
        }
        
        for lang, indicators in language_indicators.items():
            if any(indicator in code for indicator in indicators):
                return lang
                
        return "unknown"
```

## rat/research/agents/explore.py

```python
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
```

## rat/research/agents/reason.py

```python
"""
Reasoning agent for analyzing research content using DeepSeek.
Handles content analysis, parallel processing, and insight generation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
from openai import OpenAI
import os

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class AnalysisTask:
    """
    Represents a content analysis task.
    
    Attributes:
        content: Content to analyze
        priority: Analysis priority (0-1)
        rationale: Why this analysis is needed
        chunk_index: Index if this is part of a chunked analysis
    """
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()

class ReasoningAgent(BaseAgent):
    """
    Agent responsible for analyzing content using the DeepSeek API.
    
    Handles content analysis, parallel processing for large contexts,
    and insight generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning agent.
        
        Args:
            config: Optional configuration parameters
        """
        super().__init__("reason", config)
        
        # Initialize DeepSeek client
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        # Configuration
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.chunk_size = self.config.get("chunk_size", 30000)  # ~30k tokens per chunk
        self.min_priority = self.config.get("min_priority", 0.3)
        
        # Tracking
        self.analysis_tasks: Dict[str, AnalysisTask] = {}
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Analyze the research context and decide on reasoning tasks.
        
        Args:
            context: Current research context
            
        Returns:
            List of reasoning-related decisions
        """
        decisions = []
        
        # Get content needing analysis
        search_results = context.get_content(
            "main",
            ContentType.SEARCH_RESULT
        )
        explored_content = context.get_content(
            "main",
            ContentType.URL_CONTENT
        )
        
        # Analyze search results
        if search_results:
            decisions.extend(
                self._create_analysis_decisions(
                    content_items=search_results,
                    content_type="search_results",
                    base_priority=0.9
                )
            )
            
        # Analyze explored content
        if explored_content:
            decisions.extend(
                self._create_analysis_decisions(
                    content_items=explored_content,
                    content_type="url_content",
                    base_priority=0.8
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
        Execute a reasoning decision.
        
        Args:
            decision: Reasoning decision to execute
            
        Returns:
            Analysis results and insights
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            
            # Check if content needs chunking
            if len(content.split()) > self.chunk_size:
                results = self._parallel_analyze_content(content, content_type)
            else:
                results = self._analyze_content_chunk(content, content_type)
                
            success = bool(results.get("analysis"))
            if success:
                rprint(f"[green]Analysis completed for {content_type}[/green]")
            else:
                rprint(f"[yellow]No analysis generated for {content_type}[/yellow]")
                
        except Exception as e:
            rprint(f"[red]Analysis error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "analysis": "",
                "insights": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
        
    def _create_analysis_decisions(
        self,
        content_items: List[ContentItem],
        content_type: str,
        base_priority: float
    ) -> List[ResearchDecision]:
        """
        Create analysis decisions for content items.
        
        Args:
            content_items: Content items to analyze
            content_type: Type of content being analyzed
            base_priority: Base priority for these items
            
        Returns:
            List of analysis decisions
        """
        decisions = []
        
        for item in content_items:
            # Skip if priority too low
            if item.priority * base_priority < self.min_priority:
                continue
                
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.REASON,
                    priority=item.priority * base_priority,
                    context={
                        "content": item.content,
                        "content_type": content_type,
                        "metadata": item.metadata
                    },
                    rationale=f"Analyze {content_type} content"
                )
            )
            
        return decisions
        
    def _parallel_analyze_content(
        self,
        content: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Analyze large content in parallel chunks.
        
        Args:
            content: Content to analyze
            content_type: Type of content being analyzed
            
        Returns:
            Combined analysis results
        """
        # Split content into chunks
        words = content.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # Analyze chunks in parallel
        chunk_results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            future_to_chunk = {
                executor.submit(
                    self._analyze_content_chunk,
                    chunk,
                    f"{content_type}_chunk_{i}"
                ): i
                for i, chunk in enumerate(chunks)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_index = future_to_chunk[future]
                try:
                    result = future.result()
                    result["chunk_index"] = chunk_index
                    chunk_results.append(result)
                except Exception as e:
                    rprint(f"[red]Error in chunk {chunk_index}: {str(e)}[/red]")
                    
        # Combine chunk results
        return self._combine_chunk_results(chunk_results)
        
    def _analyze_content_chunk(
        self,
        content: str,
        content_type: str
    ) -> Dict[str, Any]:
        """
        Analyze a single chunk of content using DeepSeek.
        
        Args:
            content: Content chunk to analyze
            content_type: Type of content being analyzed
            
        Returns:
            Analysis results for this chunk
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{
                    "role": "system",
                    "content": (
                        "You are an expert research analyst. Analyze the following "
                        "content and extract key insights, patterns, and implications."
                    )
                }, {
                    "role": "user",
                    "content": f"Content type: {content_type}\n\nContent:\n{content}"
                }],
                temperature=0.7
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "analysis": analysis,
                "insights": self._extract_insights(analysis),
                "content_type": content_type
            }
            
        except Exception as e:
            rprint(f"[red]DeepSeek API error: {str(e)}[/red]")
            return {
                "analysis": "",
                "insights": [],
                "content_type": content_type,
                "error": str(e)
            }
            
    def _combine_chunk_results(
        self,
        chunk_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine results from multiple analyzed chunks.
        
        Args:
            chunk_results: List of chunk analysis results
            
        Returns:
            Combined analysis
        """
        # Sort chunks by index
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        
        # Combine analyses
        combined_analysis = "\n\n".join(
            chunk["analysis"] for chunk in sorted_chunks
            if chunk.get("analysis")
        )
        
        # Merge insights
        all_insights = []
        for chunk in sorted_chunks:
            all_insights.extend(chunk.get("insights", []))
            
        # Remove duplicates while preserving order
        unique_insights = list(dict.fromkeys(all_insights))
        
        return {
            "analysis": combined_analysis,
            "insights": unique_insights,
            "chunk_count": len(chunk_results)
        }
        
    def _extract_insights(self, analysis: str) -> List[str]:
        """
        Extract key insights from an analysis.
        
        Args:
            analysis: Analysis text to process
            
        Returns:
            List of extracted insights
        """
        # TODO: Use more sophisticated insight extraction
        # For now, split on newlines and filter
        lines = analysis.split("\n")
        insights = []
        
        for line in lines:
            line = line.strip()
            # Look for bullet points or numbered items
            if line.startswith(("-", "*", "•")) or (
                len(line) > 2 and line[0].isdigit() and line[1] == "."
            ):
                insights.append(line.lstrip("- *•").strip())
                
        return insights
```

## rat/research/agents/search.py

```python
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

load_dotenv()

class FirecrawlClient:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a webpage using Firecrawl API.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Make the request to scrape the URL
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                }
            )
            
            return self._process_extracted_content(result.get('data', {}), url)
            
        except Exception as e:
            rprint(f"[red]Firecrawl API request failed for {url}: {str(e)}[/red]")
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
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import time
from rich import print as rprint
from pathlib import Path

from .agents.search import SearchAgent
from .agents.explore import ExploreAgent
from .agents.reason import ReasoningAgent
from .agents.execute import ExecutionAgent
from .perplexity_client import PerplexityClient
from .firecrawl_client import FirecrawlClient
from .output_manager import OutputManager
from .agents.base import ResearchDecision, DecisionType
from .agents.context import ResearchContext, ContentType, ContentItem

@dataclass
class ResearchIteration:
    """
    Represents a single iteration of the research process.
    
    Attributes:
        iteration_number: Current iteration number
        decisions_made: List of decisions made
        content_added: New content items added
        metrics: Performance metrics for this iteration
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
    work together effectively.
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
        
        # Initialize agents
        self.search_agent = SearchAgent(
            self.perplexity,
            self.config.get("search_config")
        )
        self.explore_agent = ExploreAgent(
            self.firecrawl,
            self.config.get("explore_config")
        )
        self.reason_agent = ReasoningAgent(
            self.config.get("reason_config")
        )
        self.execute_agent = ExecutionAgent(
            self.config.get("execute_config")
        )
        
        # Initialize managers
        self.output_manager = OutputManager()
        
        # Configuration
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_new_content = self.config.get("min_new_content", 1)  # Lower threshold since we're getting good content
        self.min_confidence = self.config.get("min_confidence", 0.7)
        
        # State tracking
        self.current_context: Optional[ResearchContext] = None
        self.iterations: List[ResearchIteration] = []
        self.research_dir: Optional[Path] = None
        
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
        
        Args:
            iteration_number: Current iteration number
            
        Returns:
            Results of this iteration
        """
        iteration_start = time.time()
        decisions_made = []
        content_added = []
        
        try:
            # 1. Get decisions from all agents
            all_decisions = self._gather_agent_decisions()
            
            # 2. Sort decisions by priority
            sorted_decisions = sorted(
                all_decisions,
                key=lambda d: d.priority,
                reverse=True
            )
            
            # 3. Execute decisions
            for decision in sorted_decisions:
                agent = self._get_agent_for_decision(decision)
                if agent:
                    try:
                        result = agent.execute_decision(decision)
                        decisions_made.append(decision)
                        
                        # Add results to context
                        if result:
                            # Process result based on decision type
                            if decision.decision_type == DecisionType.SEARCH:
                                # For search results, extract content and urls
                                content_str = result.get('content', '')
                                urls = result.get('urls', [])
                                token_count = self.current_context._estimate_tokens(content_str)
                                
                                content_item = ContentItem(
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
                                
                                content_item = ContentItem(
                                    content_type=self._get_content_type(decision),
                                    content=result,
                                    metadata={
                                        "decision_type": decision.decision_type.value,
                                        "iteration": iteration_number
                                    },
                                    token_count=token_count,
                                    priority=decision.priority
                                )
                            self.current_context.add_content("main", content_item=content_item)
                            content_added.append(content_item)
                            
                    except Exception as e:
                        rprint(f"[red]Error executing decision: {str(e)}[/red]")
                        
        except Exception as e:
            rprint(f"[red]Iteration error: {str(e)}[/red]")
            
        # Calculate iteration metrics
        metrics = {
            "iteration_time": time.time() - iteration_start,
            "decisions_made": len(decisions_made),
            "content_added": len(content_added),
            "agent_metrics": self._get_agent_metrics()
        }
        
        return ResearchIteration(
            iteration_number=iteration_number,
            decisions_made=decisions_made,
            content_added=content_added,
            metrics=metrics
        )
        
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
            self.reason_agent,
            self.execute_agent
        ]:
            try:
                decisions = agent.analyze(self.current_context)
                all_decisions.extend(decisions)
            except Exception as e:
                rprint(f"[red]Error getting decisions from {agent.name}: {str(e)}[/red]")
                
        return all_decisions
        
    def _get_agent_for_decision(
        self,
        decision: ResearchDecision
    ) -> Optional[Any]:
        """
        Get the appropriate agent for a decision.
        
        Args:
            decision: Decision to handle
            
        Returns:
            Agent that can handle the decision
        """
        agent_map = {
            DecisionType.SEARCH: self.search_agent,
            DecisionType.EXPLORE: self.explore_agent,
            DecisionType.REASON: self.reason_agent,
            DecisionType.EXECUTE: self.execute_agent
        }
        
        return agent_map.get(decision.decision_type)
        
    def _get_content_type(self, decision: ResearchDecision) -> ContentType:
        """
        Map decision type to content type.
        
        Args:
            decision: Decision to map
            
        Returns:
            Appropriate content type
        """
        type_map = {
            DecisionType.SEARCH: ContentType.SEARCH_RESULT,
            DecisionType.EXPLORE: ContentType.EXPLORED_CONTENT,
            DecisionType.REASON: ContentType.ANALYSIS,
            DecisionType.EXECUTE: ContentType.STRUCTURED_OUTPUT
        }
        
        return type_map.get(decision.decision_type, ContentType.OTHER)
        
    def _should_terminate(self, iteration: ResearchIteration) -> bool:
        """
        Check if research should terminate.
        
        Args:
            iteration: Last completed iteration
            
        Returns:
            True if research should stop
        """
        # Check if we found enough content
        if len(iteration.content_added) < self.min_new_content:
            rprint("[yellow]Terminating: Not enough new content found[/yellow]")
            return True
            
        # Check if we have high confidence results
        analysis_content = self.current_context.get_content(
            "main",
            ContentType.ANALYSIS
        )
        if analysis_content:
            latest = analysis_content[-1]
            confidence = latest.content.get("confidence", 0)
            if confidence >= self.min_confidence:
                rprint("[green]Terminating: Reached confidence threshold[/green]")
                return True
                
        return False
        
    def _generate_final_output(self) -> Dict[str, Any]:
        """
        Generate the final research output.
        
        Returns:
            Research paper and metadata
        """
        # Get all content by type
        search_results = self.current_context.get_content(
            "main",
            ContentType.SEARCH_RESULT
        )
        explored_content = self.current_context.get_content(
            "main",
            ContentType.EXPLORED_CONTENT
        )
        analysis = self.current_context.get_content(
            "main",
            ContentType.ANALYSIS
        )
        structured_output = self.current_context.get_content(
            "main",
            ContentType.STRUCTURED_OUTPUT
        )
        
        # Generate paper sections
        sections = []
        
        # Introduction
        sections.append(f"# {self.current_context.initial_question}\n")
        sections.append("## Introduction\n")
        if search_results:
            sections.append(search_results[0].content)
            
        # Main findings
        sections.append("\n## Key Findings\n")
        for result in analysis:
            if isinstance(result.content, dict):
                insights = result.content.get("insights", [])
                for insight in insights:
                    sections.append(f"- {insight}\n")
            else:
                sections.append(f"- {result.content}\n")
                
        # Detailed analysis
        sections.append("\n## Detailed Analysis\n")
        for content in explored_content:
            if isinstance(content.content, dict):
                title = content.content.get("title", "")
                text = content.content.get("text", "")
                if title and text:
                    sections.append(f"\n### {title}\n\n{text}\n")
            else:
                sections.append(f"\n{content.content}\n")
                
        # Technical details
        if structured_output:
            sections.append("\n## Technical Details\n")
            for output in structured_output:
                if isinstance(output.content, dict):
                    if output.content.get("format") == "json":
                        sections.append("```json\n")
                        sections.append(
                            json.dumps(output.content["output"], indent=2)
                        )
                        sections.append("\n```\n")
                    else:
                        sections.append("```\n")
                        sections.append(output.content.get("output", ""))
                        sections.append("\n```\n")
                else:
                    sections.append(f"{output.content}\n")
                    
        # Sources
        sections.append("\n## Sources\n")
        sources = set()
        for content in explored_content:
            url = content.content.get("metadata", {}).get("url")
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
        Get metrics from all agents.
        
        Returns:
            Combined agent metrics
        """
        return {
            "search": self.search_agent.get_metrics(),
            "explore": self.explore_agent.get_metrics(),
            "reason": self.reason_agent.get_metrics(),
            "execute": self.execute_agent.get_metrics()
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
from openai import OpenAI
from typing import List, Dict, Any
from rich import print as rprint

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
            
            return {
                "content": content,
                "urls": urls
            }
            
        except Exception as e:
            rprint(f"[red]Error in Perplexity search: {str(e)}[/red]")
            return {
                "content": "",
                "urls": []
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

## rat/research/utils/deepseek_client.py

```python
"""
Utility for calling the deepseek-reasoner API.
"""

import requests
from typing import Dict, Any

class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-reasoner"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com"

    def analyze(self, user_content: str, system_instruction: str, content_type: str) -> Dict[str, Any]:
        """
        Send content to deepseek-reasoner and return a JSON response.
        Implement your actual network call here.
        """
        if not self.api_key:
            raise ValueError("No DEEPSEEK_API_KEY provided.")

        # Pseudocode for an API request:
        # response = requests.post(
        #     f"{self.base_url}/v1/completions",
        #     headers={ "Authorization": f"Bearer {self.api_key}" },
        #     json={
        #         "model": self.model,
        #         "system_instruction": system_instruction,
        #         "user_content": user_content,
        #         "metadata": {"content_type": content_type}
        #     }
        # )
        # data = response.json()
        #
        # For demonstration, we will just mock it:
        data = {
            "analysis": f"[Mock deepseek analysis for {content_type}] {user_content[:100]}..."
        }
        return data
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

