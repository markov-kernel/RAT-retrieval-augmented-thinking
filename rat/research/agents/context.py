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
        self.last_update_time = time.time()
    
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
        self.last_update_time = time.time()
        
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
