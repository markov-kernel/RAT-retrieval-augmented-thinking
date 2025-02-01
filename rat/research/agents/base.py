"""
Base agent interface and core decision-making structures.
Defines the contract that all specialized research agents must implement.
This async version uses asyncio locks and awaits where appropriate.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ResearchContext

from enum import Enum
from rich import print as rprint
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class TokenBucket:
    """
    Implements the token bucket algorithm for rate limiting.
    Provides a more flexible and efficient way to control API request rates.
    """
    def __init__(self, rate_limit: float, burst_limit: Optional[float] = None):
        """
        Initialize the token bucket.
        
        Args:
            rate_limit: Number of tokens per minute
            burst_limit: Maximum number of tokens that can be accumulated (defaults to rate_limit)
        """
        self.rate_limit = float(rate_limit)
        self.burst_limit = float(burst_limit if burst_limit is not None else rate_limit)
        self.tokens = self.burst_limit
        self.last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()
        
    async def acquire(self, tokens: float = 1.0) -> float:
        """
        Acquire tokens from the bucket. Returns the wait time if tokens aren't available.
        
        Args:
            tokens: Number of tokens to acquire (default: 1.0)
            
        Returns:
            Float: Time to wait in seconds (0 if tokens are available immediately)
        """
        async with self._lock:
            now = asyncio.get_event_loop().time()
            time_passed = now - self.last_update
            self.tokens = min(
                self.burst_limit,
                self.tokens + time_passed * (self.rate_limit / 60.0)
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0
            else:
                wait_time = (tokens - self.tokens) * 60.0 / self.rate_limit
                self.tokens = 0
                return wait_time

    async def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Try to acquire tokens without waiting.
        
        Returns:
            bool: True if tokens were acquired, False otherwise
        """
        wait_time = await self.acquire(tokens)
        return wait_time == 0.0


class DecisionType(Enum):
    """Types of decisions an agent can make during research."""
    SEARCH = "search"      # New search query needed
    EXPLORE = "explore"    # URL exploration needed
    REASON = "reason"      # Deep analysis needed (using Gemini now)
    EXECUTE = "execute"     # Execution of a decision
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
    All methods are now asynchronous with improved rate limiting.
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
            "retry_attempts": 0,
            "tokens_consumed": 0.0
        }
        self.max_workers = self.config.get("max_workers", 5)
        self.rate_limit = self.config.get("rate_limit", 100)
        self.burst_limit = self.config.get("burst_limit", self.rate_limit * 1.5)
        self._active_tasks: Set[str] = set()
        self._tasks_lock = asyncio.Lock()
        self._token_bucket = TokenBucket(self.rate_limit, self.burst_limit)
        self.logger = logging.getLogger(f"{__name__}.{name}")

    async def _enforce_rate_limit(self, tokens: float = 1.0) -> None:
        """
        Enforce rate limiting using the token bucket algorithm.
        More sophisticated than the previous implementation, allowing for bursts
        while still maintaining average rate limits.
        
        Args:
            tokens: Number of tokens to consume (default: 1.0)
        """
        if self.rate_limit <= 0:
            return

        wait_time = await self._token_bucket.acquire(tokens)
        if wait_time > 0:
            self.metrics["rate_limit_delays"] += 1
            self.logger.debug(f"Rate limit enforced, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
        
        self.metrics["tokens_consumed"] += tokens

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