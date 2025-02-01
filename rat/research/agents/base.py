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
        self._last_api_call = 0.0

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
            self._last_api_call = current_time  # Update both timestamps

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