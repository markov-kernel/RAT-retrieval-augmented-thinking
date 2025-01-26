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