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
    SEARCH = "search"      # New search query needed
    EXPLORE = "explore"    # URL exploration needed
    REASON = "reason"      # Deep analysis needed (deepseek-reasoner)
    EXECUTE = "execute"    # Code or structured output needed
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
