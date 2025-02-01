"""
Monitoring module for tracking system performance and metrics.
Provides centralized metrics collection and reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
from enum import Enum
import time
import json
from datetime import datetime


class MetricType(Enum):
    DECISION = "decision"
    API_CALL = "api_call"
    EXECUTION = "execution"
    ERROR = "error"


@dataclass
class MetricEvent:
    type: MetricType
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    total_decisions: int = 0
    successful_decisions: int = 0
    failed_decisions: int = 0
    total_execution_time: float = 0.0
    api_calls: int = 0
    api_errors: int = 0
    api_latency: float = 0.0
    events: List[MetricEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_decision(self, success: bool, execution_time: float, metadata: Dict[str, Any] = None):
        """Record a decision execution with its outcome."""
        self.total_decisions += 1
        if success:
            self.successful_decisions += 1
        else:
            self.failed_decisions += 1
        self.total_execution_time += execution_time
        self.events.append(MetricEvent(
            type=MetricType.DECISION,
            duration=execution_time,
            success=success,
            metadata=metadata or {}
        ))

    def record_api_call(self, success: bool, latency: float, metadata: Dict[str, Any] = None):
        """Record an API call with its outcome and latency."""
        self.api_calls += 1
        if not success:
            self.api_errors += 1
        self.api_latency += latency
        self.events.append(MetricEvent(
            type=MetricType.API_CALL,
            duration=latency,
            success=success,
            metadata=metadata or {}
        ))

    def record_error(self, error_type: str, error_message: str, metadata: Dict[str, Any] = None):
        """Record a system error."""
        self.events.append(MetricEvent(
            type=MetricType.ERROR,
            success=False,
            metadata={
                "error_type": error_type,
                "error_message": error_message,
                **(metadata or {})
            }
        ))

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        elapsed_time = time.time() - self.start_time
        avg_api_latency = self.api_latency / max(1, self.api_calls)
        success_rate = self.successful_decisions / max(1, self.total_decisions) * 100

        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "decisions": {
                "total": self.total_decisions,
                "successful": self.successful_decisions,
                "failed": self.failed_decisions,
                "success_rate": success_rate,
                "avg_execution_time": self.total_execution_time / max(1, self.total_decisions)
            },
            "api": {
                "total_calls": self.api_calls,
                "errors": self.api_errors,
                "avg_latency": avg_api_latency,
                "error_rate": (self.api_errors / max(1, self.api_calls)) * 100
            },
            "events": len(self.events)
        }

    def export_events(self, filepath: str):
        """Export all events to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(
                [{
                    "type": e.type.value,
                    "timestamp": e.timestamp,
                    "duration": e.duration,
                    "success": e.success,
                    "metadata": e.metadata
                } for e in self.events],
                f,
                indent=2
            )


class MetricsManager:
    """Singleton manager for system-wide metrics collection."""
    _instance = None
    _metrics: Dict[str, PerformanceMetrics] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_metrics(self, session_id: str) -> PerformanceMetrics:
        """Get or create metrics for a session."""
        if session_id not in self._metrics:
            self._metrics[session_id] = PerformanceMetrics()
        return self._metrics[session_id]

    def export_session_metrics(self, session_id: str, base_path: str):
        """Export all metrics for a session."""
        if session_id in self._metrics:
            metrics = self._metrics[session_id]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export summary
            summary_path = f"{base_path}/metrics_{timestamp}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(metrics.get_summary(), f, indent=2)
            
            # Export detailed events
            events_path = f"{base_path}/metrics_{timestamp}_events.json"
            metrics.export_events(events_path) 