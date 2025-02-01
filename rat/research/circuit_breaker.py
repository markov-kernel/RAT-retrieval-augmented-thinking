"""
Circuit breaker implementation for handling API failures gracefully.
Implements the circuit breaker pattern with three states: CLOSED, OPEN, and HALF-OPEN.
"""

import time
import logging
import asyncio
from enum import Enum
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from .monitoring import MetricsManager

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Possible states for the circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"     # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitStats:
    """Statistics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker implementation with monitoring and configurable thresholds.
    """
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 2,
        session_id: str = "default"
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.state = CircuitState.CLOSED
        self.stats = CircuitStats()
        self.session_id = session_id
        self.metrics = MetricsManager().get_metrics(session_id)
        logger.info(f"Circuit breaker '{name}' initialized with failure threshold {failure_threshold}")

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call if successful
            
        Raises:
            CircuitBreakerOpen: If the circuit is open
            Exception: Any exception from the function call
        """
        self._check_state()
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            self._handle_success()
            execution_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_api_call(
                success=True,
                latency=execution_time,
                metadata={
                    "circuit_name": self.name,
                    "state": self.state.value
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._handle_failure()
            
            # Record metrics
            self.metrics.record_api_call(
                success=False,
                latency=execution_time,
                metadata={
                    "circuit_name": self.name,
                    "state": self.state.value,
                    "error": str(e)
                }
            )
            
            raise

    def _check_state(self):
        """Check and potentially update the circuit state."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                logger.info(f"Circuit '{self.name}' attempting reset (entering half-open state)")
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit '{self.name}' is OPEN. "
                    f"Retry after {self._time_remaining_until_reset():.1f} seconds"
                )

    def _handle_success(self):
        """Handle a successful call."""
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.consecutive_failures = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.stats.consecutive_successes += 1
            if self.stats.consecutive_successes >= self.success_threshold:
                logger.info(f"Circuit '{self.name}' reset (closed) after {self.success_threshold} consecutive successes")
                self.state = CircuitState.CLOSED
                self.stats.consecutive_successes = 0

    def _handle_failure(self):
        """Handle a failed call."""
        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = time.time()
        
        if self.state == CircuitState.CLOSED and self.stats.consecutive_failures >= self.failure_threshold:
            logger.warning(
                f"Circuit '{self.name}' opened after {self.stats.consecutive_failures} consecutive failures"
            )
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit '{self.name}' reopened after failure in half-open state")
            self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt resetting the circuit."""
        if self.stats.last_failure_time is None:
            return True
        return time.time() - self.stats.last_failure_time >= self.reset_timeout

    def _time_remaining_until_reset(self) -> float:
        """Calculate time remaining until the circuit can be reset."""
        if self.stats.last_failure_time is None:
            return 0
        elapsed = time.time() - self.stats.last_failure_time
        return max(0, self.reset_timeout - elapsed)

    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.state.value,
            "stats": {
                "total_calls": self.stats.total_calls,
                "successful_calls": self.stats.successful_calls,
                "failed_calls": self.stats.failed_calls,
                "consecutive_failures": self.stats.consecutive_failures,
                "consecutive_successes": self.stats.consecutive_successes,
                "error_rate": (
                    self.stats.failed_calls / max(1, self.stats.total_calls) * 100
                ),
                "last_failure": (
                    time.strftime(
                        '%Y-%m-%d %H:%M:%S',
                        time.localtime(self.stats.last_failure_time)
                    ) if self.stats.last_failure_time else None
                )
            }
        }


class CircuitBreakerOpenError(Exception):
    """Raised when attempting to execute while the circuit is open."""
    pass 