"""Advanced circuit breaker and retry mechanisms for robust error handling.

This module provides comprehensive fault tolerance patterns including circuit breakers,
exponential backoff, timeout management, and graceful degradation for privacy-preserving
ML training systems.
"""

import time
import logging
import asyncio
import threading
from typing import Dict, Any, Callable, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import random
import functools

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, blocking requests
    HALF_OPEN = "half_open"  # Testing if service has recovered


class RetryStrategy(Enum):
    """Retry strategies."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    FIBONACCI_BACKOFF = "fibonacci_backoff"
    JITTER = "jitter"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # seconds
    expected_exception: Optional[type] = None
    fallback_function: Optional[Callable] = None
    half_open_max_calls: int = 3
    minimum_throughput: int = 10
    sliding_window_size: int = 100


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    jitter: bool = True
    timeout: Optional[float] = None
    retriable_exceptions: List[type] = field(default_factory=list)


@dataclass
class CallResult:
    """Result of a circuit breaker call."""
    success: bool
    result: Any = None
    exception: Optional[Exception] = None
    attempts: int = 1
    total_time: float = 0.0
    circuit_state: CircuitState = CircuitState.CLOSED


class CircuitBreaker:
    """Advanced circuit breaker implementation with monitoring and metrics."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
        # Sliding window for throughput calculation
        self.call_history: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        
        # Metrics
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.state_changes = []
        
        logger.info(f"CircuitBreaker initialized with failure_threshold={config.failure_threshold}")
    
    def call(self, func: Callable, *args, **kwargs) -> CallResult:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            CallResult with execution details
        """
        start_time = time.time()
        
        with self.lock:
            self.total_calls += 1
            
            # Check if circuit is open
            if self.state == CircuitState.OPEN:
                if not self._should_attempt_reset():
                    return CallResult(
                        success=False,
                        exception=Exception("Circuit breaker is OPEN"),
                        total_time=time.time() - start_time,
                        circuit_state=self.state
                    )
                else:
                    self._transition_to_half_open()
            
            # Check half-open state limits
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    return CallResult(
                        success=False,
                        exception=Exception("Circuit breaker HALF_OPEN call limit exceeded"),
                        total_time=time.time() - start_time,
                        circuit_state=self.state
                    )
                self.half_open_calls += 1
        
        # Execute the function
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Record success
            with self.lock:
                self._record_success()
                self._update_call_history(True, execution_time)
            
            return CallResult(
                success=True,
                result=result,
                total_time=execution_time,
                circuit_state=self.state
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Check if this exception should trigger circuit breaker
            should_count_failure = (
                self.config.expected_exception is None or 
                isinstance(e, self.config.expected_exception)
            )
            
            with self.lock:
                if should_count_failure:
                    self._record_failure()
                self._update_call_history(False, execution_time)
            
            # Try fallback if available
            if self.config.fallback_function and should_count_failure:
                try:
                    fallback_result = self.config.fallback_function(*args, **kwargs)
                    logger.warning(f"Circuit breaker used fallback function: {e}")
                    
                    return CallResult(
                        success=True,
                        result=fallback_result,
                        exception=e,
                        total_time=time.time() - start_time,
                        circuit_state=self.state
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback function also failed: {fallback_error}")
            
            return CallResult(
                success=False,
                exception=e,
                total_time=execution_time,
                circuit_state=self.state
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset from OPEN to HALF_OPEN."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.recovery_timeout
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit from OPEN to HALF_OPEN."""
        logger.info("Circuit breaker transitioning from OPEN to HALF_OPEN")
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        self._record_state_change(CircuitState.HALF_OPEN)
    
    def _record_success(self) -> None:
        """Record successful call."""
        self.success_count += 1
        self.total_successes += 1
        self.failure_count = 0  # Reset failure count on success
        
        # Transition from HALF_OPEN to CLOSED if enough successes
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker transitioning from HALF_OPEN to CLOSED")
            self.state = CircuitState.CLOSED
            self.half_open_calls = 0
            self._record_state_change(CircuitState.CLOSED)
    
    def _record_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.total_failures += 1
        self.last_failure_time = time.time()
        
        # Check if we should open the circuit
        if self._should_open_circuit():
            logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN
            self._record_state_change(CircuitState.OPEN)
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened."""
        # Basic threshold check
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Throughput-based check if configured
        if self.config.minimum_throughput > 0:
            recent_calls = self._get_recent_calls()
            if len(recent_calls) >= self.config.minimum_throughput:
                failure_rate = sum(1 for call in recent_calls if not call['success']) / len(recent_calls)
                return failure_rate > 0.5  # 50% failure rate threshold
        
        return False
    
    def _update_call_history(self, success: bool, execution_time: float) -> None:
        """Update call history for sliding window analysis."""
        call_record = {
            'timestamp': time.time(),
            'success': success,
            'execution_time': execution_time
        }
        
        self.call_history.append(call_record)
        
        # Maintain sliding window size
        if len(self.call_history) > self.config.sliding_window_size:
            self.call_history.pop(0)
    
    def _get_recent_calls(self, window_seconds: float = 60.0) -> List[Dict[str, Any]]:
        """Get recent calls within time window."""
        cutoff_time = time.time() - window_seconds
        return [call for call in self.call_history if call['timestamp'] > cutoff_time]
    
    def _record_state_change(self, new_state: CircuitState) -> None:
        """Record state change for monitoring."""
        self.state_changes.append({
            'timestamp': datetime.now(),
            'from_state': self.state.value if hasattr(self, 'state') else 'unknown',
            'to_state': new_state.value
        })
        
        # Keep history manageable
        if len(self.state_changes) > 100:
            self.state_changes.pop(0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self.lock:
            recent_calls = self._get_recent_calls()
            
            success_rate = (self.total_successes / self.total_calls 
                          if self.total_calls > 0 else 0.0)
            
            avg_execution_time = (
                sum(call['execution_time'] for call in recent_calls) / len(recent_calls)
                if recent_calls else 0.0
            )
            
            return {
                'state': self.state.value,
                'total_calls': self.total_calls,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'success_rate': success_rate,
                'failure_count': self.failure_count,
                'recent_calls': len(recent_calls),
                'avg_execution_time': avg_execution_time,
                'state_changes': len(self.state_changes),
                'last_failure_time': self.last_failure_time
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self.lock:
            logger.info("Circuit breaker manually reset")
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None
            self.half_open_calls = 0
            self._record_state_change(CircuitState.CLOSED)


class RetryMechanism:
    """Advanced retry mechanism with multiple backoff strategies."""
    
    def __init__(self, config: RetryConfig):
        """Initialize retry mechanism.
        
        Args:
            config: Retry configuration
        """
        self.config = config
        logger.info(f"RetryMechanism initialized with strategy={config.strategy.value}, max_attempts={config.max_attempts}")
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> CallResult:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            CallResult with execution details
        """
        last_exception = None
        total_start_time = time.time()
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # Apply timeout if configured
                if self.config.timeout:
                    result = self._execute_with_timeout(func, self.config.timeout, *args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                total_time = time.time() - total_start_time
                return CallResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_time=total_time
                )
                
            except Exception as e:
                last_exception = e
                
                # Check if exception is retriable
                if not self._is_retriable_exception(e):
                    logger.warning(f"Non-retriable exception on attempt {attempt}: {e}")
                    break
                
                # Don't retry on final attempt
                if attempt == self.config.max_attempts:
                    break
                
                # Calculate delay and wait
                delay = self._calculate_delay(attempt)
                logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
        
        total_time = time.time() - total_start_time
        return CallResult(
            success=False,
            exception=last_exception,
            attempts=self.config.max_attempts,
            total_time=total_time
        )
    
    def _execute_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """Execute function with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {timeout}s")
        
        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(*args, **kwargs)
            signal.alarm(0)  # Cancel alarm
            return result
        finally:
            signal.signal(signal.SIGALRM, old_handler)
    
    def _is_retriable_exception(self, exception: Exception) -> bool:
        """Check if exception is retriable."""
        if not self.config.retriable_exceptions:
            # Default retriable exceptions
            retriable_types = (
                ConnectionError, TimeoutError, OSError,
                # Add more default retriable exceptions
            )
            return isinstance(exception, retriable_types)
        
        return any(isinstance(exception, exc_type) for exc_type in self.config.retriable_exceptions)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate retry delay based on strategy."""
        if self.config.strategy == RetryStrategy.FIXED_INTERVAL:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            delay = self.config.base_delay * self._fibonacci(attempt)
        
        elif self.config.strategy == RetryStrategy.JITTER:
            delay = self.config.base_delay * (2 ** (attempt - 1))
            # Add jitter to prevent thundering herd
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        else:
            delay = self.config.base_delay
        
        # Apply jitter if enabled and not using jitter strategy
        if self.config.jitter and self.config.strategy != RetryStrategy.JITTER:
            jitter = delay * 0.1 * random.random()
            delay += jitter
        
        # Ensure delay doesn't exceed maximum
        return min(delay, self.config.max_delay)
    
    def _fibonacci(self, n: int) -> int:
        """Calculate fibonacci number."""
        if n <= 1:
            return n
        return self._fibonacci(n - 1) + self._fibonacci(n - 2)


class RobustExecutor:
    """Combined circuit breaker and retry mechanism for maximum robustness."""
    
    def __init__(
        self, 
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_circuit_breaker: bool = True,
        enable_retry: bool = True
    ):
        """Initialize robust executor.
        
        Args:
            circuit_config: Circuit breaker configuration
            retry_config: Retry configuration  
            enable_circuit_breaker: Enable circuit breaker
            enable_retry: Enable retry mechanism
        """
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_retry = enable_retry
        
        # Initialize circuit breaker
        if self.enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(
                circuit_config or CircuitBreakerConfig()
            )
        
        # Initialize retry mechanism
        if self.enable_retry:
            self.retry_mechanism = RetryMechanism(
                retry_config or RetryConfig()
            )
        
        logger.info(f"RobustExecutor initialized (circuit_breaker={enable_circuit_breaker}, retry={enable_retry})")
    
    def execute(self, func: Callable, *args, **kwargs) -> CallResult:
        """Execute function with robust error handling.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            CallResult with execution details
        """
        if self.enable_circuit_breaker and self.enable_retry:
            # Combine circuit breaker and retry
            def retry_func(*retry_args, **retry_kwargs):
                return self.circuit_breaker.call(func, *retry_args, **retry_kwargs)
            
            retry_result = self.retry_mechanism.execute_with_retry(retry_func, *args, **kwargs)
            
            # If retry succeeded, use its result but update with circuit breaker state
            if retry_result.success:
                return retry_result
            
            # If retry failed, return the last circuit breaker result
            return retry_result
        
        elif self.enable_circuit_breaker:
            return self.circuit_breaker.call(func, *args, **kwargs)
        
        elif self.enable_retry:
            return self.retry_mechanism.execute_with_retry(func, *args, **kwargs)
        
        else:
            # No error handling, direct execution
            try:
                result = func(*args, **kwargs)
                return CallResult(success=True, result=result)
            except Exception as e:
                return CallResult(success=False, exception=e)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics."""
        metrics = {
            'circuit_breaker_enabled': self.enable_circuit_breaker,
            'retry_enabled': self.enable_retry
        }
        
        if self.enable_circuit_breaker:
            metrics['circuit_breaker'] = self.circuit_breaker.get_metrics()
        
        return metrics
    
    def reset(self) -> None:
        """Reset all components."""
        if self.enable_circuit_breaker:
            self.circuit_breaker.reset()


# Decorator for easy integration
def robust_execution(
    circuit_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    enable_circuit_breaker: bool = True,
    enable_retry: bool = True
):
    """Decorator for robust function execution.
    
    Args:
        circuit_config: Circuit breaker configuration
        retry_config: Retry configuration
        enable_circuit_breaker: Enable circuit breaker
        enable_retry: Enable retry mechanism
    """
    def decorator(func):
        executor = RobustExecutor(
            circuit_config=circuit_config,
            retry_config=retry_config,
            enable_circuit_breaker=enable_circuit_breaker,
            enable_retry=enable_retry
        )
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = executor.execute(func, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.exception or Exception("Execution failed")
        
        # Attach executor for metrics access
        wrapper._robust_executor = executor
        return wrapper
    
    return decorator


# Async versions for async operations
class AsyncRobustExecutor:
    """Async version of robust executor."""
    
    def __init__(
        self,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None
    ):
        """Initialize async robust executor."""
        self.circuit_config = circuit_config or CircuitBreakerConfig()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(self.circuit_config)
        self.retry_mechanism = RetryMechanism(self.retry_config)
    
    async def execute(self, coro: Callable, *args, **kwargs) -> CallResult:
        """Execute coroutine with robust error handling."""
        for attempt in range(1, self.retry_config.max_attempts + 1):
            try:
                # Check circuit breaker
                if self.circuit_breaker.state == CircuitState.OPEN:
                    if not self.circuit_breaker._should_attempt_reset():
                        raise Exception("Circuit breaker is OPEN")
                
                # Execute with timeout
                if self.retry_config.timeout:
                    result = await asyncio.wait_for(
                        coro(*args, **kwargs),
                        timeout=self.retry_config.timeout
                    )
                else:
                    result = await coro(*args, **kwargs)
                
                # Record success
                with self.circuit_breaker.lock:
                    self.circuit_breaker._record_success()
                
                return CallResult(success=True, result=result, attempts=attempt)
            
            except Exception as e:
                # Record failure
                with self.circuit_breaker.lock:
                    self.circuit_breaker._record_failure()
                
                if attempt == self.retry_config.max_attempts:
                    return CallResult(success=False, exception=e, attempts=attempt)
                
                # Calculate delay and wait
                delay = self.retry_mechanism._calculate_delay(attempt)
                await asyncio.sleep(delay)
        
        return CallResult(success=False, exception=Exception("Max attempts exceeded"))


# Global instances for common use cases
default_circuit_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30.0,
    half_open_max_calls=3
)

default_retry_config = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=30.0,
    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
    jitter=True
)

# Default executor for general use
default_executor = RobustExecutor(
    circuit_config=default_circuit_config,
    retry_config=default_retry_config
)