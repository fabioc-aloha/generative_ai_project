"""
Rate Limiter Module

This module provides rate limiting functionality to control API request rates
and prevent hitting API limits. It supports multiple rate limiting strategies
and can be used as decorators or context managers.

Author: Brij Kishore Pandey
"""

import time
import threading
from typing import Dict, Optional, Callable, Any
from functools import wraps
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    A thread-safe rate limiter that controls the rate of operations.
    
    This rate limiter uses a token bucket algorithm to ensure that operations
    don't exceed specified rate limits over different time windows.
    
    Example:
        >>> limiter = RateLimiter(calls_per_minute=60, calls_per_hour=1000)
        >>> limiter.acquire()  # Blocks until request can be made
        >>> # Make API call
        >>> limiter.release()
    """
    
    def __init__(
        self,
        calls_per_second: Optional[float] = None,
        calls_per_minute: Optional[int] = None,
        calls_per_hour: Optional[int] = None,
        calls_per_day: Optional[int] = None,
        burst_size: Optional[int] = None
    ):
        """
        Initialize the rate limiter.
        
        Args:
            calls_per_second (Optional[float]): Maximum calls per second
            calls_per_minute (Optional[int]): Maximum calls per minute
            calls_per_hour (Optional[int]): Maximum calls per hour
            calls_per_day (Optional[int]): Maximum calls per day
            burst_size (Optional[int]): Maximum burst size (default: calls_per_minute or 10)
        """
        self.limits = {}
        
        # Set up rate limits
        if calls_per_second is not None:
            self.limits['second'] = {'limit': calls_per_second, 'window': 1.0}
        if calls_per_minute is not None:
            self.limits['minute'] = {'limit': calls_per_minute, 'window': 60.0}
        if calls_per_hour is not None:
            self.limits['hour'] = {'limit': calls_per_hour, 'window': 3600.0}
        if calls_per_day is not None:
            self.limits['day'] = {'limit': calls_per_day, 'window': 86400.0}
        
        # Default to 60 calls per minute if no limits specified
        if not self.limits:
            self.limits['minute'] = {'limit': 60, 'window': 60.0}
        
        # Set burst size
        self.burst_size = burst_size or calls_per_minute or 10
        
        # Initialize tracking structures
        self.call_history: Dict[str, deque] = defaultdict(deque)
        self.tokens = self.burst_size
        self.last_refill = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _refill_tokens(self) -> None:
        """Refill tokens based on the time elapsed."""
        current_time = time.time()
        time_passed = current_time - self.last_refill
        
        # Calculate refill rate based on the most restrictive per-second limit
        refill_rate = min(
            limit_info['limit'] / limit_info['window']
            for limit_info in self.limits.values()
        )
        
        tokens_to_add = time_passed * refill_rate
        self.tokens = min(self.burst_size, self.tokens + tokens_to_add)
        self.last_refill = current_time
    
    def _cleanup_old_calls(self) -> None:
        """Remove old calls from history that are outside rate limit windows."""
        current_time = time.time()
        
        for period, limit_info in self.limits.items():
            window = limit_info['window']
            cutoff_time = current_time - window
            
            # Remove calls older than the window
            history = self.call_history[period]
            while history and history[0] < cutoff_time:
                history.popleft()
    
    def _can_make_call(self) -> tuple[bool, Optional[float]]:
        """
        Check if a call can be made now.
        
        Returns:
            tuple[bool, Optional[float]]: (can_make_call, wait_time_if_not)
        """
        current_time = time.time()
        
        # Check token bucket
        if self.tokens < 1:
            # Calculate wait time for next token
            refill_rate = min(
                limit_info['limit'] / limit_info['window']
                for limit_info in self.limits.values()
            )
            wait_time = (1 - self.tokens) / refill_rate
            return False, wait_time
        
        # Check rate limits
        for period, limit_info in self.limits.items():
            history = self.call_history[period]
            limit = limit_info['limit']
            window = limit_info['window']
            
            if len(history) >= limit:
                # Calculate wait time until oldest call expires
                oldest_call = history[0]
                wait_time = oldest_call + window - current_time
                if wait_time > 0:
                    return False, wait_time
        
        return True, None
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a call.
        
        Args:
            blocking (bool): Whether to block until permission is granted
            timeout (Optional[float]): Maximum time to wait (only if blocking=True)
            
        Returns:
            bool: True if permission granted, False otherwise
        """
        start_time = time.time()
        
        with self.lock:
            while True:
                self._refill_tokens()
                self._cleanup_old_calls()
                
                can_call, wait_time = self._can_make_call()
                
                if can_call:
                    # Grant permission and record the call
                    self.tokens -= 1
                    current_time = time.time()
                    
                    for period in self.limits:
                        self.call_history[period].append(current_time)
                    
                    self.logger.debug(f"Call permitted. Tokens remaining: {self.tokens}")
                    return True
                
                if not blocking:
                    self.logger.debug("Call denied (non-blocking)")
                    return False
                
                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        self.logger.debug("Call denied (timeout)")
                        return False
                    
                    # Adjust wait time to not exceed timeout
                    wait_time = min(wait_time or 0, timeout - elapsed)
                
                # Wait before retrying
                if wait_time and wait_time > 0:
                    self.logger.debug(f"Waiting {wait_time:.2f}s before retry")
                    time.sleep(min(wait_time, 0.1))  # Cap sleep at 100ms for responsiveness
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
    
    def release(self) -> None:
        """
        Release resources (no-op for this implementation).
        
        This method is provided for compatibility with context manager patterns
        but doesn't perform any actual work in this token bucket implementation.
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current rate limiter status.
        
        Returns:
            Dict[str, Any]: Status information including available tokens and call counts
        """
        with self.lock:
            self._refill_tokens()
            self._cleanup_old_calls()
            
            status = {
                "tokens_available": self.tokens,
                "burst_size": self.burst_size,
                "limits": self.limits,
                "call_counts": {}
            }
            
            for period, history in self.call_history.items():
                status["call_counts"][period] = len(history)
            
            return status
    
    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self.lock:
            self.call_history.clear()
            self.tokens = self.burst_size
            self.last_refill = time.time()
            self.logger.info("Rate limiter reset")
    
    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


class PerKeyRateLimiter:
    """
    A rate limiter that maintains separate limits for different keys.
    
    This is useful for API clients that need different rate limits per
    API key, user, or other identifier.
    
    Example:
        >>> limiter = PerKeyRateLimiter(calls_per_minute=60)
        >>> limiter.acquire("user1")  # Separate limit for user1
        >>> limiter.acquire("user2")  # Separate limit for user2
    """
    
    def __init__(
        self,
        calls_per_second: Optional[float] = None,
        calls_per_minute: Optional[int] = None,
        calls_per_hour: Optional[int] = None,
        calls_per_day: Optional[int] = None,
        burst_size: Optional[int] = None,
        cleanup_interval: float = 300.0  # 5 minutes
    ):
        """
        Initialize the per-key rate limiter.
        
        Args:
            calls_per_second (Optional[float]): Maximum calls per second per key
            calls_per_minute (Optional[int]): Maximum calls per minute per key
            calls_per_hour (Optional[int]): Maximum calls per hour per key
            calls_per_day (Optional[int]): Maximum calls per day per key
            burst_size (Optional[int]): Maximum burst size per key
            cleanup_interval (float): Interval to cleanup unused limiters (seconds)
        """
        self.limiter_args = {
            'calls_per_second': calls_per_second,
            'calls_per_minute': calls_per_minute,
            'calls_per_hour': calls_per_hour,
            'calls_per_day': calls_per_day,
            'burst_size': burst_size
        }
        
        self.limiters: Dict[str, RateLimiter] = {}
        self.last_access: Dict[str, float] = {}
        self.cleanup_interval = cleanup_interval
        self.last_cleanup = time.time()
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_limiter(self, key: str) -> RateLimiter:
        """Get or create a rate limiter for the given key."""
        current_time = time.time()
        
        if key not in self.limiters:
            self.limiters[key] = RateLimiter(**self.limiter_args)
            self.logger.debug(f"Created new rate limiter for key: {key}")
        
        self.last_access[key] = current_time
        return self.limiters[key]
    
    def _cleanup_unused_limiters(self) -> None:
        """Remove limiters that haven't been used recently."""
        current_time = time.time()
        
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - self.cleanup_interval
        keys_to_remove = [
            key for key, last_access in self.last_access.items()
            if last_access < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.limiters[key]
            del self.last_access[key]
            self.logger.debug(f"Cleaned up unused limiter for key: {key}")
        
        self.last_cleanup = current_time
        
        if keys_to_remove:
            self.logger.info(f"Cleaned up {len(keys_to_remove)} unused limiters")
    
    def acquire(
        self,
        key: str,
        blocking: bool = True,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Acquire permission to make a call for the given key.
        
        Args:
            key (str): Identifier for the rate limit (e.g., user ID, API key)
            blocking (bool): Whether to block until permission is granted
            timeout (Optional[float]): Maximum time to wait
            
        Returns:
            bool: True if permission granted, False otherwise
        """
        with self.lock:
            self._cleanup_unused_limiters()
            limiter = self._get_limiter(key)
        
        return limiter.acquire(blocking, timeout)
    
    def release(self, key: str) -> None:
        """
        Release resources for the given key.
        
        Args:
            key (str): Identifier for the rate limit
        """
        with self.lock:
            if key in self.limiters:
                self.limiters[key].release()
    
    def get_status(self, key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get status for a specific key or all keys.
        
        Args:
            key (Optional[str]): Specific key to get status for (None for all)
            
        Returns:
            Dict[str, Any]: Status information
        """
        with self.lock:
            self._cleanup_unused_limiters()
            
            if key is not None:
                if key in self.limiters:
                    return {key: self.limiters[key].get_status()}
                else:
                    return {key: "No limiter found"}
            
            return {
                key: limiter.get_status()
                for key, limiter in self.limiters.items()
            }
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset rate limiter(s).
        
        Args:
            key (Optional[str]): Specific key to reset (None for all)
        """
        with self.lock:
            if key is not None:
                if key in self.limiters:
                    self.limiters[key].reset()
            else:
                for limiter in self.limiters.values():
                    limiter.reset()
                self.logger.info("Reset all rate limiters")


def rate_limit(
    calls_per_second: Optional[float] = None,
    calls_per_minute: Optional[int] = None,
    calls_per_hour: Optional[int] = None,
    calls_per_day: Optional[int] = None,
    burst_size: Optional[int] = None,
    per_key: bool = False,
    key_extractor: Optional[Callable] = None
):
    """
    Decorator to add rate limiting to a function.
    
    Args:
        calls_per_second (Optional[float]): Maximum calls per second
        calls_per_minute (Optional[int]): Maximum calls per minute
        calls_per_hour (Optional[int]): Maximum calls per hour
        calls_per_day (Optional[int]): Maximum calls per day
        burst_size (Optional[int]): Maximum burst size
        per_key (bool): Whether to use per-key rate limiting
        key_extractor (Optional[Callable]): Function to extract key from args/kwargs
        
    Returns:
        Callable: Decorated function with rate limiting
        
    Example:
        >>> @rate_limit(calls_per_minute=60)
        ... def api_call():
        ...     return "API response"
        
        >>> @rate_limit(calls_per_minute=100, per_key=True, 
        ...              key_extractor=lambda *args, **kwargs: kwargs.get('user_id'))
        ... def user_api_call(user_id):
        ...     return f"API response for {user_id}"
    """
    # Create the appropriate limiter
    if per_key:
        limiter = PerKeyRateLimiter(
            calls_per_second=calls_per_second,
            calls_per_minute=calls_per_minute,
            calls_per_hour=calls_per_hour,
            calls_per_day=calls_per_day,
            burst_size=burst_size
        )
    else:
        limiter = RateLimiter(
            calls_per_second=calls_per_second,
            calls_per_minute=calls_per_minute,
            calls_per_hour=calls_per_hour,
            calls_per_day=calls_per_day,
            burst_size=burst_size
        )
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract key if using per-key limiting
            if per_key:
                if key_extractor:
                    key = key_extractor(*args, **kwargs)
                else:
                    # Default: use first argument as key
                    key = str(args[0]) if args else "default"
                
                # Acquire permission for the key
                if not limiter.acquire(key):
                    raise RuntimeError(f"Rate limit exceeded for key: {key}")
                
                try:
                    return func(*args, **kwargs)
                finally:
                    limiter.release(key)
            else:
                # Global rate limiting
                if not limiter.acquire():
                    raise RuntimeError("Rate limit exceeded")
                
                try:
                    return func(*args, **kwargs)
                finally:
                    limiter.release()
        
        # Attach limiter to function for inspection
        wrapper._rate_limiter = limiter
        return wrapper
    
    return decorator


class AdaptiveRateLimiter:
    """
    An adaptive rate limiter that adjusts limits based on success/failure rates.
    
    This limiter automatically reduces rate when encountering errors and
    increases it when operations are successful, helping to avoid API limits
    while maximizing throughput.
    
    Example:
        >>> limiter = AdaptiveRateLimiter(initial_rate=60)
        >>> limiter.acquire()
        >>> # Make API call
        >>> limiter.record_success()  # or limiter.record_failure()
    """
    
    def __init__(
        self,
        initial_rate: int = 60,
        min_rate: int = 1,
        max_rate: int = 1000,
        increase_factor: float = 1.1,
        decrease_factor: float = 0.8,
        window_size: int = 10
    ):
        """
        Initialize the adaptive rate limiter.
        
        Args:
            initial_rate (int): Initial calls per minute
            min_rate (int): Minimum calls per minute
            max_rate (int): Maximum calls per minute
            increase_factor (float): Factor to increase rate on success
            decrease_factor (float): Factor to decrease rate on failure
            window_size (int): Size of sliding window for success rate calculation
        """
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.window_size = window_size
        
        self.success_history = deque(maxlen=window_size)
        self.base_limiter = RateLimiter(calls_per_minute=self.current_rate)
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire permission with current rate limit."""
        return self.base_limiter.acquire(blocking, timeout)
    
    def release(self) -> None:
        """Release resources."""
        self.base_limiter.release()
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self.lock:
            self.success_history.append(True)
            self._adjust_rate()
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        with self.lock:
            self.success_history.append(False)
            self._adjust_rate()
    
    def _adjust_rate(self) -> None:
        """Adjust rate based on recent success/failure history."""
        if len(self.success_history) < self.window_size:
            return
        
        success_rate = sum(self.success_history) / len(self.success_history)
        
        if success_rate >= 0.9:  # 90% success rate
            # Increase rate
            new_rate = min(self.max_rate, self.current_rate * self.increase_factor)
        elif success_rate <= 0.7:  # 70% success rate
            # Decrease rate
            new_rate = max(self.min_rate, self.current_rate * self.decrease_factor)
        else:
            # Keep current rate
            new_rate = self.current_rate
        
        if new_rate != self.current_rate:
            self.current_rate = int(new_rate)
            self.base_limiter = RateLimiter(calls_per_minute=self.current_rate)
            self.logger.info(
                f"Adjusted rate to {self.current_rate} calls/min "
                f"(success rate: {success_rate:.2%})"
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status including adaptive metrics."""
        base_status = self.base_limiter.get_status()
        
        success_rate = None
        if self.success_history:
            success_rate = sum(self.success_history) / len(self.success_history)
        
        base_status.update({
            "adaptive_rate": self.current_rate,
            "min_rate": self.min_rate,
            "max_rate": self.max_rate,
            "success_rate": success_rate,
            "history_size": len(self.success_history)
        })
        
        return base_status


# Global rate limiter instances for common use cases
DEFAULT_API_LIMITER = RateLimiter(calls_per_minute=60)
AGGRESSIVE_API_LIMITER = RateLimiter(calls_per_minute=100, calls_per_hour=1000)
CONSERVATIVE_API_LIMITER = RateLimiter(calls_per_minute=30, calls_per_hour=500)