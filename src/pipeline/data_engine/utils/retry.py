"""Retry decorator for handling transient API/network errors."""

import time
import functools
import logging
from typing import Callable, Any, Tuple, Type

logger = logging.getLogger("data_engine.retry")

def retry(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Multiplier applied to delay after each failure.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        A decorated function that will retry on specified exceptions.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = 1.0
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} attempts: {e}"
                        )
                        raise
                    
                    logger.warning(
                        f"Attempt {attempt}/{max_retries} for {func.__name__} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            if last_exception:
                raise last_exception
        return wrapper
    return decorator
