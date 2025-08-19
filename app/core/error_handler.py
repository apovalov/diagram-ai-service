"""Unified error handling with retry logic and fallbacks."""

import asyncio
import logging
import random
from typing import TypeVar, Callable, Any, Optional, Type, Tuple
from functools import wraps

from .exceptions import (
    DiagramError,
    RetryableError,
    FatalError,
    LLMProviderError,
    AnalysisError,
    RenderError,
    CritiqueError,
    ValidationError,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling with retry logic and metrics."""

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        max_backoff: float = 30.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
        self.jitter = jitter

    async def with_retry(
        self,
        operation: Callable[[], T],
        operation_name: str,
        fallback: Optional[Callable[[], T]] = None,
        retry_on: Tuple[Type[Exception], ...] = (RetryableError, LLMProviderError),
        no_retry_on: Tuple[Type[Exception], ...] = (FatalError, ValidationError),
    ) -> T:
        """Execute operation with retry logic and fallback."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                # Execute the operation
                if asyncio.iscoroutinefunction(operation):
                    result = await operation()
                else:
                    result = operation()

                if attempt > 0:
                    logger.info(f"{operation_name} succeeded on attempt {attempt + 1}")
                return result

            except no_retry_on as e:
                # Fatal errors - don't retry
                logger.error(f"{operation_name} failed with fatal error: {e}")
                raise e

            except retry_on as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(
                        f"{operation_name} failed after {self.max_retries + 1} attempts"
                    )
                    break

            except Exception as e:
                # Convert unknown exceptions to DiagramError
                converted_error = self._convert_exception(e, operation_name, attempt)
                last_exception = converted_error
                logger.error(f"{operation_name} failed with unexpected error: {e}")

                # Check if converted error should be retried
                if isinstance(converted_error, retry_on) and attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    await asyncio.sleep(delay)
                    continue
                else:
                    break

        # Try fallback if available
        if fallback:
            try:
                logger.info(f"Attempting fallback for {operation_name}")
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback()
                else:
                    return fallback()
            except Exception as fallback_error:
                logger.error(f"Fallback failed for {operation_name}: {fallback_error}")

        # Re-raise the last exception
        if last_exception:
            raise last_exception
        else:
            raise DiagramError(f"{operation_name} failed with unknown error")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter."""
        delay = min(self.backoff_factor**attempt, self.max_backoff)

        if self.jitter:
            # Add random jitter (±25% of delay)
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

        return max(0.1, delay)  # Minimum delay of 100ms

    def _convert_exception(
        self, e: Exception, operation: str, retry_count: int
    ) -> DiagramError:
        """Convert standard exceptions to our hierarchy."""

        # OpenAI specific errors
        if "openai" in str(type(e).__module__):
            if "rate_limit" in str(e).lower():
                return RetryableError(
                    f"OpenAI rate limit: {e}",
                    operation=operation,
                    retry_count=retry_count,
                )
            elif "timeout" in str(e).lower():
                return RetryableError(
                    f"OpenAI timeout: {e}", operation=operation, retry_count=retry_count
                )
            else:
                return LLMProviderError(
                    str(e), provider="openai", operation=operation, retry_count=retry_count
                )

        # Network/connection errors
        if any(err in str(type(e)).lower() for err in ["connection", "timeout", "network"]):
            return RetryableError(
                f"Network error: {e}", operation=operation, retry_count=retry_count
            )

        # Validation errors
        if isinstance(e, (ValueError, TypeError)) and operation in ["validation", "input"]:
            return ValidationError(
                str(e), operation=operation, retry_count=retry_count
            )

        # Default: make it retryable but with context
        return DiagramError(str(e), operation=operation, retry_count=retry_count)


# Decorator for automatic error handling
def handle_errors(
    operation_name: str,
    fallback: Optional[str] = None,
    retry_on: Tuple[Type[Exception], ...] = (RetryableError, LLMProviderError),
    no_retry_on: Tuple[Type[Exception], ...] = (FatalError, ValidationError),
):
    """Decorator for automatic error handling with retry logic."""

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get error handler from self or create default
            error_handler = getattr(self, "error_handler", ErrorHandler())

            async def operation():
                return await func(self, *args, **kwargs)

            # Create fallback if method exists
            fallback_func = None
            if fallback and hasattr(self, fallback):
                fallback_func = getattr(self, fallback)
                if not asyncio.iscoroutinefunction(fallback_func):
                    # Wrap non-async methods
                    original_fallback = fallback_func

                    async def async_fallback(*fb_args, **fb_kwargs):
                        return original_fallback(*fb_args, **fb_kwargs)

                    fallback_func = async_fallback

            return await error_handler.with_retry(
                operation=operation,
                operation_name=operation_name,
                fallback=fallback_func,
                retry_on=retry_on,
                no_retry_on=no_retry_on,
            )

        return wrapper

    return decorator