"""Unified exception hierarchy for diagram generation."""

import time
from typing import Any, Dict, Optional


class DiagramError(Exception):
    """Base exception for diagram generation with context tracking."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        retry_count: int = 0,
        timestamp: Optional[float] = None,
        **metadata
    ):
        self.operation = operation
        self.retry_count = retry_count
        self.timestamp = timestamp or time.time()
        self.metadata = metadata
        self.error_id = (
            f"{operation}_{int(self.timestamp)}" if operation else str(int(self.timestamp))
        )
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging/monitoring."""
        return {
            "error_id": self.error_id,
            "error_type": self.__class__.__name__,
            "message": str(self),
            "operation": self.operation,
            "retry_count": self.retry_count,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AnalysisError(DiagramError):
    """Analysis phase error."""

    pass


class RenderError(DiagramError):
    """Rendering phase error."""

    pass


class CritiqueError(DiagramError):
    """Critique phase error."""

    pass


class LLMProviderError(DiagramError):
    """LLM provider specific error."""

    def __init__(
        self, message: str, provider: str, status_code: Optional[int] = None, **kwargs
    ):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message, **kwargs)


class RetryableError(DiagramError):
    """Error that should trigger retry mechanism."""

    pass


class FatalError(DiagramError):
    """Error that should not trigger retry and should fail fast."""

    pass


class ValidationError(FatalError):
    """Input validation error - should not retry."""

    pass


class ConfigurationError(FatalError):
    """Configuration error - should not retry."""

    pass