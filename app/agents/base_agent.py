"""Base agent with unified error handling and common functionality."""

import logging
from abc import ABC, abstractmethod
from typing import Optional

from ..core.config import Settings
from ..core.error_handler import ErrorHandler
from ..core.exceptions import ValidationError
from ..core.schemas import DiagramAnalysis, DiagramCritique


class BaseAgent(ABC):
    """Base agent with unified error handling and common functionality."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.error_handler = ErrorHandler(
            max_retries=getattr(settings, "max_retries", 3),
            backoff_factor=getattr(settings, "backoff_factor", 1.5),
            max_backoff=getattr(settings, "max_backoff", 30.0),
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _validate_description(self, description: str) -> str:
        """Validate and sanitize description input."""
        if not description:
            raise ValidationError("Description cannot be empty", operation="validation")

        description = description.strip()
        if len(description) < 3:
            raise ValidationError(
                "Description must be at least 3 characters", operation="validation"
            )

        if len(description) > 5000:
            self.logger.warning(
                f"Description truncated from {len(description)} to 5000 characters"
            )
            description = description[:5000]

        return description

    @abstractmethod
    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        """Generate diagram analysis from description."""
        pass

    @abstractmethod
    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        """Critique diagram analysis."""
        pass

    @abstractmethod
    async def adjust_analysis(
        self, description: str, analysis: DiagramAnalysis, critique: str
    ) -> DiagramAnalysis:
        """Adjust analysis based on critique."""
        pass