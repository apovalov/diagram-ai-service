from __future__ import annotations

import asyncio
import random

from app.chains import AdjustmentChain, AnalysisChain, CritiqueChain
from app.core.config import settings
from app.core.langchain_config import get_llm
from app.core.logging import get_logger
from app.core.schemas import DiagramAnalysis, DiagramCritique
from app.agents.base_agent import BaseAgent
from app.core.error_handler import handle_errors
from app.core.exceptions import AnalysisError, CritiqueError, ValidationError

__all__ = ["LangChainDiagramAgent"]


class LangChainDiagramAgent(BaseAgent):
    """
    Enhanced LangChain-based agent with unified error handling.

    This is a drop-in replacement for the original DiagramAgent that uses LangChain
    chains instead of direct LLM API calls, now with enhanced error handling.
    """

    def __init__(self, settings_override=None) -> None:
        # Support both old and new initialization patterns for backward compatibility
        agent_settings = settings_override or settings
        super().__init__(agent_settings)
        # Keep original logger for existing code compatibility
        self.logger = get_logger(__name__)
        self.llm = get_llm()

        # Initialize chains
        self.analysis_chain = AnalysisChain(self.llm)
        self.critique_chain = CritiqueChain(self.llm)
        self.adjustment_chain = AdjustmentChain(self.llm)

    async def _retry_with_backoff(
        self, operation_name: str, max_attempts: int, operation_func
    ):
        """Retry an operation with exponential backoff and jitter."""
        for attempt in range(max_attempts):
            try:
                result = await operation_func()
                return result
            except Exception as e:
                is_last_attempt = attempt == max_attempts - 1
                if is_last_attempt:
                    self.logger.error(
                        f"{operation_name} failed after {max_attempts} attempts: {e}"
                    )
                    raise
                # Calculate backoff with exponential growth and jitter
                backoff_time = min(
                    settings.retry_backoff_base * (2**attempt),
                    settings.retry_backoff_max,
                ) + random.uniform(0, settings.retry_jitter)
                await asyncio.sleep(backoff_time)

    @handle_errors("analysis", fallback="_heuristic_analysis_fallback")
    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        """
        Generate diagram component analysis from description using LangChain.

        Args:
            description: Natural language description of the desired diagram

        Returns:
            DiagramAnalysis: Structured analysis with nodes, clusters, and connections
        """
        # Validate input using base class method
        description = self._validate_description(description)

        async def _perform_analysis():
            return await self.analysis_chain.ainvoke(description)

        try:
            return await self._retry_with_backoff(
                "analysis", settings.analysis_max_attempts, _perform_analysis
            )
        except Exception as e:
            self.logger.exception(
                "LangChain analysis failed after retries; using chain fallback: %s", e
            )
            # The chain already has fallback handling, but if it still fails,
            # we can create a minimal analysis here
            return await self.analysis_chain._create_heuristic_analysis(description)

    @handle_errors("critique", fallback="_create_default_critique")
    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        """
        Critique the current diagram analysis using vision capabilities.

        Args:
            description: Original user description
            analysis: Current diagram analysis
            image_bytes: PNG image bytes of the rendered diagram

        Returns:
            DiagramCritique: Critique result with done flag and feedback
        """
        # Validate inputs
        description = self._validate_description(description)
        if not analysis:
            raise ValidationError("Analysis is required for critique")
        if not image_bytes:
            raise ValidationError("Image bytes are required for critique")
        try:
            return await self.critique_chain.ainvoke(description, analysis, image_bytes)
        except Exception as e:
            self.logger.exception(
                "LangChain critique failed; using heuristic fallback: %s", e
            )
            return self.critique_chain._create_heuristic_critique(analysis)

    @handle_errors("adjustment", fallback="_return_original_analysis")
    async def adjust_analysis(
        self, description: str, critique: str, analysis: DiagramAnalysis
    ) -> DiagramAnalysis:
        """
        Adjust the diagram analysis based on critique feedback.

        Args:
            description: Original user description
            critique: Critique feedback to address
            analysis: Current analysis to improve

        Returns:
            DiagramAnalysis: Adjusted analysis addressing the critique
        """
        # Validate inputs
        description = self._validate_description(description)
        if not analysis:
            raise ValidationError("Analysis is required for adjustment")
        if not critique:
            self.logger.info("No critique provided, returning original analysis")
            return analysis

        async def _perform_adjustment():
            return await self.adjustment_chain.ainvoke(description, critique, analysis)

        try:
            return await self._retry_with_backoff(
                "adjustment", settings.adjust_max_attempts, _perform_adjustment
            )
        except Exception as e:
            self.logger.exception(
                "LangChain adjustment failed after retries; returning original: %s", e
            )
            return analysis

    # Compatibility methods to match original DiagramAgent interface

    def _heuristic_analysis(self, description: str) -> DiagramAnalysis:
        """
        Compatibility method for heuristic analysis fallback.

        This method maintains the same interface as the original agent
        but delegates to the chain's heuristic implementation.
        """
        return self.analysis_chain._create_heuristic_analysis(description)

    # === ADDITIONAL FALLBACK METHODS FOR ERROR HANDLING ===

    def _heuristic_analysis_fallback(self, description: str = None) -> DiagramAnalysis:
        """Fallback heuristic analysis for error handler."""
        self.logger.info("Using LangChain heuristic analysis fallback")
        if not description:
            description = "fallback analysis"
        return self.analysis_chain._create_heuristic_analysis(description)

    def _create_default_critique(self) -> DiagramCritique:
        """Fallback critique."""
        self.logger.info("Using default critique fallback")
        return DiagramCritique(
            done=True, critique="Unable to perform critique, proceeding with diagram"
        )

    def _return_original_analysis(self, *args, **kwargs) -> DiagramAnalysis:
        """Fallback for adjustment - return original analysis."""
        self.logger.info("Using original analysis fallback")
        # Try to extract analysis from arguments
        for arg in args:
            if isinstance(arg, DiagramAnalysis):
                return arg
        # If no analysis found, create a basic fallback
        return self.analysis_chain._create_heuristic_analysis("fallback")
