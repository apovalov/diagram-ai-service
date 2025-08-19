from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda

from app.core.config import settings
from app.core.constants import DIAGRAM_PHRASES, GREETING_PHRASES, IntentType
from app.core.langchain_prompts import get_intent_prompt
from app.core.logging import get_logger
from app.core.schemas import IntentResult

__all__ = ["IntentChain"]


class IntentChain:
    """LangChain-based intent classification chain."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = get_logger(__name__)
        self.prompt_template, self.parser = get_intent_prompt()

        # Create the main chain
        self.chain = self.prompt_template | self.llm | self.parser

        # Create chain with fallback to heuristic analysis
        self.chain_with_fallback = RunnableWithFallbacks(
            runnable=self.chain, fallbacks=[RunnableLambda(self._heuristic_fallback)]
        )

    async def ainvoke(self, message: str, context: dict | None = None) -> IntentResult:
        """
        Analyze message intent using LangChain.

        Args:
            message: User message to analyze
            context: Optional conversation context

        Returns:
            IntentResult: Classified intent with confidence
        """
        if settings.mock_llm:
            return self._create_mock_intent(message)

        # Include context in message if available
        context_str = ""
        if context and "messages" in context and len(context["messages"]) > 1:
            recent_messages = context["messages"][-3:]  # Last 3 messages for context
            context_str = f"\nConversation history: {recent_messages}"

        full_message = message + context_str

        start_time = time.monotonic()

        try:
            self.logger.info(
                "Requesting intent classification (provider=%s, msg_len=%d)",
                settings.llm_provider,
                len(full_message),
            )

            result = await self.chain_with_fallback.ainvoke({"message": full_message})

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            self.logger.info(
                "Intent classification completed in %d ms (intent=%s, confidence=%s)",
                elapsed_ms,
                result.intent,
                result.confidence,
            )

            return result

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            self.logger.error(
                "Intent classification failed after %d ms, using heuristic fallback: %s",
                elapsed_ms,
                e,
            )
            return self._create_fallback_intent(message)

    def _create_mock_intent(self, message: str) -> IntentResult:
        """Create mock intent for testing."""
        lower = message.lower()
        if any(word in lower for word in GREETING_PHRASES):
            return IntentResult(intent=IntentType.GREETING.value, confidence=1.0)
        if any(word in lower for word in DIAGRAM_PHRASES):
            return IntentResult(
                intent=IntentType.GENERATE_DIAGRAM.value,
                description=message,
                confidence=0.9,
            )
        return IntentResult(intent=IntentType.CLARIFICATION.value, confidence=0.6)

    def _create_fallback_intent(self, message: str) -> IntentResult:
        """Create heuristic fallback intent when LLM fails."""
        message_lower = message.lower()
        if any(word in message_lower for word in DIAGRAM_PHRASES):
            return IntentResult(
                intent=IntentType.GENERATE_DIAGRAM.value,
                confidence=0.5,
                description=message,
            )
        if any(word in message_lower for word in GREETING_PHRASES):
            return IntentResult(intent=IntentType.GREETING.value, confidence=0.8)

        return IntentResult(intent=IntentType.CLARIFICATION.value, confidence=0.3)

    async def _heuristic_fallback(self, inputs: dict[str, Any]) -> IntentResult:
        """Fallback runnable for when the main chain fails."""
        message = inputs.get("message", "")
        return self._create_fallback_intent(message)
