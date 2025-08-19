from __future__ import annotations

from app.chains import IntentChain
from app.core.constants import (
    DIAGRAM_PHRASES,
    GREETING_PHRASES,
    HELP_PHRASES,
    IntentType,
)
from app.core.langchain_config import get_llm
from app.core.logging import get_logger
from app.core.schemas import IntentResult

__all__ = ["LangChainAssistantAgent"]


class LangChainAssistantAgent:
    """
    LangChain-based agent for handling assistant conversations and intent detection.

    This is a drop-in replacement for the original AssistantAgent that uses LangChain
    chains instead of direct LLM API calls.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)
        self.llm = get_llm()

        # Initialize intent chain
        self.intent_chain = IntentChain(self.llm)

    async def get_intent(
        self, message: str, context: dict | None = None
    ) -> IntentResult:
        """
        Get intent from user message using LangChain.

        Args:
            message: User message to analyze
            context: Optional conversation context

        Returns:
            IntentResult: Classified intent with confidence score
        """
        try:
            return await self.intent_chain.ainvoke(message, context)
        except Exception as e:
            self.logger.exception(
                "LangChain intent detection failed; using heuristic fallback: %s", e
            )
            return self._create_fallback_intent(message)

    def _create_fallback_intent(self, message: str) -> IntentResult:
        """
        Create a basic fallback intent when LangChain fails.

        This method maintains compatibility with the original agent
        by using the same heuristic logic.
        """
        message_lower = message.lower()
        if any(word in message_lower for word in DIAGRAM_PHRASES):
            return IntentResult(
                intent=IntentType.GENERATE_DIAGRAM.value,
                confidence=0.5,
                description=message,
            )
        if any(word in message_lower for word in GREETING_PHRASES):
            return IntentResult(intent=IntentType.GREETING.value, confidence=0.8)
        if any(word in message_lower for word in HELP_PHRASES):
            return IntentResult(intent=IntentType.CLARIFICATION.value, confidence=0.7)

        # Default to clarification for unknown intents
        return IntentResult(intent=IntentType.CLARIFICATION.value, confidence=0.3)
