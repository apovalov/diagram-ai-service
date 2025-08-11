from __future__ import annotations

from app.core.config import settings
from app.core.constants import (
    DIAGRAM_PHRASES,
    GREETING_PHRASES,
    HELP_PHRASES,
    IntentType,
)
from app.core.llm import client
from app.core.prompts import intent_prompt
from app.core.schemas import IntentResult

__all__ = ["AssistantAgent"]


class AssistantAgent:
    """Agent for handling assistant conversations and intent detection."""

    def __init__(self) -> None:
        pass

    async def get_intent(
        self, message: str, context: dict | None = None
    ) -> IntentResult:
        """Get intent from user message using LLM."""
        # If running with mock LLM, return a simple deterministic intent result
        if settings.mock_llm:
            lower = message.lower()
            if any(word in lower for word in GREETING_PHRASES):
                return IntentResult(intent=IntentType.GREETING.value, confidence=1.0)
            if any(word in lower for word in DIAGRAM_PHRASES):
                desc = message
                return IntentResult(
                    intent=IntentType.GENERATE_DIAGRAM.value, description=desc, confidence=0.9
                )
            return IntentResult(intent=IntentType.CLARIFICATION.value, confidence=0.6)

        # Include context in prompt if available
        context_str = ""
        if context and "messages" in context and len(context["messages"]) > 1:
            recent_messages = context["messages"][-3:]  # Last 3 messages for context
            context_str = f"\nConversation history: {recent_messages}"

        prompt = intent_prompt(message + context_str)
        try:
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": IntentResult,
                },
            )
            if getattr(response, "parsed", None):
                return response.parsed
            raise ValueError("Failed to parse LLM response as JSON.")
        except Exception:
            # Runtime fallback: heuristic intent
            return self._create_fallback_intent(message)

    def _create_fallback_intent(self, message: str) -> IntentResult:
        """Create a basic fallback intent when API is unavailable."""
        message_lower = message.lower()
        if any(
            word in message_lower
            for word in DIAGRAM_PHRASES
        ):
            return IntentResult(
                intent=IntentType.GENERATE_DIAGRAM.value, confidence=0.5, description=message
            )
        if any(word in message_lower for word in HELP_PHRASES):
            return IntentResult(intent=IntentType.HELP.value, confidence=0.5)
        return IntentResult(intent=IntentType.GENERAL.value, confidence=0.3)
