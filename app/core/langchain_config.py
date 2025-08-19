from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from app.core.config import settings

__all__ = ["get_llm", "MockChatModel"]


class MockChatModel(BaseChatModel):
    """Mock LangChain ChatModel for testing without external API calls."""

    def _generate(self, messages, stop=None, **kwargs):
        """Generate a mock response for non-async calls."""
        raise RuntimeError("LLM is mocked (MOCK_LLM=true); no external calls allowed")

    async def _agenerate(self, messages, stop=None, **kwargs):
        """Generate a mock response for async calls."""
        raise RuntimeError("LLM is mocked (MOCK_LLM=true); no external calls allowed")

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"type": "mock", "mock_llm": True}


def get_llm() -> BaseChatModel:
    """
    Factory function to create appropriate LangChain LLM instance.

    Returns:
        BaseChatModel: Configured LangChain LLM instance

    Raises:
        ValueError: If invalid LLM provider is specified
        ImportError: If required dependencies are not installed
    """
    if settings.mock_llm:
        return MockChatModel()

    if settings.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=settings.openai_api_key or "dummy-key-for-testing",
                model=settings.openai_model,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout,
                max_retries=0,  # We handle retries at the chain level
            )
        except ImportError as e:
            raise ImportError(
                "langchain-openai is required for OpenAI provider. "
                "Install with: pip install langchain-openai"
            ) from e

    elif settings.llm_provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            # Configure Gemini model
            model_config = {
                "model": settings.gemini_model,
                "temperature": settings.llm_temperature,
                "timeout": settings.llm_timeout,
                "max_retries": 0,  # We handle retries at the chain level
            }

            if settings.use_vertex_ai and settings.google_cloud_project:
                # Use Vertex AI configuration
                model_config.update(
                    {
                        "project": settings.google_cloud_project,
                        "location": settings.google_cloud_location,
                    }
                )
            else:
                # Use Gemini Developer API
                model_config["google_api_key"] = settings.gemini_api_key

            return ChatGoogleGenerativeAI(**model_config)

        except ImportError as e:
            raise ImportError(
                "langchain-google-genai is required for Gemini provider. "
                "Install with: pip install langchain-google-genai"
            ) from e

    else:
        raise ValueError(
            f"Invalid LLM provider: {settings.llm_provider}. Must be 'openai' or 'gemini'"
        )


def get_llm_with_fallback() -> BaseChatModel:
    """
    Get LLM with automatic fallback to mock if initialization fails.

    Returns:
        BaseChatModel: Working LLM instance, mock if real one fails
    """
    try:
        return get_llm()
    except Exception:
        # Fallback to mock if real LLM fails to initialize
        return MockChatModel()
