from __future__ import annotations

from types import SimpleNamespace

from app.core.config import settings

__all__ = ["client"]


async def _blocked_generate_content(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("LLM is mocked (MOCK_LLM=true); no external calls allowed")


async def _blocked_chat_completions_create(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("LLM is mocked (MOCK_LLM=true); no external calls allowed")


async def _blocked_responses_create(*args, **kwargs):  # pragma: no cover
    raise RuntimeError("LLM is mocked (MOCK_LLM=true); no external calls allowed")


if settings.mock_llm:
    client = SimpleNamespace(
        # Gemini compatibility
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_blocked_generate_content)
        ),
        # OpenAI compatibility
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_blocked_chat_completions_create)
        ),
        responses=SimpleNamespace(create=_blocked_responses_create),
    )
elif settings.llm_provider == "openai":
    from openai import AsyncOpenAI

    try:
        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key or "dummy-key-for-testing",
            timeout=settings.llm_timeout,
        )
        client = SimpleNamespace(
            # OpenAI methods
            chat=openai_client.chat,
            responses=openai_client.responses,
            # Gemini compatibility (for rollback)
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=_blocked_generate_content)
            ),
        )
    except Exception:
        # Fallback to blocked methods if client initialization fails
        client = SimpleNamespace(
            # OpenAI compatibility (blocked)
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=_blocked_chat_completions_create)
            ),
            responses=SimpleNamespace(create=_blocked_responses_create),
            # Gemini compatibility (blocked)
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=_blocked_generate_content)
            ),
        )
elif settings.llm_provider == "gemini":
    from google import genai

    if settings.use_vertex_ai and settings.google_cloud_project:
        gemini_client = genai.Client(
            vertexai=True,
            project=settings.google_cloud_project,
            location=settings.google_cloud_location,
        )
    else:
        gemini_client = genai.Client(api_key=settings.gemini_api_key)

    client = SimpleNamespace(
        # Gemini methods
        aio=gemini_client.aio,
        # OpenAI compatibility (blocked when using Gemini)
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=_blocked_chat_completions_create)
        ),
        responses=SimpleNamespace(create=_blocked_responses_create),
    )
else:
    raise ValueError(
        f"Invalid LLM provider: {settings.llm_provider}. Must be 'openai' or 'gemini'"
    )
