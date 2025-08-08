from __future__ import annotations

from types import SimpleNamespace

from google import genai

from app.core.config import settings

__all__ = ["client"]


if settings.mock_llm:

    async def _blocked_generate_content(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("LLM is mocked (MOCK_LLM=true); no external calls allowed")

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_blocked_generate_content)
        )
    )
elif settings.use_vertex_ai and settings.google_cloud_project:
    client = genai.Client(
        vertexai=True,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
    )
else:
    client = genai.Client(api_key=settings.gemini_api_key)
