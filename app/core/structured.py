from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from app.core.config import settings
from app.core.llm import client

__all__ = ["ask_structured", "ask_structured_vision"]

T = TypeVar("T", bound=BaseModel)


def _json_schema_from_pydantic(model: type[BaseModel]) -> dict:
    """Extract JSON schema from Pydantic model for OpenAI structured outputs."""
    schema = model.model_json_schema()

    # OpenAI structured outputs require specific modifications for all objects
    def fix_schema_for_openai(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                # Add additionalProperties: false
                obj["additionalProperties"] = False

                # OpenAI requires ALL properties to be in required array
                if "properties" in obj:
                    obj["required"] = list(obj["properties"].keys())

            for value in obj.values():
                fix_schema_for_openai(value)
        elif isinstance(obj, list):
            for item in obj:
                fix_schema_for_openai(item)

    fix_schema_for_openai(schema)
    return schema


async def ask_structured(prompt: str, schema_cls: type[T]) -> T:
    """
    Ask OpenAI for structured output using the Chat Completions API.

    Args:
        prompt: The input prompt
        schema_cls: Pydantic model class for the expected response structure

    Returns:
        Instance of schema_cls with the structured response

    Raises:
        NotImplementedError: If using Gemini provider
        RuntimeError: If LLM is mocked
    """
    if settings.llm_provider == "gemini":
        raise NotImplementedError(
            "Structured outputs helper only supports OpenAI provider"
        )

    schema = _json_schema_from_pydantic(schema_cls)

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        },
        temperature=settings.openai_temperature,
    )

    # Extract JSON from response
    json_text = response.choices[0].message.content
    if not json_text:
        raise ValueError("No content found in OpenAI response")

    # Parse and validate JSON
    try:
        return schema_cls.model_validate_json(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}") from e


async def ask_structured_vision(
    text_prompt: str, image_data_url: str, analysis_json: str, schema_cls: type[T]
) -> T:
    """
    Ask OpenAI for structured output with vision using multimodal input.

    Args:
        text_prompt: The text prompt
        image_data_url: Base64 data URL of the image (data:image/png;base64,...)
        analysis_json: JSON string of analysis data to include
        schema_cls: Pydantic model class for the expected response structure

    Returns:
        Instance of schema_cls with the structured response

    Raises:
        NotImplementedError: If using Gemini provider
        RuntimeError: If LLM is mocked
    """
    if settings.llm_provider == "gemini":
        raise NotImplementedError(
            "Structured vision outputs helper only supports OpenAI provider"
        )

    schema = _json_schema_from_pydantic(schema_cls)

    # Build multimodal input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
                {"type": "text", "text": f"Analysis data: {analysis_json}"},
            ],
        }
    ]

    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema,
                "strict": True,
            },
        },
        temperature=settings.openai_temperature,
    )

    # Extract JSON from response
    json_text = response.choices[0].message.content
    if not json_text:
        raise ValueError("No content found in OpenAI response")

    # Parse and validate JSON
    try:
        return schema_cls.model_validate_json(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}") from e
