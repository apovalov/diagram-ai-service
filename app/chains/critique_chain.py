from __future__ import annotations

import base64
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda

from app.core.config import settings
from app.core.langchain_prompts import get_diagram_critique_prompt
from app.core.logging import get_logger
from app.core.schemas import DiagramAnalysis, DiagramCritique

__all__ = ["CritiqueChain"]


class CritiqueChain:
    """LangChain-based diagram critique chain with vision support."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = get_logger(__name__)
        self.prompt_template, self.parser = get_diagram_critique_prompt()

        # Create the main chain
        self.chain = self.prompt_template | self.llm | self.parser

        # Create chain with fallback to simple approval
        self.chain_with_fallback = RunnableWithFallbacks(
            runnable=self.chain, fallbacks=[RunnableLambda(self._heuristic_fallback)]
        )

    async def ainvoke(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        """
        Critique diagram using vision analysis.

        Args:
            description: Original user description
            analysis: Current diagram analysis
            image_bytes: PNG image bytes of rendered diagram

        Returns:
            DiagramCritique: Critique result with done flag and feedback
        """
        if settings.mock_llm:
            return DiagramCritique(done=True, critique=None)

        start_time = time.monotonic()

        try:
            self.logger.info(
                "Requesting diagram critique (provider=%s, desc_len=%d, image_size=%d)",
                settings.llm_provider,
                len(description or ""),
                len(image_bytes),
            )

            # For OpenAI, we need to create a special message with image content
            if settings.llm_provider == "openai":
                result = await self._critique_with_openai_vision(
                    description, analysis, image_bytes
                )
            else:
                # For Gemini, use the standard chain (will need special handling later)
                result = await self._critique_with_gemini_vision(
                    description, analysis, image_bytes
                )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            self.logger.info(
                "Diagram critique completed in %d ms (done=%s)",
                elapsed_ms,
                result.done,
            )

            return result

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            self.logger.error(
                "Diagram critique failed after %d ms, using heuristic fallback: %s",
                elapsed_ms,
                e,
            )
            return self._create_heuristic_critique(analysis)

    async def _critique_with_openai_vision(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        """Handle OpenAI vision-based critique."""
        # Convert image to base64 data URL
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_data_url = f"data:image/png;base64,{image_b64}"

        # Format the prompt messages
        messages = self.prompt_template.format_messages(description=description)

        # Add image content to the human message
        human_message = messages[-1]  # Last message should be human message

        # Create new human message with both text and image
        enhanced_human_message = HumanMessage(
            content=[
                {"type": "text", "text": human_message.content},
                {
                    "type": "text",
                    "text": f"\nCurrent analysis JSON:\n{analysis.model_dump_json(indent=2)}",
                },
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ]
        )

        # Replace the last message with our enhanced version
        enhanced_messages = messages[:-1] + [enhanced_human_message]

        # Invoke the LLM with enhanced messages
        response = await self.llm.ainvoke(enhanced_messages)

        # Parse the response
        return self.parser.parse(response.content)

    async def _critique_with_gemini_vision(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        """Handle Gemini vision-based critique."""
        # For now, use the standard chain (Gemini vision handling would need special implementation)
        # This is a placeholder - in real implementation, we'd need to handle Gemini's vision API
        result = await self.chain_with_fallback.ainvoke({"description": description})
        return result

    def _create_heuristic_critique(self, analysis: DiagramAnalysis) -> DiagramCritique:
        """Create heuristic critique when vision analysis fails."""
        # Simple heuristic: check for isolated nodes
        connected_nodes = set()
        for conn in analysis.connections:
            connected_nodes.add(conn.source)
            connected_nodes.add(conn.target)

        isolated_nodes = [
            node.id for node in analysis.nodes if node.id not in connected_nodes
        ]

        if isolated_nodes:
            return DiagramCritique(
                done=False,
                critique=f"Found isolated nodes that may need connections: {', '.join(isolated_nodes)}",
            )

        # If no obvious issues, approve the diagram
        return DiagramCritique(done=True, critique=None)

    async def _heuristic_fallback(self, inputs: dict[str, Any]) -> DiagramCritique:
        """Fallback runnable for when the main chain fails."""
        # For fallback, we'll just approve the diagram
        return DiagramCritique(
            done=True,
            critique="Fallback approval - unable to perform detailed critique",
        )
