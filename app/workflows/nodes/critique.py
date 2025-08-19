"""Critique node for diagram workflow."""

from __future__ import annotations

import base64
import time
from typing import Any, Dict

from app.agents.diagram_agent import DiagramAgent
from app.agents.langchain_diagram_agent import LangChainDiagramAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.core.schemas import DiagramCritique
from app.workflows.nodes.utils import handle_node_error
from app.workflows.state import DiagramWorkflowState

__all__ = ["critique_node"]

logger = get_logger(__name__)


async def critique_node(state: DiagramWorkflowState) -> Dict[str, Any]:
    """Critique rendered diagram using vision analysis.

    This node uses existing agent logic to analyze the rendered diagram
    and provide feedback for potential improvements.

    Args:
        state: Current workflow state containing rendered image

    Returns:
        Dict containing state updates with critique result
    """
    start_time = time.monotonic()
    analysis = state.get("analysis")
    description = state["description"]
    image_before = state.get("image_before")

    if not analysis:
        error = ValueError("No analysis available for critique")
        return handle_node_error(state, error, "critique")

    if not image_before:
        error = ValueError("No rendered image available for critique")
        return handle_node_error(state, error, "critique")

    # Skip critique if disabled
    if not state.get("critique_enabled", True):
        logger.info("Critique disabled, skipping critique node")
        return {
            "critique": DiagramCritique(done=True, critique="Critique disabled"),
            "current_step": "critique_complete",
        }

    # Skip critique in mock mode
    if settings.mock_llm:
        logger.info("Mock mode enabled, using mock critique")
        return {
            "critique": DiagramCritique(done=True, critique=None),
            "current_step": "critique_complete",
        }

    try:
        logger.info(
            "Starting critique node (request_id=%s, image_size=%d)",
            state["request_id"],
            len(image_before),
        )

        # Decode image for critique
        image_bytes = base64.b64decode(image_before)

        # Use appropriate agent based on configuration
        if getattr(settings, "use_langchain", False):
            agent = LangChainDiagramAgent()
            logger.debug("Using LangChain agent for critique")
        else:
            agent = DiagramAgent()
            logger.debug("Using original agent for critique")

        # Generate critique using existing agent logic
        critique = await agent.critique_analysis(
            description=description,
            analysis=analysis,
            image_bytes=image_bytes,
        )

        elapsed = time.monotonic() - start_time

        logger.info(
            "Critique completed (done=%s, has_feedback=%s, elapsed=%.3fs)",
            critique.done if critique else None,
            bool(critique.critique) if critique else False,
            elapsed,
        )

        # Update state with successful critique
        timing = dict(state.get("timing", {}))
        timing["critique_s"] = elapsed

        return {
            "critique": critique,
            "timing": timing,
            "current_step": "critique_complete",
        }

    except Exception as e:
        elapsed = time.monotonic() - start_time

        logger.error(
            "Critique node failed after %.3fs: %s",
            elapsed,
            e,
            exc_info=True,
        )

        # Update timing even on failure
        timing = dict(state.get("timing", {}))
        timing["critique_s"] = elapsed

        # Handle error with retry logic
        error_updates = handle_node_error(state, e, "critique")
        error_updates["timing"] = timing

        return error_updates
