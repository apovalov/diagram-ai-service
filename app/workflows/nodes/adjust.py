"""Adjust node for diagram workflow."""

from __future__ import annotations

import time
from typing import Any, Dict

from app.agents.diagram_agent import DiagramAgent
from app.agents.langchain_diagram_agent import LangChainDiagramAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.workflows.nodes.utils import handle_node_error
from app.workflows.state import DiagramWorkflowState

__all__ = ["adjust_node"]

logger = get_logger(__name__)


async def adjust_node(state: DiagramWorkflowState) -> Dict[str, Any]:
    """Adjust diagram analysis based on critique feedback.

    This node uses existing agent logic to modify the diagram analysis
    based on critique feedback, then triggers re-rendering.

    Args:
        state: Current workflow state containing critique feedback

    Returns:
        Dict containing state updates with adjusted analysis
    """
    start_time = time.monotonic()
    analysis = state.get("analysis")
    critique = state.get("critique")
    description = state["description"]

    if not analysis:
        error = ValueError("No analysis available for adjustment")
        return handle_node_error(state, error, "adjust")

    if not critique or not critique.critique:
        error = ValueError("No critique feedback available for adjustment")
        return handle_node_error(state, error, "adjust")

    try:
        logger.info(
            "Starting adjust node (request_id=%s, critique_len=%d)",
            state["request_id"],
            len(critique.critique),
        )

        # Use appropriate agent based on configuration
        if getattr(settings, "use_langchain", False):
            agent = LangChainDiagramAgent()
            logger.debug("Using LangChain agent for adjustment")
        else:
            agent = DiagramAgent()
            logger.debug("Using original agent for adjustment")

        # Adjust analysis using existing agent logic
        adjusted_analysis = await agent.adjust_analysis(
            description=description,
            analysis=analysis,
            critique=critique.critique,
        )

        elapsed = time.monotonic() - start_time

        logger.info(
            "Adjustment completed (nodes=%d, connections=%d, elapsed=%.3fs)",
            len(adjusted_analysis.nodes) if adjusted_analysis.nodes else 0,
            len(adjusted_analysis.connections) if adjusted_analysis.connections else 0,
            elapsed,
        )

        # Update state with adjusted analysis
        timing = dict(state.get("timing", {}))
        timing["adjust_s"] = elapsed

        return {
            "analysis": adjusted_analysis,
            "timing": timing,
            "current_step": "adjust_complete",
        }

    except Exception as e:
        elapsed = time.monotonic() - start_time

        logger.error(
            "Adjust node failed after %.3fs: %s",
            elapsed,
            e,
            exc_info=True,
        )

        # Update timing even on failure
        timing = dict(state.get("timing", {}))
        timing["adjust_s"] = elapsed

        # Handle error with retry logic
        error_updates = handle_node_error(state, e, "adjust")
        error_updates["timing"] = timing

        return error_updates
