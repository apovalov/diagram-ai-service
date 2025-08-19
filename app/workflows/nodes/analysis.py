"""Analysis node for diagram workflow."""

from __future__ import annotations

import time
from typing import Any, Dict

from app.agents.diagram_agent import DiagramAgent
from app.agents.langchain_diagram_agent import LangChainDiagramAgent
from app.core.config import settings
from app.core.logging import get_logger
from app.workflows.nodes.utils import handle_node_error
from app.workflows.state import DiagramWorkflowState

__all__ = ["analyze_node"]

logger = get_logger(__name__)


async def analyze_node(state: DiagramWorkflowState) -> Dict[str, Any]:
    """Analyze natural language description to create diagram structure.

    This node uses existing LangChain or original agent logic to generate
    structured diagram analysis from the user's description.

    Args:
        state: Current workflow state

    Returns:
        Dict containing state updates with analysis result
    """
    start_time = time.monotonic()
    description = state["description"]

    try:
        logger.info(
            "Starting analysis node (request_id=%s, desc_len=%d)",
            state["request_id"],
            len(description),
        )

        # Use appropriate agent based on configuration
        if getattr(settings, "use_langchain", False):
            agent = LangChainDiagramAgent()
            logger.debug("Using LangChain agent for analysis")
        else:
            agent = DiagramAgent()
            logger.debug("Using original agent for analysis")

        # Generate analysis using existing agent logic
        analysis = await agent.generate_analysis(description)

        # Determine analysis method for metadata
        analysis_method = "mock" if settings.mock_llm else "llm"
        if hasattr(analysis, "title") and analysis.title.lower().startswith(
            "heuristic"
        ):
            analysis_method = "heuristic"

        elapsed = time.monotonic() - start_time

        logger.info(
            "Analysis completed (method=%s, nodes=%d, connections=%d, elapsed=%.3fs)",
            analysis_method,
            len(analysis.nodes) if analysis.nodes else 0,
            len(analysis.connections) if analysis.connections else 0,
            elapsed,
        )

        # Update state with successful analysis
        timing = dict(state.get("timing", {}))
        timing["analysis_s"] = elapsed

        return {
            "analysis": analysis,
            "analysis_method": analysis_method,
            "timing": timing,
            "current_step": "analysis_complete",
        }

    except Exception as e:
        elapsed = time.monotonic() - start_time

        logger.error(
            "Analysis node failed after %.3fs: %s",
            elapsed,
            e,
            exc_info=True,
        )

        # Update timing even on failure
        timing = dict(state.get("timing", {}))
        timing["analysis_s"] = elapsed

        # Handle error with retry logic
        error_updates = handle_node_error(state, e, "analysis")
        error_updates["timing"] = timing

        return error_updates
