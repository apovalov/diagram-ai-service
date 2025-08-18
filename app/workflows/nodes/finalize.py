"""Finalize node for diagram workflow."""

from __future__ import annotations

import time
from typing import Any, Dict

from app.core.logging import get_logger
from app.core.schemas import Timing
from app.workflows.state import DiagramWorkflowState

__all__ = ["finalize_node"]

logger = get_logger(__name__)


async def finalize_node(state: DiagramWorkflowState) -> Dict[str, Any]:
    """Finalize workflow execution and format results.
    
    This node consolidates the final results, formats metadata to match
    the original service contract, and handles cleanup.
    
    Args:
        state: Current workflow state
        
    Returns:
        Dict containing final state updates
    """
    start_time = time.monotonic()
    
    logger.info(
        "Finalizing workflow (request_id=%s, step=%s)",
        state["request_id"],
        state.get("current_step", "unknown"),
    )
    
    # Determine final image
    final_image = state.get("image_after") or state.get("image_before")
    
    # Format timing information
    timing_data = state.get("timing", {})
    total_time = sum(timing_data.values())
    
    # Create timing object matching original format
    timing_obj = Timing(
        analysis_s=timing_data.get("analysis_s", 0.0),
        render_s=timing_data.get("render_s", 0.0) + timing_data.get("adjust_render_s", 0.0),
        total_s=total_time,
    )
    
    # Format metadata to match original service contract
    analysis = state.get("analysis")
    critique = state.get("critique")
    
    final_metadata = {
        "nodes_created": len(analysis.nodes) if analysis and analysis.nodes else 0,
        "clusters_created": len(analysis.clusters) if analysis and analysis.clusters else 0,
        "connections_made": len(analysis.connections) if analysis and analysis.connections else 0,
        "generation_time": total_time,
        "timing": timing_obj.model_dump(),
        "analysis_method": state.get("analysis_method", "unknown"),
        "critique_applied": bool(state.get("image_after")),
        "request_id": state["request_id"],
        **state.get("metadata", {}),
    }
    
    # Add critique information if available
    if critique:
        final_metadata["critique"] = critique.model_dump()
    
    # Add critique attempts if any were made
    if state.get("critique_attempts", 0) > 0:
        final_metadata["critique_attempts"] = state["critique_attempts"]
    
    # Add errors if any occurred
    errors = state.get("errors", [])
    if errors:
        final_metadata["errors"] = errors
        logger.warning("Workflow completed with %d errors: %s", len(errors), errors)
    
    elapsed = time.monotonic() - start_time
    
    logger.info(
        "Workflow finalized (has_image=%s, critique_applied=%s, errors=%d, elapsed=%.3fs)",
        bool(final_image),
        final_metadata["critique_applied"],
        len(errors),
        elapsed,
    )
    
    # Update final timing
    timing_data = dict(timing_data)
    timing_data["finalize_s"] = elapsed
    timing_data["total_s"] = sum(timing_data.values())
    
    final_metadata["timing"] = Timing(
        analysis_s=timing_data.get("analysis_s", 0.0),
        render_s=timing_data.get("render_s", 0.0) + timing_data.get("adjust_render_s", 0.0),
        total_s=timing_data["total_s"],
    ).model_dump()
    
    return {
        "timing": timing_data,
        "metadata": final_metadata,
        "current_step": "workflow_complete",
        "final_image": final_image,
        "final_metadata": final_metadata,
    }