"""Streaming support for LangGraph workflows."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict

from app.core.logging import get_logger
from app.workflows.state import DiagramWorkflowState

__all__ = ["stream_workflow_progress", "format_streaming_event"]

logger = get_logger(__name__)


async def stream_workflow_progress(
    workflow_stream: AsyncIterator[Dict[str, Any]],
) -> AsyncIterator[Dict[str, Any]]:
    """Stream workflow progress for real-time updates.

    Args:
        workflow_stream: LangGraph workflow stream

    Yields:
        Formatted progress events for client consumption
    """
    logger.info("Starting workflow progress streaming")

    try:
        async for event in workflow_stream:
            formatted_event = format_streaming_event(event)
            if formatted_event:
                logger.debug("Streaming event: %s", formatted_event.get("type"))
                yield formatted_event

    except Exception as e:
        logger.error("Streaming error: %s", e, exc_info=True)
        yield {
            "type": "error",
            "message": str(e),
            "timestamp": None,
        }

    logger.info("Workflow progress streaming completed")


def format_streaming_event(event: Dict[str, Any]) -> Dict[str, Any] | None:
    """Format workflow event for streaming.

    Args:
        event: Raw workflow event

    Returns:
        Formatted event or None if event should be filtered
    """
    # Extract node name and state from event
    node_name = None
    state_data = None

    # LangGraph event structure: {"node_name": {...}}
    for key, value in event.items():
        if isinstance(value, dict) and "current_step" in value:
            node_name = key
            state_data = value
            break

    if not node_name or not state_data:
        return None

    current_step = state_data.get("current_step", "unknown")
    progress = calculate_progress(current_step)

    # Create sanitized data for streaming (remove sensitive info)
    sanitized_data = sanitize_for_streaming(state_data)

    return {
        "type": "workflow_update",
        "node": node_name,
        "step": current_step,
        "progress": progress,
        "data": sanitized_data,
        "timestamp": None,  # Could add timestamp if needed
    }


def calculate_progress(current_step: str) -> float:
    """Calculate workflow progress percentage.

    Args:
        current_step: Current workflow step

    Returns:
        Progress as a float between 0.0 and 1.0
    """
    step_weights = {
        "start": 0.0,
        "analysis_complete": 0.2,
        "render_complete": 0.6,
        "critique_complete": 0.8,
        "adjust_complete": 0.9,
        "workflow_complete": 1.0,
    }

    # Handle retry and failed states
    if "_retry" in current_step:
        base_step = current_step.replace("_retry", "_complete")
        return step_weights.get(base_step, 0.0) * 0.9  # Slightly less for retries

    if "_failed" in current_step:
        return 1.0  # Failed is considered complete

    return step_weights.get(current_step, 0.0)


def sanitize_for_streaming(state_data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive information from state for streaming.

    Args:
        state_data: Raw state data

    Returns:
        Sanitized data safe for streaming
    """
    # Fields safe to include in streaming
    safe_fields = {
        "current_step",
        "analysis_method",
        "analysis_attempts",
        "critique_attempts",
        "request_id",
    }

    sanitized = {}

    for field in safe_fields:
        if field in state_data:
            sanitized[field] = state_data[field]

    # Add timing information if available
    timing = state_data.get("timing", {})
    if timing:
        sanitized["timing"] = {
            k: v for k, v in timing.items() if isinstance(v, (int, float))
        }

    # Add node/connection counts if analysis is available
    analysis = state_data.get("analysis")
    if analysis and hasattr(analysis, "nodes"):
        sanitized["analysis_summary"] = {
            "nodes": len(analysis.nodes) if analysis.nodes else 0,
            "connections": len(analysis.connections) if analysis.connections else 0,
            "clusters": len(analysis.clusters) if analysis.clusters else 0,
        }

    return sanitized
