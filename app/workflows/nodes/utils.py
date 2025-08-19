"""Utility functions for workflow nodes."""

from __future__ import annotations

import time
from typing import Any, Dict

from app.core.config import settings
from app.core.logging import get_logger
from app.workflows.state import DiagramWorkflowState

__all__ = ["handle_node_error", "create_timer_context"]

logger = get_logger(__name__)


def handle_node_error(
    state: DiagramWorkflowState, error: Exception, node_name: str
) -> Dict[str, Any]:
    """Standardized error handling for all workflow nodes.

    Args:
        state: Current workflow state
        error: Exception that occurred
        node_name: Name of the node where error occurred

    Returns:
        Dict containing state updates for error handling
    """
    errors = list(state.get("errors", []))
    errors.append(f"{node_name} failed: {str(error)}")

    attempts_key = f"{node_name}_attempts"
    attempts = state.get(attempts_key, 0) + 1
    max_attempts = state.get("max_attempts", {}).get(node_name, 3)

    logger.error(
        "Node %s failed (attempt %d/%d): %s",
        node_name,
        attempts,
        max_attempts,
        error,
        exc_info=True if attempts >= max_attempts else False,
    )

    if attempts < max_attempts:
        return {
            attempts_key: attempts,
            "errors": errors,
            "current_step": f"{node_name}_retry",
        }
    else:
        return {
            attempts_key: attempts,
            "errors": errors,
            "current_step": f"{node_name}_failed",
        }


class create_timer_context:
    """Context manager for timing node operations."""

    def __init__(self, state: DiagramWorkflowState, operation_name: str):
        self.state = state
        self.operation_name = operation_name
        self.start_time: float | None = None

    def __enter__(self):
        self.start_time = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            elapsed = time.monotonic() - self.start_time
            timing = dict(self.state.get("timing", {}))
            timing[f"{self.operation_name}_s"] = elapsed
            self.state["timing"] = timing

            logger.debug(
                "Operation %s completed in %.3fs", self.operation_name, elapsed
            )
