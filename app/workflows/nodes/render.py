"""Render node for diagram workflow."""

from __future__ import annotations

import base64
import time
from typing import Any, Dict

import anyio

from app.core.config import settings
from app.core.logging import get_logger
from app.services.diagram_service import DiagramService
from app.utils.files import save_image_bytes
from app.workflows.nodes.utils import handle_node_error
from app.workflows.state import DiagramWorkflowState

__all__ = ["render_node"]

logger = get_logger(__name__)


async def render_node(state: DiagramWorkflowState) -> Dict[str, Any]:
    """Render diagram from analysis using existing sync rendering logic.
    
    This node uses the existing DiagramService._generate_diagram_sync method
    to create PNG and DOT files from the structured analysis.
    
    Args:
        state: Current workflow state containing analysis
        
    Returns:
        Dict containing state updates with rendered image data
    """
    start_time = time.monotonic()
    analysis = state.get("analysis")
    description = state["description"]
    
    if not analysis:
        error = ValueError("No analysis available for rendering")
        return handle_node_error(state, error, "render")
    
    try:
        logger.info(
            "Starting render node (request_id=%s, nodes=%d, connections=%d)",
            state["request_id"],
            len(analysis.nodes) if analysis.nodes else 0,
            len(analysis.connections) if analysis.connections else 0,
        )
        
        # Create a temporary DiagramService instance for rendering
        diagram_service = DiagramService(settings)
        
        # Use existing sync rendering logic via thread pool
        image_data, render_metadata = await anyio.to_thread.run_sync(
            diagram_service._generate_diagram_sync,
            analysis,
            description,
        )
        
        # Persist the image to disk for critique
        try:
            image_bytes = base64.b64decode(image_data)
            image_path = save_image_bytes(
                data=image_bytes,
                base_tmp_dir=settings.tmp_dir,
            )
            logger.info("Rendered image saved to: %s", image_path)
        except Exception as e:
            logger.warning("Failed to persist rendered image: %s", e)
            image_path = None
        
        elapsed = time.monotonic() - start_time
        
        logger.info(
            "Render completed (size=%d bytes, elapsed=%.3fs)",
            len(image_data),
            elapsed,
        )
        
        # Update state with successful render
        timing = dict(state.get("timing", {}))
        timing["render_s"] = elapsed
        
        # Merge render metadata with existing metadata
        metadata = dict(state.get("metadata", {}))
        metadata.update(render_metadata)
        
        return {
            "image_before": image_data,
            "image_path": image_path,
            "timing": timing,
            "metadata": metadata,
            "current_step": "render_complete",
        }
        
    except Exception as e:
        elapsed = time.monotonic() - start_time
        
        logger.error(
            "Render node failed after %.3fs: %s",
            elapsed,
            e,
            exc_info=True,
        )
        
        # Update timing even on failure
        timing = dict(state.get("timing", {}))
        timing["render_s"] = elapsed
        
        # Handle error with retry logic
        error_updates = handle_node_error(state, e, "render")
        error_updates["timing"] = timing
        
        return error_updates