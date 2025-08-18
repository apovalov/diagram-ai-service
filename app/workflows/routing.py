"""Routing functions for LangGraph workflow conditional edges."""

from __future__ import annotations

from typing import Literal

from app.core.logging import get_logger
from app.workflows.state import DiagramWorkflowState

__all__ = [
    "should_render", 
    "should_critique", 
    "should_adjust",
    "should_retry_analysis",
    "should_retry_render",
]

logger = get_logger(__name__)

# Type aliases for routing destinations
AnalysisRoute = Literal["analyze", "render", "finalize"]
RenderRoute = Literal["critique", "finalize"]
CritiqueRoute = Literal["critique", "adjust", "finalize"]


def should_render(state: DiagramWorkflowState) -> AnalysisRoute:
    """Route after analysis based on success/failure.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node to execute: "analyze" (retry), "render", or "finalize"
    """
    current_step = state.get("current_step", "")
    
    if current_step == "analysis_retry":
        logger.debug("Routing to retry analysis")
        return "analyze"
    elif current_step == "analysis_complete":
        logger.debug("Analysis successful, routing to render")
        return "render"
    elif current_step == "analysis_failed":
        logger.warning("Analysis failed, routing to finalize")
        return "finalize"
    else:
        logger.warning("Unexpected analysis step '%s', routing to finalize", current_step)
        return "finalize"


def should_critique(state: DiagramWorkflowState) -> RenderRoute:
    """Route after rendering based on settings and success.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node to execute: "critique" or "finalize"
    """
    current_step = state.get("current_step", "")
    
    if current_step == "render_failed":
        logger.warning("Render failed, routing to finalize")
        return "finalize"
    elif not state.get("critique_enabled", True):
        logger.debug("Critique disabled, routing to finalize")
        return "finalize"
    elif current_step == "render_complete":
        logger.debug("Render successful and critique enabled, routing to critique")
        return "critique"
    else:
        logger.warning("Unexpected render step '%s', routing to finalize", current_step)
        return "finalize"


def should_adjust(state: DiagramWorkflowState) -> CritiqueRoute:
    """Route after critique based on feedback.
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node to execute: "critique" (retry), "adjust", or "finalize"
    """
    current_step = state.get("current_step", "")
    
    if current_step == "critique_retry":
        logger.debug("Routing to retry critique")
        return "critique"
    elif current_step == "critique_failed":
        logger.warning("Critique failed, routing to finalize")
        return "finalize"
    elif current_step == "critique_complete":
        critique = state.get("critique")
        
        if critique and not critique.done and critique.critique:
            logger.debug("Critique suggests improvements, routing to adjust")
            return "adjust"
        else:
            logger.debug("Critique indicates no improvements needed, routing to finalize")
            return "finalize"
    else:
        logger.warning("Unexpected critique step '%s', routing to finalize", current_step)
        return "finalize"


def should_retry_analysis(state: DiagramWorkflowState) -> bool:
    """Determine if analysis should be retried.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if analysis should be retried, False otherwise
    """
    attempts = state.get("analysis_attempts", 0)
    max_attempts = state.get("max_attempts", {}).get("analysis", 3)
    
    should_retry = attempts < max_attempts
    
    logger.debug(
        "Analysis retry check: attempts=%d, max=%d, should_retry=%s",
        attempts,
        max_attempts,
        should_retry,
    )
    
    return should_retry


def should_retry_render(state: DiagramWorkflowState) -> bool:
    """Determine if rendering should be retried.
    
    Args:
        state: Current workflow state
        
    Returns:
        True if rendering should be retried, False otherwise
    """
    attempts = state.get("render_attempts", 0)
    max_attempts = state.get("max_attempts", {}).get("render", 2)
    
    should_retry = attempts < max_attempts
    
    logger.debug(
        "Render retry check: attempts=%d, max=%d, should_retry=%s",
        attempts,
        max_attempts,
        should_retry,
    )
    
    return should_retry