"""Human-in-the-loop review capabilities for workflows."""

from __future__ import annotations

from typing import Any, Dict

from langgraph.types import Interrupt

from app.core.logging import get_logger
from app.workflows.state import DiagramWorkflowState

__all__ = ["request_human_review", "create_review_data"]

logger = get_logger(__name__)


async def request_human_review(
    state: DiagramWorkflowState,
    review_type: str = "critique_review"
) -> Dict[str, Any]:
    """Request human review by interrupting the workflow.
    
    Args:
        state: Current workflow state
        review_type: Type of review requested
        
    Returns:
        State updates (this function raises Interrupt)
        
    Raises:
        Interrupt: Always raises to pause workflow for human review
    """
    logger.info(
        "Requesting human review (type=%s, request_id=%s)",
        review_type,
        state["request_id"],
    )
    
    # Create review data for human reviewer
    review_data = create_review_data(state, review_type)
    
    # Interrupt workflow and provide review data
    raise Interrupt({
        "type": review_type,
        "request_id": state["request_id"],
        "data": review_data,
        "instructions": get_review_instructions(review_type),
    })


def create_review_data(state: DiagramWorkflowState, review_type: str) -> Dict[str, Any]:
    """Create data package for human review.
    
    Args:
        state: Current workflow state
        review_type: Type of review
        
    Returns:
        Review data dictionary
    """
    base_data = {
        "description": state["description"],
        "current_step": state.get("current_step"),
        "request_id": state["request_id"],
    }
    
    if review_type == "critique_review":
        return {
            **base_data,
            "image": state.get("image_before"),
            "analysis": state.get("analysis").model_dump() if state.get("analysis") else None,
            "critique": state.get("critique").model_dump() if state.get("critique") else None,
        }
    
    elif review_type == "analysis_review":
        return {
            **base_data,
            "analysis": state.get("analysis").model_dump() if state.get("analysis") else None,
            "analysis_method": state.get("analysis_method"),
        }
    
    else:
        return base_data


def get_review_instructions(review_type: str) -> str:
    """Get human-readable instructions for review type.
    
    Args:
        review_type: Type of review
        
    Returns:
        Instructions string
    """
    instructions = {
        "critique_review": (
            "Please review the generated diagram and critique. "
            "Decide whether to apply the suggested adjustments or proceed with the current diagram. "
            "You can also provide additional feedback for adjustments."
        ),
        "analysis_review": (
            "Please review the diagram analysis. "
            "Verify that the structure and components match the intended architecture. "
            "You can modify the analysis before proceeding to rendering."
        ),
    }
    
    return instructions.get(review_type, "Please review the workflow state and provide feedback.")


def process_human_feedback(
    state: DiagramWorkflowState,
    feedback: Dict[str, Any]
) -> Dict[str, Any]:
    """Process human feedback and update workflow state.
    
    Args:
        state: Current workflow state
        feedback: Human feedback data
        
    Returns:
        State updates based on feedback
    """
    logger.info(
        "Processing human feedback (request_id=%s, action=%s)",
        state["request_id"],
        feedback.get("action"),
    )
    
    action = feedback.get("action", "continue")
    
    if action == "approve":
        return {
            "human_review_result": "approved",
            "current_step": "human_approved",
        }
    
    elif action == "reject":
        return {
            "human_review_result": "rejected", 
            "current_step": "human_rejected",
        }
    
    elif action == "modify":
        updates = {
            "human_review_result": "modified",
            "current_step": "human_modified",
        }
        
        # Apply modifications based on feedback
        if "analysis" in feedback:
            # Human provided modified analysis
            from app.core.schemas import DiagramAnalysis
            updates["analysis"] = DiagramAnalysis(**feedback["analysis"])
        
        if "critique_override" in feedback:
            # Human overrode critique
            updates["critique_override"] = feedback["critique_override"]
        
        return updates
    
    else:
        # Default: continue workflow
        return {
            "human_review_result": "continued",
            "current_step": "human_continued", 
        }