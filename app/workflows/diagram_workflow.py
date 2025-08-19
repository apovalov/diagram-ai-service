"""Main diagram generation workflow using LangGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from app.core.config import Settings
from app.core.logging import get_logger
from app.workflows.nodes.adjust import adjust_node
from app.workflows.nodes.analysis import analyze_node
from app.workflows.nodes.critique import critique_node
from app.workflows.nodes.finalize import finalize_node
from app.workflows.nodes.render import render_node
from app.workflows.routing import should_adjust, should_critique, should_render
from app.workflows.state import DiagramWorkflowState

if TYPE_CHECKING:
    pass

__all__ = ["create_diagram_workflow"]

logger = get_logger(__name__)


def create_diagram_workflow(settings: Settings) -> StateGraph:
    """Create the complete diagram generation workflow graph.

    This workflow implements the full diagram generation pipeline:
    1. Analyze description → structured diagram
    2. Render diagram → PNG/DOT files
    3. Critique rendered diagram (if enabled)
    4. Adjust analysis based on critique (if needed)
    5. Re-render adjusted diagram
    6. Finalize results and metadata

    Args:
        settings: Application settings for configuration

    Returns:
        StateGraph: Configured workflow ready for compilation
    """
    logger.info("Creating diagram workflow graph")

    # Create the state graph
    workflow = StateGraph(DiagramWorkflowState)

    # Add workflow nodes
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("render", render_node)
    workflow.add_node("critique", critique_node)
    workflow.add_node("adjust", adjust_node)
    workflow.add_node("finalize", finalize_node)

    # Define the workflow flow

    # Start with analysis
    workflow.add_edge(START, "analyze")

    # After analysis: retry analysis, proceed to render, or finalize on failure
    workflow.add_conditional_edges(
        "analyze",
        should_render,
        {
            "analyze": "analyze",  # Retry analysis
            "render": "render",  # Proceed to rendering
            "finalize": "finalize",  # Failed, skip to finalize
        },
    )

    # After render: proceed to critique or finalize
    workflow.add_conditional_edges(
        "render",
        should_critique,
        {
            "critique": "critique",  # Proceed to critique
            "finalize": "finalize",  # Skip critique, finalize
        },
    )

    # After critique: retry critique, adjust, or finalize
    workflow.add_conditional_edges(
        "critique",
        should_adjust,
        {
            "critique": "critique",  # Retry critique
            "adjust": "adjust",  # Apply adjustments
            "finalize": "finalize",  # No adjustments needed
        },
    )

    # After adjust: re-render with adjusted analysis
    workflow.add_edge("adjust", "render")

    # Finalize always leads to end
    workflow.add_edge("finalize", END)

    logger.info(
        "Workflow graph created with %d nodes and conditional routing",
        len(workflow.nodes),
    )

    return workflow
