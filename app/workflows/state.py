"""State schema for LangGraph diagram generation workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage

from app.core.schemas import DiagramAnalysis, DiagramCritique

__all__ = ["DiagramWorkflowState"]


class DiagramWorkflowState(TypedDict):
    """Complete state schema for diagram generation workflow.
    
    This state manages the entire diagram generation pipeline from
    description to final rendered image, including critique and adjustments.
    """
    
    # Core workflow data
    description: str  # User's natural language description
    request_id: str  # Unique identifier for this workflow execution
    
    # Analysis phase
    analysis: Optional[DiagramAnalysis]  # Structured diagram analysis
    analysis_attempts: int  # Number of analysis attempts made
    analysis_method: str  # "llm" | "heuristic" | "mock"
    
    # Rendering phase
    image_before: Optional[str]  # Base64 encoded image before critique
    image_after: Optional[str]   # Base64 encoded image after adjustments
    image_path: Optional[str]    # File path to saved image
    dot_path: Optional[str]      # File path to saved DOT file
    
    # Critique phase
    critique: Optional[DiagramCritique]  # Critique result if enabled
    critique_attempts: int  # Number of critique attempts made
    critique_enabled: bool  # Whether critique is enabled for this workflow
    
    # Workflow control
    current_step: str  # Current workflow step identifier
    errors: List[str]  # List of errors encountered during workflow
    max_attempts: Dict[str, int]  # Maximum attempts per operation
    
    # Performance tracking
    timing: Dict[str, float]  # Timing data for each step in seconds
    metadata: Dict[str, Any]  # Additional metadata and metrics
    
    # Assistant context (for conversation workflows)
    messages: Optional[List[BaseMessage]]  # Conversation messages
    conversation_id: Optional[str]  # Conversation identifier
    
    # Advanced features (Phase 5)
    streaming_enabled: Optional[bool]  # Whether to stream progress updates
    human_review_enabled: Optional[bool]  # Whether human review is required
    cache_enabled: Optional[bool]  # Whether node caching is enabled