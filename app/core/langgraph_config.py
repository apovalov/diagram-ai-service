"""LangGraph configuration and factory functions."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

from app.core.config import Settings
from app.core.logging import get_logger
from app.workflows.state import DiagramWorkflowState

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

__all__ = ["LangGraphConfig", "create_initial_state"]

logger = get_logger(__name__)


class LangGraphConfig:
    """Configuration manager for LangGraph workflows."""
    
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        
        # Initialize checkpointer if enabled
        self.checkpointer = None
        if getattr(settings, "use_checkpoints", True):
            self.checkpointer = MemorySaver()
            logger.info("LangGraph checkpointer enabled (MemorySaver)")
        
        logger.info(
            "LangGraph config initialized (checkpoints=%s, streaming=%s)",
            bool(self.checkpointer),
            getattr(settings, "enable_streaming", False),
        )
    
    def create_workflow(self) -> StateGraph:
        """Factory for creating configured workflow graph.
        
        This will be implemented in Phase 3 when we create the actual workflow.
        For now, it returns a basic StateGraph that will be extended.
        """
        from app.workflows.diagram_workflow import create_diagram_workflow
        
        workflow = create_diagram_workflow(self.settings)
        logger.info("Diagram workflow graph created")
        return workflow
    
    def compile_workflow(self, workflow: StateGraph) -> CompiledGraph:
        """Compile workflow with configuration options."""
        config = {}
        
        if self.checkpointer:
            config["checkpointer"] = self.checkpointer
        
        # Note: Caching will be implemented in a future LangGraph version
        # For now, we log the intent but don't pass unsupported parameters
        if getattr(self.settings, "workflow_cache_ttl", 0) > 0:
            logger.info("Workflow caching requested (ttl=%ds) but not yet supported", self.settings.workflow_cache_ttl)
        
        compiled = workflow.compile(**config)
        logger.info("Workflow compiled with config: %s", list(config.keys()))
        return compiled


def create_initial_state(
    description: str,
    settings: Settings,
    conversation_id: str | None = None,
    messages: list | None = None,
) -> DiagramWorkflowState:
    """Create initial state for diagram workflow execution.
    
    Args:
        description: User's natural language description
        settings: Application settings
        conversation_id: Optional conversation identifier for assistant workflows
        messages: Optional conversation messages for context
        
    Returns:
        DiagramWorkflowState: Initialized state ready for workflow execution
    """
    return DiagramWorkflowState(
        # Core workflow data
        description=description,
        request_id=str(uuid.uuid4()),
        
        # Analysis phase
        analysis=None,
        analysis_attempts=0,
        analysis_method="unknown",
        
        # Rendering phase
        image_before=None,
        image_after=None,
        image_path=None,
        dot_path=None,
        
        # Critique phase
        critique=None,
        critique_attempts=0,
        critique_enabled=getattr(settings, "use_critique_generation", True),
        
        # Workflow control
        current_step="start",
        errors=[],
        max_attempts={
            "analysis": 3,
            "render": 2,
            "critique": getattr(settings, "critique_max_attempts", 3),
            "adjust": 2,
        },
        
        # Performance tracking
        timing={},
        metadata={},
        
        # Assistant context
        messages=messages,
        conversation_id=conversation_id,
        
        # Advanced features
        streaming_enabled=getattr(settings, "enable_streaming", False),
        human_review_enabled=getattr(settings, "enable_human_review", False),
        cache_enabled=getattr(settings, "workflow_cache_ttl", 0) > 0,
    )