from __future__ import annotations

from pydantic import BaseModel

__all__ = [
    "DiagramRequest",
    "DiagramMetadata",
    "DiagramResponse",
    "AssistantRequest",
    "AssistantResponse",
    "AssistantFinal",
    "IntentResult",
    "Timing",
    # llm structured output models
    "AnalysisNode",
    "AnalysisCluster",
    "AnalysisConnection",
    "DiagramAnalysis",
    "DiagramCritique",
]


class DiagramRequest(BaseModel):
    description: str
    format: str | None = "png"


class DiagramMetadata(BaseModel):
    nodes_created: int
    clusters_created: int
    connections_made: int
    generation_time: float
    timing: Timing | None = None
    analysis_method: str | None = None
    critique_applied: bool = False
    critique: dict | None = None
    adjust_render_s: float | None = None


class DiagramResponse(BaseModel):
    success: bool
    image_data: str | None = None
    image_url: str | None = None
    metadata: DiagramMetadata | None = None


class AssistantRequest(BaseModel):
    message: str
    context: dict[str, str] | None = None
    conversation_id: str | None = None


class AssistantResponse(BaseModel):
    response_type: str
    content: str
    image_data: str | None = None
    suggestions: list[str] | None = None


class AssistantFinal(BaseModel):
    """Structured finalization payload from the LLM when no tool call is made."""

    content: str
    suggestions: list[str] | None = None


# Structured intent model for assistant intent detection
class IntentResult(BaseModel):
    intent: str
    description: str | None = None
    confidence: float | None = None


# models for llm structured JSON output
class AnalysisNode(BaseModel):
    id: str
    type: str
    label: str


class AnalysisCluster(BaseModel):
    label: str
    nodes: list[str]


class AnalysisConnection(BaseModel):
    source: str
    target: str


class DiagramAnalysis(BaseModel):
    title: str
    nodes: list[AnalysisNode]
    clusters: list[AnalysisCluster]
    connections: list[AnalysisConnection]


class DiagramCritique(BaseModel):
    """Structured critique result for a rendered diagram and its analysis."""

    done: bool
    critique: str | None = None


class Timing(BaseModel):
    analysis_s: float
    render_s: float
    total_s: float
