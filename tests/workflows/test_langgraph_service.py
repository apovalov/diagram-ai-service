"""Tests for LangGraph diagram service."""

import pytest

from app.core.config import Settings
from app.services.langgraph_diagram_service import LangGraphDiagramService
from app.workflows.state import DiagramWorkflowState


@pytest.mark.asyncio
async def test_langgraph_service_initialization():
    """Test LangGraph service initializes correctly."""
    settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=False,
    )

    service = LangGraphDiagramService(settings)

    assert service.settings == settings
    assert service.workflow is not None
    assert service.langgraph_config is not None


@pytest.mark.asyncio
async def test_langgraph_basic_generation():
    """Test basic diagram generation with LangGraph."""
    settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=False,
    )

    service = LangGraphDiagramService(settings)

    description = "Create a simple web application with database"
    result = await service.generate_diagram_from_description(description)

    image_data, metadata = result

    # Verify basic structure
    assert isinstance(image_data, str)
    assert len(image_data) > 0
    assert isinstance(metadata, dict)

    # Verify metadata contains expected fields
    required_fields = [
        "nodes_created",
        "clusters_created",
        "connections_made",
        "generation_time",
        "timing",
        "analysis_method",
        "critique_applied",
        "request_id",
    ]

    for field in required_fields:
        assert field in metadata, f"Missing required field: {field}"

    # Verify timing structure
    timing = metadata["timing"]
    assert "analysis_s" in timing
    assert "render_s" in timing
    assert "total_s" in timing


@pytest.mark.asyncio
async def test_langgraph_with_critique():
    """Test diagram generation with critique enabled."""
    settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=True,
    )

    service = LangGraphDiagramService(settings)

    description = "Create a microservices architecture"
    result = await service.generate_diagram_from_description(description)

    image_data, metadata = result

    assert isinstance(image_data, str)
    assert len(image_data) > 0

    # In mock mode, critique should be skipped but field should exist
    assert "critique_applied" in metadata
    assert isinstance(metadata["critique_applied"], bool)


@pytest.mark.asyncio
async def test_langgraph_critique_workflow():
    """Test the critique-specific workflow method."""
    settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=True,
    )

    service = LangGraphDiagramService(settings)

    description = "Create an AWS serverless application"
    result = await service.generate_diagram_with_critique(description)

    (image_before, image_after), metadata = result

    # Should have at least the before image
    assert isinstance(image_before, str)
    assert len(image_before) > 0

    # image_after may be None in mock mode
    assert image_after is None or isinstance(image_after, str)

    # Verify metadata structure
    assert isinstance(metadata, dict)
    assert "critique_applied" in metadata


@pytest.mark.asyncio
async def test_metadata_formatting():
    """Test that metadata formatting matches original service contract."""
    settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=False,
    )

    service = LangGraphDiagramService(settings)

    # Create a mock final state
    final_state = DiagramWorkflowState(
        description="test",
        request_id="test-123",
        analysis=None,
        analysis_attempts=0,
        analysis_method="mock",
        image_before="test_image",
        image_after=None,
        image_path=None,
        dot_path=None,
        critique=None,
        critique_attempts=0,
        critique_enabled=False,
        current_step="workflow_complete",
        errors=[],
        max_attempts={},
        timing={"analysis_s": 0.1, "render_s": 0.2},
        metadata={"test_field": "test_value"},
        messages=None,
        conversation_id=None,
        streaming_enabled=False,
        human_review_enabled=False,
        cache_enabled=False,
    )

    metadata = service._format_metadata(final_state)

    # Verify all expected fields are present
    expected_fields = [
        "nodes_created",
        "clusters_created",
        "connections_made",
        "generation_time",
        "timing",
        "analysis_method",
        "critique_applied",
        "request_id",
        "errors",
    ]

    for field in expected_fields:
        assert field in metadata

    # Verify specific values
    assert metadata["analysis_method"] == "mock"
    assert metadata["request_id"] == "test-123"
    assert metadata["critique_applied"] is False
    assert metadata["errors"] == []
    assert metadata["test_field"] == "test_value"  # Custom metadata preserved
