"""Tests for individual workflow nodes."""

import pytest

from app.core.config import Settings
from app.core.schemas import DiagramAnalysis, AnalysisNode, AnalysisConnection
from app.workflows.nodes.analysis import analyze_node
from app.workflows.nodes.render import render_node
from app.workflows.nodes.critique import critique_node
from app.workflows.nodes.finalize import finalize_node
from app.workflows.state import DiagramWorkflowState


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    return Settings(
        mock_llm=True,
        use_langchain=False,
        use_critique_generation=True,
    )


@pytest.fixture
def sample_state():
    """Create sample workflow state for testing."""
    return DiagramWorkflowState(
        description="Create a web application with database",
        request_id="test-123",
        analysis=None,
        analysis_attempts=0,
        analysis_method="unknown",
        image_before=None,
        image_after=None,
        image_path=None,
        dot_path=None,
        critique=None,
        critique_attempts=0,
        critique_enabled=True,
        current_step="start",
        errors=[],
        max_attempts={"analysis": 3, "render": 2, "critique": 3},
        timing={},
        metadata={},
        messages=None,
        conversation_id=None,
        streaming_enabled=False,
        human_review_enabled=False,
        cache_enabled=False,
    )


@pytest.fixture
def sample_analysis():
    """Create sample analysis for testing."""
    return DiagramAnalysis(
        title="Test Application",
        nodes=[
            AnalysisNode(id="web1", type="ec2", label="Web Server"),
            AnalysisNode(id="db1", type="rds", label="Database"),
        ],
        clusters=[],
        connections=[
            AnalysisConnection(source="web1", target="db1")
        ],
    )


@pytest.mark.asyncio
async def test_analyze_node_success(sample_state, mock_settings):
    """Test successful analysis node execution."""
    # Mock the settings globally
    import app.workflows.nodes.analysis
    original_settings = app.workflows.nodes.analysis.settings
    app.workflows.nodes.analysis.settings = mock_settings
    
    try:
        result = await analyze_node(sample_state)
        
        assert "analysis" in result
        assert "analysis_method" in result
        assert "timing" in result
        assert "current_step" in result
        assert result["current_step"] == "analysis_complete"
        assert result["analysis_method"] == "mock"
        
        # Verify timing was recorded
        assert "analysis_s" in result["timing"]
        assert isinstance(result["timing"]["analysis_s"], float)
        
    finally:
        app.workflows.nodes.analysis.settings = original_settings


@pytest.mark.asyncio
async def test_render_node_success(sample_state, sample_analysis, mock_settings):
    """Test successful render node execution."""
    # Set up state with analysis
    state_with_analysis = dict(sample_state)
    state_with_analysis["analysis"] = sample_analysis
    
    # Mock the settings
    import app.workflows.nodes.render
    original_settings = app.workflows.nodes.render.settings
    app.workflows.nodes.render.settings = mock_settings
    
    try:
        result = await render_node(state_with_analysis)
        
        assert "image_before" in result
        assert "timing" in result
        assert "metadata" in result
        assert "current_step" in result
        assert result["current_step"] == "render_complete"
        
        # Verify image data
        assert isinstance(result["image_before"], str)
        assert len(result["image_before"]) > 0
        
        # Verify timing
        assert "render_s" in result["timing"]
        assert isinstance(result["timing"]["render_s"], float)
        
    finally:
        app.workflows.nodes.render.settings = original_settings


@pytest.mark.asyncio
async def test_render_node_no_analysis(sample_state):
    """Test render node with missing analysis."""
    result = await render_node(sample_state)
    
    # Should return error updates - may be retry or failed depending on attempt count
    assert "errors" in result
    assert "current_step" in result
    assert result["current_step"] in ["render_retry", "render_failed"]
    assert len(result["errors"]) > 0


@pytest.mark.asyncio
async def test_critique_node_disabled(sample_state, sample_analysis):
    """Test critique node when critique is disabled."""
    state_disabled = dict(sample_state)
    state_disabled["critique_enabled"] = False
    state_disabled["analysis"] = sample_analysis  # Add required analysis
    state_disabled["image_before"] = "fake_image_data"  # Add required image
    
    result = await critique_node(state_disabled)
    
    assert "critique" in result
    assert "current_step" in result
    assert result["current_step"] == "critique_complete"
    assert result["critique"].done is True


@pytest.mark.asyncio
async def test_critique_node_mock_mode(sample_state, sample_analysis, mock_settings):
    """Test critique node in mock mode."""
    # Set up state
    state_with_data = dict(sample_state)
    state_with_data["analysis"] = sample_analysis
    state_with_data["image_before"] = "fake_base64_image_data"
    
    # Mock settings
    import app.workflows.nodes.critique
    original_settings = app.workflows.nodes.critique.settings
    app.workflows.nodes.critique.settings = mock_settings
    
    try:
        result = await critique_node(state_with_data)
        
        assert "critique" in result
        assert "current_step" in result
        assert result["current_step"] == "critique_complete"
        assert result["critique"].done is True
        
    finally:
        app.workflows.nodes.critique.settings = original_settings


@pytest.mark.asyncio
async def test_finalize_node(sample_state, sample_analysis):
    """Test finalize node execution."""
    # Set up complete state
    final_state = dict(sample_state)
    final_state.update({
        "analysis": sample_analysis,
        "image_before": "fake_image_data",
        "analysis_method": "llm",
        "timing": {"analysis_s": 0.1, "render_s": 0.2},
        "current_step": "render_complete",
    })
    
    result = await finalize_node(final_state)
    
    # Verify finalize results
    assert "final_image" in result
    assert "final_metadata" in result
    assert "current_step" in result
    assert result["current_step"] == "workflow_complete"
    
    # Verify final metadata structure
    metadata = result["final_metadata"]
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
        assert field in metadata
    
    # Verify specific values
    assert metadata["nodes_created"] == 2
    assert metadata["connections_made"] == 1
    assert metadata["analysis_method"] == "llm"
    assert metadata["critique_applied"] is False
    assert metadata["request_id"] == "test-123"


@pytest.mark.asyncio
async def test_finalize_node_with_errors(sample_state):
    """Test finalize node with errors in state."""
    error_state = dict(sample_state)
    error_state.update({
        "errors": ["Test error 1", "Test error 2"],
        "current_step": "analysis_failed",
        "timing": {"analysis_s": 0.1},
    })
    
    result = await finalize_node(error_state)
    
    metadata = result["final_metadata"]
    assert "errors" in metadata
    assert len(metadata["errors"]) == 2
    assert metadata["errors"][0] == "Test error 1"