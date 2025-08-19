"""Compatibility tests between original, LangChain, and LangGraph implementations."""

import pytest

from app.core.config import Settings
from app.services.diagram_service import DiagramService


@pytest.mark.asyncio
async def test_langgraph_output_structure_matches_original():
    """Verify LangGraph produces similar output structure to original implementation."""
    description = "Create a web application with database and load balancer"

    # Test original implementation
    original_settings = Settings(
        use_langgraph=False,
        use_langchain=False,
        mock_llm=True,
        use_critique_generation=False,
    )

    original_service = DiagramService(original_settings)
    original_result = await original_service.generate_diagram_from_description(
        description
    )
    original_image, original_metadata = original_result

    # Test LangGraph implementation
    langgraph_settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=False,
    )

    langgraph_service = DiagramService(langgraph_settings)
    langgraph_result = await langgraph_service.generate_diagram_from_description(
        description
    )
    langgraph_image, langgraph_metadata = langgraph_result

    # Compare structure (not exact values due to randomness)
    assert isinstance(original_image, str)
    assert isinstance(langgraph_image, str)
    assert len(original_image) > 0
    assert len(langgraph_image) > 0

    # Compare metadata structures
    for key in ["nodes_created", "clusters_created", "connections_made", "timing"]:
        assert key in original_metadata
        assert key in langgraph_metadata
        assert type(original_metadata[key]) == type(langgraph_metadata[key])

    # Verify timing structure consistency
    for timing_key in ["analysis_s", "render_s", "total_s"]:
        assert timing_key in original_metadata["timing"]
        assert timing_key in langgraph_metadata["timing"]


@pytest.mark.asyncio
async def test_three_tier_fallback_behavior():
    """Test the three-tier fallback: LangGraph → LangChain → Original."""
    description = "Create a microservices architecture"

    # Test with all flags enabled - should use LangGraph
    settings_all = Settings(
        use_langgraph=True,
        use_langchain=True,
        mock_llm=True,
        langgraph_fallback=True,
        langchain_fallback=True,
    )

    service_all = DiagramService(settings_all)
    result_all = await service_all.generate_diagram_from_description(description)

    assert isinstance(result_all, tuple)
    assert len(result_all) == 2

    # Test with LangGraph disabled - should use LangChain
    settings_langchain = Settings(
        use_langgraph=False,
        use_langchain=True,
        mock_llm=True,
        langchain_fallback=True,
    )

    service_langchain = DiagramService(settings_langchain)
    result_langchain = await service_langchain.generate_diagram_from_description(
        description
    )

    assert isinstance(result_langchain, tuple)
    assert len(result_langchain) == 2

    # Test with both disabled - should use original
    settings_original = Settings(
        use_langgraph=False,
        use_langchain=False,
        mock_llm=True,
    )

    service_original = DiagramService(settings_original)
    result_original = await service_original.generate_diagram_from_description(
        description
    )

    assert isinstance(result_original, tuple)
    assert len(result_original) == 2


@pytest.mark.asyncio
async def test_critique_compatibility():
    """Test critique workflow compatibility across implementations."""
    description = "Create a serverless web application"

    # Original with critique
    original_settings = Settings(
        use_langgraph=False,
        use_langchain=False,
        mock_llm=True,
        use_critique_generation=True,
    )

    original_service = DiagramService(original_settings)
    original_result = await original_service.generate_diagram_from_description(
        description
    )
    original_image, original_metadata = original_result

    # LangGraph with critique
    langgraph_settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=True,
    )

    langgraph_service = DiagramService(langgraph_settings)
    langgraph_result = await langgraph_service.generate_diagram_from_description(
        description
    )
    langgraph_image, langgraph_metadata = langgraph_result

    # Both should have critique_applied field
    assert "critique_applied" in original_metadata
    assert "critique_applied" in langgraph_metadata

    # Both should be boolean
    assert isinstance(original_metadata["critique_applied"], bool)
    assert isinstance(langgraph_metadata["critique_applied"], bool)


@pytest.mark.asyncio
async def test_error_handling_consistency():
    """Test that error handling is consistent across implementations."""
    # This test would need to be implemented with actual error scenarios
    # For now, we'll test that all implementations handle basic cases

    description = ""  # Empty description

    settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=False,
    )

    service = DiagramService(settings)

    # Should not crash with empty description
    try:
        result = await service.generate_diagram_from_description(description)
        image, metadata = result
        assert isinstance(image, str)
        assert isinstance(metadata, dict)
    except Exception as e:
        # If it fails, it should fail gracefully
        assert isinstance(e, (ValueError, TypeError))


@pytest.mark.asyncio
async def test_performance_similar_timing():
    """Test that LangGraph performance is similar to original (within reason)."""
    description = "Create a complex microservices architecture with multiple databases"

    # Test original timing
    original_settings = Settings(
        use_langgraph=False,
        use_langchain=False,
        mock_llm=True,
        use_critique_generation=False,
    )

    original_service = DiagramService(original_settings)
    original_result = await original_service.generate_diagram_from_description(
        description
    )
    original_image, original_metadata = original_result
    original_total_time = original_metadata["timing"]["total_s"]

    # Test LangGraph timing
    langgraph_settings = Settings(
        use_langgraph=True,
        mock_llm=True,
        use_critique_generation=False,
    )

    langgraph_service = DiagramService(langgraph_settings)
    langgraph_result = await langgraph_service.generate_diagram_from_description(
        description
    )
    langgraph_image, langgraph_metadata = langgraph_result
    langgraph_total_time = langgraph_metadata["timing"]["total_s"]

    # LangGraph should not be more than 50% slower than original (generous threshold)
    # This accounts for workflow overhead while ensuring reasonable performance
    assert langgraph_total_time <= original_total_time * 1.5, (
        f"LangGraph too slow: {langgraph_total_time}s vs original {original_total_time}s"
    )
