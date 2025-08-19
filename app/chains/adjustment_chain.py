from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda

from app.core.config import settings
from app.core.langchain_prompts import get_diagram_adjustment_prompt
from app.core.logging import get_logger
from app.core.schemas import DiagramAnalysis

__all__ = ["AdjustmentChain"]


class AdjustmentChain:
    """LangChain-based diagram adjustment chain."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = get_logger(__name__)
        self.prompt_template, self.parser = get_diagram_adjustment_prompt()

        # Create the main chain
        self.chain = self.prompt_template | self.llm | self.parser

        # Create chain with fallback to return original analysis
        self.chain_with_fallback = RunnableWithFallbacks(
            runnable=self.chain, fallbacks=[RunnableLambda(self._heuristic_fallback)]
        )

    async def ainvoke(
        self, description: str, critique: str, original_analysis: DiagramAnalysis
    ) -> DiagramAnalysis:
        """
        Adjust diagram analysis based on critique.

        Args:
            description: Original user description
            critique: Critique feedback to address
            original_analysis: Current analysis to improve

        Returns:
            DiagramAnalysis: Adjusted analysis addressing the critique
        """
        if settings.mock_llm:
            # For mock mode, just return the original analysis
            return original_analysis

        start_time = time.monotonic()

        try:
            self.logger.info(
                "Requesting diagram adjustment (provider=%s, desc_len=%d, critique_len=%d)",
                settings.llm_provider,
                len(description or ""),
                len(critique or ""),
            )

            result = await self.chain_with_fallback.ainvoke(
                {"description": description, "critique": critique}
            )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            try:
                node_count = len(result.nodes)
                conn_count = len(result.connections)
                cluster_count = len(result.clusters)
            except Exception:
                node_count = conn_count = cluster_count = -1

            self.logger.info(
                "Diagram adjustment completed in %d ms (nodes=%s, conns=%s, clusters=%s)",
                elapsed_ms,
                node_count,
                conn_count,
                cluster_count,
            )

            return result

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            self.logger.error(
                "Diagram adjustment failed after %d ms, returning original analysis: %s",
                elapsed_ms,
                e,
            )
            return original_analysis

    def _create_heuristic_adjustment(
        self, description: str, critique: str, original_analysis: DiagramAnalysis
    ) -> DiagramAnalysis:
        """Create heuristic adjustment when LLM fails."""
        # Simple heuristic adjustments based on common critique patterns
        adjusted_analysis = original_analysis.model_copy(deep=True)

        critique_lower = critique.lower()

        # If critique mentions missing connections, try to add some
        if "connection" in critique_lower or "isolated" in critique_lower:
            # Find nodes that aren't connected
            connected_nodes = set()
            for conn in adjusted_analysis.connections:
                connected_nodes.add(conn.source)
                connected_nodes.add(conn.target)

            isolated_nodes = [
                node
                for node in adjusted_analysis.nodes
                if node.id not in connected_nodes
            ]

            # Connect isolated nodes to existing connected nodes
            if isolated_nodes and connected_nodes:
                from app.core.schemas import AnalysisConnection

                first_connected = next(iter(connected_nodes))
                for isolated in isolated_nodes:
                    adjusted_analysis.connections.append(
                        AnalysisConnection(source=isolated.id, target=first_connected)
                    )

        # If critique mentions missing components, we could add them
        # (This would be more complex and context-dependent)

        self.logger.info(
            "Applied heuristic adjustment based on critique pattern matching"
        )

        return adjusted_analysis

    async def _heuristic_fallback(self, inputs: dict[str, Any]) -> DiagramAnalysis:
        """Fallback runnable for when the main chain fails."""
        description = inputs.get("description", "")
        critique = inputs.get("critique", "")

        # For fallback, we need access to the original analysis
        # This is a limitation of the current setup - in practice,
        # we might need to store this in the chain context

        # For now, create a minimal analysis
        from app.core.schemas import AnalysisConnection, AnalysisNode

        return DiagramAnalysis(
            title="Fallback Adjusted Diagram",
            nodes=[
                AnalysisNode(id="web1", type="ec2", label="Web Server"),
                AnalysisNode(id="db1", type="rds", label="Database"),
            ],
            clusters=[],
            connections=[AnalysisConnection(source="web1", target="db1")],
        )
