from __future__ import annotations

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda

from app.core.config import settings
from app.core.langchain_prompts import get_diagram_analysis_prompt
from app.core.logging import get_logger
from app.core.schemas import (
    AnalysisCluster,
    AnalysisConnection,
    AnalysisNode,
    DiagramAnalysis,
)

__all__ = ["AnalysisChain"]


class AnalysisChain:
    """LangChain-based diagram analysis chain."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.logger = get_logger(__name__)
        self.prompt_template, self.parser = get_diagram_analysis_prompt()

        # Create the main chain
        self.chain = (
            self.prompt_template
            | self.llm
            | self.parser
        )

        # Create chain with fallback to heuristic analysis
        self.chain_with_fallback = RunnableWithFallbacks(
            runnable=self.chain,
            fallbacks=[RunnableLambda(self._heuristic_fallback)]
        )

    async def ainvoke(self, description: str) -> DiagramAnalysis:
        """
        Generate diagram analysis from description using LangChain.
        
        Args:
            description: Natural language description of the diagram
            
        Returns:
            DiagramAnalysis: Structured analysis with nodes, clusters, and connections
        """
        if settings.mock_llm:
            return self._create_mock_analysis()

        start_time = time.monotonic()

        try:
            self.logger.info(
                "Requesting LLM analysis (provider=%s, desc_len=%d)",
                settings.llm_provider,
                len(description or ""),
            )

            result = await self.chain_with_fallback.ainvoke({"description": description})

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            try:
                node_count = len(result.nodes)
                conn_count = len(result.connections)
                cluster_count = len(result.clusters)
            except Exception:
                node_count = conn_count = cluster_count = -1

            self.logger.info(
                "LLM analysis completed in %d ms (nodes=%s, conns=%s, clusters=%s)",
                elapsed_ms,
                node_count,
                conn_count,
                cluster_count,
            )

            return result

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            self.logger.error(
                "LLM analysis failed after %d ms, using heuristic fallback: %s",
                elapsed_ms,
                e,
            )
            return self._create_heuristic_analysis(description)

    def _create_mock_analysis(self) -> DiagramAnalysis:
        """Create mock analysis for testing."""
        return DiagramAnalysis(
            title="Application Diagram",
            nodes=[
                AnalysisNode(
                    id="alb", type="alb", label="Application Load Balancer"
                ),
                AnalysisNode(id="web1", type="ec2", label="Web 1"),
                AnalysisNode(id="db", type="rds", label="DB"),
            ],
            clusters=[AnalysisCluster(label="Web Tier", nodes=["web1"])],
            connections=[
                AnalysisConnection(source="alb", target="web1"),
                AnalysisConnection(source="web1", target="db"),
            ],
        )

    def _create_heuristic_analysis(self, description: str) -> DiagramAnalysis:
        """Create heuristic analysis when LLM fails."""
        desc_lower = description.lower()

        # Detect common patterns and build nodes
        nodes = []
        connections = []
        clusters = []
        node_counter = 1

        # Database detection
        if any(word in desc_lower for word in ["database", "db", "rds", "dynamodb", "mysql", "postgres"]):
            if "dynamo" in desc_lower:
                nodes.append(AnalysisNode(id="db1", type="dynamodb", label="Database"))
            else:
                nodes.append(AnalysisNode(id="db1", type="rds", label="Database"))

        # Web server/load balancer detection
        if any(word in desc_lower for word in ["web", "server", "load balancer", "alb", "nginx"]):
            if "load" in desc_lower or "alb" in desc_lower:
                nodes.append(AnalysisNode(id="lb1", type="alb", label="Load Balancer"))
            nodes.append(AnalysisNode(id="web1", type="ec2", label="Web Server"))

            # Connect load balancer to web server
            if any(n.type == "alb" for n in nodes):
                connections.append(AnalysisConnection(source="lb1", target="web1"))

        # Lambda detection
        if any(word in desc_lower for word in ["lambda", "function", "serverless"]):
            nodes.append(AnalysisNode(id="fn1", type="lambda", label="Lambda Function"))

        # API Gateway detection
        if any(word in desc_lower for word in ["api", "gateway", "rest"]):
            nodes.append(AnalysisNode(id="api1", type="api_gateway", label="API Gateway"))

        # S3 detection
        if any(word in desc_lower for word in ["s3", "storage", "bucket", "file"]):
            nodes.append(AnalysisNode(id="s3_1", type="s3", label="S3 Storage"))

        # Queue detection
        if any(word in desc_lower for word in ["queue", "sqs", "message"]):
            nodes.append(AnalysisNode(id="q1", type="sqs", label="Message Queue"))

        # Create default connections if we have components
        if len(nodes) >= 2:
            # Connect web to database
            web_nodes = [n for n in nodes if n.type in ["ec2", "service"]]
            db_nodes = [n for n in nodes if n.type in ["rds", "dynamodb"]]
            if web_nodes and db_nodes:
                connections.append(AnalysisConnection(source=web_nodes[0].id, target=db_nodes[0].id))

            # Connect API Gateway to Lambda
            api_nodes = [n for n in nodes if n.type == "api_gateway"]
            lambda_nodes = [n for n in nodes if n.type == "lambda"]
            if api_nodes and lambda_nodes:
                connections.append(AnalysisConnection(source=api_nodes[0].id, target=lambda_nodes[0].id))

        # Create basic clustering for web tier
        web_tier_nodes = [n.id for n in nodes if n.type in ["ec2", "alb"]]
        if len(web_tier_nodes) > 1:
            clusters.append(AnalysisCluster(label="Web Tier", nodes=web_tier_nodes))

        # Fallback: if no nodes detected, create a simple architecture
        if not nodes:
            nodes = [
                AnalysisNode(id="web1", type="ec2", label="Web Server"),
                AnalysisNode(id="db1", type="rds", label="Database"),
            ]
            connections = [AnalysisConnection(source="web1", target="db1")]

        title = "Architecture Diagram"
        if "web" in desc_lower:
            title = "Web Application Architecture"
        elif "api" in desc_lower:
            title = "API Architecture"
        elif "serverless" in desc_lower:
            title = "Serverless Architecture"

        self.logger.info(
            "Generated heuristic analysis: %d nodes, %d connections, %d clusters",
            len(nodes),
            len(connections),
            len(clusters),
        )

        return DiagramAnalysis(
            title=title,
            nodes=nodes,
            clusters=clusters,
            connections=connections,
        )

    async def _heuristic_fallback(self, inputs: dict[str, Any]) -> DiagramAnalysis:
        """Fallback runnable for when the main chain fails."""
        description = inputs.get("description", "")
        return self._create_heuristic_analysis(description)
