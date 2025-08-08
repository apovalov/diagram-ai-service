from __future__ import annotations

import base64
import os
import shutil
import tempfile
import time
from typing import Any

import anyio
from diagrams import Cluster, Diagram
from diagrams.aws.analytics import Kinesis
from diagrams.aws.compute import EC2, Lambda
from diagrams.aws.database import Dynamodb, RDS
from diagrams.aws.devtools import Codebuild, Codepipeline
from diagrams.aws.integration import SNS, SQS
from diagrams.aws.management import Cloudwatch
from diagrams.aws.network import ALB, APIGateway, ELB, InternetGateway, NLB, VPC
from diagrams.aws.security import Cognito, IAM
from diagrams.aws.storage import S3
from diagrams.generic.blank import Blank

from app.agents.diagram_agent import DiagramAgent
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import DiagramAnalysis, Timing
from app.utils.files import save_image_bytes
from app.utils.timing import Timer

__all__ = ["DiagramService"]

logger = get_logger(__name__)

BASE_GRAPH_ATTR: dict[str, str] = {
    "pad": "0.1",  # canvas padding (inches)
    "splines": "ortho",  # edge routing style
    "rankdir": "LR",  # left-to-right layout
    "nodesep": "0.8",  # separation between nodes on same rank (inches)
    "ranksep": "1.2",  # separation between ranks (inches)
}
NODE_MAP: dict[str, type] = {
    # Compute
    "ec2": EC2,
    "lambda": Lambda,
    # Database
    "rds": RDS,
    "dynamodb": Dynamodb,
    # Network & Load Balancing
    "elb": ELB,
    "alb": ALB,  # Application Load Balancer
    "nlb": NLB,  # Network Load Balancer
    "api_gateway": APIGateway,
    "apigateway": APIGateway,
    "vpc": VPC,
    "internet_gateway": InternetGateway,
    # Storage
    "s3": S3,
    # Integration & Messaging
    "sqs": SQS,
    "sns": SNS,
    # Management & Monitoring
    "cloudwatch": Cloudwatch,
    "monitoring": Cloudwatch,
    # Security
    "iam": IAM,
    "cognito": Cognito,
    # Analytics
    "kinesis": Kinesis,
    # Developer Tools
    "codebuild": Codebuild,
    "codepipeline": Codepipeline,
    # Generic service types
    "service": EC2,  # Default for microservices
    "microservice": EC2,
    "auth_service": EC2,
    "payment_service": EC2,
    "order_service": EC2,
    "web_server": EC2,
    "database": RDS,
    "queue": SQS,
    "gateway": APIGateway,
}


class DiagramService:
    """Service for generating diagrams from natural language descriptions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.temp_dir = settings.tmp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.agent = DiagramAgent()

    async def generate_diagram_from_description(
        self, description: str
    ) -> tuple[str, dict[str, Any]]:
        """Generate diagram from natural language description with end-to-end timing."""
        with Timer() as total:
            with Timer() as t_analysis:
                analysis_result: DiagramAnalysis = await self.agent.generate_analysis(
                    description
                )
            with Timer() as t_render:
                image_data, metadata = await anyio.to_thread.run_sync(
                    self._generate_diagram_sync, analysis_result, description
                )

        metadata["timing"] = Timing(
            analysis_s=t_analysis.elapsed_s,
            render_s=t_render.elapsed_s,
            total_s=total.elapsed_s,
        ).model_dump()
        return image_data, metadata

    async def generate_diagram_with_critique(
        self, description: str
    ) -> tuple[tuple[str, str | None], dict[str, Any]]:
        """Generate a diagram, run an image-based critique, optionally adjust and re-render.

        Returns ((image_before_b64, image_after_b64_or_none), metadata)
        """
        with Timer() as total:
            # Initial analysis and render
            with Timer() as t_analysis:
                analysis: DiagramAnalysis = await self.agent.generate_analysis(
                    description
                )
            with Timer() as t_render1:
                image_before_b64, metadata = await anyio.to_thread.run_sync(
                    self._generate_diagram_sync, analysis, description
                )

            # Persist initial image and log the path
            try:
                before_path = save_image_bytes(
                    data=base64.b64decode(image_before_b64),
                    base_tmp_dir=self.settings.tmp_dir,
                )
                logger.info(f"Saved pre-critique image to: {before_path}")
            except Exception as e:
                before_path = None
                logger.warning(f"Failed to save pre-critique image: {e}")

            # Critique using the rendered image
            try:
                image_bytes = base64.b64decode(image_before_b64)
                critique = await self.agent.critique_analysis(
                    description=description, analysis=analysis, image_bytes=image_bytes
                )
            except Exception:
                critique = None

            image_after_b64: str | None = None
            if critique and not critique.done and critique.critique:
                # Adjust analysis and re-render
                with Timer() as t_adjust_render:
                    updated: DiagramAnalysis = await self.agent.adjust_analysis(
                        description=description,
                        analysis=analysis,
                        critique=critique.critique,
                    )
                    image_after_b64, metadata_after = await anyio.to_thread.run_sync(
                        self._generate_diagram_sync, updated, description
                    )
                # Merge/override metadata to reflect final render
                metadata.update(metadata_after)
                metadata["critique"] = critique.model_dump()
                metadata["adjust_render_s"] = t_adjust_render.elapsed_s

        metadata["timing"] = Timing(
            analysis_s=t_analysis.elapsed_s,
            render_s=t_render1.elapsed_s,
            total_s=total.elapsed_s,
        ).model_dump()
        return (image_before_b64, image_after_b64), metadata

    def _generate_diagram_sync(
        self, analysis_result: DiagramAnalysis, description: str
    ) -> tuple[str, dict[str, Any]]:
        """Synchronous diagram generation (runs in thread pool)."""
        workdir = tempfile.mkdtemp(dir=self.temp_dir)
        diagram_path = os.path.join(workdir, "diagram")
        nodes: dict[str, Any] = {}
        start_time = time.time()

        try:
            with Diagram(
                analysis_result.title,
                filename=diagram_path,
                show=False,
                outformat="png",
                graph_attr=BASE_GRAPH_ATTR,
            ):
                for cluster_info in analysis_result.clusters:
                    with Cluster(cluster_info.label):
                        for node_id in cluster_info.nodes:
                            node_details = next(
                                (n for n in analysis_result.nodes if n.id == node_id),
                                None,
                            )
                            if node_details:
                                node_class = NODE_MAP.get(node_details.type.lower())
                                if node_class:
                                    nodes[node_id] = node_class(node_details.label)
                                else:
                                    logger.warning(
                                        f"Unknown node type '{node_details.type}' for node '{node_id}', using generic node"
                                    )
                                    nodes[node_id] = Blank(
                                        f"Unknown: {node_details.label}"
                                    )

                clustered_node_ids = [
                    node_id for c in analysis_result.clusters for node_id in c.nodes
                ]
                for node_details in analysis_result.nodes:
                    if node_details.id not in clustered_node_ids:
                        node_class = NODE_MAP.get(node_details.type.lower())
                        if node_class:
                            nodes[node_details.id] = node_class(node_details.label)
                        else:
                            logger.warning(
                                f"Unknown node type '{node_details.type}' for node '{node_details.id}', using generic node"
                            )
                            nodes[node_details.id] = Blank(
                                f"Unknown: {node_details.label}"
                            )

                for conn in analysis_result.connections:
                    source_node = nodes.get(conn.source)
                    target_node = nodes.get(conn.target)
                    if source_node and target_node:
                        source_node >> target_node

            generation_time = time.time() - start_time
            image_path = f"{diagram_path}.png"

            if not os.path.exists(image_path):
                raise FileNotFoundError("Diagram image not generated.")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        metadata = {
            "nodes_created": len(analysis_result.nodes),
            "clusters_created": len(analysis_result.clusters),
            "connections_made": len(analysis_result.connections),
            "generation_time": generation_time,
        }

        return image_data, metadata
