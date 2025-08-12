# diagram_service.py
from __future__ import annotations

import base64
import os
import shutil
import time
import uuid
from typing import Any

import anyio
from diagrams import Cluster, Diagram
from diagrams.aws.compute import EC2, Lambda
from diagrams.aws.database import RDS, Dynamodb
from diagrams.aws.integration import SNS, SQS
from diagrams.aws.management import Cloudwatch
from diagrams.aws.network import ALB, VPC, APIGateway, InternetGateway
from diagrams.aws.security import Cognito
from diagrams.aws.storage import S3
from diagrams.generic.blank import Blank

from app.agents.diagram_agent import DiagramAgent
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.schemas import (
    AnalysisCluster,
    AnalysisConnection,
    AnalysisNode,
    DiagramAnalysis,
    Timing,
)
from app.utils.cleanup import (
    cleanup_old_files,
    cleanup_outputs_directory,
    temp_file_manager,
)
from app.utils.files import save_image_bytes
from app.utils.timing import Timer

__all__ = ["DiagramService"]

logger = get_logger(__name__)

BASE_GRAPH_ATTR: dict[str, str] = {
    "pad": "0.1",
    "splines": "ortho",
    "rankdir": "LR",
    "nodesep": "0.8",
    "ranksep": "1.2",
}

# Canonical Core v1 node map used for rendering
NODE_MAP: dict[str, type] = {
    # Compute
    "ec2": EC2,
    "lambda": Lambda,

    # Database & Storage
    "rds": RDS,
    "dynamodb": Dynamodb,
    "s3": S3,

    # Networking & Routing
    "alb": ALB,
    "api_gateway": APIGateway,
    "vpc": VPC,
    "internet_gateway": InternetGateway,

    # Integration
    "sqs": SQS,
    "sns": SNS,

    # Observability & Identity
    "cloudwatch": Cloudwatch,
    "cognito": Cognito,

    # Generic microservice
    "service": EC2,  # use EC2 icon as a neutral microservice box
}

# Input aliases → canonical type keys
ALIASES: dict[str, str] = {
    "apigateway": "api_gateway",
    "gateway": "api_gateway",
    "monitoring": "cloudwatch",
    "database": "rds",
    "web_server": "ec2",
    "microservice": "service",
    "queue": "sqs",
    # Business service names collapse to generic service (labels stay descriptive)
    "auth_service": "service",
    "payment_service": "service",
    "order_service": "service",
}


def canonical_type(raw: str, *, strict: bool = True, default: str = "service") -> str:
    """Normalize a raw type to a canonical key used in NODE_MAP.

    If strict and unsupported, raises KeyError. Otherwise returns the default.
    """
    k = (raw or "").strip().lower().replace(" ", "_")
    k = ALIASES.get(k, k)
    if k not in NODE_MAP:
        if strict:
            raise KeyError(f"Unsupported type '{raw}' after normalization -> '{k}'")
        return default
    return k


class DiagramService:
    """Service for generating diagrams from natural language descriptions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.temp_dir = settings.tmp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        self.agent = DiagramAgent()

        # Run cleanup on initialization
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Clean up old temporary files on service initialization."""
        try:
            # Clean up files older than 24 hours
            cleanup_old_files(self.temp_dir, max_age_hours=24)
            # Clean up outputs directory, keeping only 50 most recent files
            cleanup_outputs_directory(self.temp_dir, max_files=50)
        except Exception as e:
            logger.warning(f"Failed to cleanup old files: {e}")

    async def generate_diagram_from_description(self, description: str) -> tuple[str, dict[str, Any]]:
        """Generate diagram from natural language description.

        Uses critique-enhanced generation if settings.use_critique_generation is True,
        otherwise uses the standard generation workflow.
        """
        if self.settings.use_critique_generation:
            # Use the critique-enhanced generation workflow
            (image_before, image_after), metadata = await self.generate_diagram_with_critique(description)

            # Return the final image (after critique adjustments if available) or the original
            final_image = image_after if image_after else image_before

            # Add a flag to indicate if critique improvements were applied
            metadata["critique_applied"] = image_after is not None

            return final_image, metadata
        else:
            # Use the standard generation workflow (original implementation)
            return await self._generate_diagram_standard(description)

    async def generate_diagram_with_critique(self, description: str) -> tuple[tuple[str, str | None], dict[str, Any]]:
        """Generate a diagram, run an image-based critique, optionally adjust and re-render.

        Returns ((image_before_b64, image_after_b64_or_none), metadata)
        """


        with Timer() as total:
            with Timer() as t_analysis:
                analysis: DiagramAnalysis = await self.agent.generate_analysis(description)


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

            # Critique using the rendered image - retry multiple times for better critique quality


            critique = None
            image_bytes = base64.b64decode(image_before_b64)
            max_critique_attempts = max(1, min(5, self.settings.critique_max_attempts))  # Clamp between 1-5



            for attempt in range(1, max_critique_attempts + 1):
                try:
                    logger.info(f"Attempting critique generation (attempt {attempt}/{max_critique_attempts})")
                    critique = await self.agent.critique_analysis(
                        description=description, analysis=analysis, image_bytes=image_bytes
                    )
                    # If we got a critique with actual feedback, use it
                    if critique and not critique.done and critique.critique:
                        logger.info(f"Critique received on attempt {attempt}: {len(critique.critique)} characters")
                        metadata["critique_attempts"] = attempt
                        break
                    # If critique says it's done (no improvements needed), that's also valid
                    elif critique and critique.done:
                        logger.info(f"Critique completed on attempt {attempt}: no improvements needed")
                        metadata["critique_attempts"] = attempt
                        break
                    else:
                        logger.info(f"Critique attempt {attempt} returned no feedback, trying again...")
                        critique = None
                except Exception as e:
                    logger.warning(
                        f"Critique attempt {attempt}/{max_critique_attempts} failed: %s",
                        e,
                        exc_info=True if attempt == max_critique_attempts else False,
                    )
                    critique = None

            if not critique:
                logger.warning("All critique attempts failed, continuing without adjustments")
                metadata["critique_attempts"] = max_critique_attempts

            image_after_b64: str | None = None
            if critique and not critique.done and critique.critique:


                with Timer() as t_adjust_render:
                    updated: DiagramAnalysis = await self.agent.adjust_analysis(
                        description=description,
                        analysis=analysis,
                        critique=critique.critique,
                    )


                    image_after_b64, metadata_after = await anyio.to_thread.run_sync(
                        self._generate_diagram_sync, updated, description
                    )


                metadata.update(metadata_after)
                metadata["critique"] = critique.model_dump()
                metadata["adjust_render_s"] = t_adjust_render.elapsed_s
            else:
                pass

        metadata["timing"] = Timing(
            analysis_s=t_analysis.elapsed_s,
            render_s=t_render1.elapsed_s,
            total_s=total.elapsed_s,
        ).model_dump()



        return (image_before_b64, image_after_b64), metadata

    async def _generate_diagram_standard(self, description: str) -> tuple[str, dict[str, Any]]:
        """Generate diagram from natural language description with standard workflow (no critique)."""
        with Timer() as total:
            with Timer() as t_analysis:
                analysis_result: DiagramAnalysis = await self.agent.generate_analysis(description)
            with Timer() as t_render:
                image_data, metadata = await anyio.to_thread.run_sync(
                    self._generate_diagram_sync, analysis_result, description
                )
            try:
                analysis_method = (
                    "heuristic" if getattr(analysis_result, "title", "").lower().startswith("heuristic") else "llm"
                )
                metadata["analysis_method"] = analysis_method
            except Exception:
                pass

        metadata["timing"] = Timing(
            analysis_s=t_analysis.elapsed_s,
            render_s=t_render.elapsed_s,
            total_s=total.elapsed_s,
        ).model_dump()
        metadata["critique_applied"] = False
        return image_data, metadata

    # -------------------- internal helpers --------------------

    def _normalize_analysis(self, analysis: DiagramAnalysis, *, strict: bool = True) -> DiagramAnalysis:
        """Return a new DiagramAnalysis with canonicalized types, validated edges/clusters, and sane defaults."""
        seen_ids: set[str] = set()
        nodes_out: list[AnalysisNode] = []

        # Nodes: dedupe ids, canonicalize types, ensure label
        for node in analysis.nodes:
            nid = node.id
            if not nid:
                continue
            if nid in seen_ids:
                raise ValueError(f"Duplicate node id: {nid}")
            seen_ids.add(nid)

            try:
                ctype = canonical_type(node.type, strict=strict)
            except KeyError:
                # In strict mode this will raise; if strict=False we fallback to 'service'
                if strict:
                    raise
                ctype = canonical_type(node.type, strict=False)

            label = (node.label or nid.replace("_", " ").title()).strip()
            nodes_out.append(AnalysisNode(id=nid, type=ctype, label=label))

        id_set = {n.id for n in nodes_out}

        # Connections: keep only edges with valid endpoints, drop self-loops
        conns_out: list[AnalysisConnection] = []
        for c in analysis.connections:
            if c.source in id_set and c.target in id_set and c.source != c.target:
                conns_out.append(AnalysisConnection(source=c.source, target=c.target))
            else:
                logger.debug("Dropping invalid connection: %s -> %s", c.source, c.target)

        # Clusters: keep only existing node ids, drop empty clusters
        clusters_out: list[AnalysisCluster] = []
        for cl in analysis.clusters:
            nodes_filtered = [nid for nid in cl.nodes if nid in id_set]
            if nodes_filtered:
                clusters_out.append(AnalysisCluster(label=cl.label, nodes=nodes_filtered))

        title = (analysis.title or "Application Diagram").strip() or "Application Diagram"

        return DiagramAnalysis(title=title, nodes=nodes_out, clusters=clusters_out, connections=conns_out)

    def _generate_diagram_sync(self, analysis_result: DiagramAnalysis, description: str) -> tuple[str, dict[str, Any]]:
        """Synchronous diagram generation (runs in thread pool)."""
        with temp_file_manager(self.temp_dir) as workdir:
            diagram_path = os.path.join(workdir, "diagram")
            nodes: dict[str, Any] = {}
            start_time = time.time()

            normalized = self._normalize_analysis(analysis_result, strict=True)

            with Diagram(
                normalized.title,
                filename=diagram_path,
                show=False,
                outformat="png",
                graph_attr=BASE_GRAPH_ATTR,
            ):
                # Clusters first
                for cluster_info in sorted(normalized.clusters, key=lambda c: c.label):
                    with Cluster(cluster_info.label):
                        for node_id in sorted(cluster_info.nodes):
                            node_details = next((n for n in normalized.nodes if n.id == node_id), None)
                            if node_details:
                                node_class = NODE_MAP.get(node_details.type)
                                nodes[node_id] = (node_class or Blank)(node_details.label)

                # Standalone nodes
                clustered_node_ids = {nid for c in normalized.clusters for nid in c.nodes}
                for node_details in sorted(normalized.nodes, key=lambda n: n.id):
                    if node_details.id in clustered_node_ids:
                        continue
                    node_class = NODE_MAP.get(node_details.type)
                    nodes[node_details.id] = (node_class or Blank)(node_details.label)

                # Connections
                for conn in sorted(normalized.connections, key=lambda c: (c.source, c.target)):
                    src = nodes.get(conn.source)
                    dst = nodes.get(conn.target)
                    if src and dst:
                        src >> dst

            generation_time = time.time() - start_time
            image_path = f"{diagram_path}.png"

            if not os.path.exists(image_path):
                raise FileNotFoundError("Diagram image not generated.")

            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode("utf-8")

            # Persist DOT (either produced by diagrams or synthesized from normalized analysis)
            dot_path = f"{diagram_path}.dot"
            if os.path.exists(dot_path):
                out = os.path.join(self.settings.tmp_dir, "outputs", f"{uuid.uuid4()}.dot")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copyfile(dot_path, out)
                logger.info("DOT saved: %s", out)
            else:
                synthesized_dot = self._build_dot_from_analysis(normalized)
                out = os.path.join(self.settings.tmp_dir, "outputs", f"{uuid.uuid4()}.dot")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w", encoding="utf-8") as dot_file:
                    dot_file.write(synthesized_dot)
                logger.info("DOT synthesized and saved: %s", out)

            # Periodic cleanup of outputs directory
            cleanup_outputs_directory(self.settings.tmp_dir, max_files=50)

            metadata = {
                "nodes_created": len(analysis_result.nodes),
                "clusters_created": len(analysis_result.clusters),
                "connections_made": len(analysis_result.connections),
                "generation_time": generation_time,
            }

            return image_data, metadata

    def _build_dot_from_analysis(self, analysis: DiagramAnalysis) -> str:
        """
        Build a Graphviz DOT representation from the analysis result.
        Handles overlapping clusters by declaring nodes only once, globally.
        """
        def q(value: str) -> str:
            safe = (value or "").replace("\"", "\\\"")
            return f'"{safe}"'

        lines: list[str] = ["digraph G {"]
        for key, val in BASE_GRAPH_ATTR.items():
            lines.append(f"  {key}={q(val)};")

        # Declare all nodes globally first, regardless of clustering
        for node in sorted(analysis.nodes, key=lambda n: n.id):
            lines.append(f"  {q(node.id)} [label={q(node.label)}];")

        # Create clusters that reference existing nodes (no node declarations inside)
        sorted_clusters = sorted(analysis.clusters, key=lambda c: c.label)
        for idx, cluster in enumerate(sorted_clusters):
            lines.append(f"  subgraph cluster_{idx} {{")
            lines.append(f"    label={q(cluster.label)};")
            # List the node IDs that belong to this cluster
            valid_cluster_nodes = [
                nid for nid in cluster.nodes
                if any(n.id == nid for n in analysis.nodes)
            ]
            for node_id in sorted(valid_cluster_nodes):
                lines.append(f"    {q(node_id)};")
            lines.append("  }")

        # Add connections
        for conn in sorted(analysis.connections, key=lambda c: (c.source, c.target)):
            lines.append(f"  {q(conn.source)} -> {q(conn.target)};")

        lines.append("}")
        return "\n".join(lines)

