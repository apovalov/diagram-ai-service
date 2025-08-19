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
from app.agents.langchain_diagram_agent import LangChainDiagramAgent
from app.core.config import Settings
from app.core.execution_strategy import StrategyFactory, ExecutionMode
from app.core.error_handler import ErrorHandler
from app.core.exceptions import DiagramError, FatalError, ValidationError
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
    "aws_lambda": "lambda",  # Handle "AWS Lambda" type from analysis
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
    """Enhanced diagram service with strategy pattern and unified error handling."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.temp_dir = settings.tmp_dir
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        # Initialize error handler
        self.error_handler = ErrorHandler(
            max_retries=getattr(settings, 'max_retries', 3),
            backoff_factor=getattr(settings, 'backoff_factor', 1.5)
        )

        # Setup monitoring if enabled
        self.langsmith_manager = None
        self.metrics = None
        if getattr(settings, 'langsmith_enabled', False):
            self._setup_monitoring()
        
        # BACKWARD COMPATIBILITY: Keep old agent initialization for existing code
        # This ensures any direct .agent access continues to work
        if self.settings.use_langchain:
            self.agent = LangChainDiagramAgent(settings)
            self.original_agent = DiagramAgent(settings)  # Keep as fallback
            logger.info("Using LangChain-based diagram agent")
        else:
            self.agent = DiagramAgent(settings)
            self.original_agent = None
            logger.info("Using original diagram agent")

        logger.info(
            f"DiagramService initialized (mode={getattr(settings, 'execution_mode', 'legacy')}, "
            f"fallbacks={getattr(settings, 'enable_fallbacks', True)}, provider={settings.llm_provider})"
        )

        # Run cleanup on initialization
        self._cleanup_old_files()

    def _setup_monitoring(self):
        """Setup LangSmith monitoring if available."""
        try:
            # Only import if LangSmith is enabled to avoid dependency issues
            from ..core.langsmith_config import LangSmithManager
            from ..core.langsmith_metrics import DiagramMetrics
            
            self.langsmith_manager = LangSmithManager(self.settings)
            if self.langsmith_manager.is_enabled():
                self.metrics = DiagramMetrics(self.langsmith_manager.client)
                logger.info("LangSmith monitoring enabled")
            else:
                logger.warning("LangSmith initialization failed, monitoring disabled")
                
        except ImportError:
            logger.info("LangSmith dependencies not available, monitoring disabled")
        except Exception as e:
            logger.warning(f"Failed to setup LangSmith monitoring: {e}")

    def _cleanup_old_files(self) -> None:
        """Clean up old temporary files on service initialization."""
        try:
            # Clean up files older than 24 hours
            cleanup_old_files(self.temp_dir, max_age_hours=24)
            # Clean up outputs directory, keeping only 50 most recent files
            cleanup_outputs_directory(self.temp_dir, max_files=50)
        except Exception as e:
            logger.warning(f"Failed to cleanup old files: {e}")

    async def generate_diagram_from_description(
        self, description: str
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate diagram with strategy pattern and fallback support.
        
        CRITICAL: This method maintains the exact same signature and behavior
        as the original implementation for backward compatibility.
        
        Uses new strategy pattern if execution_mode is set, otherwise falls back
        to legacy LangGraph → LangChain → Original workflow.
        """
        
        request_id = str(uuid.uuid4())
        start_time = time.monotonic()
        
        # Validate input (same as before)
        if not description or not description.strip():
            raise ValueError("Description cannot be empty")
        
        description = description.strip()
        if len(description) > 5000:
            logger.warning(f"Description truncated from {len(description)} to 5000 characters")
            description = description[:5000]
        
        # NEW: Use strategy pattern if execution_mode is explicitly set (not auto/legacy)
        execution_mode = getattr(self.settings, 'execution_mode', 'auto')
        if execution_mode and execution_mode != 'auto':
            try:
                return await self._generate_with_strategy_pattern(
                    description, request_id, start_time
                )
            except Exception as e:
                # If strategy pattern fails and fallbacks are disabled, raise
                if not getattr(self.settings, 'enable_fallbacks', True):
                    raise
                logger.warning(f"Strategy pattern failed, falling back to legacy workflow: {e}")
        
        # BACKWARD COMPATIBILITY: Use existing logic for legacy configurations
        # Try LangGraph first if enabled
        if getattr(self.settings, "use_langgraph", False):
            try:
                from app.services.langgraph_diagram_service import (
                    LangGraphDiagramService,
                )

                langgraph_service = LangGraphDiagramService(self.settings)
                return await langgraph_service.generate_diagram_from_description(
                    description
                )
            except Exception as e:
                if getattr(self.settings, "langgraph_fallback", True):
                    logger.warning(
                        f"LangGraph generation failed, falling back to LangChain/original: {e}"
                    )
                else:
                    raise

        # Fallback to LangChain if enabled
        if self.settings.use_langchain:
            try:
                return await self._generate_with_langchain(description)
            except Exception as e:
                if self.settings.langchain_fallback:
                    logger.warning(f"LangChain generation failed, using original: {e}")
                    return await self._generate_original(description)
                raise

        # Final fallback to original implementation
        return await self._generate_original(description)

    async def _generate_with_langchain(
        self, description: str
    ) -> tuple[str, dict[str, Any]]:
        """Generate diagram using LangChain agent."""
        if self.settings.use_critique_generation:
            # Use the critique-enhanced generation workflow
            (
                (image_before, image_after),
                metadata,
            ) = await self.generate_diagram_with_critique(description)

            # Return the final image (after critique adjustments if available) or the original
            final_image = image_after if image_after else image_before

            # Add a flag to indicate if critique improvements were applied
            metadata["critique_applied"] = image_after is not None

            return final_image, metadata
        else:
            # Use the standard generation workflow (original implementation)
            return await self._generate_diagram_standard(description)

    async def _generate_original(self, description: str) -> tuple[str, dict[str, Any]]:
        """Generate diagram using original agent."""
        # Temporarily switch to original agent
        current_agent = self.agent
        try:
            if self.original_agent:
                self.agent = self.original_agent

            if self.settings.use_critique_generation:
                # Use the critique-enhanced generation workflow
                (
                    (image_before, image_after),
                    metadata,
                ) = await self.generate_diagram_with_critique(description)

                # Return the final image (after critique adjustments if available) or the original
                final_image = image_after if image_after else image_before

                # Add a flag to indicate if critique improvements were applied
                metadata["critique_applied"] = image_after is not None

                return final_image, metadata
            else:
                # Use the standard generation workflow (original implementation)
                return await self._generate_diagram_standard(description)
        finally:
            # Restore original agent
            self.agent = current_agent

    async def generate_diagram_with_critique(
        self, description: str
    ) -> tuple[tuple[str, str | None], dict[str, Any]]:
        """Generate a diagram, run an image-based critique, optionally adjust and re-render.

        Returns ((image_before_b64, image_after_b64_or_none), metadata)
        """

        with Timer() as total:
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

            # Critique using the rendered image - retry multiple times for better critique quality

            critique = None
            image_bytes = base64.b64decode(image_before_b64)
            max_critique_attempts = max(
                1, min(5, self.settings.critique_max_attempts)
            )  # Clamp between 1-5

            for attempt in range(1, max_critique_attempts + 1):
                try:
                    logger.info(
                        f"Attempting critique generation (attempt {attempt}/{max_critique_attempts})"
                    )
                    critique = await self.agent.critique_analysis(
                        description=description,
                        analysis=analysis,
                        image_bytes=image_bytes,
                    )
                    # If we got a critique with actual feedback, use it
                    if critique and not critique.done and critique.critique:
                        logger.info(
                            f"Critique received on attempt {attempt}: {len(critique.critique)} characters"
                        )
                        metadata["critique_attempts"] = attempt
                        break
                    # If critique says it's done (no improvements needed), that's also valid
                    elif critique and critique.done:
                        logger.info(
                            f"Critique completed on attempt {attempt}: no improvements needed"
                        )
                        metadata["critique_attempts"] = attempt
                        break
                    else:
                        logger.info(
                            f"Critique attempt {attempt} returned no feedback, trying again..."
                        )
                        critique = None
                except Exception as e:
                    logger.warning(
                        f"Critique attempt {attempt}/{max_critique_attempts} failed: %s",
                        e,
                        exc_info=True if attempt == max_critique_attempts else False,
                    )
                    critique = None

            if not critique:
                logger.warning(
                    "All critique attempts failed, continuing without adjustments"
                )
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

    async def _generate_diagram_standard(
        self, description: str
    ) -> tuple[str, dict[str, Any]]:
        """Generate diagram from natural language description with standard workflow (no critique)."""
        with Timer() as total:
            with Timer() as t_analysis:
                analysis_result: DiagramAnalysis = await self.agent.generate_analysis(
                    description
                )
            with Timer() as t_render:
                image_data, metadata = await anyio.to_thread.run_sync(
                    self._generate_diagram_sync, analysis_result, description
                )
            try:
                analysis_method = (
                    "heuristic"
                    if getattr(analysis_result, "title", "")
                    .lower()
                    .startswith("heuristic")
                    else "llm"
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

    def _normalize_analysis(
        self, analysis: DiagramAnalysis, *, strict: bool = True
    ) -> DiagramAnalysis:
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
                logger.debug(
                    "Dropping invalid connection: %s -> %s", c.source, c.target
                )

        # Clusters: keep only existing node ids, drop empty clusters
        clusters_out: list[AnalysisCluster] = []
        for cl in analysis.clusters:
            nodes_filtered = [nid for nid in cl.nodes if nid in id_set]
            if nodes_filtered:
                clusters_out.append(
                    AnalysisCluster(label=cl.label, nodes=nodes_filtered)
                )

        title = (
            analysis.title or "Application Diagram"
        ).strip() or "Application Diagram"

        return DiagramAnalysis(
            title=title, nodes=nodes_out, clusters=clusters_out, connections=conns_out
        )

    def _generate_diagram_sync(
        self, analysis_result: DiagramAnalysis, description: str
    ) -> tuple[str, dict[str, Any]]:
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
                            node_details = next(
                                (n for n in normalized.nodes if n.id == node_id), None
                            )
                            if node_details:
                                node_class = NODE_MAP.get(node_details.type)
                                nodes[node_id] = (node_class or Blank)(
                                    node_details.label
                                )

                # Standalone nodes
                clustered_node_ids = {
                    nid for c in normalized.clusters for nid in c.nodes
                }
                for node_details in sorted(normalized.nodes, key=lambda n: n.id):
                    if node_details.id in clustered_node_ids:
                        continue
                    node_class = NODE_MAP.get(node_details.type)
                    nodes[node_details.id] = (node_class or Blank)(node_details.label)

                # Connections
                for conn in sorted(
                    normalized.connections, key=lambda c: (c.source, c.target)
                ):
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
                out = os.path.join(
                    self.settings.tmp_dir, "outputs", f"{uuid.uuid4()}.dot"
                )
                os.makedirs(os.path.dirname(out), exist_ok=True)
                shutil.copyfile(dot_path, out)
                logger.info("DOT saved: %s", out)
            else:
                synthesized_dot = self._build_dot_from_analysis(normalized)
                out = os.path.join(
                    self.settings.tmp_dir, "outputs", f"{uuid.uuid4()}.dot"
                )
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
            safe = (value or "").replace('"', '\\"')
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
                nid for nid in cluster.nodes if any(n.id == nid for n in analysis.nodes)
            ]
            for node_id in sorted(valid_cluster_nodes):
                lines.append(f"    {q(node_id)};")
            lines.append("  }")

        # Add connections
        for conn in sorted(analysis.connections, key=lambda c: (c.source, c.target)):
            lines.append(f"  {q(conn.source)} -> {q(conn.target)};")

        lines.append("}")
        return "\n".join(lines)

    # === NEW STRATEGY PATTERN METHODS ===

    async def _generate_with_strategy_pattern(
        self, description: str, request_id: str, start_time: float
    ) -> tuple[str, dict[str, Any]]:
        """Generate using new strategy pattern."""
        
        # Convert string execution_mode to enum
        try:
            mode = ExecutionMode(self.settings.execution_mode)
        except ValueError:
            logger.warning(f"Unknown execution_mode '{self.settings.execution_mode}', using AUTO")
            mode = ExecutionMode.AUTO
        
        # Get strategy execution chain
        if getattr(self.settings, 'enable_fallbacks', True):
            strategy_chain = StrategyFactory.get_fallback_chain(mode, self.settings)
        else:
            strategy_chain = [mode]
        
        logger.info(
            f"Starting diagram generation (request_id={request_id}, "
            f"strategies={[s.value for s in strategy_chain]})"
        )
        
        last_error = None
        
        # Try each strategy in the chain
        for i, mode in enumerate(strategy_chain):
            try:
                # Create strategy
                strategy = StrategyFactory.create_agent_strategy(mode, self.settings)
                
                logger.info(f"Attempting strategy {mode.value} (attempt {i+1}/{len(strategy_chain)})")
                
                # For LangGraph, use full service approach for now
                if mode == ExecutionMode.LANGGRAPH:
                    try:
                        from app.services.langgraph_diagram_service import LangGraphDiagramService
                        service = LangGraphDiagramService(self.settings)
                        return await service.generate_diagram_from_description(description)
                    except ImportError:
                        logger.warning("LangGraph service not available, skipping")
                        continue
                
                # For other strategies, use existing workflow but with new agent
                if self.settings.use_critique_generation:
                    result = await self._generate_with_critique_using_agent(strategy, description)
                else:
                    result = await self._generate_standard_using_agent(strategy, description)
                
                # Log success metrics
                duration_ms = (time.monotonic() - start_time) * 1000
                await self._log_success_metrics(
                    request_id, description, mode.value, duration_ms, result[1]
                )
                
                logger.info(
                    f"Diagram generation successful with {mode.value} "
                    f"(duration={duration_ms:.1f}ms, request_id={request_id})"
                )
                
                return result
                
            except FatalError as e:
                # Fatal errors should not trigger fallbacks
                logger.error(f"Fatal error in {mode.value}: {e}")
                await self._log_error_metrics(request_id, "diagram_generation", e, mode.value)
                raise ValueError(str(e))  # Convert to original exception type for compatibility
                
            except Exception as e:
                last_error = e
                logger.warning(f"Strategy {mode.value} failed: {e}")
                await self._log_error_metrics(request_id, "diagram_generation", e, mode.value)
                
                # If this is the last strategy, break and raise
                if i == len(strategy_chain) - 1:
                    break
                
                # Log fallback attempt
                next_mode = strategy_chain[i + 1]
                logger.info(f"Falling back from {mode.value} to {next_mode.value}")
        
        # All strategies failed
        total_duration = (time.monotonic() - start_time) * 1000
        error_msg = f"All strategies failed after {total_duration:.1f}ms. Last error: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg) from last_error

    async def _generate_with_critique_using_agent(
        self, agent_strategy, description: str
    ) -> tuple[str, dict[str, Any]]:
        """Generate with critique using provided agent strategy."""
        # Temporarily replace the agent and use existing workflow
        original_agent = self.agent
        try:
            # Replace agent with strategy agent
            self.agent = agent_strategy.agent if hasattr(agent_strategy, 'agent') else agent_strategy
            
            # Use existing critique workflow
            (
                (image_before, image_after),
                metadata
            ) = await self.generate_diagram_with_critique(description)
            
            final_image = image_after if image_after else image_before
            metadata["critique_applied"] = bool(image_after)
            return final_image, metadata
        except AttributeError:
            # Fallback if agent doesn't have expected interface
            return await self._generate_standard_using_agent(agent_strategy, description)
        finally:
            # Always restore original agent
            self.agent = original_agent

    async def _generate_standard_using_agent(
        self, agent_strategy, description: str
    ) -> tuple[str, dict[str, Any]]:
        """Generate without critique using provided agent strategy."""
        # Use the existing _generate_diagram_standard method but with the strategy agent
        original_agent = self.agent
        try:
            # Temporarily replace agent
            self.agent = agent_strategy.agent if hasattr(agent_strategy, 'agent') else agent_strategy
            return await self._generate_diagram_standard(description)
        finally:
            # Restore original agent
            self.agent = original_agent

    async def _log_success_metrics(
        self, 
        request_id: str, 
        description: str, 
        strategy: str, 
        duration_ms: float, 
        metadata: dict[str, Any]
    ):
        """Log success metrics to monitoring system."""
        if self.metrics:
            try:
                await self.metrics.log_diagram_generation(
                    request_id=request_id,
                    description=description,
                    execution_mode=strategy,
                    success=True,
                    duration_ms=duration_ms,
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to log success metrics: {e}")

    async def _log_error_metrics(
        self, 
        request_id: str, 
        operation: str, 
        error: Exception, 
        strategy: str
    ):
        """Log error metrics to monitoring system."""
        if self.metrics:
            try:
                await self.metrics.log_error(
                    request_id=request_id,
                    operation=operation,
                    error=error,
                    execution_mode=strategy
                )
            except Exception as e:
                logger.warning(f"Failed to log error metrics: {e}")
