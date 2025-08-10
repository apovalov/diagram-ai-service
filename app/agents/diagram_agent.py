# from __future__ import annotations

# from google.genai import types as genai_types
# import time

# from app.core.config import settings
# from app.core.llm import client
# from app.core.logging import get_logger
# from app.core.schemas import (
#     AnalysisCluster,
#     AnalysisConnection,
#     AnalysisNode,
#     DiagramAnalysis,
#     DiagramCritique,
# )
# from app.core.prompts import (
#     diagram_adjustment_prompt,
#     diagram_analysis_prompt,
#     diagram_critique_prompt,
# )

# __all__ = ["DiagramAgent"]


# class DiagramAgent:
#     """Agent for analyzing diagram descriptions and extracting components."""

#     def __init__(self) -> None:
#         self.logger = get_logger(__name__)

#     async def generate_analysis(self, description: str) -> DiagramAnalysis:
#         """Generate diagram component analysis from description."""
#         # Mocked local analysis for development/test when mock mode is enabled
#         if settings.mock_llm:
#             return DiagramAnalysis(
#                 title="Application Diagram",
#                 nodes=[
#                     AnalysisNode(id="alb", type="alb", label="ALB"),
#                     AnalysisNode(id="web1", type="ec2", label="Web 1"),
#                     AnalysisNode(id="db", type="rds", label="DB"),
#                 ],
#                 clusters=[AnalysisCluster(label="Web Tier", nodes=["web1"])],
#                 connections=[
#                     AnalysisConnection(source="alb", target="web1"),
#                     AnalysisConnection(source="web1", target="db"),
#                 ],
#             )

#         prompt = diagram_analysis_prompt(description)
#         try:
#             start = time.monotonic()
#             self.logger.info(
#                 "Requesting LLM analysis (model=%s, desc_len=%d)",
#                 settings.gemini_model,
#                 len(description or ""),
#             )
#             response = await client.aio.models.generate_content(
#                 model=settings.gemini_model,
#                 contents=prompt,
#                 config={
#                     "response_mime_type": "application/json",
#                     "response_schema": DiagramAnalysis,
#                 },
#             )
#             elapsed_ms = int((time.monotonic() - start) * 1000)
#             if getattr(response, "parsed", None):
#                 try:
#                     node_count = len(response.parsed.nodes)
#                     conn_count = len(response.parsed.connections)
#                     cluster_count = len(response.parsed.clusters)
#                 except Exception:
#                     node_count = conn_count = cluster_count = -1
#                 self.logger.info(
#                     "LLM analysis parsed successfully in %d ms (nodes=%s, conns=%s, clusters=%s)",
#                     elapsed_ms,
#                     node_count,
#                     conn_count,
#                     cluster_count,
#                 )
#                 return response.parsed
#             else:
#                 raw_text = None
#                 # Best-effort extraction of text for debugging
#                 try:
#                     raw_text = getattr(response, "text", None)
#                     if not raw_text and hasattr(response, "candidates"):
#                         parts_text: list[str] = []
#                         for c in getattr(response, "candidates", []) or []:
#                             content = getattr(c, "content", None) or getattr(c, "output", None)
#                             if content and hasattr(content, "parts"):
#                                 for p in content.parts:
#                                     t = getattr(p, "text", None)
#                                     if t:
#                                         parts_text.append(t)
#                         raw_text = "\n".join(parts_text) if parts_text else None
#                 except Exception:
#                     # Ignore extraction errors
#                     pass
#                 preview = (raw_text or "<no-text>")[:500]
#                 self.logger.warning(
#                     "LLM response had no parsed content after %d ms. Preview: %s",
#                     elapsed_ms,
#                     preview,
#                 )
#                 raise ValueError("Failed to parse LLM response as JSON.")
#         except Exception as e:
#             self.logger.exception("LLM analysis failed; using heuristic fallback: %s", e)
#             return self._heuristic_analysis(description)

#     async def critique_analysis(
#         self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
#     ) -> DiagramCritique:
#         if settings.mock_llm:
#             return DiagramCritique(done=True, critique=None)

#         prompt = diagram_critique_prompt(description)
#         response = await client.aio.models.generate_content(
#             model=settings.gemini_model,
#             contents=[
#                 prompt,
#                 genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
#                 genai_types.Part.from_text(text=analysis.model_dump_json()),
#             ],
#             config={
#                 "response_mime_type": "application/json",
#                 "response_schema": DiagramCritique,
#             },
#         )
#         if getattr(response, "parsed", None):
#             return response.parsed
#         raise ValueError("Failed to parse critique as JSON.")

#     async def adjust_analysis(
#         self, description: str, analysis: DiagramAnalysis, critique: str
#     ) -> DiagramAnalysis:
#         if settings.mock_llm:
#             return analysis

#         prompt = diagram_adjustment_prompt(description, critique)
#         response = await client.aio.models.generate_content(
#             model=settings.gemini_model,
#             contents=[prompt, analysis.model_dump_json()],
#             config={
#                 "response_mime_type": "application/json",
#                 "response_schema": DiagramAnalysis,
#             },
#         )
#         if getattr(response, "parsed", None):
#             return response.parsed
#         raise ValueError("Failed to parse adjusted analysis as JSON.")

#     def _heuristic_analysis(self, description: str) -> DiagramAnalysis:
#         # AI-generated fallback, didn't check it thoroughly enough
#         """Very basic keyword-based extraction to enable a best-effort render when LLM is unavailable."""
#         text = description.lower()

#         nodes: list[AnalysisNode] = []
#         connections: list[AnalysisConnection] = []
#         clusters: list[AnalysisCluster] = []

#         def add(node_id: str, type_: str, label: str) -> None:
#             if not any(n.id == node_id for n in nodes):
#                 nodes.append(AnalysisNode(id=node_id, type=type_, label=label))

#         # Common components
#         if any(
#             k in text for k in ["alb", "load balancer", "application load balancer"]
#         ):
#             add("alb", "alb", "Application Load Balancer")
#         if any(k in text for k in ["api gateway", "apigateway", "gateway"]):
#             add("api_gw", "api_gateway", "API Gateway")
#         if any(k in text for k in ["ec2", "server", "service", "microservice", "web"]):
#             add("web1", "ec2", "Web / Service")
#         if "lambda" in text:
#             add("lambda", "lambda", "Lambda Function")
#         if any(k in text for k in ["rds", "postgres", "mysql", "database", "db"]):
#             add("db", "rds", "Database")
#         if any(k in text for k in ["s3", "bucket", "object storage"]):
#             add("s3", "s3", "S3 Bucket")
#         if "sqs" in text or "queue" in text:
#             add("queue", "sqs", "Queue")
#         if "sns" in text:
#             add("sns", "sns", "SNS Topic")
#         if any(k in text for k in ["cloudwatch", "monitoring", "metrics"]):
#             add("cw", "cloudwatch", "CloudWatch")

#         # Defaults if nothing matched
#         if not nodes:
#             add("alb", "alb", "Application Load Balancer")
#             add("web1", "ec2", "Web Server")
#             add("db", "rds", "Database")

#         # Simple connections
#         def maybe_connect(src: str, dst: str) -> None:
#             if any(n.id == src for n in nodes) and any(n.id == dst for n in nodes):
#                 connections.append(AnalysisConnection(source=src, target=dst))

#         maybe_connect("alb", "web1")
#         maybe_connect("api_gw", "web1")
#         maybe_connect("web1", "db")
#         maybe_connect("lambda", "db")
#         maybe_connect("web1", "queue")
#         maybe_connect("sns", "lambda")

#         # Optional clustering
#         if any(n.id == "web1" for n in nodes):
#             clusters.append(AnalysisCluster(label="Web Tier", nodes=["web1"]))

#         return DiagramAnalysis(
#             title="Heuristic Diagram",
#             nodes=nodes,
#             clusters=clusters,
#             connections=connections,
#         )

# diagram_agent.py
from __future__ import annotations

import time
from google.genai import types as genai_types

from app.core.config import settings
from app.core.llm import client
from app.core.logging import get_logger
from app.core.schemas import (
    AnalysisCluster,
    AnalysisConnection,
    AnalysisNode,
    DiagramAnalysis,
    DiagramCritique,
)
from app.core.prompts import (
    diagram_adjustment_prompt,
    diagram_analysis_prompt,
    diagram_critique_prompt,
)

__all__ = ["DiagramAgent"]


class DiagramAgent:
    """Agent for analyzing diagram descriptions and extracting components."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        """Generate diagram component analysis from description."""
        if settings.mock_llm:
            return DiagramAnalysis(
                title="Application Diagram",
                nodes=[
                    AnalysisNode(id="alb", type="alb", label="Application Load Balancer"),
                    AnalysisNode(id="web1", type="ec2", label="Web 1"),
                    AnalysisNode(id="db", type="rds", label="DB"),
                ],
                clusters=[AnalysisCluster(label="Web Tier", nodes=["web1"])],
                connections=[
                    AnalysisConnection(source="alb", target="web1"),
                    AnalysisConnection(source="web1", target="db"),
                ],
            )

        prompt = diagram_analysis_prompt(description)
        try:
            start = time.monotonic()
            self.logger.info(
                "Requesting LLM analysis (model=%s, desc_len=%d)",
                settings.gemini_model,
                len(description or ""),
            )
            response = await client.aio.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": DiagramAnalysis,
                },
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)
            if getattr(response, "parsed", None):
                try:
                    node_count = len(response.parsed.nodes)
                    conn_count = len(response.parsed.connections)
                    cluster_count = len(response.parsed.clusters)
                except Exception:
                    node_count = conn_count = cluster_count = -1
                self.logger.info(
                    "LLM analysis parsed successfully in %d ms (nodes=%s, conns=%s, clusters=%s)",
                    elapsed_ms,
                    node_count,
                    conn_count,
                    cluster_count,
                )
                return response.parsed
            else:
                raw_text = None
                try:
                    raw_text = getattr(response, "text", None)
                    if not raw_text and hasattr(response, "candidates"):
                        parts_text: list[str] = []
                        for c in getattr(response, "candidates", []) or []:
                            content = getattr(c, "content", None) or getattr(c, "output", None)
                            if content and hasattr(content, "parts"):
                                for p in content.parts:
                                    t = getattr(p, "text", None)
                                    if t:
                                        parts_text.append(t)
                        raw_text = "\n".join(parts_text) if parts_text else None
                except Exception:
                    pass
                preview = (raw_text or "<no-text>")[:500]
                self.logger.warning(
                    "LLM response had no parsed content after %d ms. Preview: %s",
                    elapsed_ms,
                    preview,
                )
                raise ValueError("Failed to parse LLM response as JSON.")
        except Exception as e:
            self.logger.exception("LLM analysis failed; using heuristic fallback: %s", e)
            return self._heuristic_analysis(description)

    async def critique_analysis(self, description: str, analysis: DiagramAnalysis, image_bytes: bytes) -> DiagramCritique:
        if settings.mock_llm:
            return DiagramCritique(done=True, critique=None)

        prompt = diagram_critique_prompt(description)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[
                prompt,
                genai_types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                genai_types.Part.from_text(text=analysis.model_dump_json()),
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": DiagramCritique,
            },
        )
        if getattr(response, "parsed", None):
            return response.parsed
        raise ValueError("Failed to parse critique as JSON.")

    async def adjust_analysis(self, description: str, analysis: DiagramAnalysis, critique: str) -> DiagramAnalysis:
        if settings.mock_llm:
            return analysis

        prompt = diagram_adjustment_prompt(description, critique)
        response = await client.aio.models.generate_content(
            model=settings.gemini_model,
            contents=[prompt, analysis.model_dump_json()],
            config={
                "response_mime_type": "application/json",
                "response_schema": DiagramAnalysis,
            },
        )
        if getattr(response, "parsed", None):
            return response.parsed
        raise ValueError("Failed to parse adjusted analysis as JSON.")

    # -------------------- heuristic fallback --------------------

    def _heuristic_analysis(self, description: str) -> DiagramAnalysis:
        """Very basic keyword-based extraction to enable a best-effort render when LLM is unavailable."""
        text = (description or "").lower()

        nodes: list[AnalysisNode] = []
        connections: list[AnalysisConnection] = []
        clusters: list[AnalysisCluster] = []

        def add(node_id: str, type_: str, label: str) -> None:
            if not any(n.id == node_id for n in nodes):
                nodes.append(AnalysisNode(id=node_id, type=type_, label=label))

        # Common components (canonical types only)
        if any(k in text for k in ["alb", "load balancer", "application load balancer"]):
            add("alb", "alb", "Application Load Balancer")
        if any(k in text for k in ["api gateway", "apigateway", "gateway"]):
            add("api_gw", "api_gateway", "API Gateway")
        if any(k in text for k in ["ec2", "server", "service", "microservice", "web"]):
            add("web1", "ec2", "Web / Service")
        if "lambda" in text:
            add("lambda", "lambda", "Lambda Function")
        if any(k in text for k in ["rds", "postgres", "mysql", "database", "db"]):
            add("db", "rds", "Database")
        if any(k in text for k in ["s3", "bucket", "object storage"]):
            add("s3", "s3", "S3 Bucket")
        if "sqs" in text or "queue" in text:
            add("queue", "sqs", "Queue")
        if "sns" in text:
            add("sns", "sns", "SNS Topic")
        if any(k in text for k in ["cloudwatch", "monitoring", "metrics"]):
            add("cw", "cloudwatch", "CloudWatch")
        if "cognito" in text or "auth" in text:
            add("cognito", "cognito", "Cognito")

        # Defaults if nothing matched
        if not nodes:
            add("alb", "alb", "Application Load Balancer")
            add("web1", "ec2", "Web Server")
            add("db", "rds", "Database")

        # Simple connections
        def maybe_connect(src: str, dst: str) -> None:
            if any(n.id == src for n in nodes) and any(n.id == dst for n in nodes):
                connections.append(AnalysisConnection(source=src, target=dst))

        maybe_connect("alb", "web1")
        maybe_connect("api_gw", "web1")
        maybe_connect("web1", "db")
        maybe_connect("lambda", "db")
        maybe_connect("web1", "queue")
        maybe_connect("sns", "lambda")

        # Optional clustering
        if any(n.id == "web1" for n in nodes):
            clusters.append(AnalysisCluster(label="Web Tier", nodes=["web1"]))

        return DiagramAnalysis(
            title="Heuristic Diagram",
            nodes=nodes,
            clusters=clusters,
            connections=connections,
        )