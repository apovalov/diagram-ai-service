# diagram_agent.py
from __future__ import annotations

import asyncio
import base64
import random
import time

from app.core.config import settings
from app.core.constants import AWS_COMPONENTS
from app.core.llm import client
from app.core.logging import get_logger
from app.core.prompts import (
    diagram_adjustment_prompt,
    diagram_analysis_prompt,
    diagram_critique_prompt,
)
from app.core.schemas import (
    AnalysisCluster,
    AnalysisConnection,
    AnalysisNode,
    DiagramAnalysis,
    DiagramCritique,
)
from app.core.structured import ask_structured, ask_structured_vision

__all__ = ["DiagramAgent"]


class DiagramAgent:
    """Agent for analyzing diagram descriptions and extracting components."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    async def _retry_with_backoff(
        self, operation_name: str, max_attempts: int, operation_func
    ):
        """Retry an operation with exponential backoff and jitter."""
        for attempt in range(max_attempts):
            try:
                result = await operation_func()
                return result
            except Exception as e:
                is_last_attempt = attempt == max_attempts - 1
                if is_last_attempt:
                    self.logger.error(
                        f"{operation_name} failed after {max_attempts} attempts: {e}"
                    )
                    raise
                # Calculate backoff with exponential growth and jitter
                backoff_time = min(
                    settings.retry_backoff_base * (2**attempt),
                    settings.retry_backoff_max,
                ) + random.uniform(0, settings.retry_jitter)
                await asyncio.sleep(backoff_time)

    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        """Generate diagram component analysis from description."""
        if settings.mock_llm:
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

        async def _perform_analysis():
            prompt = diagram_analysis_prompt(description)

            start = time.monotonic()
            model_name = (
                settings.openai_model
                if settings.llm_provider == "openai"
                else settings.gemini_model
            )
            self.logger.info(
                "Requesting LLM analysis (provider=%s, model=%s, desc_len=%d)",
                settings.llm_provider,
                model_name,
                len(description or ""),
            )

            if settings.llm_provider == "openai":
                response = await ask_structured(prompt, DiagramAnalysis)
                elapsed_ms = int((time.monotonic() - start) * 1000)

                try:
                    node_count = len(response.nodes)
                    conn_count = len(response.connections)
                    cluster_count = len(response.clusters)
                except Exception:
                    node_count = conn_count = cluster_count = -1

                self.logger.info(
                    "LLM analysis parsed successfully in %d ms (nodes=%s, conns=%s, clusters=%s)",
                    elapsed_ms,
                    node_count,
                    conn_count,
                    cluster_count,
                )
                return response
            else:  # Gemini fallback
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=settings.gemini_model,
                        contents=prompt,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": DiagramAnalysis,
                            "temperature": settings.llm_temperature,
                        },
                    ),
                    timeout=settings.llm_timeout,
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
                                content = getattr(c, "content", None) or getattr(
                                    c, "output", None
                                )
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

        try:
            return await self._retry_with_backoff(
                "analysis", settings.analysis_max_attempts, _perform_analysis
            )
        except Exception as e:
            self.logger.exception(
                "LLM analysis failed after retries; using heuristic fallback: %s", e
            )
            return self._heuristic_analysis(description)

    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        if settings.mock_llm:
            return DiagramCritique(done=True, critique=None)

        prompt = diagram_critique_prompt(description)

        if settings.llm_provider == "openai":
            # Convert image bytes to base64 data URL
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            image_data_url = f"data:image/png;base64,{image_b64}"

            return await ask_structured_vision(
                text_prompt=prompt,
                image_data_url=image_data_url,
                analysis_json=analysis.model_dump_json(),
                schema_cls=DiagramCritique,
            )
        else:  # Gemini fallback
            from google.genai import types as genai_types

            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=settings.gemini_model,
                    contents=[
                        prompt,
                        genai_types.Part.from_bytes(
                            data=image_bytes, mime_type="image/png"
                        ),
                        genai_types.Part.from_text(text=analysis.model_dump_json()),
                    ],
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": DiagramCritique,
                        "temperature": settings.llm_temperature,
                    },
                ),
                timeout=settings.llm_timeout,
            )

            if getattr(response, "parsed", None):
                return response.parsed
            raise ValueError("Failed to parse critique as JSON.")

    async def adjust_analysis(
        self, description: str, analysis: DiagramAnalysis, critique: str
    ) -> DiagramAnalysis:
        if settings.mock_llm:
            return analysis

        async def _perform_adjust():
            prompt = diagram_adjustment_prompt(description, critique)
            full_prompt = f"{prompt}\n\nOriginal analysis: {analysis.model_dump_json()}"

            if settings.llm_provider == "openai":
                return await ask_structured(full_prompt, DiagramAnalysis)
            else:  # Gemini fallback
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=settings.gemini_model,
                        contents=[prompt, analysis.model_dump_json()],
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": DiagramAnalysis,
                            "temperature": settings.llm_temperature,
                        },
                    ),
                    timeout=settings.llm_timeout,
                )

                if getattr(response, "parsed", None):
                    return response.parsed
                raise ValueError("Failed to parse adjusted analysis as JSON.")

        return await self._retry_with_backoff(
            "adjust", settings.adjust_max_attempts, _perform_adjust
        )

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
        if any(k in text for k in AWS_COMPONENTS["alb"]["keywords"]):
            add("alb", AWS_COMPONENTS["alb"]["type"], AWS_COMPONENTS["alb"]["label"])
        if any(k in text for k in AWS_COMPONENTS["api_gateway"]["keywords"]):
            add(
                "api_gw",
                AWS_COMPONENTS["api_gateway"]["type"],
                AWS_COMPONENTS["api_gateway"]["label"],
            )
        if any(k in text for k in AWS_COMPONENTS["ec2"]["keywords"]):
            add("web1", AWS_COMPONENTS["ec2"]["type"], AWS_COMPONENTS["ec2"]["label"])
        if any(k in text for k in AWS_COMPONENTS["lambda"]["keywords"]):
            add(
                "lambda",
                AWS_COMPONENTS["lambda"]["type"],
                AWS_COMPONENTS["lambda"]["label"],
            )
        if any(k in text for k in AWS_COMPONENTS["rds"]["keywords"]):
            add("db", AWS_COMPONENTS["rds"]["type"], AWS_COMPONENTS["rds"]["label"])
        if any(k in text for k in AWS_COMPONENTS["s3"]["keywords"]):
            add("s3", AWS_COMPONENTS["s3"]["type"], AWS_COMPONENTS["s3"]["label"])
        if any(k in text for k in AWS_COMPONENTS["sqs"]["keywords"]):
            add("queue", AWS_COMPONENTS["sqs"]["type"], AWS_COMPONENTS["sqs"]["label"])
        if any(k in text for k in AWS_COMPONENTS["sns"]["keywords"]):
            add("sns", AWS_COMPONENTS["sns"]["type"], AWS_COMPONENTS["sns"]["label"])
        if any(k in text for k in AWS_COMPONENTS["cloudwatch"]["keywords"]):
            add(
                "cw",
                AWS_COMPONENTS["cloudwatch"]["type"],
                AWS_COMPONENTS["cloudwatch"]["label"],
            )
        if any(k in text for k in AWS_COMPONENTS["cognito"]["keywords"]):
            add(
                "cognito",
                AWS_COMPONENTS["cognito"]["type"],
                AWS_COMPONENTS["cognito"]["label"],
            )

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
