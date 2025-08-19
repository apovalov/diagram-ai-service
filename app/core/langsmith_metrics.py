"""LangSmith metrics system that never fails core functionality."""

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)


class DiagramMetrics:
    """Safe metrics collector that never breaks core functionality."""

    def __init__(self, langsmith_client=None):
        self.client = langsmith_client
        self.enabled = langsmith_client is not None
        self._local_fallback = LocalMetricsCollector()

    async def log_diagram_generation(
        self,
        request_id: str,
        description: str,
        execution_mode: str,
        success: bool,
        duration_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log diagram generation metrics safely."""
        metrics_data = {
            "request_id": request_id,
            "description_length": len(description),
            "execution_mode": execution_mode,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Always log locally as fallback
        success_local = await self._local_fallback.log_generation(metrics_data)

        # Try LangSmith if available
        success_remote = True
        if self.enabled:
            success_remote = await self._safe_log_to_langsmith(
                "diagram_generation",
                metrics_data
            )

        return success_local or success_remote

    async def log_error(
        self,
        request_id: str,
        operation: str,
        error: Union[Exception, str],
        execution_mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Log error metrics safely."""
        error_data = {
            "request_id": request_id,
            "operation": operation,
            "error_type": type(error).__name__ if isinstance(error, Exception) else "Unknown",
            "error_message": str(error),
            "execution_mode": execution_mode,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }

        # Always log locally as fallback
        success_local = await self._local_fallback.log_error(error_data)

        # Try LangSmith if available
        success_remote = True
        if self.enabled:
            success_remote = await self._safe_log_to_langsmith(
                "diagram_error",
                error_data
            )

        return success_local or success_remote

    async def log_strategy_performance(
        self,
        strategy: str,
        success_rate: float,
        avg_duration_ms: float,
        total_requests: int,
        time_window: str = "1h"
    ) -> bool:
        """Log strategy performance metrics safely."""
        performance_data = {
            "strategy": strategy,
            "success_rate": success_rate,
            "avg_duration_ms": avg_duration_ms,
            "total_requests": total_requests,
            "time_window": time_window,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Always log locally as fallback
        success_local = await self._local_fallback.log_performance(performance_data)

        # Try LangSmith if available
        success_remote = True
        if self.enabled:
            success_remote = await self._safe_log_to_langsmith(
                "strategy_performance",
                performance_data
            )

        return success_local or success_remote

    async def _safe_log_to_langsmith(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Safely log to LangSmith with comprehensive error handling."""
        if not self.enabled or not self.client:
            return False

        try:
            # Use asyncio to avoid blocking
            loop = asyncio.get_event_loop()

            # Create run in LangSmith
            def sync_log():
                try:
                    run_id = self.client.create_run(
                        name=event_type,
                        run_type="tool",
                        inputs={"event_data": data},
                        project_name="diagram-ai-service"
                    )
                    return run_id is not None
                except Exception as e:
                    logger.debug(f"LangSmith sync logging failed: {e}")
                    return False

            # Run in thread pool to avoid blocking async operations
            success = await loop.run_in_executor(None, sync_log)
            
            if success:
                logger.debug(f"Successfully logged {event_type} to LangSmith")
            
            return success

        except Exception as e:
            # Log the error but don't let it break anything
            logger.debug(f"Failed to log {event_type} to LangSmith: {e}")
            return False

    def create_trace(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Create a trace context that never fails."""
        return SafeTraceContext(self, operation_name, metadata)


class SafeTraceContext:
    """A trace context that never raises exceptions."""

    def __init__(self, metrics_client: DiagramMetrics, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        self.metrics_client = metrics_client
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
        self.trace_id = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.start_time = time.time()
        try:
            if self.metrics_client.enabled:
                # Try to create LangSmith trace
                self.trace_id = await self._create_trace()
        except Exception as e:
            logger.debug(f"Failed to create trace for {self.operation_name}: {e}")
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        try:
            duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0
            
            if exc_type:
                # Log error if operation failed
                await self.metrics_client.log_error(
                    request_id=str(self.trace_id) if self.trace_id else "unknown",
                    operation=self.operation_name,
                    error=exc_val or exc_type.__name__,
                    metadata=self.metadata
                )
            else:
                # Log success
                await self.metrics_client._local_fallback.log_operation({
                    "operation": self.operation_name,
                    "duration_ms": duration_ms,
                    "success": True,
                    "metadata": self.metadata,
                    "trace_id": self.trace_id
                })
        except Exception as e:
            logger.debug(f"Failed to finalize trace for {self.operation_name}: {e}")

    async def _create_trace(self):
        """Create trace in LangSmith if available."""
        try:
            if self.metrics_client.client:
                run = self.metrics_client.client.create_run(
                    name=self.operation_name,
                    run_type="chain",
                    inputs={"metadata": self.metadata}
                )
                return str(run.id) if run else None
        except Exception as e:
            logger.debug(f"Failed to create LangSmith trace: {e}")
        return None


class LocalMetricsCollector:
    """Local fallback metrics collector for when LangSmith is unavailable."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LocalMetrics")

    async def log_generation(self, data: Dict[str, Any]) -> bool:
        """Log generation metrics locally."""
        try:
            self.logger.info(
                f"Diagram Generation - "
                f"Mode: {data.get('execution_mode', 'unknown')}, "
                f"Success: {data.get('success', False)}, "
                f"Duration: {data.get('duration_ms', 0):.1f}ms, "
                f"Request: {data.get('request_id', 'unknown')}"
            )
            return True
        except Exception as e:
            logger.debug(f"Local generation logging failed: {e}")
            return False

    async def log_error(self, data: Dict[str, Any]) -> bool:
        """Log error metrics locally."""
        try:
            self.logger.warning(
                f"Diagram Error - "
                f"Operation: {data.get('operation', 'unknown')}, "
                f"Type: {data.get('error_type', 'unknown')}, "
                f"Message: {data.get('error_message', 'unknown')[:100]}, "
                f"Request: {data.get('request_id', 'unknown')}"
            )
            return True
        except Exception as e:
            logger.debug(f"Local error logging failed: {e}")
            return False

    async def log_performance(self, data: Dict[str, Any]) -> bool:
        """Log performance metrics locally."""
        try:
            self.logger.info(
                f"Strategy Performance - "
                f"Strategy: {data.get('strategy', 'unknown')}, "
                f"Success Rate: {data.get('success_rate', 0):.2%}, "
                f"Avg Duration: {data.get('avg_duration_ms', 0):.1f}ms, "
                f"Requests: {data.get('total_requests', 0)}"
            )
            return True
        except Exception as e:
            logger.debug(f"Local performance logging failed: {e}")
            return False

    async def log_operation(self, data: Dict[str, Any]) -> bool:
        """Log general operation metrics locally."""
        try:
            self.logger.debug(
                f"Operation - "
                f"Name: {data.get('operation', 'unknown')}, "
                f"Duration: {data.get('duration_ms', 0):.1f}ms, "
                f"Success: {data.get('success', False)}"
            )
            return True
        except Exception as e:
            logger.debug(f"Local operation logging failed: {e}")
            return False