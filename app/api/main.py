from __future__ import annotations

import time
import uuid
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import Settings, settings
from app.core.logging import get_logger, setup_logging
from app.core.schemas import (
    AssistantRequest,
    AssistantResponse,
    DiagramMetadata,
    DiagramRequest,
    DiagramResponse,
)
from app.services.assistant_service import AssistantService
from app.services.diagram_service import DiagramService
from app.utils.files import save_image_base64

# Setup logging
setup_logging()
logger = get_logger(__name__)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Optional monitoring middleware that never breaks core functionality."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.metrics = None
        
        # Only initialize if monitoring is enabled
        if getattr(settings, 'langsmith_enabled', False):
            self._setup_monitoring()

    def _setup_monitoring(self):
        """Setup monitoring components safely."""
        try:
            from ..core.langsmith_config import LangSmithManager
            from ..core.langsmith_metrics import DiagramMetrics
            
            manager = LangSmithManager(self.settings)
            if manager.is_enabled():
                self.metrics = DiagramMetrics(manager.client)
                logger.info("API monitoring middleware enabled")
        except Exception as e:
            logger.info(f"Monitoring middleware initialization failed: {e}")
            self.metrics = None

    async def dispatch(self, request: Request, call_next):
        """Process request with optional monitoring."""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Add request ID to headers for tracing
        request.state.request_id = request_id

        try:
            # Call the next middleware/handler
            response = await call_next(request)
            
            # Log successful request if monitoring is enabled
            if self.metrics:
                duration_ms = (time.time() - start_time) * 1000
                await self._log_request_success(request, response, duration_ms, request_id)
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response
            
        except Exception as e:
            # Log error if monitoring is enabled
            if self.metrics:
                duration_ms = (time.time() - start_time) * 1000
                await self._log_request_error(request, e, duration_ms, request_id)
            
            # Re-raise the exception to maintain normal error handling
            raise

    async def _log_request_success(self, request: Request, response: Response, duration_ms: float, request_id: str):
        """Log successful request safely."""
        try:
            await self.metrics.log_diagram_generation(
                request_id=request_id,
                description=f"{request.method} {request.url.path}",
                execution_mode="api_request",
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "method": request.method,
                    "path": str(request.url.path),
                    "status_code": response.status_code,
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            )
        except Exception as e:
            logger.debug(f"Failed to log request success: {e}")

    async def _log_request_error(self, request: Request, error: Exception, duration_ms: float, request_id: str):
        """Log request error safely."""
        try:
            await self.metrics.log_error(
                request_id=request_id,
                operation="api_request",
                error=error,
                execution_mode="api_request",
                metadata={
                    "method": request.method,
                    "path": str(request.url.path),
                    "duration_ms": duration_ms,
                    "user_agent": request.headers.get("user-agent", "unknown")
                }
            )
        except Exception as e:
            logger.debug(f"Failed to log request error: {e}")


# Initialize FastAPI app
app = FastAPI(title="Diagram API Service", version="0.1.0")

# Add monitoring middleware if enabled (this is safe - it handles its own failures)
try:
    if getattr(settings, 'langsmith_enabled', False):
        app.add_middleware(MonitoringMiddleware, settings=settings)
        logger.info("Added monitoring middleware")
except Exception as e:
    logger.warning(f"Failed to add monitoring middleware: {e}")


def get_settings() -> Settings:
    """Dependency to get application settings."""
    return settings


def get_diagram_service(settings: Settings = Depends(get_settings)) -> DiagramService:
    """Dependency to get diagram service."""
    return DiagramService(settings)


def get_assistant_service(
    settings: Settings = Depends(get_settings),
) -> AssistantService:
    """Dependency to get assistant service."""
    return AssistantService(settings)


@app.post("/api/v1/generate-diagram", response_model=DiagramResponse)
async def generate_diagram(
    request: DiagramRequest,
    diagram_service: DiagramService = Depends(get_diagram_service),
    settings: Settings = Depends(get_settings),
):
    """
    Generate diagram image from natural language description.
    """
    if not request.description:
        raise HTTPException(
            status_code=400, detail="Invalid diagram description provided"
        )

    try:
        image_data, metadata = await diagram_service.generate_diagram_from_description(
            request.description
        )

        image_url: str | None = save_image_base64(
            image_data=image_data,
            settings=settings,
            file_extension=request.format,
        )

        return DiagramResponse(
            success=True,
            image_data=image_data,
            image_url=image_url,
            metadata=DiagramMetadata(**metadata),
        )
    except Exception as e:
        logger.error(f"Error generating diagram: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/assistant", response_model=AssistantResponse)
async def assistant(
    request: AssistantRequest,
    assistant_service: AssistantService = Depends(get_assistant_service),
):
    """
    Assistant-style endpoint with context awareness.
    """
    if not request.message:
        raise HTTPException(status_code=400, detail="Invalid message provided")

    return await assistant_service.process_message(request)
