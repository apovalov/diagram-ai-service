from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException

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

app = FastAPI(title="Diagram API Service", version="0.1.0")


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
