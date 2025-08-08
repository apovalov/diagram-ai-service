from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["settings", "Settings"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    gemini_api_key: str | None = Field(
        default=None, description="Google Gemini API key (optional in mock mode)"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash", description="Gemini model name"
    )
    tmp_dir: str = Field(
        default="/tmp/diagrams", description="Temporary directory for diagram files"
    )
    use_vertex_ai: bool = Field(
        default=False, description="Use Vertex AI instead of Gemini Developer API"
    )
    mock_llm: bool = Field(
        default=False,
        description="If true, avoid external LLM calls and use deterministic local mocks",
    )
    google_cloud_project: str = Field(
        default="", description="Google Cloud Project ID for Vertex AI"
    )
    google_cloud_location: str = Field(
        default="us-central1", description="Google Cloud location for Vertex AI"
    )


# Global settings instance
settings = Settings()
