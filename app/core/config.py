from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["settings", "Settings"]


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # LLM Provider Configuration
    llm_provider: str = Field(
        default="openai", description="LLM provider: 'openai' or 'gemini'"
    )

    # OpenAI Configuration
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (optional in mock mode)"
    )
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openai_timeout: int = Field(
        default=60, description="Timeout in seconds for OpenAI requests"
    )
    openai_temperature: float = Field(
        default=0.1, description="Temperature parameter for OpenAI generation (0.0-2.0)"
    )

    # Gemini Configuration (kept for rollback)
    gemini_api_key: str | None = Field(
        default=None, description="Google Gemini API key (optional in mock mode)"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash", description="Gemini model name"
    )
    gemini_timeout: int = Field(
        default=60, description="Timeout in seconds for LLM requests"
    )
    gemini_temperature: float = Field(
        default=0.1, description="Temperature parameter for LLM generation (0.0-2.0)"
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
    use_critique_generation: bool = Field(
        default=True,
        description="If true, use critique-enhanced diagram generation; otherwise use standard generation",
    )
    critique_max_attempts: int = Field(
        default=3,
        description="Maximum number of attempts for critique generation (1-5)",
    )
    analysis_max_attempts: int = Field(default=2, description="Retries for analysis")
    adjust_max_attempts: int = Field(default=2, description="Retries for adjust")
    retry_backoff_base: float = Field(
        default=0.5, description="Exponential backoff base (s)"
    )
    retry_backoff_max: float = Field(default=4.0, description="Backoff cap (s)")
    retry_jitter: float = Field(default=0.25, description="Jitter (s)")
    google_cloud_project: str = Field(
        default="", description="Google Cloud Project ID for Vertex AI"
    )
    google_cloud_location: str = Field(
        default="us-central1", description="Google Cloud location for Vertex AI"
    )


# Global settings instance
settings = Settings()
