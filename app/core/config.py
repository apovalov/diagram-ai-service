from __future__ import annotations

import logging
from enum import Enum
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["settings", "Settings"]


class Settings(BaseSettings):
    """Enhanced application settings with strategy pattern and migration support."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # === NEW CORE EXECUTION SETTINGS ===
    execution_mode: str = Field(
        default="auto",
        description="Execution strategy: auto (recommended), original, langchain, langgraph"
    )
    
    enable_fallbacks: bool = Field(
        default=True,
        description="Enable automatic fallback to other strategies on failure"
    )

    # === LLM PROVIDER CONFIGURATION (UNCHANGED) ===
    llm_provider: str = Field(
        default="openai", description="LLM provider: 'openai' or 'gemini'"
    )

    # OpenAI Configuration
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (optional in mock mode)"
    )
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")

    # Gemini Configuration (kept for rollback)
    gemini_api_key: str | None = Field(
        default=None, description="Google Gemini API key (optional in mock mode)"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash", description="Gemini model name"
    )

    # === NEW PERFORMANCE SETTINGS ===
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed operations"
    )
    
    backoff_factor: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Exponential backoff factor for retries"
    )
    
    # === UNIFIED LLM CONFIGURATION (ENHANCED) ===
    llm_timeout: int = Field(
        default=60, 
        ge=10,
        le=300,
        description="Timeout for LLM requests in seconds"
    )
    llm_temperature: float = Field(
        default=0.1, description="Temperature parameter for LLM generation (0.0-2.0)"
    )
    
    # === NEW OBSERVABILITY & MONITORING ===
    langsmith_enabled: bool = Field(
        default=False,
        description="Enable LangSmith tracing and monitoring"
    )
    
    langsmith_project: str = Field(
        default="diagram-ai-service",
        description="LangSmith project name"
    )
    
    langsmith_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key (required if langsmith_enabled=True)"
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

    # === DEPRECATED FIELDS FOR BACKWARD COMPATIBILITY ===
    use_langchain: bool = Field(
        default=False,
        description="DEPRECATED: Use execution_mode='langchain' instead"
    )
    langchain_fallback: bool = Field(
        default=True,
        description="DEPRECATED: Use enable_fallbacks instead"
    )
    langchain_verbose: bool = Field(
        default=False, description="Enable LangChain verbose logging"
    )

    use_langgraph: bool = Field(
        default=False,
        description="DEPRECATED: Use execution_mode='langgraph' instead"
    )
    langgraph_fallback: bool = Field(
        default=True,
        description="DEPRECATED: Use enable_fallbacks instead"
    )
    use_checkpoints: bool = Field(
        default=True, description="Enable workflow state persistence"
    )
    enable_streaming: bool = Field(
        default=False, description="Enable real-time workflow streaming"
    )
    enable_human_review: bool = Field(
        default=False, description="Enable human-in-the-loop capabilities"
    )
    workflow_cache_ttl: int = Field(
        default=3600, description="Node cache TTL in seconds (0 to disable)"
    )

    @model_validator(mode='after')
    def validate_and_migrate_config(self) -> 'Settings':
        """Validate configuration and migrate deprecated settings."""
        
        # === MIGRATE DEPRECATED SETTINGS WITH WARNINGS ===
        # Check langchain first since it was the original migration flag
        if self.use_langchain and self.execution_mode == "auto":
            logger.warning(
                "DEPRECATED: use_langchain=true is deprecated. "
                "Use EXECUTION_MODE=langchain instead. "
                "The old setting will be removed in a future version."
            )
            self.execution_mode = "langchain"
            
        elif self.use_langgraph and self.execution_mode == "auto":
            logger.warning(
                "DEPRECATED: use_langgraph=true is deprecated. "
                "Use EXECUTION_MODE=langgraph instead. "
                "The old setting will be removed in a future version."
            )
            self.execution_mode = "langgraph"
        
        # Migrate fallback settings
        if not self.langchain_fallback or not self.langgraph_fallback:
            logger.warning(
                "DEPRECATED: langchain_fallback and langgraph_fallback are deprecated. "
                "Use ENABLE_FALLBACKS instead. "
                "The old settings will be removed in a future version."
            )
            if not self.langchain_fallback and not self.langgraph_fallback:
                self.enable_fallbacks = False

        # === VALIDATE API KEYS (EXISTING LOGIC) ===
        if not self.mock_llm:
            available_providers = []
            if self.openai_api_key:
                available_providers.append("openai")
            if self.gemini_api_key:
                available_providers.append("gemini")
            
            if not available_providers:
                raise ValueError(
                    "At least one API key must be provided when mock_llm=False. "
                    "Set OPENAI_API_KEY or GEMINI_API_KEY"
                )
            
            # Warn if primary provider has no key
            provider_key_map = {
                "openai": self.openai_api_key,
                "gemini": self.gemini_api_key,
            }
            
            if self.llm_provider in provider_key_map and not provider_key_map[self.llm_provider]:
                logger.warning(
                    f"Primary provider '{self.llm_provider}' has no API key. "
                    f"Available providers: {available_providers}"
                )

        # === VALIDATE LANGSMITH ===
        if self.langsmith_enabled and not self.langsmith_api_key:
            logger.warning(
                "LangSmith enabled but no API key provided. "
                "Set LANGSMITH_API_KEY or disable with LANGSMITH_ENABLED=false"
            )

        # === LOG CONFIGURATION SUMMARY ===
        logger.info(
            f"Configuration loaded: execution_mode={self.execution_mode}, "
            f"provider={self.llm_provider}, fallbacks={self.enable_fallbacks}, "
            f"langsmith={self.langsmith_enabled}, mock={self.mock_llm}"
        )

        return self

    @property
    def is_production_ready(self) -> bool:
        """Check if configuration is suitable for production."""
        return (
            not self.mock_llm
            and (self.openai_api_key or self.gemini_api_key)
            and self.llm_timeout >= 30
        )


# Global settings instance (UNCHANGED)
settings = Settings()
