"""LangSmith configuration and setup with safe initialization."""

import logging
import os
from typing import Optional, Any, Dict
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)


class LangSmithManager:
    """Safe LangSmith manager that gracefully handles missing dependencies."""

    def __init__(self, settings):
        self.settings = settings
        self.client: Optional[Any] = None
        self.enabled = False
        
        if getattr(settings, 'langsmith_enabled', False):
            self._initialize_langsmith()
    
    def _initialize_langsmith(self):
        """Initialize LangSmith with safe error handling."""
        try:
            # Only import when needed to avoid dependency errors
            from langsmith import Client
            import langsmith
            
            api_key = getattr(self.settings, 'langsmith_api_key', None)
            if not api_key:
                logger.warning("LangSmith API key not provided, monitoring disabled")
                return
            
            # Set environment variables for LangSmith
            os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
            os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
            os.environ.setdefault("LANGCHAIN_PROJECT", 
                                getattr(self.settings, 'langsmith_project', 'diagram-ai-service'))
            
            # Initialize client
            self.client = Client(
                api_key=api_key,
                api_url=os.environ.get("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            )
            
            # Test connection
            try:
                # Simple health check
                self.client.list_runs(limit=1)
                self.enabled = True
                logger.info(f"LangSmith initialized successfully (project: {os.environ.get('LANGCHAIN_PROJECT')})")
            except Exception as e:
                logger.warning(f"LangSmith connection test failed: {e}")
                self.enabled = False
                
        except ImportError as e:
            logger.info(f"LangSmith dependencies not available: {e}")
            self.enabled = False
        except Exception as e:
            logger.warning(f"Failed to initialize LangSmith: {e}")
            self.enabled = False
    
    def is_enabled(self) -> bool:
        """Check if LangSmith is properly initialized and enabled."""
        return self.enabled and self.client is not None
    
    @contextmanager
    def trace_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracing operations."""
        if not self.is_enabled():
            # No-op context manager when disabled
            yield None
            return
            
        try:
            from langsmith import traceable
            
            # Use LangSmith's tracing
            with traceable(
                name=operation_name,
                metadata=metadata or {},
                run_type="chain"
            ) as tracer:
                yield tracer
                
        except Exception as e:
            logger.debug(f"Tracing failed for {operation_name}: {e}")
            # Yield None to ensure code continues working
            yield None
    
    @asynccontextmanager
    async def async_trace_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Async context manager for tracing operations."""
        if not self.is_enabled():
            # No-op async context manager when disabled
            yield None
            return
            
        try:
            from langsmith import traceable
            
            # Use LangSmith's async tracing
            async with traceable(
                name=operation_name,
                metadata=metadata or {},
                run_type="chain"
            ) as tracer:
                yield tracer
                
        except Exception as e:
            logger.debug(f"Async tracing failed for {operation_name}: {e}")
            # Yield None to ensure code continues working
            yield None
    
    def create_run(
        self, 
        name: str, 
        run_type: str = "chain",
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a run and return run ID, or None if disabled."""
        if not self.is_enabled():
            return None
            
        try:
            run = self.client.create_run(
                name=name,
                run_type=run_type,
                inputs=inputs or {},
                project_name=os.environ.get('LANGCHAIN_PROJECT', 'diagram-ai-service'),
                extra=metadata or {}
            )
            return str(run.id) if run else None
            
        except Exception as e:
            logger.debug(f"Failed to create run for {name}: {e}")
            return None
    
    def update_run(
        self,
        run_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        end_time: Optional[float] = None
    ) -> bool:
        """Update a run with outputs/error, return success status."""
        if not self.is_enabled() or not run_id:
            return False
            
        try:
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                error=error,
                end_time=end_time
            )
            return True
            
        except Exception as e:
            logger.debug(f"Failed to update run {run_id}: {e}")
            return False


# Global instance - will be None if LangSmith is not available
_langsmith_manager: Optional[LangSmithManager] = None


def get_langsmith_manager(settings=None) -> Optional[LangSmithManager]:
    """Get global LangSmith manager instance."""
    global _langsmith_manager
    
    if _langsmith_manager is None and settings:
        _langsmith_manager = LangSmithManager(settings)
    
    return _langsmith_manager


def is_langsmith_available() -> bool:
    """Check if LangSmith is available and enabled."""
    try:
        import langsmith
        return True
    except ImportError:
        return False