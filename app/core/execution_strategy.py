"""Strategy pattern for execution modes with automatic selection."""

from enum import Enum
from typing import Protocol, Type, List, Dict, Any
from abc import abstractmethod
import logging

from .config import Settings
from .exceptions import ConfigurationError
from .schemas import DiagramAnalysis, DiagramCritique

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution strategies for diagram generation."""

    ORIGINAL = "original"
    LANGCHAIN = "langchain"
    LANGGRAPH = "langgraph"
    AUTO = "auto"  # Intelligent automatic selection


class AgentStrategy(Protocol):
    """Protocol defining the agent strategy interface."""

    @abstractmethod
    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        """Generate diagram analysis from description."""
        ...

    @abstractmethod
    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        """Critique the diagram analysis."""
        ...

    @abstractmethod
    async def adjust_analysis(
        self, description: str, analysis: DiagramAnalysis, critique: str
    ) -> DiagramAnalysis:
        """Adjust analysis based on critique."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging."""
        ...


class StrategyFactory:
    """Factory for creating and managing execution strategies."""

    @staticmethod
    def create_agent_strategy(mode: ExecutionMode, settings: Settings) -> AgentStrategy:
        """Create appropriate agent strategy based on mode and settings."""

        if mode == ExecutionMode.AUTO:
            mode = StrategyFactory._auto_select_mode(settings)
            logger.info(f"AUTO mode selected: {mode.value}")

        strategies = {
            ExecutionMode.ORIGINAL: lambda s: OriginalAgentStrategy(s),
            ExecutionMode.LANGCHAIN: lambda s: LangChainAgentStrategy(s),
            ExecutionMode.LANGGRAPH: lambda s: LangGraphAgentStrategy(s),
        }

        strategy_factory = strategies.get(mode)
        if not strategy_factory:
            raise ConfigurationError(f"Unknown execution mode: {mode}")

        try:
            return strategy_factory(settings)
        except ImportError as e:
            # If strategy dependencies are missing, try fallback
            logger.warning(f"Strategy {mode.value} unavailable: {e}")
            if mode != ExecutionMode.ORIGINAL:
                logger.info("Falling back to original strategy")
                return OriginalAgentStrategy(settings)
            raise ConfigurationError(f"No execution strategies available: {e}")

    @staticmethod
    def _auto_select_mode(settings: Settings) -> ExecutionMode:
        """Intelligently select the best execution mode."""

        try:
            # Try LangGraph first (most advanced)
            if hasattr(settings, "langsmith_enabled") and settings.langsmith_enabled:
                try:
                    import langgraph

                    return ExecutionMode.LANGGRAPH
                except ImportError:
                    pass

            # Try LangChain next
            try:
                import langchain_core

                if settings.llm_provider in ["openai", "anthropic"]:
                    return ExecutionMode.LANGCHAIN
            except ImportError:
                pass

            # Fallback to original
            return ExecutionMode.ORIGINAL

        except Exception:
            return ExecutionMode.ORIGINAL

    @staticmethod
    def get_fallback_chain(
        primary_mode: ExecutionMode, settings: Settings
    ) -> List[ExecutionMode]:
        """Get fallback chain for the given primary mode."""

        base_chains = {
            ExecutionMode.LANGGRAPH: [
                ExecutionMode.LANGGRAPH,
                ExecutionMode.LANGCHAIN,
                ExecutionMode.ORIGINAL,
            ],
            ExecutionMode.LANGCHAIN: [ExecutionMode.LANGCHAIN, ExecutionMode.ORIGINAL],
            ExecutionMode.ORIGINAL: [ExecutionMode.ORIGINAL],
            ExecutionMode.AUTO: [
                ExecutionMode.LANGGRAPH,
                ExecutionMode.LANGCHAIN,
                ExecutionMode.ORIGINAL,
            ],
        }

        chain = base_chains[primary_mode]

        # Filter to only available strategies
        available_chain = []
        for mode in chain:
            try:
                StrategyFactory.create_agent_strategy(mode, settings)
                available_chain.append(mode)
            except (ConfigurationError, ImportError):
                continue

        if not available_chain:
            # Always have original as final fallback
            available_chain = [ExecutionMode.ORIGINAL]

        return available_chain


# Strategy implementations (wrappers around existing agents)
class OriginalAgentStrategy:
    """Strategy using the original Python implementation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from ..agents.diagram_agent import DiagramAgent

        self.agent = DiagramAgent(settings)

    @property
    def name(self) -> str:
        return "original"

    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        return await self.agent.generate_analysis(description)

    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        return await self.agent.critique_analysis(description, analysis, image_bytes)

    async def adjust_analysis(
        self, description: str, analysis: DiagramAnalysis, critique: str
    ) -> DiagramAnalysis:
        return await self.agent.adjust_analysis(description, analysis, critique)


class LangChainAgentStrategy:
    """Strategy using LangChain implementation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from ..agents.langchain_diagram_agent import LangChainDiagramAgent

        self.agent = LangChainDiagramAgent(settings)

    @property
    def name(self) -> str:
        return "langchain"

    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        return await self.agent.generate_analysis(description)

    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        return await self.agent.critique_analysis(description, analysis, image_bytes)

    async def adjust_analysis(
        self, description: str, analysis: DiagramAnalysis, critique: str
    ) -> DiagramAnalysis:
        return await self.agent.adjust_analysis(description, critique, analysis)


class LangGraphAgentStrategy:
    """Strategy using LangGraph workflow implementation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        from ..services.langgraph_diagram_service import LangGraphDiagramService

        self.service = LangGraphDiagramService(settings)

    @property
    def name(self) -> str:
        return "langgraph"

    async def generate_analysis(self, description: str) -> DiagramAnalysis:
        # For individual analysis, use workflow nodes directly
        from ..workflows.nodes.analysis import analyze_node
        from ..workflows.langgraph_config import create_initial_state

        initial_state = create_initial_state(description, self.settings)
        result = await analyze_node(initial_state)
        return result.get("analysis")

    async def critique_analysis(
        self, description: str, analysis: DiagramAnalysis, image_bytes: bytes
    ) -> DiagramCritique:
        # Similar approach for critique
        from ..workflows.nodes.critique import critique_node
        from ..workflows.langgraph_config import create_initial_state
        import base64

        initial_state = create_initial_state(description, self.settings)
        initial_state["analysis"] = analysis
        initial_state["image_before"] = base64.b64encode(image_bytes).decode()

        result = await critique_node(initial_state)
        return result.get("critique")

    async def adjust_analysis(
        self, description: str, analysis: DiagramAnalysis, critique: str
    ) -> DiagramAnalysis:
        # Adjust using workflow node
        from ..workflows.nodes.adjust import adjust_node
        from ..workflows.langgraph_config import create_initial_state
        from ..core.schemas import DiagramCritique

        initial_state = create_initial_state(description, self.settings)
        initial_state["analysis"] = analysis
        initial_state["critique"] = DiagramCritique(done=False, critique=critique)

        result = await adjust_node(initial_state)
        return result.get("analysis")