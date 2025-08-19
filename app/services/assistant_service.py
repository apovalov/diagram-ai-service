from __future__ import annotations

import asyncio
import json

from app.agents.assistant_agent import AssistantAgent
from app.agents.langchain_assistant_agent import LangChainAssistantAgent
from app.core.config import Settings
from app.core.constants import IntentType
from app.core.llm import client
from app.core.logging import get_logger
from app.core.schemas import (
    AssistantFinal,
    AssistantRequest,
    AssistantResponse,
    IntentResult,
)
from app.services.diagram_service import DiagramService

__all__ = ["AssistantService"]

logger = get_logger(__name__)


class AssistantService:
    """Service for handling assistant conversations and routing to diagram generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        # Initialize assistant agents based on feature flag
        if self.settings.use_langchain:
            self.assistant_agent = LangChainAssistantAgent()
            self.original_assistant_agent = AssistantAgent()  # Keep as fallback
            logger.info("Using LangChain-based assistant agent")
        else:
            self.assistant_agent = AssistantAgent()
            self.original_assistant_agent = None
            logger.info("Using original assistant agent")

        self.diagram_service = DiagramService(settings)
        # Simple in-memory conversation store (for stateless service with session-like behavior)
        self._conversation_context: dict[str, dict] = {}

    async def process_message(self, request: AssistantRequest) -> AssistantResponse:
        # Handle conversation context and memory
        conversation_id = request.conversation_id or "default"
        context = self._get_conversation_context(conversation_id)

        # Add current message to context
        if "messages" not in context:
            context["messages"] = []
        context["messages"].append({"role": "user", "content": request.message})

        # Include context in intent detection if available
        message_with_context = request.message
        if request.context:
            context.update(request.context)

        intent_result: IntentResult = await self._get_intent_with_fallback(
            message_with_context, context
        )
        intent = intent_result.intent

        if intent == IntentType.GENERATE_DIAGRAM.value:
            description = intent_result.description
            if not description:
                return AssistantResponse(
                    response_type="question",
                    content="I can help with that! What would you like the diagram to show?",
                )

            # Route through a tool-enabled flow so the model can request functions.
            response = await self._handle_with_tools(description)

            # Store response in context
            context["messages"].append(
                {"role": "assistant", "content": response.content, "type": "image"}
            )
            self._update_conversation_context(conversation_id, context)
            return response
        elif intent == IntentType.CLARIFICATION.value:
            response = AssistantResponse(
                response_type="text",
                content="I am an AI assistant that can generate diagrams from natural language descriptions. How can I help you?",
                suggestions=[
                    "Create a system architecture diagram",
                    "Generate a microservices diagram",
                    "Design a web application flow",
                ],
            )
        elif intent == IntentType.GREETING.value:
            response = AssistantResponse(
                response_type="text",
                content="Hello! How can I help you create a diagram today?",
                suggestions=[
                    "Show me an example diagram",
                    "Create a cloud architecture",
                    "Design a database schema",
                ],
            )
        else:
            response = AssistantResponse(
                response_type="text",
                content="I'm not sure how to help with that. Please try describing the diagram you would like to create.",
                suggestions=[
                    "Try: 'Create a web application with database'",
                    "Try: 'Show me a microservices architecture'",
                ],
            )

        # Store response in context
        context["messages"].append({"role": "assistant", "content": response.content})
        self._update_conversation_context(conversation_id, context)
        return response

    async def _handle_with_tools(self, description: str) -> AssistantResponse:
        """Enable function/tool-calling for diagram generation."""
        # In mock mode, skip LLM tool loop and directly generate via service
        if self.settings.mock_llm:
            (
                image_data,
                metadata,
            ) = await self.diagram_service.generate_diagram_from_description(
                description
            )
            return AssistantResponse(
                response_type="image",
                content="Here is the diagram you requested:",
                image_data=image_data,
                suggestions=["Would you like me to adjust layout or add components?"],
            )

        if self.settings.llm_provider == "openai":
            return await self._handle_with_tools_openai(description)
        else:
            return await self._handle_with_tools_gemini(description)

    async def _handle_with_tools_openai(self, description: str) -> AssistantResponse:
        """Handle tool calling with OpenAI Chat Completions."""
        tools = self._build_openai_tools()
        messages = [{"role": "user", "content": f"Create a diagram: {description}"}]
        latest_image_data: str | None = None
        latest_metadata: dict | None = None

        for _ in range(3):
            try:
                response = await client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=self.settings.llm_temperature,
                    timeout=self.settings.llm_timeout,
                )
            except Exception as e:
                logger.warning(
                    f"Assistant OpenAI tool loop failure, falling back to direct generation: {e}"
                )
                (
                    latest_image_data,
                    latest_metadata,
                ) = await self.diagram_service.generate_diagram_from_description(
                    description
                )
                return AssistantResponse(
                    response_type="image",
                    content="Here is the diagram you requested:",
                    image_data=latest_image_data,
                    suggestions=[
                        "Would you like me to adjust layout or add components?"
                    ],
                )

            message = response.choices[0].message

            if message.tool_calls:
                # Execute the first tool call
                tool_call = message.tool_calls[0]
                if tool_call.function.name == IntentType.GENERATE_DIAGRAM.value:
                    args = json.loads(tool_call.function.arguments)
                    desc = args.get("description", description)
                    (
                        latest_image_data,
                        latest_metadata,
                    ) = await self.diagram_service.generate_diagram_from_description(
                        desc
                    )

                    # Add assistant message with tool call
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments,
                                    },
                                }
                            ],
                        }
                    )

                    # Add tool response
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(
                                {
                                    "image_data": latest_image_data,
                                    "metadata": latest_metadata,
                                }
                            ),
                        }
                    )
                    continue
                else:
                    # Unknown tool, break out
                    break
            else:
                # No tool calls, finalize
                return AssistantResponse(
                    response_type="image" if latest_image_data else "text",
                    content=message.content or "Here is the diagram you requested:",
                    image_data=latest_image_data,
                    suggestions=[
                        "Would you like me to adjust layout or add components?"
                    ]
                    if latest_image_data
                    else None,
                )

        # Safety fallback
        if latest_image_data:
            return AssistantResponse(
                response_type="image",
                content="Here is the diagram you requested:",
                image_data=latest_image_data,
                suggestions=["Would you like me to adjust layout or add components?"],
            )
        (
            latest_image_data,
            _,
        ) = await self.diagram_service.generate_diagram_from_description(description)
        return AssistantResponse(
            response_type="image",
            content="Here is the diagram you requested:",
            image_data=latest_image_data,
        )

    async def _handle_with_tools_gemini(self, description: str) -> AssistantResponse:
        """Handle tool calling with Gemini (fallback)."""
        from google.genai import types as genai_types

        tool, tool_config = self._build_gemini_tools()
        contents = self._start_conversation(description)
        latest_image_data: str | None = None
        latest_metadata: dict | None = None

        for _ in range(3):
            try:
                resp = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self.settings.gemini_model,
                        contents=contents,
                        config=genai_types.GenerateContentConfig(
                            tools=[tool],
                            tool_config=tool_config,
                            response_mime_type="application/json",
                            response_schema=AssistantFinal,
                            temperature=self.settings.llm_temperature,
                        ),
                    ),
                    timeout=self.settings.llm_timeout,
                )
            except Exception as e:
                logger.warning(
                    f"Assistant Gemini tool loop failure, falling back to direct generation: {e}"
                )
                (
                    latest_image_data,
                    latest_metadata,
                ) = await self.diagram_service.generate_diagram_from_description(
                    description
                )
                return AssistantResponse(
                    response_type="image",
                    content="Here is the diagram you requested:",
                    image_data=latest_image_data,
                    suggestions=[
                        "Would you like me to adjust layout or add components?"
                    ],
                )

            (
                did_call,
                contents,
                latest_image_data,
                latest_metadata,
            ) = await self._maybe_execute_tool_gemini(
                resp, description, contents, latest_image_data, latest_metadata
            )
            if did_call:
                continue

            # No tool call → finalize via structured output
            final: AssistantFinal | None = getattr(resp, "parsed", None)
            if not final:
                # Defensive fallback in case SDK returns non-parsed
                text_fallback = (
                    getattr(resp, "text", None) or "Here is the diagram you requested:"
                )
                return AssistantResponse(
                    response_type="image" if latest_image_data else "text",
                    content=text_fallback,
                    image_data=latest_image_data,
                    suggestions=(
                        ["Would you like me to adjust layout or add components?"]
                        if latest_image_data
                        else None
                    ),
                )
            return AssistantResponse(
                response_type="image" if latest_image_data else "text",
                content=final.content,
                image_data=latest_image_data,
                suggestions=final.suggestions,
            )

        # Safety fallback
        if latest_image_data:
            return AssistantResponse(
                response_type="image",
                content="Here is the diagram you requested:",
                image_data=latest_image_data,
                suggestions=["Would you like me to adjust layout or add components?"],
            )
        (
            latest_image_data,
            _,
        ) = await self.diagram_service.generate_diagram_from_description(description)
        return AssistantResponse(
            response_type="image",
            content="Here is the diagram you requested:",
            image_data=latest_image_data,
        )

    def _build_openai_tools(self) -> list[dict]:
        """Build OpenAI Chat Completions tools definition."""
        return [
            {
                "type": "function",
                "function": {
                    "name": IntentType.GENERATE_DIAGRAM.value,
                    "description": "Generate a diagram from a natural language description.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Natural language description of the diagram to generate",
                            }
                        },
                        "required": ["description"],
                    },
                },
            }
        ]

    def _build_gemini_tools(self) -> tuple:
        """Declare tools and return Tool plus ToolConfig set to ANY mode.

        Using FunctionCallingConfig(mode='ANY') strongly encourages tool use.
        """
        from google.genai import types as genai_types

        generate_fn = genai_types.FunctionDeclaration(
            name=IntentType.GENERATE_DIAGRAM.value,
            description="Generate a diagram from a natural language description.",
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "description": genai_types.Schema(type=genai_types.Type.STRING)
                },
                required=["description"],
            ),
        )
        tool = genai_types.Tool(function_declarations=[generate_fn])
        tool_config = genai_types.ToolConfig(
            function_calling_config=genai_types.FunctionCallingConfig(
                mode=genai_types.FunctionCallingConfigMode.ANY
            )
        )
        return tool, tool_config

    def _start_conversation(self, description: str) -> list:
        """Seed the conversation with the user request."""
        from google.genai import types as genai_types

        return [
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_text(text=f"Create a diagram: {description}")
                ],
            )
        ]

    async def _maybe_execute_tool_gemini(
        self,
        resp,
        description: str,
        contents: list,
        latest_image_data: str | None,
        latest_metadata: dict | None,
    ) -> tuple[bool, list, str | None, dict | None]:
        """Execute the first function call if present; return whether a call was made.

        Appends the model's tool call and our tool response to contents when executed.
        """
        from google.genai import types as genai_types

        if not getattr(resp, "function_calls", None):
            return False, contents, latest_image_data, latest_metadata

        fn_call = resp.function_calls[0]
        args = fn_call.args or {}

        if fn_call.name == IntentType.GENERATE_DIAGRAM.value:
            desc = str(args.get("description", description))
            (
                latest_image_data,
                latest_metadata,
            ) = await self.diagram_service.generate_diagram_from_description(desc)

            tool_part = genai_types.Part.from_function_response(
                name=IntentType.GENERATE_DIAGRAM.value,
                response={
                    "image_data": latest_image_data,
                    "metadata": latest_metadata,
                },
            )
        else:
            return False, contents, latest_image_data, latest_metadata

        return (
            True,
            contents
            + [
                resp.candidates[0].content,
                genai_types.Content(role="tool", parts=[tool_part]),
            ],
            latest_image_data,
            latest_metadata,
        )

    async def _get_intent_with_fallback(
        self, message: str, context: dict | None = None
    ) -> IntentResult:
        """Get intent with fallback to original agent if LangChain fails."""
        if self.settings.use_langchain:
            try:
                return await self.assistant_agent.get_intent(message, context)
            except Exception as e:
                if self.settings.langchain_fallback and self.original_assistant_agent:
                    logger.warning(
                        f"LangChain intent detection failed, using original: {e}"
                    )
                    return await self.original_assistant_agent.get_intent(
                        message, context
                    )
                raise
        else:
            return await self.assistant_agent.get_intent(message, context)

    def _get_conversation_context(self, conversation_id: str) -> dict:
        """Get conversation context for a given conversation ID."""
        return self._conversation_context.get(conversation_id, {})

    def _update_conversation_context(self, conversation_id: str, context: dict) -> None:
        """Update conversation context for a given conversation ID."""
        # Keep only last 10 messages to prevent memory bloat
        if "messages" in context and len(context["messages"]) > 10:
            context["messages"] = context["messages"][-10:]
        self._conversation_context[conversation_id] = context
