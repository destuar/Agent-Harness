"""Agent Harness - A simple, reusable agent loop.

Purpose:
    Provide a minimal, project-agnostic framework for building LLM agents.
    The harness implements the core agent loop (call model -> execute tools -> repeat)
    while remaining decoupled from any specific use case or domain.

Design Principle:
    LLM + loop + tools = agent

    The harness handles the repetitive parts (streaming, tool accumulation,
    iteration control) while the caller provides the intelligence via
    system prompts and tool implementations.

Installation:
    pip install agent-harness              # Core only (AgentHarness, Tool, Message, Hook)
    pip install agent-harness[openai]      # + OpenAI/Azure clients
    pip install agent-harness[anthropic]   # + Anthropic client
    pip install agent-harness[all]         # Everything

Usage:
    from agent_harness import AgentHarness, Tool, OpenAICompatibleClient, Message

    client = OpenAICompatibleClient(
        base_url="https://your-endpoint.com/api/openai/v1",
        api_key="...",
        model="gpt-5.4",
    )

    @tool(name="get_weather", description="Get weather", parameters={...})
    def get_weather(args):
        return f"Weather in {args['city']}: Sunny"

    harness = AgentHarness(
        client=client,
        system_prompt="You are a helpful assistant.",
        tools=[get_weather],
    )

    for chunk in harness.run_stream([Message(role="user", content="What's the weather in LA?")]):
        print(chunk, end="")

Exports:
    - AgentHarness: The core agent loop
    - Message, ToolCall, StreamChunk: Data structures
    - Tool, tool, create_tool: Tool definition helpers
    - Hook, HookResult: Safety check base classes
    - ModelClient, OpenAICompatibleClient: LLM provider interfaces
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .clients import BaseModelClient, ModelClient
from .core import AgentHarness
from .hooks import Hook, HookResult
from .messages import Message, StreamChunk, ToolCall, ToolCallDelta
from .tools import Tool, create_tool, tool

if TYPE_CHECKING:
    from .clients.anthropic import AnthropicClient
    from .clients.azure import AzureModelClient
    from .clients.openai_compat import OpenAICompatibleClient


def __getattr__(name: str):
    if name == "OpenAICompatibleClient":
        from .clients.openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient
    if name == "AzureModelClient":
        from .clients.azure import AzureModelClient

        return AzureModelClient
    if name == "AnthropicClient":
        from .clients.anthropic import AnthropicClient

        return AnthropicClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core
    "AgentHarness",
    # Messages
    "Message",
    "StreamChunk",
    "ToolCall",
    "ToolCallDelta",
    # Tools
    "Tool",
    "tool",
    "create_tool",
    # Hooks
    "Hook",
    "HookResult",
    # Clients
    "ModelClient",
    "BaseModelClient",
    "AnthropicClient",
    "AzureModelClient",
    "OpenAICompatibleClient",
]
