"""Base interface for model clients.

Purpose:
    Define the ModelClient protocol and BaseModelClient abstract class that
    all provider implementations must follow. This abstraction allows the
    AgentHarness to work with any LLM provider without code changes.

Key relationships:
    - ``core.py``: AgentHarness accepts any ModelClient for LLM calls
    - ``openai_compat.py``: OpenAICompatibleClient implements this interface
    - ``messages.py``: StreamChunk is the return type for chat_stream()

Interface Contract:
    Implementations must provide chat_stream() which:
    - Accepts messages in OpenAI API format
    - Accepts optional tools in OpenAI function calling format
    - Yields StreamChunk objects with text and/or tool call deltas
    - Handles all provider-specific authentication and error handling
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator, Protocol

from ..messages import StreamChunk


class ModelClient(Protocol):
    """Protocol for model clients.

    Implementations must provide chat_stream() which yields StreamChunks.
    """

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion.

        Args:
            messages: List of messages in OpenAI API format
            tools: Optional list of tools in OpenAI API format

        Yields:
            StreamChunk objects with text deltas and/or tool call deltas
        """
        ...


class BaseModelClient(ABC):
    """Abstract base class for model clients.

    Provides common functionality and enforces the interface.
    """

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion."""
        ...

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Non-streaming chat completion.

        Returns:
            Tuple of (response_text, tool_calls)
        """
        text = ""
        tool_calls: list[dict[str, Any]] = []
        tool_call_buffer: dict[int, dict[str, Any]] = {}

        for chunk in self.chat_stream(messages, tools):
            if chunk.text:
                text += chunk.text

            if chunk.tool_call:
                tc = chunk.tool_call
                if tc.index not in tool_call_buffer:
                    tool_call_buffer[tc.index] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                    }
                buf = tool_call_buffer[tc.index]
                if tc.id:
                    buf["id"] = tc.id
                if tc.name:
                    buf["name"] = tc.name
                if tc.arguments:
                    buf["arguments"] += tc.arguments

        # Convert accumulated tool calls to list
        for idx in sorted(tool_call_buffer.keys()):
            tool_calls.append(tool_call_buffer[idx])

        return text, tool_calls
