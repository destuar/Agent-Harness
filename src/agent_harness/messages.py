"""Message types for the agent harness.

Purpose:
    Define provider-agnostic data structures for conversation messages,
    tool calls, and streaming response chunks. These types flow through
    the harness and can be serialized to any OpenAI-compatible API format.

Key relationships:
    - ``core.py``: AgentHarness uses Message for conversation history
    - ``clients/openai_compat.py``: Yields StreamChunk from API responses
    - ``core.py``: Accumulates ToolCallDelta into complete ToolCall objects

Type Mapping:
    - Message -> OpenAI messages array items
    - ToolCall -> function_call in assistant messages
    - StreamChunk -> parsed SSE chunks from streaming API
    - ToolCallDelta -> incremental tool_calls from streaming
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A conversation message.

    Supports all OpenAI-compatible message types:
    - system: System prompt
    - user: User input (can include images)
    - assistant: Model response (can include tool_calls)
    - tool: Tool execution result
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_api_format(self) -> dict[str, Any]:
        """Convert to OpenAI API message format."""
        msg: dict[str, Any] = {"role": self.role}

        if self.content is not None:
            msg["content"] = self.content

        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in self.tool_calls
            ]

        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id

        if self.name:
            msg["name"] = self.name

        return msg


@dataclass
class StreamChunk:
    """A chunk from a streaming response.

    Either text or tool_call will be set, not both.
    For tool calls, chunks arrive incrementally - accumulate them.
    """

    text: str | None = None
    tool_call: ToolCallDelta | None = None
    finish_reason: str | None = None


@dataclass
class ToolCallDelta:
    """Incremental tool call data from streaming.

    During streaming, tool calls arrive in pieces:
    - First chunk has index and id
    - Subsequent chunks have name and arguments fragments
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str | None = None
