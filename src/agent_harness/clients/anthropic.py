"""Anthropic Claude client for the agent harness.

Purpose:
    Provide an interface to Anthropic's Claude models via the Messages API.
    Translates between the harness's OpenAI-format messages/tools and
    Anthropic's native format.

Key relationships:
    - ``base.py``: Implements the ModelClient protocol
    - ``core.py``: AgentHarness uses this client for model calls
    - ``messages.py``: StreamChunk and ToolCallDelta for streaming output

Format Differences (OpenAI vs Anthropic):
    - System prompt: OpenAI puts in messages array, Anthropic uses separate param
    - Tool definitions: OpenAI wraps in {"type": "function", "function": {...}},
      Anthropic uses {"name": ..., "input_schema": ...}
    - Tool results: OpenAI uses role="tool", Anthropic uses role="user" with
      tool_result content blocks
    - Images: OpenAI uses image_url with data URI, Anthropic uses image with
      base64 source

Dependencies:
    - anthropic: Official Anthropic Python SDK
"""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, Generator

import anthropic

from ..messages import StreamChunk, ToolCallDelta
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class AnthropicClient(BaseModelClient):
    """Client for Anthropic's Claude models via the Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-7",
        max_tokens: int = 16384,
        base_url: str | None = None,
    ):
        """Initialize the Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., claude-opus-4-7, claude-sonnet-4-6)
            max_tokens: Maximum tokens in response (required by Anthropic API)
            base_url: Optional custom API endpoint (e.g., Azure Anthropic proxy).
                      If not provided, uses default api.anthropic.com.
        """
        self.model = model
        self.max_tokens = max_tokens

        # Support custom endpoints (e.g., Azure AI Services Anthropic proxy)
        client_kwargs = {"api_key": api_key}
        if base_url:
            # Normalize: SDK appends /v1/messages, so strip that if present
            if base_url.endswith("/v1/messages"):
                base_url = base_url.rsplit("/v1/messages", 1)[0]
            elif base_url.endswith("/v1"):
                base_url = base_url.rsplit("/v1", 1)[0]
            elif base_url.endswith("/messages"):
                base_url = base_url.rsplit("/messages", 1)[0]
            client_kwargs["base_url"] = base_url
            logger.info(f"Initialized Anthropic client with custom endpoint: {base_url}, model: {model}")
        else:
            logger.info(f"Initialized Anthropic client with model: {model}")

        self.client = anthropic.Anthropic(**client_kwargs)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion from Claude.

        Translates OpenAI-format messages and tools to Anthropic format,
        streams the response, and yields StreamChunk objects compatible
        with the agent harness.

        Args:
            messages: Conversation history in OpenAI format
            tools: Optional list of tool definitions in OpenAI format

        Yields:
            StreamChunk objects with text deltas and/or tool call deltas
        """
        # Extract system prompt (first message if role=="system")
        system_prompt = None
        conversation_messages = messages
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0].get("content", "")
            conversation_messages = messages[1:]

        # Convert messages to Anthropic format
        anthropic_messages = self._convert_messages(conversation_messages)

        # Convert tools to Anthropic format
        anthropic_tools = self._convert_tools(tools) if tools else None

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        # Stream the response
        with self.client.messages.stream(**kwargs) as stream:
            # Track tool use blocks for yielding ToolCallDelta
            current_tool_index = -1
            tool_blocks: dict[int, dict[str, Any]] = {}

            for event in stream:
                # Handle different event types
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "text":
                        # Text block starting - nothing to yield yet
                        pass
                    elif block.type == "tool_use":
                        # Tool use block starting
                        current_tool_index += 1
                        tool_blocks[current_tool_index] = {
                            "id": block.id,
                            "name": block.name,
                            "input_json": "",
                        }
                        # Yield initial tool call delta with id and name
                        yield StreamChunk(
                            tool_call=ToolCallDelta(
                                index=current_tool_index,
                                id=block.id,
                                name=block.name,
                                arguments=None,
                            )
                        )

                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        # Text content fragment
                        yield StreamChunk(text=delta.text)
                    elif delta.type == "input_json_delta":
                        # Tool input JSON fragment
                        if current_tool_index >= 0:
                            tool_blocks[current_tool_index]["input_json"] += delta.partial_json
                            yield StreamChunk(
                                tool_call=ToolCallDelta(
                                    index=current_tool_index,
                                    id=None,
                                    name=None,
                                    arguments=delta.partial_json,
                                )
                            )

                elif event.type == "message_delta":
                    # Message complete - check stop reason
                    stop_reason = event.delta.stop_reason
                    if stop_reason == "end_turn":
                        yield StreamChunk(finish_reason="stop")
                    elif stop_reason == "tool_use":
                        yield StreamChunk(finish_reason="tool_calls")
                    elif stop_reason:
                        yield StreamChunk(finish_reason=stop_reason)

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format messages to Anthropic format.

        Key differences:
        - Tool results: OpenAI uses role="tool", Anthropic uses role="user"
          with tool_result content blocks
        - Assistant tool calls: OpenAI uses tool_calls array, Anthropic uses
          tool_use content blocks
        - Images: OpenAI uses image_url, Anthropic uses image with base64 source

        Args:
            messages: Messages in OpenAI format

        Returns:
            Messages in Anthropic format
        """
        result = []
        pending_tool_results: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                # Flush any pending tool results first
                if pending_tool_results:
                    result.append({
                        "role": "user",
                        "content": pending_tool_results,
                    })
                    pending_tool_results = []

                # Convert user message content
                anthropic_content = self._convert_content(content)
                result.append({
                    "role": "user",
                    "content": anthropic_content,
                })

            elif role == "assistant":
                # Flush any pending tool results first
                if pending_tool_results:
                    result.append({
                        "role": "user",
                        "content": pending_tool_results,
                    })
                    pending_tool_results = []

                # Build assistant content blocks
                assistant_content = []

                # Add text content if present
                if content:
                    if isinstance(content, str):
                        assistant_content.append({
                            "type": "text",
                            "text": content,
                        })
                    elif isinstance(content, list):
                        # Multi-part content
                        for part in content:
                            if part.get("type") == "text":
                                assistant_content.append({
                                    "type": "text",
                                    "text": part.get("text", ""),
                                })

                # Add tool use blocks if present
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func = tc.get("function", {})
                    # Parse arguments from JSON string
                    args_str = func.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}

                    assistant_content.append({
                        "type": "tool_use",
                        "id": tc.get("id"),
                        "name": func.get("name"),
                        "input": args,
                    })

                if assistant_content:
                    result.append({
                        "role": "assistant",
                        "content": assistant_content,
                    })

            elif role == "tool":
                # Anthropic expects tool results as user messages with tool_result blocks
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id"),
                    "content": content if isinstance(content, str) else json.dumps(content),
                }
                pending_tool_results.append(tool_result)

        # Flush any remaining tool results
        if pending_tool_results:
            result.append({
                "role": "user",
                "content": pending_tool_results,
            })

        return result

    def _convert_content(
        self, content: str | list[dict[str, Any]] | None
    ) -> str | list[dict[str, Any]]:
        """Convert message content to Anthropic format.

        Handles:
        - Simple strings (pass through)
        - Multi-modal content with images (convert image_url to image)

        Args:
            content: OpenAI-format content (string or list of content blocks)

        Returns:
            Anthropic-format content
        """
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        # Multi-modal content - convert each block
        result = []
        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                result.append({
                    "type": "text",
                    "text": block.get("text", ""),
                })

            elif block_type == "image_url":
                # Convert OpenAI image_url to Anthropic image format
                image_url = block.get("image_url", {})
                url = image_url.get("url", "")

                # Parse data URI: data:image/jpeg;base64,/9j/4AAQ...
                if url.startswith("data:"):
                    match = re.match(r"data:([^;]+);base64,(.+)", url)
                    if match:
                        media_type = match.group(1)
                        base64_data = match.group(2)
                        result.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        })
                    else:
                        # Malformed data URI - skip or log warning
                        logger.warning(f"Malformed image data URI: {url[:50]}...")
                else:
                    # URL-based image - Anthropic also supports this
                    result.append({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": url,
                        },
                    })

        return result if result else ""

    def _convert_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format tools to Anthropic format.

        OpenAI format:
            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}

        Anthropic format:
            {"name": ..., "description": ..., "input_schema": ...}

        Args:
            tools: Tools in OpenAI format

        Returns:
            Tools in Anthropic format
        """
        result = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                result.append({
                    "name": func.get("name"),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                })
        return result
