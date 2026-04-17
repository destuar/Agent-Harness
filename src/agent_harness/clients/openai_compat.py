"""OpenAI-compatible client for custom endpoints.

Purpose:
    Provide a unified interface to any API that implements the OpenAI chat
    completions protocol. This enables the AgentHarness to work with multiple
    providers without code changes.

Supported Endpoints:
    - Azure AI Foundry (recommended for enterprise)
    - Azure OpenAI
    - OpenAI
    - Local LLM servers (Ollama, vLLM, LM Studio, etc.)

Key relationships:
    - ``base.py``: Implements the ModelClient protocol
    - ``core.py``: AgentHarness uses this client for model calls
    - ``messages.py``: StreamChunk and ToolCallDelta for streaming output

Streaming Protocol:
    The OpenAI streaming API sends response chunks incrementally:
    1. Text chunks arrive as delta.content (yielded as StreamChunk.text)
    2. Tool calls arrive as delta.tool_calls with incremental arguments
    3. finish_reason indicates completion ("stop" or "tool_calls")

    Tool calls are streamed in pieces:
    - First chunk has id and function name
    - Subsequent chunks have argument fragments
    - The harness accumulates these into complete ToolCall objects

Dependencies:
    - openai: Official OpenAI Python SDK (works with compatible endpoints)
"""

from __future__ import annotations

import logging
from typing import Any, Generator

from openai import OpenAI

from ..messages import StreamChunk, ToolCallDelta
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class OpenAICompatibleClient(BaseModelClient):
    """Client for any OpenAI-compatible API endpoint.

    This is the most flexible client - works with any endpoint that
    implements the OpenAI chat completions API.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "gpt-5.4",
    ):
        """Initialize the OpenAI-compatible client.

        Args:
            base_url: Base URL for the API (e.g., https://api.openai.com/v1)
            api_key: API key for authentication
            model: Model name to use for completions
        """
        self.model = model

        # Ensure base_url ends properly for the OpenAI client
        # The client will append /chat/completions
        if base_url.endswith("/responses"):
            # Azure AI Foundry uses /responses, convert to standard path
            base_url = base_url.rsplit("/responses", 1)[0]
        if not base_url.endswith("/v1") and "/v1" not in base_url:
            base_url = base_url.rstrip("/") + "/v1"

        logger.info(f"Initializing OpenAI-compatible client with base_url: {base_url}")

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion from the model.

        Opens a streaming connection to the API and yields response chunks
        as they arrive. The harness accumulates these into complete responses.

        Args:
            messages: Conversation history in OpenAI format
                      (list of {"role": "...", "content": "..."} dicts)
            tools: Optional list of tool definitions in OpenAI function calling format

        Yields:
            StreamChunk objects containing one of:
            - text: Text content fragment
            - tool_call: ToolCallDelta with id/name/arguments fragment
            - finish_reason: "stop" (done) or "tool_calls" (needs tool execution)

        Note:
            Tool calls arrive incrementally - the harness accumulates deltas
            into complete ToolCall objects before execution.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools

        response = self.client.chat.completions.create(**kwargs)

        for chunk in response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            # Handle text content
            if delta.content:
                yield StreamChunk(text=delta.content)

            # Handle tool calls
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    yield StreamChunk(
                        tool_call=ToolCallDelta(
                            index=tc.index,
                            id=tc.id,
                            name=tc.function.name if tc.function else None,
                            arguments=tc.function.arguments if tc.function else None,
                        )
                    )

            # Handle finish reason
            if choice.finish_reason:
                yield StreamChunk(finish_reason=choice.finish_reason)
