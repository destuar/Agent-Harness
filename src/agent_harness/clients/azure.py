"""Azure OpenAI client implementation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Generator

from openai import AzureOpenAI

from ..messages import StreamChunk, ToolCallDelta
from .base import BaseModelClient

logger = logging.getLogger(__name__)


class AzureModelClient(BaseModelClient):
    """Azure OpenAI client for the agent harness.

    Supports both API key and Azure AD token authentication.
    """

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        api_key: str | None = None,
        api_version: str = "2024-08-01-preview",
        azure_ad_token_provider: Callable[[], str] | None = None,
    ):
        """Initialize the Azure OpenAI client.

        Args:
            endpoint: Azure OpenAI endpoint URL
            deployment: Model deployment name
            api_key: API key (optional if using Azure AD)
            api_version: API version
            azure_ad_token_provider: Callable that returns Azure AD token
        """
        self.deployment = deployment

        kwargs: dict[str, Any] = {
            "azure_endpoint": endpoint,
            "api_version": api_version,
        }

        if api_key:
            kwargs["api_key"] = api_key
        elif azure_ad_token_provider:
            kwargs["azure_ad_token_provider"] = azure_ad_token_provider
        else:
            raise ValueError("Either api_key or azure_ad_token_provider required")

        self.client = AzureOpenAI(**kwargs)

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Generator[StreamChunk, None, None]:
        """Stream a chat completion from Azure OpenAI.

        Yields StreamChunk objects with text and tool call deltas.
        """
        kwargs: dict[str, Any] = {
            "model": self.deployment,
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
