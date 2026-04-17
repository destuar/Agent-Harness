"""Model client implementations.

Provider clients are lazy-loaded so the package works without
installing provider SDKs. Install extras to enable them:

    pip install agent-harness[openai]      # OpenAICompatibleClient, AzureModelClient
    pip install agent-harness[anthropic]   # AnthropicClient
    pip install agent-harness[all]         # Everything
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import BaseModelClient, ModelClient

if TYPE_CHECKING:
    from .anthropic import AnthropicClient
    from .azure import AzureModelClient
    from .openai_compat import OpenAICompatibleClient


def __getattr__(name: str):
    if name == "OpenAICompatibleClient":
        from .openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient
    if name == "AzureModelClient":
        from .azure import AzureModelClient

        return AzureModelClient
    if name == "AnthropicClient":
        from .anthropic import AnthropicClient

        return AnthropicClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicClient",
    "AzureModelClient",
    "BaseModelClient",
    "ModelClient",
    "OpenAICompatibleClient",
]
