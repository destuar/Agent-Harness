"""Tool definitions for the agent harness.

Purpose:
    Define the Tool dataclass and helper functions for creating tools that
    the model can invoke during agent execution. Tools bridge LLM reasoning
    to external capabilities (APIs, databases, sub-agents, etc.).

Key relationships:
    - ``core.py``: AgentHarness holds tools dict and executes them
    - ``clients/openai_compat.py``: Tool.to_api_format() used for API calls

Tool Components:
    - name: Unique identifier the model uses to invoke the tool
    - description: Helps the model understand when and why to use it
    - parameters: JSON Schema defining expected arguments (OpenAI format)
    - handler: Function(args: dict) -> str that executes the tool

Creation Patterns:
    1. @tool decorator: For simple, standalone tool functions
    2. create_tool(): For dynamic tools or closures capturing state
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    """A tool that can be called by the model."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]

    def to_api_format(self) -> dict[str, Any]:
        """Convert to OpenAI API tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable], Tool]:
    """Decorator to create a Tool from a function.

    Usage:
        @tool(
            name="get_weather",
            description="Get the weather for a city",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        def get_weather(args: dict) -> str:
            return f"Weather in {args['city']}: Sunny"
    """
    if parameters is None:
        parameters = {"type": "object", "properties": {}}

    def decorator(fn: Callable[[dict[str, Any]], str]) -> Tool:
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=fn,
        )

    return decorator


def create_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: Callable[[dict[str, Any]], str],
) -> Tool:
    """Factory function to create a Tool.

    Alternative to the @tool decorator for when you need to create
    tools dynamically or with closures.
    """
    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        handler=handler,
    )
