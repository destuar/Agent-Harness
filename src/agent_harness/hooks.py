"""Hook system for the agent harness.

Purpose:
    Provide a mechanism for safety checks that run automatically before
    and after tool execution. Hooks can block dangerous tool calls,
    sanitize arguments, and redact sensitive data from results.

Key relationships:
    - ``core.py``: AgentHarness runs hooks in _execute_tool()
    - ``hooks/builtin/``: Built-in hook implementations

Hook Lifecycle:
    1. Agent requests a tool call
    2. on_tool_call callback fires (for logging/progress)
    3. All hooks' before_tool_call() run in order
       - If any returns allowed=False, tool is blocked
       - Hooks can modify args via modified_args
    4. Tool handler executes (if not blocked)
    5. All hooks' after_tool_call() run in order
       - Hooks can modify/redact the result
    6. on_tool_result callback fires (for history)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HookResult:
    """Result from a before_tool_call hook.

    Attributes:
        allowed: Whether the tool call should proceed.
        reason: Explanation if blocked (returned to the model as the tool result).
        modified_args: If set, replaces the original args for downstream hooks
                       and the tool handler. Use this to sanitize inputs.
    """

    allowed: bool
    reason: str = ""
    modified_args: dict[str, Any] | None = None


class Hook:
    """Base class for safety hooks.

    Subclass this and override before_tool_call and/or after_tool_call
    to implement custom safety checks.

    Example:
        class NoDeleteHook(Hook):
            def before_tool_call(self, tool_name, args):
                if "DELETE" in str(args).upper():
                    return HookResult(allowed=False, reason="DELETE operations are blocked")
                return HookResult(allowed=True)
    """

    def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """Run before tool execution. Return allowed=False to block.

        Args:
            tool_name: Name of the tool being called.
            args: Arguments the model passed to the tool.

        Returns:
            HookResult indicating whether to proceed.
        """
        return HookResult(allowed=True)

    def after_tool_call(self, tool_name: str, args: dict[str, Any], result: str) -> str:
        """Run after tool execution. Can modify/redact the result.

        Args:
            tool_name: Name of the tool that was called.
            args: Arguments the tool was called with.
            result: The string result from the tool handler.

        Returns:
            The (potentially modified) result string.
        """
        return result
