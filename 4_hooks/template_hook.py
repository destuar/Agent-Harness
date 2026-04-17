"""Example custom hook -- content filter.

This template demonstrates how to create a hook that inspects tool call
arguments before execution and optionally modifies results afterward.

Usage:
    from 4_hooks.template_hook import ContentFilterHook

    harness = AgentHarness(
        client=client,
        system_prompt="...",
        tools=[...],
        hooks=[ContentFilterHook(blocked_terms=["DROP TABLE", "rm -rf"])],
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agent_harness import Hook, HookResult

logger = logging.getLogger(__name__)


class ContentFilterHook(Hook):
    """Block tool calls whose arguments contain prohibited terms.

    This is a simple keyword-based filter. For production use, consider
    combining it with more sophisticated checks (regex patterns, an LLM
    classifier, etc.).

    Args:
        blocked_terms: List of strings that should never appear in tool arguments.
        case_sensitive: Whether matching should be case-sensitive (default: False).
        redact_in_results: If True, also redact blocked terms found in tool results.
    """

    def __init__(
        self,
        blocked_terms: list[str],
        case_sensitive: bool = False,
        redact_in_results: bool = True,
    ) -> None:
        self.blocked_terms_original = blocked_terms
        self.case_sensitive = case_sensitive
        self.redact_in_results = redact_in_results

        if case_sensitive:
            self.blocked_terms = blocked_terms
        else:
            self.blocked_terms = [t.lower() for t in blocked_terms]

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison based on case_sensitive setting."""
        return text if self.case_sensitive else text.lower()

    def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """Check tool arguments for prohibited terms.

        Scans the string representation of the entire args dict so it catches
        terms regardless of which parameter they appear in.
        """
        args_str = self._normalize(str(args))

        for i, term in enumerate(self.blocked_terms):
            if term in args_str:
                original_term = self.blocked_terms_original[i]
                logger.warning(
                    "ContentFilterHook blocked tool '%s': "
                    "args contain prohibited term '%s'",
                    tool_name,
                    original_term,
                )
                return HookResult(
                    allowed=False,
                    reason=f"Blocked: argument contains prohibited term '{original_term}'",
                )

        return HookResult(allowed=True)

    def after_tool_call(self, tool_name: str, args: dict[str, Any], result: str) -> str:
        """Optionally redact prohibited terms from tool results."""
        if not self.redact_in_results:
            return result

        redacted = result
        for i, term in enumerate(self.blocked_terms):
            original_term = self.blocked_terms_original[i]
            # Case-insensitive replacement via manual scan
            lower_result = redacted.lower() if not self.case_sensitive else redacted
            idx = 0
            while True:
                search_in = lower_result if not self.case_sensitive else redacted
                pos = search_in.find(term, idx)
                if pos == -1:
                    break
                redacted = redacted[:pos] + "[REDACTED]" + redacted[pos + len(term) :]
                lower_result = redacted.lower() if not self.case_sensitive else redacted
                idx = pos + len("[REDACTED]")

        if redacted != result:
            logger.info("ContentFilterHook redacted content from tool '%s' result", tool_name)

        return redacted


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hook = ContentFilterHook(blocked_terms=["DROP TABLE", "rm -rf", "password123"])

    # Test before_tool_call -- should block
    result = hook.before_tool_call("run_sql", {"query": "DROP TABLE users;"})
    print(f"Blocked: {not result.allowed}, Reason: {result.reason}")

    # Test before_tool_call -- should allow
    result = hook.before_tool_call("run_sql", {"query": "SELECT * FROM users LIMIT 10"})
    print(f"Allowed: {result.allowed}")

    # Test after_tool_call -- should redact
    raw_result = json.dumps({"output": "The password is password123, do not share it."})
    redacted = hook.after_tool_call("some_tool", {}, raw_result)
    print(f"Redacted result: {redacted}")
