"""Safety hooks: block dangerous SQL and redact emails from tool results.

Demonstrates how to subclass Hook and register multiple hooks with
AgentHarness.  When a hook blocks a call the model receives the rejection
reason so it can adjust its approach.

Requirements:
    pip install agent-harness[openai]
"""

import os
import re

from agent_harness import AgentHarness, Hook, HookResult, OpenAICompatibleClient, tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(name="run_sql", description="Execute a SQL query and return the result rows.")
def run_sql(query: str) -> str:
    # Mock -- just echo the query
    return f"[mock result for: {query}]"


@tool(name="lookup_user", description="Return profile info for a user id.")
def lookup_user(user_id: str) -> str:
    return f"User {user_id}: name=Jane Doe, email=jane.doe@example.com, role=admin"


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class SqlSafetyHook(Hook):
    """Block SQL statements that modify or destroy data."""

    DANGEROUS = re.compile(r"\b(DELETE|DROP|TRUNCATE|ALTER)\b", re.IGNORECASE)

    def before_tool_call(self, tool_name: str, args: dict) -> HookResult:
        if tool_name == "run_sql":
            query = args.get("query", "")
            if self.DANGEROUS.search(query):
                return HookResult(
                    allowed=False,
                    reason="Destructive SQL is not permitted. Use SELECT queries only.",
                )
        return HookResult(allowed=True)


class RedactEmailHook(Hook):
    """Strip email addresses from any tool result before the model sees it."""

    EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}")

    def after_tool_call(self, tool_name: str, args: dict, result: str) -> str:
        return self.EMAIL_RE.sub("[REDACTED]", result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model="gpt-4o",
    )

    harness = AgentHarness(
        client=client,
        system_prompt=(
            "You are a database assistant. Use the run_sql tool for queries "
            "and lookup_user for user info."
        ),
        tools=[run_sql, lookup_user],
        hooks=[SqlSafetyHook(), RedactEmailHook()],
    )

    # The model will attempt a DELETE, get blocked, then retry with SELECT.
    print("--- Dangerous query test ---")
    result = harness.run("Delete all inactive users from the users table.")
    print(result)

    # Email in the tool result will be redacted before the model sees it.
    print("\n--- Redaction test ---")
    result = harness.run("Look up user 42 and tell me their email.")
    print(result)


if __name__ == "__main__":
    main()
