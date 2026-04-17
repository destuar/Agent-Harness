# Hooks

Hooks define **safety checks that run automatically** before and after every tool execution. They can block dangerous tool calls, sanitize arguments, and redact sensitive data from results -- without any changes to the tools themselves.

## How Hooks Work

Hooks are built on the `Hook` base class. You subclass it and override one or both methods:

- **`before_tool_call(tool_name, args) -> HookResult`** -- Runs before the tool handler executes. Can block the call (`allowed=False`) or modify the arguments (`modified_args`).
- **`after_tool_call(tool_name, args, result) -> str`** -- Runs after the tool handler returns. Can modify or redact the result string.

Multiple hooks run in registration order. If any `before_tool_call` returns `allowed=False`, the tool is blocked immediately and remaining hooks are skipped.

## Hook Lifecycle

```
Agent requests tool call
    |
    v
on_tool_call callback (logging/progress)
    |
    v
Hook 1: before_tool_call() --[blocked]--> return blocked reason to model
Hook 2: before_tool_call()       |
Hook N: before_tool_call()       |
    |                            |
    v (all allowed)              |
Tool handler executes            |
    |                            |
    v                            |
Hook 1: after_tool_call()        |
Hook 2: after_tool_call()        |
Hook N: after_tool_call()        |
    |                            |
    v                            |
on_tool_result callback (history)
```

## Creating a Custom Hook

```python
from agent_harness import Hook, HookResult

class MaxRowsHook(Hook):
    """Ensure SQL queries don't return unbounded result sets."""

    def __init__(self, max_rows: int = 1000):
        self.max_rows = max_rows

    def before_tool_call(self, tool_name, args):
        if tool_name == "run_sql" and "query" in args:
            query = args["query"].strip().rstrip(";")
            if "LIMIT" not in query.upper() and "TOP" not in query.upper():
                args = {**args, "query": f"{query} LIMIT {self.max_rows}"}
                return HookResult(allowed=True, modified_args=args)
        return HookResult(allowed=True)
```

## Registering Hooks

Pass hooks when creating the harness. They apply to every tool call:

```python
from 4_hooks.builtin.secrets_detector import SecretsDetectorHook
from 4_hooks.builtin.sql_guard import SQLGuardHook

harness = AgentHarness(
    client=client,
    system_prompt="...",
    tools=[sql_tool, search_tool],
    hooks=[
        SecretsDetectorHook(),
        SQLGuardHook(allowed_tables=["patients", "encounters"], max_rows=500),
    ],
)
```

## HookResult Fields

| Field | Type | Purpose |
|-------|------|---------|
| `allowed` | `bool` | Whether the tool call should proceed |
| `reason` | `str` | Explanation if blocked (returned to the model as the tool result) |
| `modified_args` | `dict \| None` | If set, replaces the original args for downstream hooks and the handler |

## Files in This Directory

| File | Description |
|------|-------------|
| `template_hook.py` | Template for a custom content-filter hook |
| `builtin/secrets_detector.py` | Built-in hook that detects API keys, passwords, and tokens |
| `builtin/sql_guard.py` | Built-in hook that enforces SQL safety (SELECT-only, table whitelist, row limits) |
