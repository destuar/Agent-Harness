# Tools

Tools define **what an agent can do**. A tool is a callable function that the agent can invoke during its loop to interact with external systems -- APIs, databases, file systems, sub-agents, or anything else.

## How Tools Work

When the LLM decides it needs to take an action, it emits a tool call with a name and JSON arguments. The harness matches the name to a registered `Tool`, passes the arguments to its handler, and returns the string result back to the model as context for its next turn.

**Handler signature:**

```python
def my_handler(args: dict[str, Any]) -> str:
    ...
```

- **Input**: A `dict` parsed from the JSON arguments the model provided.
- **Output**: A `str` returned to the model. Typically `json.dumps(...)` for structured data.
- **Parameters**: Defined using [JSON Schema](https://json-schema.org/) in the OpenAI function-calling format.

## Creation Patterns

### 1. `@tool` Decorator -- Simple, Static Tools

Best for standalone functions that don't need external state:

```python
from agent_harness import tool

@tool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)
def get_weather(args: dict) -> str:
    city = args["city"]
    return json.dumps({"city": city, "temperature": 72, "condition": "sunny"})
```

The decorator returns a `Tool` instance (not a plain function), so you pass it directly to `AgentHarness(tools=[get_weather])`.

### 2. `create_tool()` Factory -- Dynamic Tools with Closures

Best when the handler needs to capture external state (a database connection, an API client, configuration):

```python
from agent_harness import create_tool, Tool

def make_db_tool(connection) -> Tool:
    def handler(args: dict) -> str:
        rows = connection.execute(args["query"]).fetchall()
        return json.dumps([dict(r) for r in rows])

    return create_tool(
        name="query_database",
        description="Run a read-only SQL query",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL SELECT query"},
            },
            "required": ["query"],
        },
        handler=handler,
    )
```

## Registering Tools

Pass tools as a list when creating the harness:

```python
harness = AgentHarness(
    client=client,
    system_prompt="You are a helpful assistant.",
    tools=[get_weather, make_db_tool(conn)],
)
```

## Examples in This Directory

| File | Pattern | Description |
|------|---------|-------------|
| `template_tool.py` | `@tool` decorator | Simple weather lookup tool |
| `template_dynamic_tool.py` | `create_tool()` factory | Search tool with injected service |

## Tips

- **Keep handlers focused.** One tool should do one thing well.
- **Return structured JSON.** Models parse structured output more reliably.
- **Write clear descriptions.** The model relies on the `description` and parameter descriptions to decide when and how to call your tool.
- **Validate inputs in the handler.** The model may omit optional fields or send unexpected values.
- **Use `create_tool()` for testability.** Injecting dependencies via closures makes tools easy to unit test with mocks.
