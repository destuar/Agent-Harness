# Agent Harness

> LLM + loop + tools = agent

A minimal, reusable framework for building LLM agents. The harness handles the repetitive parts (streaming, tool call accumulation, iteration control, safety hooks) while you provide the intelligence via system prompts, tools, skills, and roles.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import os
from agent_harness import AgentHarness, Message, OpenAICompatibleClient, tool

client = OpenAICompatibleClient(
    base_url=os.environ["API_ENDPOINT"],
    api_key=os.environ["API_KEY"],
    model="gpt-5.4",
)

@tool(
    name="get_weather",
    description="Get the weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
)
def get_weather(args):
    return f'{{"city": "{args["city"]}", "temp": 72, "condition": "sunny"}}'

harness = AgentHarness(
    client=client,
    system_prompt="You are a helpful assistant.",
    tools=[get_weather],
)

for chunk in harness.run_stream([Message(role="user", content="What's the weather in LA?")]):
    print(chunk, end="")
```

## Core Concepts

The framework organizes agent capabilities into four distinct concepts:

### Tools — What the agent can *do*

Tools are callable functions the agent invokes during its loop. Each tool has a name, description, JSON Schema parameters, and a handler function.

```python
# Static tool via decorator
@tool(name="search", description="Search docs", parameters={...})
def search(args):
    return json.dumps(results)

# Dynamic tool via factory (for closures capturing state)
search_tool = create_tool("search", "Search docs", {...}, handler=my_handler)
```

See [tools/](3_tools/) for templates.

### Skills — *How* to execute a task

Skills are instruction documents (markdown) loaded into the agent's system prompt. They define procedures, output formats, constraints, and examples.

```python
skill_text = Path("2_skills/data_analysis.md").read_text()
harness = AgentHarness(
    client=client,
    system_prompt=f"You are an analyst.\n\n{skill_text}",
    tools=[...],
)
```

See [skills/](2_skills/) for templates.

### Roles — *Who* a sub-agent is

Roles are system prompts that define sub-agent identities for the agent-as-tool pattern. A tool handler creates a new AgentHarness with a specialized role.

```python
role_prompt = Path("1_roles/researcher.md").read_text()

@tool(name="run_researcher", description="Research a topic", parameters={...})
def run_researcher(args):
    sub = AgentHarness(client, role_prompt, tools=[search_tool])
    return sub.run([Message(role="user", content=args["topic"])])

main_agent = AgentHarness(client, "You are an orchestrator.", tools=[run_researcher])
```

See [roles/](1_roles/) for templates.

### Hooks — Automatic safety checks

Hooks run before and after every tool execution. They can block dangerous calls, sanitize arguments, and redact sensitive results.

```python
from agent_harness import Hook, HookResult

class SQLGuard(Hook):
    def before_tool_call(self, tool_name, args):
        if "DROP" in str(args).upper():
            return HookResult(allowed=False, reason="DROP statements are blocked")
        return HookResult(allowed=True)

harness = AgentHarness(client, prompt, tools=[...], hooks=[SQLGuard()])
```

See [hooks/](4_hooks/) for templates and built-in hooks.

## The Agent Loop

```
User Message
    │
    ▼
┌──────────────────────────────────┐
│  AgentHarness.run_stream()       │
│                                  │
│  1. Call model with messages +   │
│     tools schema                 │
│  2. Stream text chunks to caller │
│  3. If tool calls requested:     │
│     a. on_tool_call callback     │
│     b. Run before_tool_call      │◄── Hooks can BLOCK here
│        hooks                     │
│     c. Execute tool handler      │
│     d. Run after_tool_call       │◄── Hooks can REDACT here
│        hooks                     │
│     e. on_tool_result callback   │
│     f. Add result to messages    │
│     g. Go to step 1              │
│  4. If no tool calls: done       │
│                                  │
│  Safety: max_iterations limit    │
└──────────────────────────────────┘
    │
    ▼
  Response
```

## API Reference

### AgentHarness

```python
AgentHarness(
    client: ModelClient,           # LLM provider client
    system_prompt: str,            # System prompt for the agent
    tools: list[Tool] = None,      # Tools the agent can call
    max_iterations: int = 30,      # Safety limit for tool loops
    hooks: list[Hook] = None,      # Safety hooks (before/after tools)
    on_tool_call: Callable = None, # Callback: tool is about to execute
    on_tool_result: Callable = None,   # Callback: tool finished
    on_tool_call_message: Callable = None,  # Callback: assistant requested tools
)
```

**Methods:**
- `run_stream(messages) -> Generator[str]` — Streaming agent loop, yields text chunks
- `run(messages) -> str` — Blocking agent loop, returns complete text

### Tool

```python
# Decorator
@tool(name="...", description="...", parameters={...})
def my_tool(args: dict[str, Any]) -> str: ...

# Factory
my_tool = create_tool(name="...", description="...", parameters={...}, handler=fn)
```

Handler signature: `(args: dict[str, Any]) -> str`

### Hook

```python
class MyHook(Hook):
    def before_tool_call(self, tool_name: str, args: dict) -> HookResult:
        return HookResult(allowed=True)  # or allowed=False, reason="..."

    def after_tool_call(self, tool_name: str, args: dict, result: str) -> str:
        return result  # or modified result
```

### Message

```python
Message(role="user", content="Hello")
Message(role="user", content=[  # Multi-modal
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
    {"type": "text", "text": "What's in this image?"},
])
```

### Clients

```python
# OpenAI-compatible (OpenAI, Azure AI Foundry, Ollama, vLLM, etc.)
client = OpenAICompatibleClient(base_url="...", api_key="...", model="gpt-5.4")

# Azure OpenAI
client = AzureModelClient(endpoint="...", deployment="...", api_key="...")

# Anthropic Claude
client = AnthropicClient(api_key="...", model="claude-opus-4-7")
```

All clients implement the `ModelClient` protocol: `chat_stream(messages, tools) -> Generator[StreamChunk]`

## Examples

| Example | Pattern |
|---------|---------|
| [01_basic_agent.py](3_tools/examples/01_basic_agent.py) | Minimal agent: client + prompt + tool + run |
| [02_hooks.py](4_hooks/examples/02_hooks.py) | Safety hooks that block/filter tool calls |
| [03_sub_agent.py](1_roles/examples/03_sub_agent.py) | Agent-as-tool with roles |
| [04_skills.py](2_skills/examples/04_skills.py) | Loading skill documents into prompts |
| [05_parallel_batch.py](src/examples/05_parallel_batch.py) | Parallel batch processing |
| [06_sse_streaming.py](src/examples/06_sse_streaming.py) | Background thread + SSE streaming |
| [07_provider_switching.py](src/examples/07_provider_switching.py) | Config-driven provider selection |
| [08_multimodal.py](src/examples/08_multimodal.py) | Images + text in messages |

## Design Principles

- **Minimal**: ~1,300 lines total, no magic
- **Provider-agnostic**: OpenAI protocol as common format, Anthropic translation built-in
- **No framework lock-in**: Works with any web framework or as standalone scripts
- **Composable**: Agents are tools, tools are functions, skills are text, hooks are classes
- **Backward-compatible**: Hooks and callbacks are optional — existing code works without them
