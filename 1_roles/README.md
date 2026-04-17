# Roles

Roles define **who a sub-agent is**. A role is a system prompt that establishes an agent's identity, capabilities, decision-making rules, and behavioral constraints. Roles are the foundation of the **agent-as-tool** pattern, where a top-level agent delegates work to specialized sub-agents.

## Role vs. Skill

| Concept | Defines | Analogy |
|---------|---------|---------|
| **Role** | WHO the agent is | A job title and job description |
| **Skill** | HOW to do a task | A page from the employee handbook |

A role is the core system prompt. Skills are optional additions that give the agent specific procedural knowledge.

## The Agent-as-Tool Pattern

The key use case for roles is creating sub-agents that a parent agent can invoke as tools. Each sub-agent gets its own `AgentHarness` with a role prompt, its own set of tools, and a focused mission.

```python
from pathlib import Path
from agent_harness import AgentHarness, Message, Tool, create_tool

# Load the role definition
role_prompt = Path("1_roles/researcher.md").read_text()

def create_researcher_tool(client, search_tool: Tool) -> Tool:
    """Create a tool that delegates to a Research Agent sub-agent."""

    def run_researcher(args: dict) -> str:
        sub_harness = AgentHarness(
            client=client,
            system_prompt=role_prompt,
            tools=[search_tool],
            max_iterations=10,
        )
        topic = args["topic"]
        result = sub_harness.run([Message(role="user", content=topic)])
        return result

    return create_tool(
        name="research",
        description="Delegate a research question to the Research Agent",
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research question or topic to investigate",
                },
            },
            "required": ["topic"],
        },
        handler=run_researcher,
    )
```

The parent agent sees `research` as a normal tool. Under the hood, it spins up a full agent loop with its own identity and capabilities.

## Writing Good Roles

1. **Start with identity.** The first sentence should tell the agent who it is: "You are a Research Agent specialized in..."
2. **List capabilities.** Enumerate what the agent can do and which tools it has access to.
3. **Define decision rules.** When should it search? When should it stop? How should it handle ambiguity?
4. **Specify output format.** What should the sub-agent return to its caller?
5. **Set safety constraints.** What should the agent never do?
6. **Keep roles focused.** A role should describe one specialist. If you need a generalist, compose multiple sub-agents under an orchestrator.

## Example in This Directory

| File | Description |
|------|-------------|
| `template_role.md` | Template showing the recommended role document structure |
