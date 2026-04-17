# Skills

Skills define **how an agent executes a task**. A skill is an instruction document -- typically Markdown -- that is loaded into the agent's system prompt or message context to give it step-by-step guidance for a specific kind of work.

## Skills Are Not Code

Unlike tools (which are Python functions), skills are **text**. They live in `.md` files and contain:

- Step-by-step procedures
- Output format specifications
- Constraints and guardrails
- Examples of expected input/output

The agent reads the skill text and follows the instructions using its reasoning and any available tools.

## Loading a Skill

```python
from pathlib import Path
from agent_harness import AgentHarness, Message

# Read the skill document
skill_text = Path("2_skills/data_analysis.md").read_text()

# Inject it into the system prompt
system_prompt = f"""You are a data analyst assistant.

{skill_text}
"""

harness = AgentHarness(
    client=client,
    system_prompt=system_prompt,
    tools=[query_tool, chart_tool],
)
```

## Composing Multiple Skills

Skills are composable. Load several skill documents into one prompt to give the agent a broader set of capabilities:

```python
analysis_skill = Path("2_skills/data_analysis.md").read_text()
formatting_skill = Path("2_skills/report_formatting.md").read_text()

system_prompt = f"""You are a data analyst assistant.

{analysis_skill}

{formatting_skill}
"""
```

Order can matter -- place higher-priority skills first, since models tend to weight earlier context more heavily.

## Skill vs. Role

| Concept | What it defines | Format | How it is used |
|---------|----------------|--------|----------------|
| **Skill** | HOW to do something | Procedure / instructions | Appended to any agent's prompt |
| **Role** | WHO the agent is | Identity / persona | Used as the core system prompt |

A single agent often combines one **role** with one or more **skills**:

```
Role: "You are a Research Agent..."
  + Skill: "How to search and cite sources"
  + Skill: "How to format findings as structured JSON"
```

## Writing Good Skills

1. **Be specific.** Vague instructions produce vague results. Say "Return a JSON object with keys `summary`, `confidence`, and `sources`" rather than "Return structured output."
2. **Include examples.** A single concrete input/output example is worth a paragraph of explanation.
3. **State constraints explicitly.** If the agent should never do something, say so.
4. **Keep skills focused.** One skill, one task. Compose them for complex workflows.
5. **Test iteratively.** Run the agent with the skill, read the output, and tighten the instructions.

## Example in This Directory

| File | Description |
|------|-------------|
| `template_skill.md` | Template showing the recommended skill document structure |
