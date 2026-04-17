"""Skills: compose behavioural documents into an agent's system prompt.

A "skill" is just a block of text (typically loaded from a .md file) that
gives the agent domain knowledge or behavioural rules.  This example
shows how to combine a base prompt with one or more skill documents.

Requirements:
    pip install agent-harness[openai]
"""

import os

from agent_harness import AgentHarness, OpenAICompatibleClient, tool

# ---------------------------------------------------------------------------
# Skills (in production these would be read from files)
# ---------------------------------------------------------------------------

SQL_SKILL = """\
## SQL Skill
- Always use parameterized queries; never interpolate user input.
- Prefer CTEs over nested sub-selects for readability.
- Always include a LIMIT clause on exploratory queries.
- When asked for "recent" data, default to the last 30 days.
"""

STYLE_SKILL = """\
## Communication Style
- Be concise: aim for 1-3 sentences unless asked for detail.
- Use plain language; avoid jargon unless the user is technical.
- When showing SQL, format it with uppercase keywords.
"""

# ---------------------------------------------------------------------------
# Prompt composition
# ---------------------------------------------------------------------------

BASE_PROMPT = "You are a data analyst assistant."


def build_prompt(*skills: str) -> str:
    parts = [BASE_PROMPT, ""]  # blank line separator
    for skill in skills:
        parts.append(skill)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@tool(name="run_sql", description="Execute SQL against the analytics database.")
def run_sql(query: str) -> str:
    return f"[mock] 12 rows returned for: {query[:80]}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model="gpt-4o",
    )

    prompt = build_prompt(SQL_SKILL, STYLE_SKILL)

    harness = AgentHarness(
        client=client,
        system_prompt=prompt,
        tools=[run_sql],
    )

    result = harness.run("Show me recent signups grouped by country.")
    print(result)


if __name__ == "__main__":
    main()
