"""Agent-as-tool: a main orchestrator delegates research to a sub-agent.

The sub-agent has its own system prompt, tools, and iteration budget.
It runs synchronously inside the tool handler and returns its final
answer to the outer agent.

Requirements:
    pip install agent-harness[openai]
"""

import os

from agent_harness import AgentHarness, OpenAICompatibleClient, tool

# ---------------------------------------------------------------------------
# Shared client (reused by both agents)
# ---------------------------------------------------------------------------

def _make_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model="gpt-4o",
    )

# ---------------------------------------------------------------------------
# Sub-agent: researcher
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = """\
You are a research assistant.  When given a topic, use the search tool to
gather facts, then write a concise 2-3 sentence summary.  Cite your sources
inline.
"""


@tool(name="search", description="Search the web for a query and return snippets.")
def search(query: str) -> str:
    # Mock search results
    return (
        f"Results for '{query}':\n"
        "1. Example Corp reported record Q3 earnings (source: reuters.com)\n"
        "2. The global market grew 4% YoY (source: bloomberg.com)\n"
    )


@tool(name="research", description="Delegate a research question to a specialist agent.")
def research(question: str) -> str:
    """Spin up a sub-agent, run to completion, return its answer."""
    sub = AgentHarness(
        client=_make_client(),
        system_prompt=RESEARCHER_PROMPT,
        tools=[search],
        max_iterations=10,
    )
    return sub.run(question)

# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = """\
You are a senior analyst.  Use the research tool to gather background on a
topic, then synthesize the findings into a final recommendation.
"""


def main() -> None:
    orchestrator = AgentHarness(
        client=_make_client(),
        system_prompt=ORCHESTRATOR_PROMPT,
        tools=[research],
        max_iterations=10,
    )

    result = orchestrator.run(
        "Should we invest in Example Corp?  Research their recent performance."
    )
    print(result)


if __name__ == "__main__":
    main()
