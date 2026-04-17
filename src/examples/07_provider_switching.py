"""Config-driven client selection: OpenAI, Azure, or Anthropic.

Set API_PROVIDER to one of "openai", "azure", or "anthropic" (defaults to
"openai").  Each provider reads its own set of environment variables.

Requirements:
    pip install agent-harness[openai]      # for OpenAI / Azure
    pip install agent-harness[anthropic]   # for Anthropic
"""

import os

from agent_harness import AgentHarness, ModelClient, tool


def make_client() -> ModelClient:
    """Build the right client based on API_PROVIDER."""
    provider = os.environ.get("API_PROVIDER", "openai").lower()

    if provider == "anthropic":
        from agent_harness import AnthropicClient

        return AnthropicClient(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        )

    elif provider == "azure":
        from agent_harness import AzureModelClient

        return AzureModelClient(
            api_endpoint=os.environ["AZURE_ENDPOINT"],
            api_key=os.environ["AZURE_API_KEY"],
            deployment=os.environ["AZURE_DEPLOYMENT"],
            api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
        )

    else:  # default: openai-compatible
        from agent_harness import OpenAICompatibleClient

        return OpenAICompatibleClient(
            api_endpoint=os.environ["API_ENDPOINT"],
            api_key=os.environ["API_KEY"],
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        )


# ---------------------------------------------------------------------------
# Tool + harness
# ---------------------------------------------------------------------------

@tool(name="echo", description="Echo the input back.")
def echo(text: str) -> str:
    return text


def main() -> None:
    client = make_client()
    print(f"Using client: {type(client).__name__}")

    harness = AgentHarness(
        client=client,
        system_prompt="You are a helpful assistant.",
        tools=[echo],
    )

    result = harness.run("Say hello using the echo tool.")
    print(result)


if __name__ == "__main__":
    main()
