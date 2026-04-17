"""Minimal agent: one tool, streaming and non-streaming usage.

Requirements:
    pip install agent-harness[openai]

Set API_ENDPOINT and API_KEY in your environment before running.
"""

import os

from agent_harness import AgentHarness, OpenAICompatibleClient, tool


@tool(name="get_weather", description="Return the current weather for a city.")
def get_weather(city: str) -> str:
    # Mock implementation
    forecasts = {
        "seattle": "62 F, cloudy",
        "phoenix": "105 F, sunny",
        "new york": "74 F, partly cloudy",
    }
    return forecasts.get(city.lower(), f"No data for {city}")


def main() -> None:
    client = OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model="gpt-4o",
    )

    harness = AgentHarness(
        client=client,
        system_prompt="You are a helpful weather assistant.",
        tools=[get_weather],
    )

    # --- Streaming usage ---
    print("=== Streaming ===")
    for chunk in harness.run_stream("What is the weather in Seattle?"):
        print(chunk, end="", flush=True)
    print()

    # --- Non-streaming usage ---
    print("\n=== Non-streaming ===")
    result = harness.run("Compare the weather in Phoenix and New York.")
    print(result)


if __name__ == "__main__":
    main()
