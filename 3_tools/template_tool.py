"""Example tool using the @tool decorator.

This template demonstrates the simplest way to create a tool: the @tool decorator.
Use this pattern for standalone functions that don't need external state or
dependency injection.

Usage:
    from 3_tools.template_tool import get_weather

    harness = AgentHarness(client=client, system_prompt="...", tools=[get_weather])
"""

from __future__ import annotations

import json

from agent_harness import tool


@tool(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g. 'Los Angeles', 'New York')",
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units (default: fahrenheit)",
            },
        },
        "required": ["city"],
    },
)
def get_weather(args: dict) -> str:
    """Look up current weather for a city.

    Args:
        args: Dict with keys "city" (required) and "units" (optional).

    Returns:
        JSON string with weather data.
    """
    city = args["city"]
    units = args.get("units", "fahrenheit")

    # -----------------------------------------------------------------
    # Replace this block with an actual weather API call.
    # Example: response = requests.get(f"https://api.weather.com/...", params={...})
    # -----------------------------------------------------------------
    weather_data = {
        "city": city,
        "temperature": 72 if units == "fahrenheit" else 22,
        "units": units,
        "condition": "sunny",
        "humidity": 45,
    }

    return json.dumps(weather_data)


# ---------------------------------------------------------------------------
# Quick demo -- run this file directly to see how the tool integrates
# with AgentHarness.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from agent_harness import AgentHarness, Message

    # -- You need a real client to run this end-to-end. Example with OpenAI: --
    #
    # from agent_harness import OpenAICompatibleClient
    #
    # client = OpenAICompatibleClient(
    #     base_url="https://api.openai.com/v1",
    #     api_key="sk-...",
    #     model="gpt-4o",
    # )
    #
    # harness = AgentHarness(
    #     client=client,
    #     system_prompt="You are a helpful weather assistant.",
    #     tools=[get_weather],
    # )
    #
    # response = harness.run([Message(role="user", content="What's the weather in LA?")])
    # print(response)

    # -- Without a client, you can still test the tool handler directly: --
    print("Tool name:", get_weather.name)
    print("Tool description:", get_weather.description)
    print()

    test_args = {"city": "Los Angeles", "units": "fahrenheit"}
    result = get_weather.handler(test_args)
    print(f"Handler result for {test_args}:")
    print(json.dumps(json.loads(result), indent=2))
