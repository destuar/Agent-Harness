"""Example tool using the create_tool() factory with a closure.

This template demonstrates how to create tools dynamically when the handler
needs access to external state -- a database connection, an API client, a
configuration object, etc. The closure captures the dependency so the handler
signature stays clean: (args: dict) -> str.

Usage:
    search_service = MySearchService(...)
    search_tool = create_search_tool(search_service)

    harness = AgentHarness(client=client, system_prompt="...", tools=[search_tool])
"""

from __future__ import annotations

import json
from typing import Any, Protocol

from agent_harness import Tool, create_tool


# ---------------------------------------------------------------------------
# Define the interface your tool depends on.  Using a Protocol keeps the
# tool decoupled from any concrete implementation.
# ---------------------------------------------------------------------------
class SearchService(Protocol):
    """Protocol for a search backend the tool can call."""

    def search(self, query: str, top: int = 5) -> list[dict[str, Any]]:
        """Return a list of result dicts with at least 'title' and 'snippet'."""
        ...


# ---------------------------------------------------------------------------
# Factory function -- this is the pattern
# ---------------------------------------------------------------------------
def create_search_tool(search_service: SearchService) -> Tool:
    """Create a knowledge-base search tool backed by *search_service*.

    The returned Tool captures *search_service* in a closure so the agent
    harness can invoke it without knowing anything about the underlying
    search implementation.

    Args:
        search_service: Any object implementing the SearchService protocol.

    Returns:
        A Tool instance ready to pass to AgentHarness.
    """

    def search_handler(args: dict) -> str:
        query = args.get("query", "")
        top = args.get("top", 5)

        if not query.strip():
            return json.dumps({"error": "Query must not be empty"})

        results = search_service.search(query, top=top)
        return json.dumps({"query": query, "count": len(results), "results": results})

    return create_tool(
        name="search_knowledge_base",
        description="Search the knowledge base for relevant documents",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query",
                },
                "top": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5)",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["query"],
        },
        handler=search_handler,
    )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # A stub search service for demonstration purposes.
    class StubSearchService:
        """In-memory search service that matches on substring."""

        def __init__(self, documents: list[dict[str, str]]) -> None:
            self.documents = documents

        def search(self, query: str, top: int = 5) -> list[dict[str, Any]]:
            query_lower = query.lower()
            matches = [
                doc for doc in self.documents if query_lower in doc["title"].lower() or query_lower in doc["snippet"].lower()
            ]
            return matches[:top]

    # Create a service with sample data
    service = StubSearchService(
        documents=[
            {"title": "Agent Architecture", "snippet": "An overview of multi-agent system design patterns."},
            {"title": "Tool Use in LLMs", "snippet": "How large language models invoke external tools."},
            {"title": "RAG Pipelines", "snippet": "Retrieval-augmented generation for knowledge-grounded agents."},
            {"title": "Prompt Engineering", "snippet": "Techniques for crafting effective system prompts."},
        ]
    )

    # Build the tool via the factory
    search_tool = create_search_tool(service)

    print("Tool name:", search_tool.name)
    print("Tool description:", search_tool.description)
    print()

    # Call the handler directly
    test_args = {"query": "agent", "top": 3}
    result = search_tool.handler(test_args)
    print(f"Handler result for {test_args}:")
    print(json.dumps(json.loads(result), indent=2))
