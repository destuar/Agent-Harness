"""Background-thread streaming with dual queues and SSE-formatted output.

Runs AgentHarness in a background thread.  Text chunks and tool-call
progress events are pushed into separate queues.  A generator polls
both queues and yields Server-Sent Events (SSE) strings, including
keepalive comments every 15 seconds.

No web framework is required -- the generator can be plugged into any
ASGI/WSGI handler that accepts an iterable of bytes.

Requirements:
    pip install agent-harness[openai]
"""

import json
import os
import queue
import threading
import time

from agent_harness import AgentHarness, OpenAICompatibleClient, tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool(name="fetch_data", description="Fetch rows from the analytics API.")
def fetch_data(metric: str) -> str:
    time.sleep(1)  # simulate latency
    return f'{{"metric": "{metric}", "value": 42, "trend": "up"}}'

# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(event_type: str, data: dict) -> str:
    """Format a single SSE event."""
    payload = json.dumps({"type": event_type, **data})
    return f"data: {payload}\n\n"


def sse_stream(
    client: OpenAICompatibleClient,
    user_message: str,
) -> "Generator[str, None, None]":
    """Yield SSE-formatted events from a harness run."""

    chunk_q: queue.Queue[str | None] = queue.Queue()
    progress_q: queue.Queue[dict] = queue.Queue()
    done_event = threading.Event()

    # -- callbacks ----------------------------------------------------------

    def on_tool_call(call_id: str, name: str, args: dict) -> None:
        progress_q.put({"tool": name, "status": "started", "args": args})

    def on_tool_result(call_id: str, name: str, result: str) -> None:
        progress_q.put({"tool": name, "status": "completed"})

    # -- background thread --------------------------------------------------

    def _run() -> None:
        harness = AgentHarness(
            client=client,
            system_prompt="You are an analytics assistant. Use fetch_data for metrics.",
            tools=[fetch_data],
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
        for chunk in harness.run_stream(user_message):
            chunk_q.put(chunk)
        chunk_q.put(None)  # sentinel
        done_event.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # -- main-thread generator ----------------------------------------------

    last_keepalive = time.monotonic()
    KEEPALIVE_INTERVAL = 15.0

    while not done_event.is_set() or not chunk_q.empty() or not progress_q.empty():
        # Drain progress events first
        try:
            while True:
                evt = progress_q.get_nowait()
                yield _sse("progress", evt)
        except queue.Empty:
            pass

        # Drain text chunks
        try:
            while True:
                chunk = chunk_q.get(timeout=0.1)
                if chunk is None:
                    break
                yield _sse("text", {"content": chunk})
        except queue.Empty:
            pass

        # Keepalive
        now = time.monotonic()
        if now - last_keepalive >= KEEPALIVE_INTERVAL:
            yield ": keepalive\n\n"
            last_keepalive = now

    yield _sse("done", {})

# ---------------------------------------------------------------------------
# Main -- demonstrate the generator
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model="gpt-4o",
    )

    print("--- SSE event stream ---\n")
    for event in sse_stream(client, "What is the current value of signups?"):
        print(event, end="")


if __name__ == "__main__":
    main()
