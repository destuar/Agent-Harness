"""Parallel batch processing with ThreadPoolExecutor.

Splits a workload into batches, processes each batch in its own
AgentHarness instance (with its own conversation state), and
synthesizes the results with a final harness call.

Requirements:
    pip install agent-harness[openai]
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent_harness import AgentHarness, OpenAICompatibleClient, tool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ITEMS = [
    "Quarterly revenue rose 12% YoY.",
    "Customer churn increased to 8%.",
    "New product launch exceeded targets by 20%.",
    "Operating costs grew 5%, below inflation.",
    "Employee satisfaction score dropped 3 points.",
    "Market share in APAC expanded to 15%.",
    "R&D spend as % of revenue hit a 5-year high.",
    "Net promoter score improved by 7 points.",
]

BATCH_SIZE = 3
MAX_WORKERS = 3
BATCH_TIMEOUT = 60  # seconds

# ---------------------------------------------------------------------------
# Shared progress counter
# ---------------------------------------------------------------------------

progress_lock = threading.Lock()
progress_done = 0


def _make_client() -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model="gpt-4o",
    )

# ---------------------------------------------------------------------------
# Tool available to each batch agent
# ---------------------------------------------------------------------------

@tool(name="classify", description="Classify a business metric as positive, negative, or neutral.")
def classify(metric: str) -> str:
    # Mock classifier
    neg_words = {"dropped", "churn", "increased to 8"}
    if any(w in metric.lower() for w in neg_words):
        return "negative"
    return "positive"

# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(batch_id: int, items: list[str]) -> str:
    global progress_done
    harness = AgentHarness(
        client=_make_client(),
        system_prompt=(
            "You are an analyst.  For each item, use the classify tool, "
            "then return a bullet list of item + classification."
        ),
        tools=[classify],
        max_iterations=15,
    )
    prompt = "Classify these metrics:\n" + "\n".join(f"- {i}" for i in items)
    result = harness.run(prompt)
    with progress_lock:
        progress_done += 1
        print(f"  [batch {batch_id}] done ({progress_done} batches complete)")
    return result


def split_batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    batches = split_batches(ITEMS, BATCH_SIZE)
    print(f"Processing {len(ITEMS)} items in {len(batches)} batches ...\n")

    results: dict[int, str] = {}
    failures: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_batch, idx, batch): idx
            for idx, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result(timeout=BATCH_TIMEOUT)
            except Exception as exc:
                failures[idx] = str(exc)
                print(f"  [batch {idx}] FAILED: {exc}")

    if failures:
        print(f"\n{len(failures)} batch(es) failed; synthesizing from partial results.")

    # Synthesize with a final harness call
    combined = "\n\n".join(
        f"### Batch {i}\n{text}" for i, text in sorted(results.items())
    )

    synthesizer = AgentHarness(
        client=_make_client(),
        system_prompt="Summarize the batch results into a single executive brief.",
    )
    summary = synthesizer.run(f"Here are the batch analyses:\n\n{combined}")
    print("\n=== Executive Summary ===")
    print(summary)


if __name__ == "__main__":
    main()
