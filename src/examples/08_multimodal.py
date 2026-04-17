"""Multi-modal messages: send images alongside text to an agent.

Reads a local image file, encodes it as a base64 data URI, and builds
a Message with a content list containing both image and text blocks
(following the OpenAI chat-completions content format).

Requirements:
    pip install agent-harness[openai]
"""

import base64
import mimetypes
import os
import sys

from agent_harness import AgentHarness, Message, OpenAICompatibleClient


def image_to_data_uri(path: str) -> str:
    """Read a local image and return a base64 data URI."""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{encoded}"


def build_image_message(image_path: str, prompt: str) -> Message:
    """Build a user Message with an image and a text block."""
    data_uri = image_to_data_uri(image_path)
    return Message(
        role="user",
        content=[
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ],
    )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python 08_multimodal.py <image_path> [prompt]")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image in detail."

    client = OpenAICompatibleClient(
        api_endpoint=os.environ["API_ENDPOINT"],
        api_key=os.environ["API_KEY"],
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
    )

    harness = AgentHarness(
        client=client,
        system_prompt="You are a vision assistant that analyzes images.",
    )

    message = build_image_message(image_path, prompt)
    result = harness.run(message)
    print(result)


if __name__ == "__main__":
    main()
