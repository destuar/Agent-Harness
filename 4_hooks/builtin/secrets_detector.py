"""Built-in hook: secrets detector.

Detects API keys, passwords, bearer tokens, AWS credentials, and other
secrets in tool arguments (before execution) and tool results (after
execution). Blocks tool calls that would send secrets outbound and redacts
secrets that appear in results.

Usage:
    from 4_hooks.builtin.secrets_detector import SecretsDetectorHook

    harness = AgentHarness(
        client=client,
        system_prompt="...",
        tools=[...],
        hooks=[SecretsDetectorHook()],
    )
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from agent_harness import Hook, HookResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for known secret formats
# ---------------------------------------------------------------------------
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # AWS Access Key ID (starts with AKIA, 20 uppercase alphanumeric chars)
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}")),
    # AWS Secret Access Key (40 base64-ish chars following common prefixes)
    ("AWS Secret Key", re.compile(r"(?:aws_secret_access_key|secret_key)\s*[:=]\s*[A-Za-z0-9/+=]{40}")),
    # Generic API key patterns (key=... or api_key=... with 20+ chars)
    ("API Key (generic)", re.compile(r"(?:api[_-]?key|apikey)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{20,}[\"']?", re.IGNORECASE)),
    # OpenAI API key
    ("OpenAI API Key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    # Bearer tokens
    ("Bearer Token", re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{20,}")),
    # GitHub personal access tokens
    ("GitHub Token", re.compile(r"gh[pousr]_[A-Za-z0-9_]{36,}")),
    # Slack tokens
    ("Slack Token", re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}")),
    # Azure connection strings
    ("Azure Connection String", re.compile(r"AccountKey=[A-Za-z0-9+/=]{40,}", re.IGNORECASE)),
    # Generic password fields in JSON/dicts
    ("Password field", re.compile(r"[\"']?(?:password|passwd|pwd|secret)[\"']?\s*[:=]\s*[\"'][^\"']{4,}[\"']", re.IGNORECASE)),
    # Private keys (PEM format)
    ("Private Key", re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----")),
    # JWT tokens (three base64url segments separated by dots)
    ("JWT Token", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    # Basic auth in URLs
    ("URL Credentials", re.compile(r"https?://[^:]+:[^@]+@")),
]


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string.

    High entropy (> 4.5 on 20+ character strings) is a heuristic signal
    that a value may be a secret key or token.
    """
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _find_high_entropy_strings(text: str, min_length: int = 20, threshold: float = 4.5) -> list[str]:
    """Find substrings that look like secrets based on entropy.

    Scans for contiguous alphanumeric+symbol tokens of at least *min_length*
    characters and returns those whose Shannon entropy exceeds *threshold*.
    """
    # Match long contiguous tokens that look like keys/tokens
    token_pattern = re.compile(r"[A-Za-z0-9+/=_\-]{%d,}" % min_length)
    suspicious: list[str] = []
    for match in token_pattern.finditer(text):
        token = match.group()
        # Skip common non-secret patterns (all digits, simple words, UUIDs)
        if token.isdigit():
            continue
        if token.isalpha() and token.islower():
            continue
        entropy = _shannon_entropy(token)
        if entropy >= threshold:
            suspicious.append(token)
    return suspicious


def _scan_for_secrets(text: str) -> list[tuple[str, str]]:
    """Scan text for secrets. Returns list of (pattern_name, matched_value)."""
    findings: list[tuple[str, str]] = []

    for pattern_name, pattern in _SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append((pattern_name, match.group()))

    # Entropy-based detection as a catch-all
    for token in _find_high_entropy_strings(text):
        # Skip if already caught by a named pattern
        already_caught = any(token in matched for _, matched in findings)
        if not already_caught:
            findings.append(("High-entropy string", token))

    return findings


def _redact_secrets(text: str) -> str:
    """Replace detected secrets in text with [REDACTED] placeholders."""
    redacted = text
    findings = _scan_for_secrets(text)

    # Sort by length descending so longer matches are replaced first,
    # preventing partial replacements of overlapping patterns
    findings.sort(key=lambda f: len(f[1]), reverse=True)

    for pattern_name, matched in findings:
        placeholder = f"[REDACTED:{pattern_name}]"
        redacted = redacted.replace(matched, placeholder)

    return redacted


class SecretsDetectorHook(Hook):
    """Detect and block/redact secrets in tool calls.

    Behavior:
        - **before_tool_call**: If secrets are detected in the arguments,
          the tool call is BLOCKED and the reason is returned to the model.
        - **after_tool_call**: If secrets are detected in the result, they
          are REDACTED before the result is returned to the model.

    Args:
        block_on_args: If True (default), block tool calls with secrets in args.
        redact_results: If True (default), redact secrets found in results.
        entropy_detection: If True (default), also flag high-entropy strings.
        extra_patterns: Additional (name, regex_pattern) tuples to scan for.
    """

    def __init__(
        self,
        block_on_args: bool = True,
        redact_results: bool = True,
        entropy_detection: bool = True,
        extra_patterns: list[tuple[str, str]] | None = None,
    ) -> None:
        self.block_on_args = block_on_args
        self.redact_results = redact_results
        self.entropy_detection = entropy_detection

        # Register any extra patterns provided by the caller
        if extra_patterns:
            for name, pattern_str in extra_patterns:
                _SECRET_PATTERNS.append((name, re.compile(pattern_str)))

    def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """Block tool calls that contain secrets in their arguments."""
        if not self.block_on_args:
            return HookResult(allowed=True)

        args_text = json.dumps(args, default=str)
        findings = _scan_for_secrets(args_text)

        if not self.entropy_detection:
            findings = [(name, val) for name, val in findings if name != "High-entropy string"]

        if findings:
            pattern_names = sorted(set(name for name, _ in findings))
            reason = (
                f"Blocked: tool arguments contain potential secrets "
                f"({', '.join(pattern_names)}). Remove credentials before retrying."
            )
            logger.warning(
                "SecretsDetectorHook blocked tool '%s': detected %s",
                tool_name,
                pattern_names,
            )
            return HookResult(allowed=False, reason=reason)

        return HookResult(allowed=True)

    def after_tool_call(self, tool_name: str, args: dict[str, Any], result: str) -> str:
        """Redact secrets found in tool results."""
        if not self.redact_results:
            return result

        redacted = _redact_secrets(result)
        if redacted != result:
            logger.info(
                "SecretsDetectorHook redacted secrets from tool '%s' result",
                tool_name,
            )
        return redacted


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hook = SecretsDetectorHook()

    # Test 1: Block args containing an AWS key
    print("--- Test 1: AWS key in args ---")
    result = hook.before_tool_call("http_request", {
        "url": "https://api.example.com",
        "headers": {"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnopqrstuvwxyz"},
    })
    print(f"  Allowed: {result.allowed}")
    print(f"  Reason: {result.reason}")

    # Test 2: Allow clean args
    print("\n--- Test 2: Clean args ---")
    result = hook.before_tool_call("search", {"query": "annual revenue 2024"})
    print(f"  Allowed: {result.allowed}")

    # Test 3: Redact secrets in results
    print("\n--- Test 3: Redact results ---")
    raw = json.dumps({
        "config": "api_key=sk-abc123def456ghi789jkl012mno",
        "status": "connected",
    })
    cleaned = hook.after_tool_call("read_config", {}, raw)
    print(f"  Original: {raw}")
    print(f"  Redacted: {cleaned}")
