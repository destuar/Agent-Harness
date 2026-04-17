"""Built-in hook: SQL safety guard.

Enforces SQL safety policies on any tool that accepts SQL queries:
- Only SELECT statements are allowed
- Dangerous keywords (INSERT, UPDATE, DELETE, DROP, etc.) are blocked
- Optional table whitelist restricts which tables can be queried
- Optional row limit injects TOP N or LIMIT N to prevent unbounded results

This is a generalized version of the SQL validator pattern, suitable for
any project where an LLM agent executes SQL against a real database.

Usage:
    from 4_hooks.builtin.sql_guard import SQLGuardHook

    harness = AgentHarness(
        client=client,
        system_prompt="...",
        tools=[sql_tool],
        hooks=[
            SQLGuardHook(
                tool_names=["run_sql", "query_database"],
                allowed_tables=["patients", "encounters", "departments"],
                max_rows=500,
            ),
        ],
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent_harness import Hook, HookResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dangerous SQL keywords that indicate non-SELECT operations
# ---------------------------------------------------------------------------
_DANGEROUS_KEYWORDS: list[str] = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "EXEC",
    "EXECUTE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "CALL",
    "REPLACE",  # MySQL REPLACE INTO
    "RENAME",
    "sp_",      # SQL Server stored procedures
    "xp_",      # SQL Server extended stored procedures
]

# Regex that matches any of the dangerous keywords as whole words,
# case-insensitive, at the start of a statement or after a semicolon.
_DANGEROUS_PATTERN: re.Pattern[str] = re.compile(
    r"(?:^|;\s*)\s*(" + "|".join(re.escape(kw) for kw in _DANGEROUS_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# Patterns for SQL comment injection that could hide dangerous statements
_COMMENT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"/\*.*?\*/", re.DOTALL),  # Block comments
    re.compile(r"--[^\n]*"),               # Line comments
]


def _strip_sql_comments(sql: str) -> str:
    """Remove SQL comments to prevent comment-based injection."""
    cleaned = sql
    for pattern in _COMMENT_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return cleaned


def _extract_table_names(sql: str) -> list[str]:
    """Extract table names from a SQL SELECT statement.

    This is a best-effort heuristic that catches the common patterns:
    - FROM table_name
    - JOIN table_name
    - FROM schema.table_name (extracts table_name)

    It does not handle every edge case (subqueries aliased as tables, CTEs,
    etc.) but is sufficient for most agent-generated SQL.
    """
    # Normalize whitespace
    normalized = re.sub(r"\s+", " ", sql.strip())

    tables: list[str] = []

    # Match FROM and JOIN clauses
    # Handles: FROM table, FROM schema.table, JOIN table, LEFT JOIN table, etc.
    table_ref_pattern = re.compile(
        r"(?:FROM|JOIN)\s+"
        r"(?:\[?(\w+)\]?\.)??"   # optional schema (group 1)
        r"\[?(\w+)\]?"           # table name (group 2)
        r"(?:\s+(?:AS\s+)?\w+)?",  # optional alias
        re.IGNORECASE,
    )

    for match in table_ref_pattern.finditer(normalized):
        table = match.group(2)
        if table and table.upper() not in ("SELECT", "WHERE", "ON", "SET", "VALUES"):
            tables.append(table.lower())

    return list(dict.fromkeys(tables))  # deduplicate, preserve order


def _has_row_limit(sql: str) -> bool:
    """Check if the query already has a TOP or LIMIT clause."""
    upper = sql.upper()
    # LIMIT at the end (MySQL, PostgreSQL, SQLite)
    if re.search(r"\bLIMIT\s+\d+", upper):
        return True
    # TOP after SELECT (SQL Server)
    if re.search(r"\bSELECT\s+TOP\s+\d+", upper):
        return True
    # FETCH FIRST N ROWS (ANSI SQL)
    if re.search(r"\bFETCH\s+FIRST\s+\d+\s+ROWS", upper):
        return True
    return False


def _inject_row_limit(sql: str, max_rows: int, dialect: str = "auto") -> str:
    """Inject a row limit into a SQL query.

    Args:
        sql: The SQL query string.
        max_rows: Maximum rows to return.
        dialect: "mssql" for TOP N, "ansi" for LIMIT N, "auto" to guess.

    Returns:
        The query with a row limit injected.
    """
    stripped = sql.strip().rstrip(";")

    if dialect == "auto":
        # Heuristic: if query uses square brackets or dbo., assume SQL Server
        if re.search(r"\[|\bdbo\b|\bTOP\b", sql, re.IGNORECASE):
            dialect = "mssql"
        else:
            dialect = "ansi"

    if dialect == "mssql":
        # Inject TOP N after SELECT (handles SELECT DISTINCT too)
        modified = re.sub(
            r"\bSELECT(\s+DISTINCT)?\s+",
            rf"SELECT\1 TOP {max_rows} ",
            stripped,
            count=1,
            flags=re.IGNORECASE,
        )
        return modified
    else:
        # ANSI / MySQL / PostgreSQL: append LIMIT N
        return f"{stripped} LIMIT {max_rows}"


class SQLGuardHook(Hook):
    """Enforce SQL safety policies on tool calls.

    Args:
        tool_names: Names of tools this hook applies to. If None, applies to
            all tools whose args contain a "query" key.
        query_param: The key in args that holds the SQL string (default: "query").
        allowed_tables: If set, only queries against these tables are permitted.
            Table names are matched case-insensitively.
        max_rows: If set, queries without a LIMIT/TOP clause will have one
            injected automatically. Set to None to skip row limiting.
        sql_dialect: "auto" (default), "mssql", or "ansi". Controls how row
            limits are injected.
        allow_comments: If False (default), SQL comments are stripped before
            validation to prevent comment-based injection.
    """

    def __init__(
        self,
        tool_names: list[str] | None = None,
        query_param: str = "query",
        allowed_tables: list[str] | None = None,
        max_rows: int | None = 1000,
        sql_dialect: str = "auto",
        allow_comments: bool = False,
    ) -> None:
        self.tool_names = tool_names
        self.query_param = query_param
        self.allowed_tables = [t.lower() for t in allowed_tables] if allowed_tables else None
        self.max_rows = max_rows
        self.sql_dialect = sql_dialect
        self.allow_comments = allow_comments

    def _applies_to(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if this hook should run for the given tool call."""
        if self.tool_names is not None:
            return tool_name in self.tool_names
        # If no tool_names specified, apply to any tool with the query param
        return self.query_param in args

    def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> HookResult:
        """Validate SQL query before execution."""
        if not self._applies_to(tool_name, args):
            return HookResult(allowed=True)

        query = args.get(self.query_param, "")
        if not isinstance(query, str) or not query.strip():
            return HookResult(allowed=False, reason="SQL query is empty or not a string.")

        # Strip comments unless explicitly allowed
        clean_query = query if self.allow_comments else _strip_sql_comments(query)

        # ---------------------------------------------------------------
        # Check 1: Only SELECT statements allowed
        # ---------------------------------------------------------------
        # Check the first non-whitespace keyword (after stripping comments)
        first_keyword = clean_query.strip().split()[0].upper() if clean_query.strip() else ""
        # Allow WITH (CTEs) that lead to SELECT
        if first_keyword not in ("SELECT", "WITH"):
            return HookResult(
                allowed=False,
                reason=f"Only SELECT statements are allowed. Got: {first_keyword}",
            )

        # ---------------------------------------------------------------
        # Check 2: No dangerous keywords anywhere in the query
        # ---------------------------------------------------------------
        dangerous_match = _DANGEROUS_PATTERN.search(clean_query)
        if dangerous_match:
            keyword = dangerous_match.group(1).upper()
            return HookResult(
                allowed=False,
                reason=f"Dangerous SQL keyword detected: {keyword}. Only read-only SELECT queries are permitted.",
            )

        # ---------------------------------------------------------------
        # Check 3: Table whitelist
        # ---------------------------------------------------------------
        if self.allowed_tables is not None:
            referenced_tables = _extract_table_names(clean_query)
            unauthorized = [t for t in referenced_tables if t not in self.allowed_tables]
            if unauthorized:
                return HookResult(
                    allowed=False,
                    reason=(
                        f"Query references unauthorized table(s): {', '.join(unauthorized)}. "
                        f"Allowed tables: {', '.join(self.allowed_tables)}"
                    ),
                )

        # ---------------------------------------------------------------
        # Check 4: Row limit enforcement
        # ---------------------------------------------------------------
        modified_args = None
        if self.max_rows is not None and not _has_row_limit(clean_query):
            limited_query = _inject_row_limit(query, self.max_rows, self.sql_dialect)
            modified_args = {**args, self.query_param: limited_query}
            logger.info(
                "SQLGuardHook injected row limit (%d) into query for tool '%s'",
                self.max_rows,
                tool_name,
            )

        return HookResult(allowed=True, modified_args=modified_args)

    def after_tool_call(self, tool_name: str, args: dict[str, Any], result: str) -> str:
        """Pass through results unchanged.

        Override this method if you need to redact or transform SQL results
        (e.g., masking PII columns).
        """
        return result


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    hook = SQLGuardHook(
        allowed_tables=["patients", "encounters", "departments"],
        max_rows=500,
    )

    test_cases: list[tuple[str, dict[str, Any]]] = [
        # Should PASS -- clean SELECT
        ("run_sql", {"query": "SELECT name, age FROM patients WHERE age > 30"}),
        # Should PASS with modification -- missing LIMIT
        ("run_sql", {"query": "SELECT * FROM encounters"}),
        # Should BLOCK -- DELETE statement
        ("run_sql", {"query": "DELETE FROM patients WHERE id = 5"}),
        # Should BLOCK -- DROP TABLE
        ("run_sql", {"query": "DROP TABLE patients"}),
        # Should BLOCK -- unauthorized table
        ("run_sql", {"query": "SELECT * FROM secret_admin_table"}),
        # Should PASS -- CTE with SELECT
        ("run_sql", {"query": "WITH recent AS (SELECT * FROM encounters WHERE date > '2024-01-01') SELECT * FROM recent LIMIT 10"}),
        # Should BLOCK -- comment injection hiding dangerous keyword
        ("run_sql", {"query": "SELECT 1; /* */ DROP TABLE patients"}),
        # Should PASS -- unrelated tool
        ("search", {"query": "find all patients"}),
    ]

    for tool_name, args in test_cases:
        result = hook.before_tool_call(tool_name, args)
        status = "ALLOWED" if result.allowed else "BLOCKED"
        detail = ""
        if result.reason:
            detail = f" -- {result.reason}"
        if result.modified_args:
            detail += f" (modified query: {result.modified_args.get('query', '')!r})"
        print(f"  [{status}] {tool_name}({args.get('query', '')!r}){detail}")
