# Skill: Data Analysis

## Purpose

Enable the agent to analyze tabular datasets by writing and executing SQL queries, interpreting results, and producing structured summaries with key findings.

## When to Use

Activate this skill when the user asks to:
- Explore, summarize, or profile a dataset
- Answer quantitative questions about data
- Identify trends, outliers, or distributions
- Compare groups or time periods within the data

## Instructions

Follow these steps in order:

1. **Understand the request.** Restate the user's question in your own words to confirm understanding. Identify the key metrics, dimensions, and filters involved.

2. **Inspect the schema.** Use the available schema/metadata tool to list tables and columns relevant to the question. Note column types and any NULL patterns.

3. **Write the query.** Construct a SQL SELECT statement that answers the question. Follow these rules:
   - Use only SELECT statements (no INSERT, UPDATE, DELETE, or DDL).
   - Apply appropriate filters (WHERE), groupings (GROUP BY), and sorting (ORDER BY).
   - Limit results to a reasonable size (no more than 1,000 rows unless explicitly requested).
   - Use aliases for readability.

4. **Execute and validate.** Run the query. Check that the result set is non-empty and that values are plausible. If no rows are returned, re-examine filters and try a broader query.

5. **Interpret results.** Describe what the data shows in plain language. Highlight:
   - Key numbers (totals, averages, min/max)
   - Notable patterns or anomalies
   - Caveats or limitations (e.g., missing data, small sample size)

6. **Format the output.** Present findings in the output format defined below.

## Output Format

Return a JSON object with the following structure:

```json
{
  "question": "The user's original question, restated clearly",
  "sql": "The SQL query that was executed",
  "summary": "A 2-4 sentence plain-language summary of the findings",
  "key_metrics": [
    {"label": "Total Records", "value": 12345},
    {"label": "Average Score", "value": 87.3}
  ],
  "caveats": ["Any limitations or notes about data quality"]
}
```

## Constraints

- Never execute INSERT, UPDATE, DELETE, DROP, or any DDL statement.
- Never expose raw personally identifiable information (PII) in the summary. Aggregate or anonymize.
- If the dataset is too large to scan efficiently, sample or apply filters rather than querying the entire table.
- If you are uncertain about a finding, state the uncertainty explicitly in the `caveats` field.
- Do not fabricate data. If the query returns no results, say so.

## Examples

### Input

> "What is the average length of stay by department for 2024?"

### Expected Output

```json
{
  "question": "What is the average length of stay, broken down by department, for the year 2024?",
  "sql": "SELECT department, AVG(length_of_stay) AS avg_los FROM encounters WHERE discharge_date >= '2024-01-01' AND discharge_date < '2025-01-01' GROUP BY department ORDER BY avg_los DESC",
  "summary": "In 2024, the ICU had the longest average length of stay at 6.8 days, followed by Surgery at 4.2 days. The Emergency department had the shortest at 0.9 days. Overall average across all departments was 3.1 days.",
  "key_metrics": [
    {"label": "ICU Avg LOS", "value": 6.8},
    {"label": "Surgery Avg LOS", "value": 4.2},
    {"label": "Overall Avg LOS", "value": 3.1}
  ],
  "caveats": ["Excludes encounters still in progress (no discharge date). Departments with fewer than 10 encounters may have unreliable averages."]
}
```
