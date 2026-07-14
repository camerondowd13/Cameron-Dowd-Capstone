"""
AccountResearcher: given a company name, research it and return structured
findings for downstream sales agents (industry, size, growth signals, etc.).

Search is backed by Exa (not Claude's built-in web_search tool) — Claude
Sonnet 5 runs an agentic tool-use loop where it decides what to search for,
we execute those searches against Exa, and it submits its final findings
through a schema-enforced tool call so the output is always exactly the
6 requested fields.
"""
import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv(".env.local")

MODEL = "claude-sonnet-5"
MAX_SEARCHES = 4
MAX_TURNS = MAX_SEARCHES + 2  # spare turns for the "limit reached" nudge + final submit

FIELDS = [
    "industry",
    "size_range",
    "growth_signals",
    "hiring_status",
    "tech_stack_hints",
    "recent_news",
]

SYSTEM_PROMPT = """You are a SaaS AE researching target accounts.
Given an account name, use web_search to find: industry, size_range,
growth_signals, hiring_status, tech_stack_hints, recent_news
(last 90 days). Return as structured JSON with EXACTLY those 6
fields. Use max 4 searches. If a field can't be found, use null —
NEVER hallucinate."""

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web via Exa and get back titles, URLs, and page text "
        "for the top results. Use focused queries, e.g. '<company> funding', "
        "'<company> hiring', '<company> tech stack', '<company> news'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query to run."}
        },
        "required": ["query"],
    },
}

SUBMIT_TOOL = {
    "name": "submit_account_research",
    "description": "Submit the final structured research findings for the account.",
    "input_schema": {
        "type": "object",
        "properties": {
            field: {
                "type": ["string", "null"],
                "description": f"The account's {field.replace('_', ' ')}, or null if not found.",
            }
            for field in FIELDS
        },
        "required": FIELDS,
    },
}

_NULLISH = {"null", "none", "n/a", "unknown", ""}


def _clean(value):
    """Normalize model output so 'null'-as-string collapses to real None."""
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def _run_exa_search(exa: Exa, query: str) -> str:
    response = exa.search(
        query,
        type="neural",
        num_results=5,
        contents={"text": {"maxCharacters": 800}},
    )
    if not response.results:
        return "No results found."
    return "\n\n".join(
        f"- {r.title} ({r.url})\n  {(r.text or '').strip()}"
        for r in response.results
    )


def research_account(account_name: str) -> dict:
    """Research a target account by name and return structured findings.

    Runs an agentic loop: Claude decides what to search for (via Exa,
    capped at MAX_SEARCHES calls) and submits its findings through a
    schema-enforced tool call, so the return value always has exactly the
    6 requested fields — missing data as None, never guessed.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY not set in environment.")

    client = Anthropic(api_key=anthropic_key)
    exa = Exa(exa_key)

    messages = [{"role": "user", "content": f"Research this account: {account_name}"}]
    search_count = 0

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1536,
            system=SYSTEM_PROMPT,
            tools=[WEB_SEARCH_TOOL, SUBMIT_TOOL],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            break

        tool_results = []
        submitted = None
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "submit_account_research":
                submitted = block.input
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "Received."}
                )

            elif block.name == "web_search":
                if search_count >= MAX_SEARCHES:
                    content = (
                        "Search limit reached (4/4). Submit your findings now via "
                        "submit_account_research, using null for anything unconfirmed."
                    )
                else:
                    search_count += 1
                    content = _run_exa_search(exa, block.input.get("query", account_name))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if submitted is not None:
            return {field: _clean(submitted.get(field)) for field in FIELDS}

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"research_account('{account_name}') did not submit findings within {MAX_TURNS} turns."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python account_researcher.py "Account Name"')
        sys.exit(1)

    name = sys.argv[1]
    print(f"Researching '{name}'...", file=sys.stderr)
    findings = research_account(name)
    print(json.dumps(findings, indent=2))
