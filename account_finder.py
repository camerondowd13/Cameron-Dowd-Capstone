"""
AccountFinder: given ICP filters (territory + company size, optionally
city/industry), find real candidate companies that fit — trigger-first.

Mirrors the PRD's step 1 (trigger search) with step 3 as fallback (plain
ICP-fit industry search when no trigger is found). This is the discovery
step: it returns candidate company names, not deep research on any one of
them. Feed each returned name into account_researcher.research_account()
for the deep dive (that's a separate call, by design — keeps discovery
and enrichment independently tunable/debuggable).
"""
import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from exa_py import Exa

from icp import MAX_SIZE, MIN_SIZE, VALID_INDUSTRIES
from search_utils import run_exa_search

load_dotenv(".env.local")

MODEL = "claude-sonnet-5"
MAX_SEARCHES = 4
MAX_TURNS = MAX_SEARCHES + 2  # spare turns for the "limit reached" nudge + final submit

SYSTEM_PROMPT = """You are a SaaS AE prospecting for target accounts.
Given a territory (state, optionally city) and a company size range, use
web_search to find real, named companies that fit.

Prioritize companies showing an active buying trigger — recent funding,
leadership changes, hiring surges in Finance/AP/Procurement roles,
expansion into new offices or markets, a recent product launch, or public
complaints about their current tools. If you can't find enough companies
with a clear trigger, fall back to companies that simply fit the
territory + size (and industry, if given) criteria, and mark those with
a null buying_trigger rather than inventing one.

For every candidate, report your best estimate of employee_count as a
plain integer (from LinkedIn headcount, company site, news articles,
etc.). If you truly can't find any headcount signal, use null — never
guess a number. A candidate below the requested minimum size can still be
included ONLY if it has a real, specific buying_trigger — never include
an undersized company with a null trigger.

Use max 4 searches. Return real companies only — NEVER invent a company
that didn't actually show up in your search results. If you find fewer
than the requested number, return fewer — never pad the list to hit the
count."""

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web via Exa. Build queries around territory + industry "
        "+ size signals, e.g. 'construction companies hiring AP Manager "
        "Florida', 'manufacturing company expansion Virginia 2026', "
        "'healthcare company Series B funding New York'."
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
    "name": "submit_candidates",
    "description": "Submit the final list of candidate companies found.",
    "input_schema": {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "description": "Candidate companies found, most promising (strongest trigger) first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Company name."},
                        "location": {
                            "type": ["string", "null"],
                            "description": "City, State if known, else just State, else null.",
                        },
                        "employee_count": {
                            "type": ["integer", "null"],
                            "description": "Best estimate of headcount, or null if no signal found.",
                        },
                        "buying_trigger": {
                            "type": ["string", "null"],
                            "description": (
                                "The specific signal suggesting active buying intent, "
                                "or null if this is a plain ICP-fit fallback match "
                                "with no known trigger."
                            ),
                        },
                    },
                    "required": ["name", "location", "employee_count", "buying_trigger"],
                },
            }
        },
        "required": ["companies"],
    },
}


def _within_size_range(company: dict, min_size: int, max_size: int) -> bool:
    """Enforce the ICP size range in code, with the PRD's sub-min exception:
    a company below min_size still qualifies if it has a real buying_trigger.
    Unknown headcount (null) is kept — Exa often can't find exact headcount
    for smaller/private companies, so dropping those would gut real matches."""
    count = company.get("employee_count")
    if count is None:
        return True
    if min_size <= count <= max_size:
        return True
    if count < min_size and company.get("buying_trigger"):
        return True
    return False


def find_accounts(
    state: str,
    min_size: int = MIN_SIZE,
    max_size: int = MAX_SIZE,
    city: str | None = None,
    industry: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Find candidate companies matching ICP filters, trigger-first.

    Returns up to `limit` dicts: {"name", "location", "employee_count",
    "buying_trigger"}, hard-filtered against [min_size, max_size] in code
    (not just prompted) — companies below min_size are kept only if they
    have a real buying_trigger, per the PRD's exception. Feed each "name"
    into account_researcher.research_account() next.
    """
    if industry is not None and industry.lower() not in VALID_INDUSTRIES:
        raise ValueError(
            f"industry must be one of {VALID_INDUSTRIES} or None, got {industry!r}"
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY not set in environment.")

    client = Anthropic(api_key=anthropic_key)
    exa = Exa(exa_key)

    territory = f"{city}, {state}" if city else state
    request = (
        f"Find up to {limit} candidate companies.\n"
        f"Territory: {territory}\n"
        f"Company size: {min_size}-{max_size} employees "
        f"(a company under {min_size} still qualifies if it has a real buying trigger)"
    )
    if industry:
        request += f"\nIndustry: {industry}"

    messages = [{"role": "user", "content": request}]
    search_count = 0

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
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

            if block.name == "submit_candidates":
                submitted = block.input.get("companies", [])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "Received."}
                )

            elif block.name == "web_search":
                if search_count >= MAX_SEARCHES:
                    content = (
                        "Search limit reached (4/4). Submit whatever candidates "
                        "you've found now via submit_candidates."
                    )
                else:
                    search_count += 1
                    content = run_exa_search(exa, block.input.get("query", territory))
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if submitted is not None:
            filtered = [c for c in submitted if _within_size_range(c, min_size, max_size)]
            return filtered[:limit]

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"find_accounts(state={state!r}) did not submit candidates within {MAX_TURNS} turns."
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python account_finder.py "State" min_size max_size [city] [industry]')
        sys.exit(1)

    state = sys.argv[1]
    min_size = int(sys.argv[2])
    max_size = int(sys.argv[3])
    city = sys.argv[4] if len(sys.argv) > 4 else None
    industry = sys.argv[5] if len(sys.argv) > 5 else None

    where = f"{city + ', ' if city else ''}{state}"
    print(f"Finding accounts in {where} ({min_size}-{max_size} employees)...", file=sys.stderr)
    candidates = find_accounts(state, min_size, max_size, city=city, industry=industry)
    print(json.dumps(candidates, indent=2))
