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
import math
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from exa_py import Exa

from icp import MAX_SIZE, MIN_SIZE, VALID_INDUSTRIES
from search_utils import run_exa_search, strip_linkedin

load_dotenv(".env.local")

MODEL = "claude-sonnet-5"
DEFAULT_LIMIT = 20
EXA_NUM_RESULTS = 8  # per search -- up from 5, so each query pulls more raw material
MIN_SEARCH_BUDGET = 4

# Search budget scales with how many candidates are actually requested --
# a fixed 4 searches (the old default) only ever surfaces ~20 raw results
# total, nowhere near enough to reliably yield 20 *verified, distinct*
# in-ICP companies once overlap/noise/off-ICP results are filtered out.
def _search_budget(limit: int) -> int:
    return max(MIN_SEARCH_BUDGET, math.ceil(limit / 2))


def _build_system_prompt(search_budget: int) -> str:
    return f"""You are a SaaS AE prospecting for target accounts.
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

For every candidate, also report source_url: the exact URL (from your
search results) that backs up the buying_trigger claim (or, if there's no
trigger, any URL confirming the company/location/size). This must be a
real URL that appeared in a web_search result — never invent one.

You have {search_budget} searches available -- to actually cover a large
requested volume, vary your queries across different angles (different
cities within the territory, different trigger types, different industry
sub-segments) rather than repeating similar queries. Return real
companies only — NEVER invent a company that didn't actually show up in
your search results, and never return the same company twice. If you find
fewer than the requested number even after using your full search budget,
return fewer — never pad the list to hit the count."""

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
                        "source_url": {
                            "type": ["string", "null"],
                            "description": "Real URL from a search result backing up this candidate, or null.",
                        },
                    },
                    "required": ["name", "location", "employee_count", "buying_trigger", "source_url"],
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
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    """Find candidate companies matching ICP filters, trigger-first.

    Returns up to `limit` dicts: {"name", "location", "employee_count",
    "buying_trigger", "source_url"}, hard-filtered against [min_size,
    max_size] in code (not just prompted) — companies below min_size are
    kept only if they have a real buying_trigger, per the PRD's exception.
    Feed each "name" into account_researcher.research_account() next.

    Search budget scales with `limit` (see _search_budget) -- requesting
    20 candidates runs meaningfully longer than requesting 5, since it
    needs more distinct searches to find that many real, verified matches.
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

    search_budget = _search_budget(limit)
    max_turns = search_budget + 2  # spare turns for the "limit reached" nudge + final submit
    system_prompt = _build_system_prompt(search_budget)

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

    for _ in range(max_turns):
        # Once the search budget is spent, force the model to submit --
        # a text-only nudge isn't reliable at larger budgets (it can keep
        # calling web_search past the limit instead of wrapping up).
        if search_count >= search_budget:
            tools = [SUBMIT_TOOL]
            tool_choice = {"type": "tool", "name": "submit_candidates"}
        else:
            tools = [WEB_SEARCH_TOOL, SUBMIT_TOOL]
            tool_choice = {"type": "auto"}

        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
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
                if search_count >= search_budget:
                    content = (
                        f"Search limit reached ({search_budget}/{search_budget}). Submit "
                        "whatever candidates you've found now via submit_candidates."
                    )
                else:
                    search_count += 1
                    content = run_exa_search(
                        exa, block.input.get("query", territory), num_results=EXA_NUM_RESULTS
                    )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if submitted is not None:
            filtered = [c for c in submitted if _within_size_range(c, min_size, max_size)]
            deduped = []
            seen_names = set()
            for c in filtered:
                key = (c.get("name") or "").strip().lower()
                if key and key not in seen_names:
                    seen_names.add(key)
                    c["source_url"] = strip_linkedin(c.get("source_url"))
                    deduped.append(c)
            return deduped[:limit]

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"find_accounts(state={state!r}) did not submit candidates within {max_turns} turns."
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
