"""
AccountResearcher: given a company name, research it and return structured
findings for downstream sales agents (industry, size, buying triggers, etc.),
plus a deterministic meets_icp flag computed in code against icp.py.

Search is backed by Exa (not Claude's built-in web_search tool) — Claude
Sonnet 5 runs an agentic tool-use loop where it decides what to search for,
we execute those searches against Exa, and it submits its final findings
through a schema-enforced tool call. meets_icp is NOT self-reported by the
model — it's computed in code from the structured employee_count and
industry_category the model provides, per the PRD's "binary ICP match/
no-match, no confidence scoring" requirement.
"""
import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from exa_py import Exa

from icp import VALID_INDUSTRIES, meets_size
from search_utils import clean_nullish, run_exa_search

load_dotenv(".env.local")

MODEL = "claude-sonnet-5"
MAX_SEARCHES = 4
MAX_TURNS = MAX_SEARCHES + 2  # spare turns for the "limit reached" nudge + final submit

FIELDS = [
    "industry",
    "size_range",
    "buying_triggers",
    "hiring_status",
    "tech_stack_hints",
    "recent_news",
]

SYSTEM_PROMPT = f"""You are a SaaS AE researching target accounts.
Given an account name, use web_search to find: industry, size_range,
buying_triggers, hiring_status, tech_stack_hints, recent_news
(last 90 days). Return as structured JSON with EXACTLY those 6 fields,
plus two structured fields used for ICP scoring:
- employee_count: your best estimate as a plain integer, or null if truly
  unknown. Never guess a number you can't back with a search result.
- industry_category: exactly one of {VALID_INDUSTRIES + ["other"]}, based
  on the account's actual primary business.
Use max 4 searches. If a field can't be found, use null — NEVER hallucinate.

buying_triggers means specific signals suggesting this account may be
evaluating new software right now — recent funding, leadership changes,
rapid hiring in Finance/AP/Procurement roles, expansion into new offices
or markets, a recent product launch, or public complaints about their
current tools. Not generic company growth — only signals that plausibly
indicate active buying intent. Null if none found.

Many company names are shared by unrelated businesses (e.g. "Apex Solutions").
Before reporting anything, you must be confident all 6 fields describe the
SAME single company. Disambiguation rules:
- If a known official domain is given in the research request, treat it as
  ground truth. Discard any search result that isn't clearly that company,
  even if the name matches.
- If no domain is given, identify the single most likely company (the one
  with the strongest, most consistent signal across your searches — e.g.
  most coverage, most specific matching details) and report on that company
  ONLY. Never blend facts from two different companies that share a name.
- If you cannot confidently settle on one company — name is generic, no
  domain given, and results are split across clearly unrelated businesses
  with no clear leader — return null for ALL 6 fields. Reporting confident
  facts about the wrong company is worse than returning nothing.

Also report sources: a list of the real URLs (from your search results)
that back up your findings, so a human can click through and verify.
Never invent a URL — only include ones that actually appeared in a
web_search result."""

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
            **{
                field: {
                    "type": ["string", "null"],
                    "description": f"The account's {field.replace('_', ' ')}, or null if not found.",
                }
                for field in FIELDS
            },
            "employee_count": {
                "type": ["integer", "null"],
                "description": "Best estimate of headcount as a plain integer, or null if unknown.",
            },
            "industry_category": {
                "type": ["string", "null"],
                "enum": VALID_INDUSTRIES + ["other", None],
                "description": "Primary business category, for ICP scoring.",
            },
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Real URLs from search results backing up the findings above.",
            },
        },
        "required": FIELDS + ["employee_count", "industry_category", "sources"],
    },
}

def _compute_meets_icp(submitted: dict, buying_triggers) -> bool:
    category = (submitted.get("industry_category") or "").lower()
    if category not in VALID_INDUSTRIES:
        return False
    return meets_size(submitted.get("employee_count"), has_trigger=bool(buying_triggers))


def research_account(account_name: str, domain: str | None = None) -> dict:
    """Research a target account by name and return structured findings.

    Runs an agentic loop: Claude decides what to search for (via Exa,
    capped at MAX_SEARCHES calls) and submits its findings through a
    schema-enforced tool call. Returns the 6 narrative FIELDS, plus
    employee_count (int|None) and meets_icp (bool, computed in code from
    icp.py — never self-reported by the model).

    domain: optional official website domain (e.g. "stampli.com"). When
    given, the first search is locked to that domain to anchor company
    identity before broader searches run — pass this whenever the caller
    has it (e.g. derived from a contact email) to cut down ambiguity on
    generic company names.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY not set in environment.")

    client = Anthropic(api_key=anthropic_key)
    exa = Exa(exa_key)

    initial_request = f"Research this account: {account_name}"
    if domain:
        initial_request += (
            f"\nKnown official domain: {domain}. Confirm you have the right "
            "company against this domain before trusting any other source."
        )
    messages = [{"role": "user", "content": initial_request}]
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
                    # Anchor identity on the known domain for the first search only —
                    # later searches need the open web (news, hiring sites, etc.)
                    lock_domains = [domain] if (domain and search_count == 1) else None
                    content = run_exa_search(
                        exa, block.input.get("query", account_name), lock_domains
                    )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if submitted is not None:
            result = {field: clean_nullish(submitted.get(field)) for field in FIELDS}
            result["employee_count"] = submitted.get("employee_count")
            result["meets_icp"] = _compute_meets_icp(submitted, result["buying_triggers"])
            result["sources"] = submitted.get("sources", [])
            return result

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"research_account('{account_name}') did not submit findings within {MAX_TURNS} turns."
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print('Usage: python account_researcher.py "Account Name" [domain]')
        sys.exit(1)

    name = sys.argv[1]
    account_domain = sys.argv[2] if len(sys.argv) == 3 else None
    print(f"Researching '{name}'" + (f" (domain: {account_domain})" if account_domain else "") + "...", file=sys.stderr)
    findings = research_account(name, domain=account_domain)
    print(json.dumps(findings, indent=2))
