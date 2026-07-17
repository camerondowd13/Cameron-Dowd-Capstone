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
from urllib.parse import urlparse

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from exa_py import Exa
from rich.console import Console

import account_researcher
import apollo_client
import contact_finder
from enrich_pipeline import load_supabase_config
from icp import MAX_SIZE, MIN_SIZE, VALID_INDUSTRIES
from search_utils import run_exa_search, strip_linkedin

console = Console()

# SIC major-group prefixes per industry, used to post-filter Apollo
# Organization Search results -- q_organization_keyword_tags is a loose
# match (confirmed pulling in a trade association and staffing agencies
# alongside real construction companies), so keyword hits alone aren't
# trustworthy. SIC codes are Apollo's own structured classification, not a
# fuzzy tag, so filtering on them is a precise industry check.
INDUSTRY_SIC_PREFIXES = {
    "construction": ("15", "16", "17"),
    "manufacturing": tuple(str(n) for n in range(20, 40)),
    "healthcare": ("80",),
}


def _matches_industry_sic(org: dict, industry: str) -> bool:
    # INDUSTRY_SIC_PREFIXES keys are lowercase; callers pass industry
    # capitalized (e.g. "Construction", matching the site's dropdown and
    # find_accounts()'s own VALID_INDUSTRIES check). A bare .get() without
    # lowercasing silently returned () for every call -- meaning every
    # candidate failed this check regardless of its actual SIC codes.
    prefixes = INDUSTRY_SIC_PREFIXES.get((industry or "").lower(), ())
    return any(
        (code or "").startswith(prefixes) for code in (org.get("sic_codes") or [])
    )


def _domain(url: str) -> str:
    """Normalize a URL down to its bare domain for comparison (strips
    'www.', scheme, path). Handles both full URLs (Exa's format) and bare
    domains with no scheme (Supabase's stored format, e.g. 'company.com'
    with no 'https://') -- urlparse treats a schemeless string as a path,
    not a netloc, so bare domains need the scheme added first."""
    if not url:
        return ""
    if "//" not in url:
        url = "//" + url
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def _fetch_known_companies() -> tuple[set, set]:
    """Every company already in the Supabase accounts table, regardless of
    status -- so AccountFinder never resurfaces one you or your SDR are
    already working. Returns (normalized_names, domains). Best-effort: on
    any failure (Supabase unreachable, config missing), returns empty sets
    rather than raising -- dedup is a nice-to-have, not something that
    should break discovery if Supabase is temporarily down."""
    try:
        supabase_url, supabase_key = load_supabase_config()
        resp = requests.get(
            f"{supabase_url}/rest/v1/accounts",
            params={"select": "name,website"},
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
    except Exception as e:
        print(f"Warning: could not fetch known accounts from Supabase for dedup: {e}", file=sys.stderr)
        return set(), set()

    names = {(r.get("name") or "").strip().lower() for r in rows if r.get("name")}
    domains = {_domain(r["website"]) for r in rows if r.get("website")}
    domains.discard("")
    return names, domains

load_dotenv(".env.local")

MODEL = "claude-opus-4-8"
DEFAULT_LIMIT = 20
EXA_NUM_RESULTS = 8  # per search -- up from 5, so each query pulls more raw material
MIN_SEARCH_BUDGET = 8  # raised from 4 (2026-07-16) -- more search turns per
# request, to reduce thin/zero-candidate results on narrower state+industry
# combos. Trades more time/cost per search for a higher odds of hitting the
# requested limit.

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

Also report, if seen in your search results:
- website: the company's own official homepage domain (e.g.
  "https://company.com"), separate from source_url (which may be a news
  article, not the company's own site). Null if not seen.
- contacts_seen: any real people (name + title) you happened to notice
  for this company while searching, even without email/phone confirmed —
  this is a lighter signal than full contact verification, just names
  worth knowing about. Only include names that actually appeared in your
  search results — never invent one. Empty list if none seen.

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
                        "website": {
                            "type": ["string", "null"],
                            "description": "The company's own official homepage domain, if seen. Null if unknown.",
                        },
                        "contacts_seen": {
                            "type": "array",
                            "description": "Real people (name + title) noticed for this company, even without confirmed email/phone. Empty if none.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "title": {"type": ["string", "null"]},
                                },
                                "required": ["name", "title"],
                            },
                        },
                    },
                    "required": [
                        "name", "location", "employee_count", "buying_trigger",
                        "source_url", "website", "contacts_seen",
                    ],
                },
            }
        },
        "required": ["companies"],
    },
}


VERIFY_TOOL = {
    "name": "submit_verification",
    "description": "Submit verified details extracted strictly from the provided search text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "companies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "employee_count": {"type": ["integer", "null"]},
                        "website": {"type": ["string", "null"]},
                        "contacts_seen": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "title": {"type": ["string", "null"]},
                                },
                                "required": ["name", "title"],
                            },
                        },
                    },
                    "required": ["name", "employee_count", "website", "contacts_seen"],
                },
            }
        },
        "required": ["companies"],
    },
}

# Stage 1 (broad discovery, many companies at once) reliably finds a trigger
# for a company, but employee_count/website/contacts_seen tend to live on
# different pages than whatever surfaced the trigger -- a handful of broad
# queries covering 20 companies rarely happens to hit all three for the same
# one. Stage 2 goes deep on each surviving candidate individually: dedicated
# searches per company, then one batched extraction call (not one call per
# company, to keep cost/latency down) to actually fill the gaps.
MAX_VERIFY_CANDIDATES_MULTIPLIER = 5  # raised from 3 (2026-07-16) -- verify a
# bigger slice of stage 1's raw candidates, so more of what was already
# found gets a real shot at clearing stage 2/3, instead of only the first
# `limit*3` ever being examined more deeply.


def _verify_candidate_details(client, exa, candidates, seen_urls, search_text_corpus):
    to_verify = [
        c for c in candidates
        if c.get("employee_count") is None or not c.get("website") or not c.get("contacts_seen")
    ]
    if not to_verify:
        return candidates

    per_company_text = {}
    for c in to_verify:
        name = c["name"]
        blocks = []
        for query in (f"{name} official website", f"{name} leadership team employees staff"):
            text = run_exa_search(exa, query, num_results=5, seen_urls=seen_urls)
            search_text_corpus.append(text)
            blocks.append(text)
        per_company_text[name] = "\n\n".join(blocks)

    verify_prompt = (
        "For each company below, using ONLY the search results provided for "
        "it, report employee_count (integer or null), website (the official "
        "homepage URL, exactly as it appears in the results, or null), and "
        "contacts_seen (list of real {name, title} people found in the "
        "results, or empty list). Never guess -- if the given text doesn't "
        "show it, use null/empty, even if you think you know the answer "
        "from general knowledge.\n\n"
    )
    for name, text in per_company_text.items():
        verify_prompt += f"=== {name} ===\n{text}\n\n"

    response = client.messages.create(
        model=MODEL,
        # 16000 (not the old 8192) -- with limit*3 candidates batched into one
        # call (up to 30 at limit=10), 8192 was too tight and risked truncating
        # mid-JSON, which surfaced as "string indices must be integers" further
        # down when a cut-off companies[] entry parsed as something other than
        # a full object. 16000 is the safe non-streaming ceiling.
        max_tokens=16000,
        system="You extract structured company facts strictly from the search text you're given. Never use outside knowledge, never guess.",
        tools=[VERIFY_TOOL],
        tool_choice={"type": "tool", "name": "submit_verification"},
        messages=[{"role": "user", "content": verify_prompt}],
    )
    result_block = next((b for b in response.content if b.type == "tool_use"), None)
    # Defensive: even with more headroom, a truncated or malformed response
    # could still produce a non-dict entry -- skip it rather than crash the
    # whole batch (better to lose one company's verification than all of them).
    verified_by_name = {
        v["name"]: v
        for v in (result_block.input.get("companies", []) if result_block else [])
        if isinstance(v, dict) and "name" in v
    }

    for c in to_verify:
        v = verified_by_name.get(c["name"])
        if not v:
            continue
        if c.get("employee_count") is None:
            c["employee_count"] = v.get("employee_count")
        if not c.get("website"):
            c["website"] = v.get("website")
        if not c.get("contacts_seen"):
            c["contacts_seen"] = v.get("contacts_seen") or []

    return candidates


def _discover_via_apollo(
    state: str,
    industry: str,
    min_size: int,
    max_size: int,
    limit: int,
    known_names: set,
    known_domains: set,
    target_titles: list[str] | None = None,
    trace: bool = False,
    events: list | None = None,
) -> list[dict]:
    """Discovery via Apollo's Organization Search instead of freeform
    Claude+Exa web search. The web-search discovery loop is bottlenecked by
    a small search-turn budget trying to *find* companies that exist --
    Apollo already has a structured database of them, filterable by
    location/size/industry (SIC-verified, not just the keyword tag) in one
    call, no LLM required for that part.

    What Apollo's org search can't give us -- a real, current buying
    trigger -- still requires per-candidate research, so each SIC-matched
    org gets run through account_researcher.research_account() (Claude+Exa,
    but scoped to ONE already-known-real company, a much narrower task than
    discovering companies from scratch). Stops once `limit` candidates
    clear both research_account's meets_icp gate and contact_finder's
    reachability bar, same qualification standard as the web-search path.

    trace=True prints a colored, step-by-step console log of every stage --
    critically, API/infrastructure errors print in [bold red] with an "API
    ERROR" label distinct from a genuine [yellow] disqualification, so the
    two can never be silently confused again (they were, in production,
    before this was added -- an Anthropic usage-limit rejection looked
    identical to "this candidate doesn't qualify" from the outside).

    events: if a list is passed, each step also appends a structured
    {"stage", "company", "status", "detail"} dict to it -- same information
    as the console trace, but machine-readable, for rendering a visual
    report (see render_trace_html.py) instead of just reading terminal text.

    Returns candidates in the same shape find_accounts() already produces,
    or [] if industry is None (Apollo's keyword-tag search needs something
    to search for) or the Apollo API isn't configured."""
    def emit(stage, company, status, detail):
        if events is not None:
            events.append({"stage": stage, "company": company, "status": status, "detail": detail})

    if not industry or not apollo_client.APOLLO_API_KEY:
        if trace:
            console.print("[dim]Skipping Apollo discovery -- no industry given or APOLLO_API_KEY not set.[/dim]")
        return []

    if trace:
        console.rule("[bold cyan]DISCOVERY — Apollo Organization Search[/bold cyan]")
        console.print(f"[cyan]Searching Apollo: {state}, {industry}, {min_size}-{max_size} employees...[/cyan]")

    # Organization Search doesn't consume Apollo credits (per their pricing
    # docs), so there's no cost to pulling extra pages -- confirmed in
    # testing that results for identical parameters aren't stable between
    # calls (one pull returned 0 real SIC-matches out of 100, an
    # immediately-following identical call returned several in its first
    # 5), so a single page isn't reliable enough to trust on its own.
    MAX_PAGES = 3
    candidates = []
    for page in range(1, MAX_PAGES + 1):
        if len(candidates) >= limit * 3:
            break
        try:
            orgs = apollo_client.search_organizations(state, industry, min_size, max_size, per_page=100, page=page)
        except Exception as e:
            if trace:
                console.print(f"[bold red]API ERROR — Apollo org search (page {page}) failed: {e}[/bold red]")
            else:
                print(f"Warning: Apollo org search failed for {state!r}/{industry!r}: {e}", file=sys.stderr)
            emit("discovery", None, "error", f"Apollo org search (page {page}) failed: {e}")
            break

        sic_misses = 0
        dedup_misses = 0
        page_candidates = []
        for org in orgs:
            name = org.get("name")
            domain = _domain(org.get("website_url") or "")
            if not name or not domain:
                continue
            if name.strip().lower() in known_names or domain in known_domains:
                dedup_misses += 1
                continue
            if not _matches_industry_sic(org, industry):
                sic_misses += 1
                continue
            page_candidates.append({"name": name, "domain": domain, "website": org.get("website_url")})

        candidates.extend(page_candidates)
        if trace:
            console.print(
                f"[cyan]  page {page}: {len(orgs)} raw -> {len(page_candidates)} passed "
                f"(sic mismatch: {sic_misses}, already known: {dedup_misses})[/cyan]"
            )
        emit(
            "discovery", None, "info",
            f"Page {page}: Apollo returned {len(orgs)} companies matching location+size, but Apollo's "
            f"keyword search is loose -- it also returns tangentially-related companies (staffing agencies, "
            f"trade associations). Checking each one's actual government SIC industry code narrowed this "
            f"down to {len(page_candidates)} genuine matches ({sic_misses} wrong industry despite the keyword "
            f"hit, {dedup_misses} already in your Supabase accounts table).",
        )
        if not orgs:
            break  # ran out of pages

    if trace:
        console.print(f"[cyan]  -> {len(candidates)} total candidates across {page} page(s)[/cyan]")
        console.rule("[bold blue]RESEARCH — per-candidate ICP + trigger check[/bold blue]")

    qualified = []
    for c in candidates:
        if len(qualified) >= limit:
            if trace:
                console.print(f"[dim]Reached limit ({limit}) -- stopping.[/dim]")
            break
        try:
            research = account_researcher.research_account(c["name"], domain=c["domain"])
        except Exception as e:
            if trace:
                console.print(f"[bold red]⚠ API ERROR — {c['name']}: research_account failed: {e}[/bold red]")
            else:
                print(f"Warning: research_account failed for {c['name']!r}: {e}", file=sys.stderr)
            emit("research", c["name"], "error",
                 f"Claude+Exa research call itself failed (infrastructure problem, not a disqualification): {e}")
            continue
        if not research.get("meets_icp"):
            if trace:
                console.print(f"[yellow]✗ {c['name']} — does not meet ICP ({research.get('industry')})[/yellow]")
            emit("research", c["name"], "disqualified",
                 f"Real company, but doesn't fit the ICP -- Claude found its actual industry/size to be "
                 f"\"{research.get('industry')}\" ({research.get('size_range')}), which fails the size/industry bar.")
            continue
        trigger = research.get("buying_triggers") or "no specific trigger found, but size+industry alone qualify it"
        if trace:
            console.print(f"[green]✓ {c['name']} — meets_icp, {research.get('employee_count')} employees[/green]")
        emit("research", c["name"], "success",
             f"Confirmed real and ICP-fit ({research.get('employee_count')} employees). "
             f"Why it's a good lead right now: {trigger}")

        try:
            contacts = contact_finder.find_contacts(c["name"], domain=c["domain"], target_titles=target_titles)
        except Exception as e:
            if trace:
                console.print(f"[bold red]⚠ API ERROR — {c['name']}: find_contacts failed: {e}[/bold red]")
            else:
                print(f"Warning: find_contacts failed for {c['name']!r}: {e}", file=sys.stderr)
            emit("contact", c["name"], "error",
                 f"Contact lookup call itself failed (infrastructure problem, not a disqualification): {e}")
            continue

        general_office = contacts["general_office"]
        verified_contacts = contacts["contacts"]
        if not verified_contacts and not general_office:
            if trace:
                console.print(f"[yellow]  ○ {c['name']} — no reachable contact found[/yellow]")
            emit("contact", c["name"], "disqualified",
                 "Tried Apollo, then a web-search fallback -- neither found a named person with confirmed "
                 "email+phone, nor even a general company phone/email. Dropped: a real ICP-fit company with "
                 "no way to actually reach them isn't useful as a lead.")
            continue
        if verified_contacts:
            names = ", ".join(f"{p['name']} ({p['title']})" for p in verified_contacts)
            detail = (f"Found {len(verified_contacts)} named person(s) with a confirmed direct email AND phone "
                      f"(the strict bar -- one without the other doesn't count): {names}.")
        else:
            detail = ("No named person had both email+phone confirmed, but found the company's general "
                      "office phone/email as a fallback -- still a real way to reach them, just not a direct dial.")
        if trace:
            console.print(f"[green]  ✓ {c['name']} — {len(verified_contacts)} verified contact(s), general_office={bool(general_office)}[/green]")
        emit("contact", c["name"], "success", detail)

        qualified.append({
            "name": c["name"],
            "location": state,
            "employee_count": research.get("employee_count"),
            "buying_trigger": research.get("buying_triggers"),
            "source_url": (research.get("sources") or [None])[0],
            "website": c["website"],
            "contacts_seen": [],
            "verified_contacts": verified_contacts,
            "general_office": general_office,
            # Company background for the salesperson -- already generated by
            # research_account above, previously discarded. Surfaced in the
            # frontend's lead detail modal (a click into any board/dashboard
            # company). Only the Apollo discovery path populates these; the
            # Claude+Exa fallback path leaves them null and the UI just omits
            # the section.
            "recent_news": research.get("recent_news"),
            "hiring_status": research.get("hiring_status"),
            "tech_stack_hints": research.get("tech_stack_hints"),
            "industry_detail": research.get("industry"),
        })

    if trace:
        console.rule("[bold]RESULT[/bold]")
        style = "bold green" if len(qualified) >= limit else "bold yellow"
        console.print(f"[{style}]{len(qualified)}/{limit} qualified leads found via Apollo discovery[/{style}]")

    return qualified


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
    target_titles: list[str] | None = None,
    trace: bool = False,
    events: list | None = None,
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

    trace=True prints a colored, step-by-step log of every stage (Apollo
    discovery, per-candidate research, contact lookup, and — if Apollo
    discovery falls short of `limit` — the Claude+Exa fallback loop) to the
    console. Off by default since it's meant for local debugging, not the
    production server path.

    events: if a list is passed, structured {"stage", "company", "status",
    "detail"} dicts get appended to it as each step runs -- pass this (with
    trace=True or on its own) to also render a visual HTML report after the
    call via render_trace_html.py, instead of only reading console text.
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

    known_names, known_domains = _fetch_known_companies()

    # Apollo Organization Search first: a structured database query for
    # "real companies matching location/size/industry" beats asking an LLM
    # to discover them one web search at a time. What Apollo's org search
    # can't give us -- a current buying trigger -- still goes through
    # account_researcher per candidate. If this alone covers `limit`, skip
    # the Claude+Exa discovery loop below entirely.
    apollo_qualified = _discover_via_apollo(
        state, industry, min_size, max_size, limit, known_names, known_domains,
        target_titles=target_titles, trace=trace, events=events
    )
    if len(apollo_qualified) >= limit:
        if events is not None:
            events.append({"stage": "result", "company": None, "status": "success",
                            "detail": f"{limit}/{limit} qualified via Apollo discovery alone"})
        return apollo_qualified[:limit]

    if trace and apollo_qualified:
        console.print(f"[dim]Apollo discovery found {len(apollo_qualified)}/{limit} -- falling back to Claude+Exa web search for the rest.[/dim]")
    elif trace:
        console.print("[dim]Apollo discovery found 0 -- falling back to Claude+Exa web search.[/dim]")

    remaining_limit = limit - len(apollo_qualified)
    known_names = known_names | {(c["name"] or "").strip().lower() for c in apollo_qualified}
    known_domains = known_domains | {_domain(c.get("website") or "") for c in apollo_qualified}

    search_budget = _search_budget(remaining_limit)
    max_turns = search_budget + 2  # spare turns for the "limit reached" nudge + final submit
    system_prompt = _build_system_prompt(search_budget)

    territory = f"{city}, {state}" if city else state
    request = (
        f"Find up to {remaining_limit} candidate companies.\n"
        f"Territory: {territory}\n"
        f"Company size: {min_size}-{max_size} employees "
        f"(a company under {min_size} still qualifies if it has a real buying trigger)"
    )
    if industry:
        request += f"\nIndustry: {industry}"

    messages = [{"role": "user", "content": request}]
    search_count = 0
    seen_urls = set()
    search_text_corpus = []  # raw text from every search, for name-grounding contacts_seen

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
            max_tokens=8192,
            system=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            raise RuntimeError(
                f"find_accounts(state={state!r}) got stop_reason={response.stop_reason!r} "
                "instead of a tool call -- likely ran out of output tokens mid-response "
                "if this was a forced submit_candidates call with many candidates."
            )

        tool_results = []
        submitted = None
        for block in response.content:
            if block.type != "tool_use":
                continue

            if block.name == "submit_candidates":
                # Defensive: Claude's tool call is expected to return each
                # entry as an object per the schema, but (as seen with this
                # same tool's stage-2 counterpart, submit_verification) it can
                # occasionally return a malformed entry -- e.g. a plain
                # string -- which would otherwise crash every c["..."] /
                # c.get("...") access below. Skip anything that isn't a dict
                # with a name, rather than trusting the schema was honored.
                submitted = [
                    c for c in block.input.get("companies", [])
                    if isinstance(c, dict) and c.get("name")
                ]
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
                        exa, block.input.get("query", territory),
                        num_results=EXA_NUM_RESULTS, seen_urls=seen_urls,
                    )
                    search_text_corpus.append(content)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if submitted is not None:
            def _ground(candidates_list):
                """(Re)apply grounding to website + contacts_seen against the
                CURRENT seen_urls/search_text_corpus -- called again after
                stage 2 adds more search data, so newly-verified facts get
                checked too, not just what stage 1 opportunistically found."""
                corpus = "\n".join(search_text_corpus).lower()
                seen_domains = {_domain(u) for u in seen_urls if _domain(u)}
                for c in candidates_list:
                    website = strip_linkedin(c.get("website"))
                    c["website"] = website if website and _domain(website) in seen_domains else None
                    c["contacts_seen"] = [
                        p for p in (c.get("contacts_seen") or [])
                        if isinstance(p, dict) and (p.get("name") or "").strip().lower() in corpus
                    ]

            filtered = [c for c in submitted if _within_size_range(c, min_size, max_size)]
            deduped = []
            seen_names = set()
            for c in filtered:
                key = (c.get("name") or "").strip().lower()
                if key and key not in seen_names:
                    c["source_url"] = strip_linkedin(c.get("source_url"))
                    # Grounding check: the cited source_url must be a URL Exa
                    # actually returned during this run, not just a plausible
                    # string the model wrote. Catches the case a plain
                    # non-null check can't -- a real-looking but never-shown URL.
                    if not c["source_url"] or c["source_url"] not in seen_urls:
                        continue
                    seen_names.add(key)
                    deduped.append(c)

            _ground(deduped)

            # Dedup against Supabase: never resurface a company already in
            # the accounts table (any status) -- avoids Cameron/his SDR
            # duplicating work on a company already being handled. Checked
            # early (before stages 2-3) so no verification cost is wasted
            # on companies we're about to throw away anyway.
            not_already_known = [
                c for c in deduped
                if (c.get("name") or "").strip().lower() not in known_names
                and _domain(c.get("website") or "") not in known_domains
            ]

            # Stage 2: go deep on each surviving candidate (capped, to bound
            # cost) to fill in whatever stage 1's broad sweep missed.
            to_verify_cap = not_already_known[: remaining_limit * MAX_VERIFY_CANDIDATES_MULTIPLIER]
            _verify_candidate_details(client, exa, to_verify_cap, seen_urls, search_text_corpus)
            _ground(to_verify_cap)  # re-check stage 2's additions against the now-larger corpus

            stage2_qualified = [
                c for c in to_verify_cap
                if c.get("employee_count") is not None and c.get("website") and c.get("contacts_seen")
            ]

            # Stage 3: realistic "reachable" bar, based on what today's testing
            # actually proved achievable without a paid contact-data provider
            # (Apollo/ZoomInfo). A named person + confirmed direct email+phone
            # was tested repeatedly and consistently returned zero -- that
            # data mostly isn't published anywhere public. What DOES work
            # reliably: a real named person (contacts_seen, already grounded)
            # to ask for by name, plus a real verified way to actually reach
            # the company (general phone or email). That's a legitimate
            # cold-call workflow, just not a direct dial.
            fully_qualified = []
            for c in stage2_qualified:
                try:
                    result = contact_finder.find_contacts(
                        c["name"], domain=_domain(c["website"]), target_titles=target_titles
                    )
                except Exception as e:
                    # A single company's contact lookup failing (e.g. the model
                    # not calling submit_contacts within its turn budget)
                    # shouldn't lose every other already-verified candidate --
                    # skip just this one and keep going.
                    print(f"Warning: find_contacts failed for {c['name']!r}: {e}", file=sys.stderr)
                    continue
                # Prefer any fully-verified direct contact if one happens to
                # exist, but don't require it -- general_office satisfies
                # "reachable" too.
                c["verified_contacts"] = result["contacts"]
                c["general_office"] = result["general_office"]
                if c["contacts_seen"] and (result["contacts"] or result["general_office"]):
                    fully_qualified.append(c)
                if len(fully_qualified) >= remaining_limit:
                    break

            final = (apollo_qualified + fully_qualified)[:limit]
            if events is not None:
                events.append({"stage": "result", "company": None,
                                "status": "success" if len(final) >= limit else "disqualified",
                                "detail": f"{len(final)}/{limit} qualified ({len(apollo_qualified)} via Apollo, {len(fully_qualified)} via Claude+Exa fallback)"})
            return final

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
