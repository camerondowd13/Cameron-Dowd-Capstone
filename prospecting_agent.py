"""
ProspectingAgent: end-to-end pipeline. Given ICP filters (state, size range,
optionally city/industry), finds brand-new qualifying companies via
AccountFinder (which already does discovery + verification + contact-finding,
and dedups against Supabase) and fully processes each one: deep research via
AccountResearcher (+ targeted Exa fallback for any gap), best-contact
selection, a new Supabase row, and a personalized Gmail draft. Per-company
outreach emails are always DRAFTS, never sent.

Outer loop: one find_accounts() call.
Inner loop per company (max 4 tool calls / 4 iterations): iteration 1
always runs AccountResearcher for a full first pass (industry,
hiring_status, tech_stack_hints, recent_news -- employee_count already
came from AccountFinder). Iterations 2-4 each run one targeted Exa search
for a single still-missing field, only while a gap remains.

On a single company's failure, whatever was gathered is still saved to
Supabase with status='partial' -- one bad company never stops the batch.

run_daily() is the unattended entry point (`--daily` on the CLI, or the
scheduled 8am cloud routine): runs the pipeline across DAILY_TERRITORIES
and SENDS (not drafts) one summary email to Cameron -- that email is a
self-notification, not outreach to a prospect, so it's exempt from the
draft-only rule above.
"""
import os
import sys
from urllib.parse import urlparse

from anthropic import Anthropic
from composio import Composio
from dotenv import load_dotenv
from exa_py import Exa
from rich.console import Console
from rich.panel import Panel

import account_finder
import account_researcher
from enrich_pipeline import load_supabase_config
from icp import TARGET_TITLES
from search_utils import run_exa_search

load_dotenv(".env.local")

MODEL = "claude-opus-4-8"
COMPOSIO_USER_ID = "cameron_test_trimmed"
GMAIL_TOOLKIT_VERSION = "20260702_01"
MAX_ITERATIONS = 4
DEFAULT_LIMIT = 3  # first real test run -- larger batches once this is proven

# employee_count is excluded here -- AccountFinder already produced it.
RESEARCH_FIELDS = ["industry", "hiring_status", "tech_stack_hints", "recent_news"]

FIELD_QUERIES = {
    "industry": "{name} industry sector what does the company do",
    "hiring_status": "{name} hiring jobs openings 2026",
    "tech_stack_hints": "{name} software tools technology stack",
    "recent_news": "{name} news 2026",
}

console = Console()


def _panel(tag: str, body: str, color: str) -> None:
    console.print(Panel(body, title=tag, border_style=color))


def _bare_domain(url: str | None) -> str | None:
    """Strip scheme/www down to a bare domain, matching Supabase's stored
    website format (e.g. "company.com", no "https://")."""
    if not url:
        return None
    if "//" not in url:
        url = "//" + url
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc or None


EXTRACT_TOOL = {
    "name": "submit_field",
    "description": "Submit the extracted value for the requested field, strictly from the given search text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "value": {
                "type": ["string", "null"],
                "description": "The extracted value, or null if the text doesn't support it.",
            }
        },
        "required": ["value"],
    },
}


def _extract_field(client: Anthropic, field: str, company_name: str, search_text: str) -> str | None:
    """Pull a single field out of raw Exa search text via a schema-forced
    Claude call -- grounds the targeted-search fallback the same way
    AccountFinder/AccountResearcher ground their own findings: never trust
    free text, only what's demonstrably in the search results."""
    prompt = (
        f"Using ONLY the search text below, extract {company_name}'s "
        f"{field.replace('_', ' ')}. Never guess or use outside knowledge -- "
        f"if the text doesn't support it, return null.\n\n{search_text}"
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system="You extract a single structured fact strictly from the search text you're given. Never use outside knowledge, never guess.",
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "submit_field"},
        messages=[{"role": "user", "content": prompt}],
    )
    block = next((b for b in response.content if b.type == "tool_use"), None)
    value = block.input.get("value") if block else None
    return value if value and str(value).strip().lower() not in {"null", "none", "n/a", ""} else None


def _pick_best_contact(candidate: dict) -> dict:
    """Prefer verified_contacts (name+title+email+phone all confirmed by
    ContactFinder), else general_office (a real reachable line, no named
    person), else the best-titled contacts_seen name (no phone/email --
    just who to ask for). contact_name/title/phone/email are nullable in
    Supabase (see supabase-migrations.sql) precisely so these fallback
    tiers can be saved, not just the rare fully-verified case.

    email_verified/phone_verified are only set true for the
    verified_contacts tier -- that's the one case ContactFinder confirmed
    both fields directly for this specific named person."""
    verified = candidate.get("verified_contacts") or []
    if verified:
        c = verified[0]
        return {
            "contact_name": c.get("name"),
            "contact_title": c.get("title"),
            "contact_email": c.get("email"),
            "contact_phone": c.get("phone"),
            "email_verified": True,
            "phone_verified": True,
        }

    general_office = candidate.get("general_office")
    if general_office and (general_office.get("email") or general_office.get("phone")):
        return {
            "contact_name": None,
            "contact_title": None,
            "contact_email": general_office.get("email"),
            "contact_phone": general_office.get("phone"),
        }

    contacts_seen = candidate.get("contacts_seen") or []
    if contacts_seen:
        def rank(person):
            title = (person.get("title") or "").lower()
            for i, t in enumerate(TARGET_TITLES):
                if t.lower() in title:
                    return i
            return len(TARGET_TITLES)

        best = min(contacts_seen, key=rank)
        return {
            "contact_name": best.get("name"),
            "contact_title": best.get("title"),
            "contact_email": None,
            "contact_phone": None,
        }

    return {"contact_name": None, "contact_title": None, "contact_email": None, "contact_phone": None}


def _draft_email(client: Anthropic, name: str, row: dict, buying_trigger: str | None) -> str:
    contact_line = (
        f"Contact: {row['contact_name']}, {row['contact_title'] or 'unknown title'}"
        if row.get("contact_name") else "Contact: unnamed (general office)"
    )
    trigger_instruction = (
        f"Buying trigger: {buying_trigger}\nReference this specifically -- don't be generic."
        if buying_trigger
        else "No known buying trigger -- keep it focused on a plausible AP/Finance pain point for a company this size, not generic filler."
    )
    prompt = f"""You are a SaaS AE at Stampli (AP automation software). Draft a personalized
~120-word outreach email to this account.

Account: {name}
{contact_line}
{trigger_instruction}

Respond with ONLY the email body text, no subject line, no preamble."""
    response = client.messages.create(model=MODEL, max_tokens=512, messages=[{"role": "user", "content": prompt}])
    return next(b.text for b in response.content if b.type == "text")


def _create_gmail_draft(composio: Composio, name: str, recipient_email: str, body: str) -> None:
    composio.tools.execute(
        slug="GMAIL_CREATE_EMAIL_DRAFT",
        user_id=COMPOSIO_USER_ID,
        arguments={
            "recipient_email": recipient_email,
            "subject": f"Quick note for {name}",
            "body": body,
        },
    )


def _insert_account(supabase_url: str, supabase_key: str, row: dict) -> dict:
    import requests

    resp = requests.post(
        f"{supabase_url}/rest/v1/accounts",
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        json=row,
    )
    if not resp.ok:
        raise RuntimeError(f"Supabase insert failed ({resp.status_code}): {resp.text}")
    return resp.json()[0]


def _contact_tier(row: dict) -> str:
    if row.get("email_verified"):
        return "verified_contact"
    if row.get("contact_email") or row.get("contact_phone"):
        return "general_office"
    if row.get("contact_name"):
        return "named_no_contact_info"
    return "none"


def _process_company(
    client: Anthropic,
    exa: Exa,
    composio: Composio,
    supabase_url: str,
    supabase_key: str,
    index: int,
    total: int,
    candidate: dict,
    state: str,
    city: str | None,
) -> dict:
    """Returns a result summary dict for daily-digest aggregation:
    {name, territory, status, contact_tier, gmail_draft, error}."""
    name = candidate["name"]
    domain = _bare_domain(candidate.get("website"))
    header = f"Company {index}/{total}: {name}"
    console.rule(f"[bold cyan]{header}[/bold cyan]")

    # Baseline row, built up as data arrives so a mid-loop failure still has
    # something real to save (status gets flipped to 'partial' below).
    row = {
        "name": name,
        "website": domain,
        "territory": candidate.get("location") or (f"{city}, {state}" if city else state),
        "status": "new",
    }
    if candidate.get("employee_count") is not None:
        row["size"] = candidate["employee_count"]
    row.update(_pick_best_contact(candidate))

    buying_trigger = candidate.get("buying_trigger")
    gmail_draft = False

    try:
        findings = {f: None for f in RESEARCH_FIELDS}

        for iteration in range(1, MAX_ITERATIONS + 1):
            missing = [f for f in RESEARCH_FIELDS if not findings[f]]
            tag = f"{header} · iteration {iteration}"
            _panel(tag, f"OBSERVE: still missing = {missing}", "blue")

            if iteration == 1:
                _panel(tag, "THINK: first pass -- run AccountResearcher for a full sweep.", "blue")
                _panel(tag, f"ACT: account_researcher.research_account({name!r}, domain={domain!r})", "yellow")
                research = account_researcher.research_account(name, domain=domain)
                for f in RESEARCH_FIELDS:
                    if findings[f] is None and research.get(f):
                        findings[f] = research[f]
                if research.get("buying_triggers"):
                    buying_trigger = research["buying_triggers"]
            else:
                field = missing[0]
                query = FIELD_QUERIES[field].format(name=name)
                _panel(tag, f"THINK: still missing '{field}' -- run one targeted Exa search for it.", "blue")
                _panel(tag, f"ACT: Exa search: {query!r}", "yellow")
                text = run_exa_search(exa, query, num_results=5)
                value = _extract_field(client, field, name, text)
                if value:
                    findings[field] = value

            still_missing = [f for f in RESEARCH_FIELDS if not findings[f]]
            if not still_missing:
                _panel(tag, "CHECK: all fields filled -- moving to wrap-up.", "green")
                break
            if iteration == MAX_ITERATIONS:
                _panel(tag, f"CHECK: tool-call budget spent ({MAX_ITERATIONS}/{MAX_ITERATIONS}) -- wrapping up with what we have.", "green")
                break
            _panel(tag, f"CHECK: still missing {still_missing} -- back to observe.", "green")

        # industry/size: only fill if currently blank, never overwrite once set.
        if findings["industry"]:
            row["industry"] = findings["industry"]

        # buying_triggers/research_notes: always refresh with fresh findings.
        if buying_trigger:
            row["buying_triggers"] = buying_trigger
        notes = " | ".join(
            f"{label}: {findings[f]}"
            for f, label in (("hiring_status", "Hiring"), ("tech_stack_hints", "Tech"), ("recent_news", "News"))
            if findings[f]
        )
        if notes:
            row["research_notes"] = notes

        if row.get("contact_email"):
            _panel(header, "ACT: drafting personalized email + Gmail draft.", "yellow")
            email_body = _draft_email(client, name, row, buying_trigger)
            _create_gmail_draft(composio, name, row["contact_email"], email_body)
            gmail_draft = True

        inserted = _insert_account(supabase_url, supabase_key, row)
        _panel(header, f"CHECK/STOP: saved to Supabase (id={inserted.get('id')}), status={row['status']}.", "green")
        return {
            "name": name, "territory": row["territory"], "status": row["status"],
            "contact_tier": _contact_tier(row), "gmail_draft": gmail_draft, "error": None,
            "contact_name": row.get("contact_name"), "contact_title": row.get("contact_title"),
            "contact_email": row.get("contact_email"), "contact_phone": row.get("contact_phone"),
            "buying_trigger": buying_trigger,
        }

    except Exception as e:
        console.print(f"[bold red]Company {index}/{total} ({name}) failed: {e}[/bold red]")
        row["status"] = "partial"
        # territory/industry/size are NOT NULL in Supabase -- a failure before
        # research runs (e.g. AccountResearcher itself erroring) can leave
        # industry unset, which would make even this partial save fail the
        # same way and silently lose the company. status='partial' already
        # signals "incomplete, go verify" -- a labeled placeholder here is
        # honest, not a fabrication, unlike inventing data in a normal row.
        row.setdefault("industry", "Unknown (research failed)")
        row.setdefault("size", candidate.get("employee_count") or 0)
        result = {
            "name": name, "territory": row["territory"], "status": "partial",
            "contact_tier": _contact_tier(row), "gmail_draft": gmail_draft, "error": str(e),
            "contact_name": row.get("contact_name"), "contact_title": row.get("contact_title"),
            "contact_email": row.get("contact_email"), "contact_phone": row.get("contact_phone"),
            "buying_trigger": row.get("buying_triggers"),
        }
        try:
            inserted = _insert_account(supabase_url, supabase_key, row)
            console.print(f"[yellow]Saved partial row for {name} (id={inserted.get('id')}).[/yellow]")
        except Exception as e2:
            console.print(f"[bold red]Also failed to save partial row for {name}: {e2}[/bold red]")
            result["status"] = "lost"
            result["error"] = f"{e} | partial-save also failed: {e2}"
        return result


def _init_clients() -> tuple:
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    composio_key = os.getenv("COMPOSIO_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY not set in environment.")
    if not composio_key:
        raise RuntimeError("COMPOSIO_API_KEY not set in environment.")

    client = Anthropic(api_key=anthropic_key)
    exa = Exa(exa_key)
    composio = Composio(api_key=composio_key, toolkit_versions={"gmail": GMAIL_TOOLKIT_VERSION})
    supabase_url, supabase_key = load_supabase_config()
    return client, exa, composio, supabase_url, supabase_key


def _run_state(
    client: Anthropic, exa: Exa, composio: Composio, supabase_url: str, supabase_key: str,
    state: str, min_size: int, max_size: int, city: str | None = None, industry: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> list[dict]:
    where = f"{city + ', ' if city else ''}{state}"
    console.print(f"[bold]Finding up to {limit} new qualifying companies in {where} ({min_size}-{max_size} employees)...[/bold]")
    candidates = account_finder.find_accounts(state, min_size, max_size, city=city, industry=industry, limit=limit)
    console.print(f"[bold]AccountFinder returned {len(candidates)} candidate(s).[/bold]")

    results = []
    for i, candidate in enumerate(candidates, start=1):
        results.append(
            _process_company(client, exa, composio, supabase_url, supabase_key, i, len(candidates), candidate, state, city)
        )
    return results


def run(state: str, min_size: int, max_size: int, city: str | None = None, industry: str | None = None, limit: int = DEFAULT_LIMIT) -> None:
    client, exa, composio, supabase_url, supabase_key = _init_clients()
    _run_state(client, exa, composio, supabase_url, supabase_key, state, min_size, max_size, city=city, industry=industry, limit=limit)
    console.rule("[bold]Done[/bold]")


# Daily unattended run (see run_daily / the /schedule cloud routine): fixed
# territories Cameron chose, 20-399 employees (the standing ICP floor -- he
# only specified the max), any industry, up to 3/state/day.
DAILY_TERRITORIES = ["California", "New York", "Florida"]
DAILY_MIN_SIZE = 20
DAILY_MAX_SIZE = 399
DAILY_LIMIT_PER_STATE = 3
DAILY_RECIPIENT = "camerondowd13@gmail.com"


def _send_summary_email(composio: Composio, results: list[dict], failed_states: list[tuple[str, str]] | None = None) -> None:
    """Sends (not drafts) the daily digest directly to Cameron -- a
    self-notification, not outreach to a prospect, so the codebase's
    'draft only, never send' rule (which governs the per-company outreach
    emails above) doesn't apply here.

    failed_states: territories whose entire run_state() call raised before
    returning any results (e.g. AccountFinder itself erroring) -- these
    produce zero entries in `results`, so without surfacing them here the
    email would silently look like that territory was just empty today
    instead of broken."""
    import datetime

    failed_states = failed_states or []
    today_iso = datetime.date.today().isoformat()
    today_human = datetime.date.today().strftime("%B %d, %Y")
    saved = [r for r in results if r["status"] in ("new", "partial")]
    lost = [r for r in results if r["status"] == "lost"]
    drafts = sum(1 for r in results if r["gmail_draft"])

    # Flush-left block layout, no leading-space indentation: Gmail renders
    # plain-text email in a proportional font and collapses leading
    # whitespace, so indentation-based hierarchy flattens in the inbox.
    # Hierarchy here comes from section headers, blank lines between blocks,
    # and full-width divider rules (runs of dashes survive, unlike spaces).
    divider = "-" * 52

    lines = ["Daily Prospecting Report", today_human, ""]
    lines.append("SUMMARY")
    lines.append(f"{len(saved)} compan{'y' if len(saved) == 1 else 'ies'} saved to the board. {drafts} outreach draft(s) created.")
    if failed_states:
        lines.append(f"{len(failed_states)} territor{'y' if len(failed_states) == 1 else 'ies'} could not run at all today: {', '.join(s for s, _ in failed_states)}.")
    if lost:
        lines.append(f"{len(lost)} compan{'y' if len(lost) == 1 else 'ies'} could not be saved (see failures below).")

    if saved:
        lines.append("")
        lines.append(divider)
        lines.append("NEW LEADS")
        for r in saved:
            lines.append("")
            flag = "  [PARTIAL -- needs a look]" if r["status"] == "partial" else ""
            lines.append(f"{r['name']} -- {r['territory']}{flag}")
            if r.get("contact_tier"):
                lines.append(f"Contact type: {r['contact_tier'].capitalize()}")
            contact_line = ", ".join(
                p for p in (r.get("contact_name"), r.get("contact_title")) if p
            )
            if contact_line:
                lines.append(f"Contact: {contact_line}")
            if r.get("contact_email"):
                lines.append(f"Email: {r['contact_email']}")
            if r.get("contact_phone"):
                lines.append(f"Phone: {r['contact_phone']}")
            if r.get("buying_trigger"):
                lines.append(f"Trigger: {r['buying_trigger']}")
    else:
        lines.append("")
        lines.append("No new qualifying companies found today.")

    if lost:
        lines.append("")
        lines.append(divider)
        lines.append("COULD NOT BE SAVED")
        for r in lost:
            lines.append("")
            lines.append(f"{r['name']} -- {r['territory']}")
            lines.append(r["error"])

    if failed_states:
        lines.append("")
        lines.append(divider)
        lines.append("TERRITORIES THAT DIDN'T RUN")
        for state, error in failed_states:
            lines.append("")
            lines.append(state)
            lines.append(error)

    body = "\n".join(lines)
    subject = f"Daily Prospecting Report -- {today_iso} -- {len(saved)} new lead(s)"
    composio.tools.execute(
        slug="GMAIL_SEND_EMAIL",
        user_id=COMPOSIO_USER_ID,
        arguments={"recipient_email": DAILY_RECIPIENT, "subject": subject, "body": body},
    )


def send_lead_report(results: list[dict], failed_states: list[tuple[str, str]] | None = None) -> None:
    """Public wrapper: build a Composio client and send the leads report,
    reusing _send_summary_email's exact formatting. Lets callers that don't
    already have a Composio client (e.g. server.py's /email-leads demo
    endpoint) send the same report the scheduled daily run sends."""
    composio = Composio(
        api_key=os.getenv("COMPOSIO_API_KEY"),
        toolkit_versions={"gmail": GMAIL_TOOLKIT_VERSION},
    )
    _send_summary_email(composio, results, failed_states)


def run_daily() -> None:
    """Entry point for the scheduled 8am cloud routine: runs the full
    pipeline across DAILY_TERRITORIES, then sends (not drafts) one summary
    email covering the whole day's results."""
    client, exa, composio, supabase_url, supabase_key = _init_clients()

    all_results = []
    failed_states = []
    for state in DAILY_TERRITORIES:
        console.rule(f"[bold magenta]Daily run: {state}[/bold magenta]")
        try:
            all_results.extend(
                _run_state(client, exa, composio, supabase_url, supabase_key,
                           state, DAILY_MIN_SIZE, DAILY_MAX_SIZE, limit=DAILY_LIMIT_PER_STATE)
            )
        except Exception as e:
            console.print(f"[bold red]Daily run for {state} failed entirely:[/bold red]")
            console.print_exception()
            failed_states.append((state, str(e)))

    console.rule("[bold]Sending daily summary email[/bold]")
    _send_summary_email(composio, all_results, failed_states)
    console.rule("[bold]Done[/bold]")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--daily":
        run_daily()
        sys.exit(0)

    if len(sys.argv) < 4:
        print('Usage: python prospecting_agent.py "State" min_size max_size [city] [industry] [limit]')
        print('       python prospecting_agent.py --daily')
        sys.exit(1)

    state = sys.argv[1]
    min_size = int(sys.argv[2])
    max_size = int(sys.argv[3])
    city = (sys.argv[4] or None) if len(sys.argv) > 4 else None
    industry = (sys.argv[5] or None) if len(sys.argv) > 5 else None
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else DEFAULT_LIMIT

    run(state, min_size, max_size, city=city, industry=industry, limit=limit)
