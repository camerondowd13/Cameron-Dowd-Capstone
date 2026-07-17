"""
ApolloClient: thin wrapper around Apollo.io's People Search + People
Enrichment APIs. Used by contact_finder.py as its first-choice contact
source when APOLLO_API_KEY is set -- falls back to the existing Exa+Claude
search otherwise (see contact_finder.py's module docstring, which already
flagged low real-contact hit rate as the signal to bring in a dedicated
provider like Apollo).

Email reveal (people/match with reveal_personal_emails) is synchronous.
Phone reveal is NOT: Apollo only delivers a revealed mobile/direct-dial
number via an async webhook POST, minutes after the request -- there's no
phone number in the immediate response at all. api/apollo-phone-webhook.js
receives that callback and writes it into the apollo_phone_reveals
Supabase table; poll_phone_reveals() below polls that table for several
person ids at once (one shared timeout window, not one per person), since
this script has no long-running listener of its own to receive the callback
directly.
"""
import os
import time

import requests
from dotenv import load_dotenv

from enrich_pipeline import load_supabase_config

load_dotenv(".env.local")

APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")
# Full webhook URL Apollo will POST the revealed phone number to,
# including the auth token query param api/apollo-phone-webhook.js
# checks against APOLLO_WEBHOOK_SECRET -- see .env.local.
APOLLO_WEBHOOK_URL = os.getenv("APOLLO_WEBHOOK_URL")
APOLLO_BASE = "https://api.apollo.io/api/v1"

# Phone reveal is async (Apollo POSTs the number to a webhook). We poll for
# it, but only briefly: a live web search can't wait minutes per company, and
# stacking long waits across several companies blows past the backend's 300s
# limit and gets the whole request killed. If the number hasn't landed in this
# window, we fall back to the company's general office line (guaranteed from
# Apollo's org record), so the lead is still complete -- just not a direct dial.
PHONE_POLL_TIMEOUT = 25  # seconds (was 150)
PHONE_POLL_INTERVAL = 5


def _headers() -> dict:
    return {
        "x-api-key": APOLLO_API_KEY,
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
    }


def search_people(domain: str, titles: list[str], per_page: int = 5) -> list[dict]:
    """People Search: real people at `domain` matching `titles`. Returns
    raw Apollo person dicts (id, name, title, has_email, has_direct_phone,
    etc.) -- email/phone VALUES aren't included here, those require
    enrich_email() / request_phone_reveal() per person id.

    Filtered to contact_email_status verified/likely-to-engage so we don't
    burn an enrichment call discovering a match Apollo already knows has no
    email (e.g. has_email: false) -- confirmed happening in production
    testing. Not filtered any tighter than that: Apollo's own coverage is
    already thin for smaller companies, so over-filtering risks zero
    candidates before enrichment even runs."""
    resp = requests.post(
        f"{APOLLO_BASE}/mixed_people/api_search",
        headers=_headers(),
        json={
            "q_organization_domains_list": [domain],
            "person_titles": titles,
            "include_similar_titles": False,
            "contact_email_status": ["verified", "likely to engage"],
            "per_page": per_page,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("people", [])


def search_organizations(
    state: str, keyword: str, min_size: int, max_size: int, per_page: int = 25, page: int = 1
) -> list[dict]:
    """Organization Search: real companies matching location + employee
    range + a keyword tag. Doesn't consume Apollo credits (Organization
    Search is free per Apollo's pricing docs). Returns raw Apollo org
    dicts (name, website_url, sic_codes, naics_codes, primary_phone, etc.).

    q_organization_keyword_tags is a loose/fuzzy match -- confirmed in
    testing it pulls in tangentially-related companies (a trade
    association, staffing agencies) alongside real matches. Callers MUST
    post-filter by sic_codes (see INDUSTRY_SIC_PREFIXES in account_finder.py)
    for a precise industry match; don't trust the keyword tag alone.

    Confirmed in testing: results for identical parameters are NOT stable
    between calls -- one page-1 pull returned 0 real SIC matches out of
    100, an immediately-following call to the same query returned several
    in its first 5. Callers needing a reliable yield should paginate (see
    `page`) rather than trust a single page."""
    resp = requests.post(
        f"{APOLLO_BASE}/mixed_companies/search",
        headers=_headers(),
        json={
            "organization_locations": [f"{state}, US"],
            "organization_num_employees_ranges": [f"{min_size},{max_size}"],
            "q_organization_keyword_tags": [keyword],
            "per_page": per_page,
            "page": page,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("organizations", [])


def enrich_person(person_id: str) -> dict:
    """Synchronous email reveal for a specific person id (from
    search_people). Returns {"email", "name"} -- email is None if Apollo
    doesn't have one (never a locked placeholder like
    'email_not_unlocked@domain.com'). name is the real, unmasked full name
    -- search_people only gives an obfuscated last name (e.g. 'Ow***s'),
    this /people/match response is the first point a full name is
    available. Response body is nested under a top-level "person" key."""
    resp = requests.post(
        f"{APOLLO_BASE}/people/match",
        headers=_headers(),
        json={"id": person_id, "reveal_personal_emails": True},
        timeout=20,
    )
    resp.raise_for_status()
    person = resp.json().get("person") or {}
    email = person.get("email")
    return {
        "email": email if email and "not_unlocked" not in email else None,
        "name": person.get("name"),
    }


def request_phone_reveal(person_id: str) -> bool:
    """Kicks off Apollo's ASYNC phone reveal for a person id -- the actual
    number is delivered later via webhook, not in this call's response.
    Returns False (no-op, nothing requested) if APOLLO_WEBHOOK_URL isn't
    configured, since there'd be nowhere for Apollo to deliver it."""
    if not APOLLO_WEBHOOK_URL:
        return False
    resp = requests.post(
        f"{APOLLO_BASE}/people/match",
        headers=_headers(),
        json={
            "id": person_id,
            "reveal_phone_number": True,
            "webhook_url": APOLLO_WEBHOOK_URL,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return True


def poll_phone_reveals(person_ids: list[str], timeout: int = PHONE_POLL_TIMEOUT) -> dict[str, str]:
    """Polls the apollo_phone_reveals table for MULTIPLE person ids sharing
    one timeout window, instead of one full timeout per person -- since
    Apollo's webhook lands independently of when we start polling for it,
    requesting reveals for several candidates up front and polling for all
    of them at once means the wait is ~150s total regardless of how many
    candidates we're trying, not 150s stacked per candidate.

    Returns {person_id: phone} for whichever ids resolved before the
    deadline; ids that never resolved are simply absent from the result --
    same as a single-person timeout, not an error."""
    if not person_ids:
        return {}
    supabase_url, supabase_key = load_supabase_config()
    deadline = time.monotonic() + timeout
    resolved: dict[str, str] = {}
    remaining = set(person_ids)
    while remaining and time.monotonic() < deadline:
        resp = requests.get(
            f"{supabase_url}/rest/v1/apollo_phone_reveals",
            params={
                "apollo_person_id": f"in.({','.join(remaining)})",
                "select": "apollo_person_id,phone",
            },
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        for row in resp.json():
            phone = row.get("phone")
            pid = row.get("apollo_person_id")
            if phone and pid in remaining:
                resolved[pid] = phone
                remaining.discard(pid)
        if remaining:
            time.sleep(PHONE_POLL_INTERVAL)
    return resolved
