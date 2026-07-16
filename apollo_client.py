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
Supabase table; poll_phone_reveal() below polls that table, since this
script has no long-running listener of its own to receive the callback
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

PHONE_POLL_TIMEOUT = 150  # seconds -- Apollo's webhook typically lands within a few minutes
PHONE_POLL_INTERVAL = 10


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
    enrich_email() / request_phone_reveal() per person id."""
    resp = requests.post(
        f"{APOLLO_BASE}/mixed_people/api_search",
        headers=_headers(),
        json={
            "q_organization_domains_list": [domain],
            "person_titles": titles,
            "include_similar_titles": False,
            "per_page": per_page,
        },
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json().get("people", [])


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


def poll_phone_reveal(person_id: str, timeout: int = PHONE_POLL_TIMEOUT) -> str | None:
    """Polls the apollo_phone_reveals table (populated by
    api/apollo-phone-webhook.js) for up to `timeout` seconds. Returns the
    phone number once the webhook lands, or None on timeout -- the caller
    treats that the same as Apollo never having a number (falls back to
    general_office), not as an error."""
    supabase_url, supabase_key = load_supabase_config()
    deadline = time.monotonic() + timeout
    while True:
        resp = requests.get(
            f"{supabase_url}/rest/v1/apollo_phone_reveals",
            params={"apollo_person_id": f"eq.{person_id}", "select": "phone"},
            headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows and rows[0].get("phone"):
            return rows[0]["phone"]
        if time.monotonic() >= deadline:
            return None
        time.sleep(PHONE_POLL_INTERVAL)
