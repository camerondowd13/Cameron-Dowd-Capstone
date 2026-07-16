"""
ContactFinder: given a company (name + optional domain), find reachable
decision-makers — name, title, email, phone.

Per the PRD, a contact is only valid if BOTH email and phone are found;
one without the other is a dead end and is dropped in code (never
returned as partial).

First choice: Apollo (apollo_client.py), when APOLLO_API_KEY is set and a
domain is known -- this is the "bring a dedicated provider" path the old
version of this docstring flagged as the fix for a low real-contact hit
rate on the open web. Falls back to the Exa+Claude web search below
whenever Apollo isn't configured, errors, or simply doesn't have a
reachable contact for this company (e.g. trial credits exhausted) --
same honest caveat as before applies to that fallback: direct phone
numbers for a named individual are rarely published on the open web, so
expect it to return few or zero contacts on its own.
"""
import json
import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv
from exa_py import Exa

import apollo_client
from icp import TARGET_TITLES
from search_utils import clean_nullish, run_exa_search, strip_linkedin

load_dotenv(".env.local")

MODEL = "claude-opus-4-8"
MAX_SEARCHES = 4
MAX_TURNS = MAX_SEARCHES + 2  # spare turns for the "limit reached" nudge + final submit

SYSTEM_PROMPT = f"""You are a SaaS AE finding reachable contacts at a target account.
Given a company, use web_search to find real people in decision-making
roles — prioritize titles like {", ".join(TARGET_TITLES)}, or others
plausibly involved in purchasing finance/AP software.

If a known official domain is given: your first 2 searches are locked to
that company's own site. Use them on the pages most likely to actually
list contact info -- queries like "team", "leadership", "about us",
"contact us", "staff directory". Company sites are the best source for
directly-listed emails/phones, better than general open-web search. Only
after those 2 searches should you broaden to the open web (press
releases, conference bios, public filings) for names you still need
contact info on.

For each person, only report an email or phone number if you found it
directly in a search result (company "team"/"about" page, press release,
conference bio, public filing, etc.) — NEVER guess an email pattern
(like first.last@company.com) and NEVER invent a phone number. If you
can't confirm a real email or phone for someone, use null for that field
— a person is still worth reporting with just a name/title if that's all
you found; the caller will filter incomplete contacts.

Also report source_url for each person: the real URL (from your search
results) where you found their name/title/contact info, so a human can
click through and verify. Never invent a URL.

Separately, also report general_office: the company's general/main
office phone number AND general email (e.g. info@, sales@, contact@),
if you saw either anywhere in your searches (e.g. on a "Contact Us"
page) — even if you couldn't find a direct line for any specific
person. This is a fallback so there's still a real way to reach the
company even when no named contact is directly reachable. Only report a
real phone/email you actually saw, with its source_url — null for
anything you didn't see. Never invent one.

Use max 4 searches. Report real people only — never invent a name."""

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Search the web via Exa. When a domain is locked (your first 2 "
        "searches, if a domain is known), use site-focused queries like "
        "'<company> leadership team', '<company> about us', '<company> "
        "contact us staff directory'. Once open, broaden to "
        "'<company> CFO', '<company> press release [name] [title]'."
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
    "name": "submit_contacts",
    "description": "Submit the final list of contacts found.",
    "input_schema": {
        "type": "object",
        "properties": {
            "contacts": {
                "type": "array",
                "description": "People found, most senior/relevant first.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full name."},
                        "title": {"type": ["string", "null"]},
                        "email": {
                            "type": ["string", "null"],
                            "description": "Only if directly confirmed in a search result, else null.",
                        },
                        "phone": {
                            "type": ["string", "null"],
                            "description": "Only if directly confirmed in a search result, else null.",
                        },
                        "source_url": {
                            "type": ["string", "null"],
                            "description": "Real URL where this person's info was found, or null.",
                        },
                    },
                    "required": ["name", "title", "email", "phone", "source_url"],
                },
            },
            "general_office": {
                "type": ["object", "null"],
                "description": (
                    "The company's general/main office phone and/or general "
                    "email, if seen anywhere in search results -- null for "
                    "anything never seen."
                ),
                "properties": {
                    "phone": {"type": ["string", "null"]},
                    "email": {"type": ["string", "null"]},
                    "source_url": {"type": ["string", "null"]},
                },
                "required": ["phone", "email", "source_url"],
            },
        },
        "required": ["contacts", "general_office"],
    },
}


def _is_reachable(contact: dict) -> bool:
    """PRD rule: phone/email resolution — both required, or the contact is invalid."""
    return bool(clean_nullish(contact.get("email"))) and bool(clean_nullish(contact.get("phone")))


def _rank_by_title(person: dict, titles: list[str]) -> int:
    title = (person.get("title") or "").lower()
    for i, t in enumerate(titles):
        if t.lower() in title:
            return i
    return len(titles)


def _find_contacts_via_apollo(domain: str, titles: list[str], limit: int) -> list[dict]:
    """Apollo-first contact lookup: real named people, ranked by title
    match, with a synchronously-revealed verified email. Phone reveal
    (async, credits-limited -- see apollo_client.py) is only attempted for
    the single best-ranked match per company, to conserve Apollo's
    separately-metered mobile-reveal credits across a whole run rather
    than spend them 1-for-1 on every person searched.

    Same PRD bar as the Exa path (_is_reachable: both email AND phone
    required) -- a person with only one of the two is dropped here too,
    not returned as partial."""
    people = apollo_client.search_people(domain, titles, per_page=limit * 2)
    if not people:
        return []
    people.sort(key=lambda p: _rank_by_title(p, titles))

    contacts = []
    for i, person in enumerate(people):
        if len(contacts) >= limit:
            break
        person_id = person.get("id")
        if not person_id:
            continue

        enriched = apollo_client.enrich_person(person_id)
        email = enriched["email"]
        if not email:
            continue

        phone = None
        if i == 0 and apollo_client.request_phone_reveal(person_id):
            phone = apollo_client.poll_phone_reveal(person_id)
        if not phone:
            continue

        contacts.append({
            "name": enriched["name"],
            "title": person.get("title"),
            "email": email,
            "phone": phone,
            "source_url": f"https://app.apollo.io/#/people/{person_id}",
        })

    return contacts


def find_contacts(
    account_name: str,
    domain: str | None = None,
    target_titles: list[str] | None = None,
    limit: int = 3,
) -> dict:
    """Find reachable contacts at an account, plus a general-office phone
    fallback. Returns {"contacts": [...], "general_office": {...} | None}.

    contacts: up to `limit` dicts {"name", "title", "email", "phone",
    "source_url"} — only contacts with BOTH email and phone confirmed are
    included (PRD: no partial-credit contacts).

    general_office: {"phone", "email", "source_url"} for the company's main
    line/general inbox, or None if neither was ever seen. This is a
    fallback so there's still a real way to reach the company even when
    zero named contacts are directly reachable -- direct lines/emails for
    a specific person are rarely published, but a company's main
    phone/email almost always is.
    """
    titles = target_titles or TARGET_TITLES

    if domain and apollo_client.APOLLO_API_KEY:
        try:
            apollo_contacts = _find_contacts_via_apollo(domain, titles, limit)
        except Exception as e:
            # Apollo being down/out-of-credits/misconfigured shouldn't
            # block contact-finding entirely -- fall through to the Exa
            # path below, same as when Apollo simply finds nothing.
            print(f"Warning: Apollo lookup failed for {account_name!r}, falling back to web search: {e}", file=sys.stderr)
            apollo_contacts = []
        if apollo_contacts:
            return {"contacts": apollo_contacts, "general_office": None}
        # Apollo found nothing reachable -- fall through to Exa+Claude,
        # which also covers general_office (Apollo's People Search has no
        # equivalent to a company's main line/inbox).

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    exa_key = os.getenv("EXA_API_KEY")
    if not anthropic_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment.")
    if not exa_key:
        raise RuntimeError("EXA_API_KEY not set in environment.")

    client = Anthropic(api_key=anthropic_key)
    exa = Exa(exa_key)

    request = f"Find contacts at: {account_name}\nTarget titles: {', '.join(titles)}"
    if domain:
        request += (
            f"\nKnown official domain: {domain}. Your first 2 searches are locked "
            "to this domain -- use them on the company's own team/leadership/"
            "contact pages before broadening to the open web."
        )

    messages = [{"role": "user", "content": request}]
    search_count = 0

    for _ in range(MAX_TURNS):
        # Once the search budget is spent, force the model to submit --
        # a text-only nudge alone isn't reliable (it can keep calling
        # web_search past the limit instead of wrapping up).
        if search_count >= MAX_SEARCHES:
            tools = [SUBMIT_TOOL]
            tool_choice = {"type": "tool", "name": "submit_contacts"}
        else:
            tools = [WEB_SEARCH_TOOL, SUBMIT_TOOL]
            tool_choice = {"type": "auto"}

        response = client.messages.create(
            model=MODEL,
            max_tokens=1536,
            system=SYSTEM_PROMPT,
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

            if block.name == "submit_contacts":
                submitted = block.input
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": "Received."}
                )

            elif block.name == "web_search":
                if search_count >= MAX_SEARCHES:
                    content = (
                        "Search limit reached (4/4). Submit whatever contacts "
                        "you've found now via submit_contacts."
                    )
                else:
                    search_count += 1
                    lock_domains = [domain] if (domain and search_count <= 2) else None
                    content = run_exa_search(
                        exa, block.input.get("query", account_name), lock_domains
                    )
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": content}
                )

        if submitted is not None:
            all_contacts = submitted.get("contacts", [])
            reachable = [c for c in all_contacts if _is_reachable(c)]
            for c in reachable:
                c["source_url"] = strip_linkedin(c.get("source_url"))

            general_office = submitted.get("general_office")
            if general_office:
                phone = clean_nullish(general_office.get("phone"))
                email = clean_nullish(general_office.get("email"))
                source_url = strip_linkedin(general_office.get("source_url"))
                general_office = (
                    {"phone": phone, "email": email, "source_url": source_url}
                    if (phone or email) and source_url else None
                )

            return {"contacts": reachable[:limit], "general_office": general_office}

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(
        f"find_contacts('{account_name}') did not submit contacts within {MAX_TURNS} turns."
    )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print('Usage: python contact_finder.py "Account Name" [domain]')
        sys.exit(1)

    name = sys.argv[1]
    account_domain = sys.argv[2] if len(sys.argv) == 3 else None
    print(f"Finding contacts at '{name}'...", file=sys.stderr)
    contacts = find_contacts(name, domain=account_domain)
    print(json.dumps(contacts, indent=2))
