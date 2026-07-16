"""ScoutAgent: weekly personal digest. For each of Cameron's three interests
(Fitness, AI, Comedy), searches Exa, judges each batch of results against
this-week/this-city/actually-his-taste, and keeps only what passes. A thin
category triggers a new search angle -- never a repeated query. Stops once
every category has 2-3 solid items or 8 searches total have run, then
composes one short digest email and sends it via Composio Gmail.

Only two tools in the loop: Exa search and Composio Gmail. Claude itself is
used purely for judgment (propose_query / judge_search_results) and for
writing the final email -- no other layers.
"""
import datetime
import os

from anthropic import Anthropic
from composio import Composio
from dotenv import load_dotenv
from exa_py import Exa
from rich.console import Console
from rich.panel import Panel

from search_utils import run_exa_search

load_dotenv(".env.local")

MODEL = "claude-sonnet-5"
COMPOSIO_USER_ID = "cameron_test_trimmed"
GMAIL_TOOLKIT_VERSION = "20260702_01"
RECIPIENT = "camerondowd13@gmail.com"

MAX_SEARCHES = 8
MIN_ITEMS = 2
MAX_ITEMS = 3

CATEGORIES = [
    {
        "key": "fitness",
        "label": "Fitness",
        "brief": (
            "Weightlifting/yoga news (national or general-interest), AND yoga "
            "events/classes happening in Nashville, TN this coming week. "
            "Either type counts toward this category."
        ),
    },
    {
        "key": "ai",
        "label": "AI",
        "brief": (
            "AI news, new product launches/updates, or other genuinely notable "
            "AI developments from this coming week or very recently."
        ),
    },
    {
        "key": "comedy",
        "label": "Comedy",
        "brief": "Stand-up comedy shows/performances happening in Nashville, TN this coming week.",
    },
]

console = Console()


def _panel(tag: str, body: str, color: str) -> None:
    console.print(Panel(body, title=tag, border_style=color))


PROPOSE_QUERY_TOOL = {
    "name": "propose_query",
    "description": "Propose the next Exa search query to try for this category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "reasoning": {
                "type": "string",
                "description": "One sentence: why this query, and how it differs from previous attempts.",
            },
        },
        "required": ["query", "reasoning"],
    },
}

JUDGE_TOOL = {
    "name": "judge_search_results",
    "description": "Judge raw search results against the category criteria and extract only items that genuinely qualify.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["good", "thin", "empty"]},
            "reasoning": {"type": "string", "description": "One or two sentences on why items were kept or rejected."},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "link": {"type": "string", "description": "Exact URL as it appears in the search results -- never invented."},
                        "when": {"type": "string", "description": "The date/day this occurs, or e.g. 'newly launched' for AI items."},
                        "why_it_fits": {"type": "string", "description": "One short phrase on why this is genuinely current and on-taste."},
                    },
                    "required": ["title", "link", "when", "why_it_fits"],
                },
            },
        },
        "required": ["verdict", "reasoning", "items"],
    },
}

COMPOSE_TOOL = {
    "name": "compose_digest_email",
    "description": "Compose the final weekly digest email.",
    "input_schema": {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "body": {
                "type": "string",
                "description": "Plain text email body, short and friendly, organized by category with links.",
            },
        },
        "required": ["subject", "body"],
    },
}


def _propose_query(client: Anthropic, label: str, brief: str, date_context: str, tried: list[str], current_items: list[dict]) -> tuple[str, str]:
    tried_block = "\n".join(f"- {q}" for q in tried) or "(none yet)"
    items_block = "\n".join(f"- {i['title']} ({i['when']})" for i in current_items) or "(none found yet)"
    prompt = f"""{date_context}

Category: {label} -- {brief}

Queries already tried for this category (never repeat one -- pick a meaningfully different angle):
{tried_block}

Items already accepted for this category:
{items_block}

Propose ONE new Exa web search query likely to surface genuinely current, specific items for this category. Prefer specific event-listing / news-source angles over generic roundup queries."""
    response = client.messages.create(
        model=MODEL, max_tokens=300, tools=[PROPOSE_QUERY_TOOL],
        tool_choice={"type": "tool", "name": "propose_query"},
        messages=[{"role": "user", "content": prompt}],
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return block.input["query"], block.input["reasoning"]


def _judge(client: Anthropic, label: str, brief: str, date_context: str, query: str, raw_text: str) -> dict:
    prompt = f"""{date_context}

Category: {label} -- {brief}

Search query used: {query!r}

Raw search results:
{raw_text}

Judge these results strictly:
- Must be genuinely CURRENT: a specific dated event/class this coming week, or (for AI) genuinely recent news/launch -- not evergreen or generic content.
- Must actually match this category, not just tangentially related.
- Any Nashville-specific item must actually be in/near Nashville, TN.
- Reject vague listicles, "best of" roundup articles, or anything without a real link/date grounding it.
- Only extract items truly supported by the text above -- never invent a link or date. Use the exact URL shown in the results.

Return verdict "good" (1+ solid items), "thin" (weak/partial), or "empty" (nothing qualifies). List only the items that pass."""
    response = client.messages.create(
        model=MODEL, max_tokens=800, tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "judge_search_results"},
        messages=[{"role": "user", "content": prompt}],
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return block.input


def _compose_email(client: Anthropic, state: dict, date_context: str, thin_labels: list[str]) -> tuple[str, str]:
    sections = []
    for c in CATEGORIES:
        items = state[c["key"]]["items"]
        block = "\n".join(
            f"- {i['title']} ({i['when']}) -- {i['link']} -- {i['why_it_fits']}" for i in items
        ) or "(nothing solid found this week)"
        sections.append(f"{c['label']}:\n{block}")
    items_block = "\n\n".join(sections)
    thin_note = (
        f"Categories that came up thin this week: {', '.join(thin_labels)}."
        if thin_labels else "Every category came up solid this week."
    )

    prompt = f"""{date_context}

You're drafting a short, friendly weekly personal digest email for Cameron, covering the items his scout agent found and vetted this week. Write it TO Cameron, casual and brief -- like a friend texting recs, not a marketing newsletter. Organize by category with each item's name, when it is, and its link. {thin_note}

Vetted items:
{items_block}

Keep it tight -- no long intro, no filler."""
    response = client.messages.create(
        model=MODEL, max_tokens=900, tools=[COMPOSE_TOOL],
        tool_choice={"type": "tool", "name": "compose_digest_email"},
        messages=[{"role": "user", "content": prompt}],
    )
    block = next(b for b in response.content if b.type == "tool_use")
    return block.input["subject"], block.input["body"]


def _init_clients() -> tuple[Anthropic, Exa, Composio]:
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
    return client, exa, composio


def run() -> None:
    client, exa, composio = _init_clients()

    today = datetime.date.today()
    week_end = today + datetime.timedelta(days=7)
    date_context = (
        f"Today is {today.strftime('%A, %B %d, %Y')}. \"This coming week\" means "
        f"between now and {week_end.strftime('%A, %B %d, %Y')}."
    )

    console.rule("[bold magenta]Weekly Scout[/bold magenta]")
    _panel("SETUP", date_context, "blue")

    state = {c["key"]: {"items": [], "queries_tried": []} for c in CATEGORIES}
    seen_links: set[str] = set()
    searches_used = 0

    while searches_used < MAX_SEARCHES:
        pending = [c for c in CATEGORIES if len(state[c["key"]]["items"]) < MIN_ITEMS]
        if not pending:
            break
        category = pending[searches_used % len(pending)]
        key, label, brief = category["key"], category["label"], category["brief"]
        cat_state = state[key]
        tag = f"{label} · search {searches_used + 1}/{MAX_SEARCHES}"

        _panel(
            tag,
            f"OBSERVE: {len(cat_state['items'])} accepted item(s) so far. "
            f"Queries tried: {cat_state['queries_tried'] or 'none'}.",
            "blue",
        )

        query, reasoning = _propose_query(client, label, brief, date_context, cat_state["queries_tried"], cat_state["items"])
        _panel(tag, f"THINK: {reasoning}\nQuery -> {query!r}", "blue")
        cat_state["queries_tried"].append(query)

        _panel(tag, f"ACT: running Exa search for {query!r}", "yellow")
        try:
            raw_text = run_exa_search(exa, query, num_results=6)
        except Exception as e:
            raw_text = f"Search failed: {e}"
        searches_used += 1

        judgment = _judge(client, label, brief, date_context, query, raw_text)
        verdict = judgment.get("verdict", "empty")
        items = judgment.get("items") or []
        color = "green" if verdict == "good" else ("yellow" if verdict == "thin" else "red")
        _panel(tag, f"CHECK ({verdict}): {judgment.get('reasoning', '')}", color)

        added = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            if not link or link in seen_links:
                continue
            if len(cat_state["items"]) >= MAX_ITEMS:
                break
            seen_links.add(link)
            cat_state["items"].append(item)
            added += 1
        if added:
            console.print(f"[green]  + kept {added} item(s) for {label}[/green]")

    console.rule("[bold]Search loop complete[/bold]")
    thin_labels = []
    for c in CATEGORIES:
        n = len(state[c["key"]]["items"])
        satisfied = n >= MIN_ITEMS
        if not satisfied:
            thin_labels.append(c["label"])
        _panel(c["label"], f"{n} item(s) accepted -- {'OK' if satisfied else 'THIN'}", "green" if satisfied else "red")

    subject, body = _compose_email(client, state, date_context, thin_labels)
    _panel("EMAIL", f"Subject: {subject}\n\n{body}", "cyan")

    composio.tools.execute(
        slug="GMAIL_SEND_EMAIL",
        user_id=COMPOSIO_USER_ID,
        arguments={"recipient_email": RECIPIENT, "subject": subject, "body": body},
    )
    console.rule(f"[bold green]Sent to {RECIPIENT}[/bold green]")


if __name__ == "__main__":
    run()
