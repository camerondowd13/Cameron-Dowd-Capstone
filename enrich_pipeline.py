import os
import re
import requests
from dotenv import load_dotenv
from exa_py import Exa
from anthropic import Anthropic
from composio import Composio

load_dotenv(".env.local")

COMPOSIO_USER_ID = "cameron_test_trimmed"
GMAIL_TOOLKIT_VERSION = "20260702_01"


def load_supabase_config():
    with open("config.js") as f:
        contents = f.read()
    url = re.search(r"SUPABASE_URL = '([^']+)'", contents).group(1)
    key = re.search(r"SUPABASE_ANON_KEY = '([^']+)'", contents).group(1)
    return url, key


def get_new_account(supabase_url, supabase_key):
    print("Step 1: Reading one 'new' account from Supabase...")
    resp = requests.get(
        f"{supabase_url}/rest/v1/accounts",
        params={"status": "eq.new", "limit": 1, "select": "*"},
        headers={"apikey": supabase_key, "Authorization": f"Bearer {supabase_key}"},
    )
    resp.raise_for_status()
    rows = resp.json()
    if not rows:
        raise SystemExit("No accounts with status='new' found in Supabase.")
    account = rows[0]
    print(f"  -> {account['name']} ({account['contact_email']})")
    return account


def mark_account_contacted(supabase_url, supabase_key, account_id):
    resp = requests.patch(
        f"{supabase_url}/rest/v1/accounts",
        params={"id": f"eq.{account_id}"},
        headers={
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
        },
        json={"status": "contacted"},
    )
    resp.raise_for_status()


def search_exa(account_name):
    print(f"Step 2: Exa search for '{account_name} recent funding, hiring, news'...")
    exa = Exa(os.getenv("EXA_API_KEY"))
    result = exa.search(
        f"{account_name} recent funding, hiring, news",
        type="neural",
        num_results=5,
    )
    print(f"  -> {len(result.results)} results")
    return result.results


def draft_email(account, search_results):
    print("Step 3: Drafting personalized email with Claude...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    search_summary = "\n\n".join(
        f"- {r.title} ({r.url})\n  {(r.text or '')[:500]}" for r in search_results
    )

    prompt = f"""You are a SaaS AE. Given this account: {account['name']} and this fresh context from Exa search:

{search_summary}

Draft a 4-sentence personalized outreach email. Reference something specific from the
search results — don't be generic.
Contact: {account['contact_name']}, {account['contact_title']}.
Respond with ONLY the email body text, no subject line, no preamble."""

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    email_body = next(b.text for b in response.content if b.type == "text")
    print(f"  -> drafted {len(email_body)} chars")
    return email_body


def save_gmail_draft(account, email_body):
    print("Step 4: Saving as Gmail draft via Composio...")
    composio = Composio(
        api_key=os.getenv("COMPOSIO_API_KEY"),
        toolkit_versions={"gmail": GMAIL_TOOLKIT_VERSION},
    )
    result = composio.tools.execute(
        slug="GMAIL_CREATE_EMAIL_DRAFT",
        user_id=COMPOSIO_USER_ID,
        arguments={
            "recipient_email": account["contact_email"],
            "subject": f"Quick note for {account['name']}",
            "body": email_body,
        },
    )
    print(f"  -> {result.get('data', {}).get('display_url', result)}")
    return result


def main():
    supabase_url, supabase_key = load_supabase_config()
    account = get_new_account(supabase_url, supabase_key)
    search_results = search_exa(account["name"])
    email_body = draft_email(account, search_results)
    save_gmail_draft(account, email_body)
    mark_account_contacted(supabase_url, supabase_key, account["id"])
    print("Done.")


if __name__ == "__main__":
    main()
