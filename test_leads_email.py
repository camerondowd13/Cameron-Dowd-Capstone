"""
Manual test harness for the Daily Prospecting Report email (the "new leads"
summary). Builds the report from fixed sample lead data and sends it to
DAILY_RECIPIENT so you can see the real rendered result in your inbox,
WITHOUT running the full daily pipeline (no searches, no real API cost
beyond the one Gmail send).

Run: .venv/bin/python3 test_leads_email.py
Sends one real email (subject prefixed with [TEST]) to your own inbox.
Not part of the scheduled cron -- purely a local format check.
"""
import os

from composio import Composio

import prospecting_agent

# Representative sample results -- structure matches what _run_state builds.
SAMPLE_RESULTS = [
    {
        "status": "new", "name": "Vaughn Construction", "territory": "Texas",
        "contact_tier": "verified direct contact", "gmail_draft": True,
        "contact_name": "Jane Okafor", "contact_title": "CFO",
        "contact_email": "jokafor@vaughnconstruction.com", "contact_phone": "(214) 382-3700",
        "buying_trigger": "Opened a new Dallas regional office in Q1, expanding finance headcount.",
        "error": None,
    },
    {
        "status": "new", "name": "Ohio Gratings, Inc.", "territory": "Ohio",
        "contact_tier": "verified direct contact", "gmail_draft": True,
        "contact_name": "Mark Reilly", "contact_title": "Director of Finance",
        "contact_email": "mreilly@ohiogratings.com", "contact_phone": "(330) 477-6707",
        "buying_trigger": "Announced an ERP transformation project spanning multiple entities.",
        "error": None,
    },
    {
        "status": "partial", "name": "Kinley Construction", "territory": "Texas",
        "contact_tier": "general office", "gmail_draft": False,
        "contact_name": None, "contact_title": None,
        "contact_email": None, "contact_phone": "(817) 416-2100",
        "buying_trigger": "Multi-market expansion into four new cities.",
        "error": None,
    },
    {
        "status": "lost", "name": "Some Failed Co", "territory": "Florida",
        "contact_tier": None, "gmail_draft": False,
        "error": "Supabase insert failed: duplicate key",
    },
]

SAMPLE_FAILED_STATES = [("Nevada", "AccountFinder timed out after 300s")]


def main() -> None:
    composio = Composio(
        api_key=os.getenv("COMPOSIO_API_KEY"),
        toolkit_versions={"gmail": prospecting_agent.GMAIL_TOOLKIT_VERSION},
    )

    # Wrap the send so the test email is clearly marked [TEST] in the inbox,
    # while the body/subject still come from the real _send_summary_email
    # (no duplicated formatting logic that could drift from production).
    orig_execute = composio.tools.execute

    def patched(*args, **kwargs):
        args_dict = kwargs.get("arguments")
        if args_dict and args_dict.get("subject"):
            print("SUBJECT:", args_dict["subject"])
            print("=" * 60)
            print(args_dict["body"])
            print("=" * 60)
            args_dict["subject"] = f"[TEST] {args_dict['subject']}"
        return orig_execute(*args, **kwargs)

    composio.tools.execute = patched

    prospecting_agent._send_summary_email(composio, SAMPLE_RESULTS, SAMPLE_FAILED_STATES)
    print(f"\nSent test email to {prospecting_agent.DAILY_RECIPIENT}")


if __name__ == "__main__":
    main()
