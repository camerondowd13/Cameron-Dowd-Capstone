"""
Renders the structured trace events collected by account_finder.find_accounts()
(events=[...] param) into a standalone HTML report -- a visual, color-coded
step-by-step breakdown of one search run, grouped by pipeline stage.

Local-only diagnostic tool: generates a file, opens it in the default
browser. Not part of the production server/demo path.
"""
import html
import webbrowser
from datetime import datetime

STAGE_LABELS = {
    "discovery": "Discovery — Apollo Organization Search",
    "research": "Research — per-candidate ICP + trigger check",
    "contact": "Contact — reachability check",
    "result": "Result",
}

STAGE_DESCRIPTIONS = {
    "discovery": "Queries Apollo's own database of real companies directly, filtered by state, employee "
                 "count, and industry — instead of asking an AI to search the web and \"discover\" companies "
                 "one at a time, which is slow and can run out of search budget before finding enough.",
    "research": "Apollo can't tell us WHY to call a company today. For each real company Apollo returned, "
                "Claude reads actual news/web sources to confirm it truly fits the ICP and find a specific, "
                "current reason to reach out — a funding round, a new hire, an expansion. No real trigger "
                "and no ICP fit means the company gets dropped here, even though it's a real company.",
    "contact": "A company can be a perfect fit with a great trigger and still not be a usable lead if there's "
               "no way to actually reach anyone there. This step looks for a real named person with a "
               "confirmed email + phone, falling back to the company's general office line if no direct "
               "contact is found.",
    "result": "What actually came out of this run.",
}

STATUS_STYLE = {
    "success": ("#16a34a", "#f0fdf4", "✓"),
    "disqualified": ("#ca8a04", "#fefce8", "○"),
    "error": ("#dc2626", "#fef2f2", "⚠"),
    "info": ("#64748b", "#f8fafc", "•"),
}


def render_trace_html(events: list[dict], meta: dict, output_path: str) -> str:
    """meta: {"state", "industry", "limit", "min_size", "max_size"} -- shown
    in the report header. Returns the output_path for convenience."""
    stages = ["discovery", "research", "contact", "result"]
    sections = []
    for stage in stages:
        stage_events = [e for e in events if e["stage"] == stage]
        if not stage_events:
            continue
        rows = []
        for e in stage_events:
            color, bg, icon = STATUS_STYLE.get(e["status"], STATUS_STYLE["info"])
            company = f'<strong>{html.escape(e["company"])}</strong> — ' if e.get("company") else ""
            rows.append(f'''
                <div class="event" style="border-left-color:{color}; background:{bg};">
                    <span class="icon" style="color:{color};">{icon}</span>
                    <span class="text">{company}{html.escape(e["detail"])}</span>
                </div>
            ''')
        sections.append(f'''
            <section class="stage">
                <h2>{html.escape(STAGE_LABELS.get(stage, stage))}</h2>
                <p class="stage-desc">{html.escape(STAGE_DESCRIPTIONS.get(stage, ""))}</p>
                {"".join(rows)}
            </section>
        ''')

    result_events = [e for e in events if e["stage"] == "result"]
    result_summary = result_events[-1]["detail"] if result_events else "No result recorded"

    doc = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline trace — {html.escape(meta.get("state", ""))} / {html.escape(meta.get("industry") or "any industry")}</title>
<style>
  body {{
    font-family: -apple-system, "Inter", system-ui, sans-serif;
    background: #0f172a;
    color: #e2e8f0;
    margin: 0;
    padding: 40px 24px;
  }}
  .wrap {{ max-width: 760px; margin: 0 auto; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 4px; }}
  .meta {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 8px; }}
  .summary {{
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 20px 0 32px;
    font-weight: 600;
    font-size: 1.05rem;
  }}
  .stage {{ margin-bottom: 32px; }}
  .stage h2 {{
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #94a3b8;
    margin-bottom: 8px;
    padding-bottom: 0;
  }}
  .stage-desc {{
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.5;
    margin: 0 0 14px;
    padding-bottom: 12px;
    border-bottom: 1px solid #334155;
  }}
  .event {{
    display: flex;
    gap: 10px;
    align-items: baseline;
    border-left: 3px solid;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 6px;
    color: #1e293b;
    font-size: 0.92rem;
  }}
  .event .icon {{ font-weight: 700; flex-shrink: 0; }}
  .event .text strong {{ color: #0f172a; }}
  .footer {{ color: #64748b; font-size: 0.8rem; margin-top: 40px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Pipeline trace</h1>
  <div class="meta">
    {html.escape(meta.get("state", ""))} · {html.escape(meta.get("industry") or "any industry")} ·
    {meta.get("min_size", "")}-{meta.get("max_size", "")} employees · limit={meta.get("limit", "")}
  </div>
  <div class="summary">{html.escape(result_summary)}</div>
  {"".join(sections)}
  <div class="footer">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
</div>
</body>
</html>'''

    with open(output_path, "w") as f:
        f.write(doc)
    return output_path


def run_and_open(state: str, industry: str | None = None, limit: int = 3,
                  min_size: int | None = None, max_size: int | None = None,
                  output_path: str = "trace_report.html") -> None:
    """Convenience entry point: run a real find_accounts() call, capture its
    trace events, render the HTML report, and open it in the browser."""
    import account_finder
    from icp import MAX_SIZE, MIN_SIZE

    min_size = min_size if min_size is not None else MIN_SIZE
    max_size = max_size if max_size is not None else MAX_SIZE

    events = []
    account_finder.find_accounts(
        state=state, industry=industry, limit=limit,
        min_size=min_size, max_size=max_size,
        trace=True, events=events,
    )
    path = render_trace_html(
        events,
        {"state": state, "industry": industry, "limit": limit, "min_size": min_size, "max_size": max_size},
        output_path,
    )
    webbrowser.open(f"file://{__import__('os').path.abspath(path)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a traced find_accounts() search and open a visual HTML report.")
    parser.add_argument("--state", required=True)
    parser.add_argument("--industry", default=None)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    run_and_open(args.state, args.industry, args.limit)
