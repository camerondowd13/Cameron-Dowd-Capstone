"""
HTTP wrapper exposing the prospecting skills (AccountFinder, AccountResearcher,
ContactFinder) plus the enrich_pipeline cron script as endpoints on Render,
so tomorrow's agent -- and a local demo page -- can call them live instead of
importing the modules directly.

Every route requires X-API-Key (checked against SERVICE_API_KEY) since each
one costs real Anthropic/Exa API usage, and /run additionally creates a real
Gmail draft -- this is not meant to be a public-facing endpoint.

CORS is wide open (Access-Control-Allow-Origin: *) so a local demo.html file
(opened via file://, not hosted anywhere) can call this cross-origin. That's
safe here specifically because the real gate is the API key, not the origin.
"""
import os
from functools import wraps

from flask import Flask, jsonify, request
from flask_cors import CORS

import account_finder
import account_researcher
import contact_finder
import enrich_pipeline

app = Flask(__name__)
CORS(app)

API_KEY = os.getenv("SERVICE_API_KEY")


def require_api_key(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if API_KEY and request.headers.get("X-API-Key") != API_KEY:
            return jsonify(error="unauthorized"), 401
        return fn(*args, **kwargs)
    return wrapper


@app.get("/")
def health():
    return jsonify(status="ok")


@app.get("/debug-env")
def debug_env():
    # TEMPORARY diagnostic -- reports only length + a one-way SHA256 hash of
    # each value, never the value itself. A hash can't be reversed back into
    # the real key, so this is safe to leave reachable while debugging.
    # Remove once keys are confirmed correct.
    import hashlib
    keys = ["ANTHROPIC_API_KEY", "EXA_API_KEY", "COMPOSIO_API_KEY", "SERVICE_API_KEY"]
    result = {}
    for k in keys:
        v = os.getenv(k) or ""
        result[k] = {"length": len(v), "sha256": hashlib.sha256(v.encode()).hexdigest()}
    return jsonify(result)


@app.post("/run")
@require_api_key
def run_pipeline():
    try:
        enrich_pipeline.main()
        return jsonify(status="done")
    except SystemExit as e:
        # enrich_pipeline.get_new_account() raises SystemExit when there's
        # no 'new' account to process -- that's a normal outcome, not a crash.
        return jsonify(status="no accounts to process", detail=str(e))
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/find-accounts")
@require_api_key
def find_accounts_route():
    body = request.get_json(force=True, silent=True) or {}
    state = body.get("state")
    if not state:
        return jsonify(error="'state' is required"), 400

    try:
        candidates = account_finder.find_accounts(
            state=state,
            min_size=body.get("min_size", account_finder.MIN_SIZE),
            max_size=body.get("max_size", account_finder.MAX_SIZE),
            city=body.get("city"),
            industry=body.get("industry"),
            limit=body.get("limit", account_finder.DEFAULT_LIMIT),
            target_titles=body.get("target_titles"),
        )
        return jsonify(candidates=candidates)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/research-account")
@require_api_key
def research_account_route():
    body = request.get_json(force=True, silent=True) or {}
    account_name = body.get("account_name")
    if not account_name:
        return jsonify(error="'account_name' is required"), 400

    try:
        result = account_researcher.research_account(account_name, domain=body.get("domain"))
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


@app.post("/find-contacts")
@require_api_key
def find_contacts_route():
    body = request.get_json(force=True, silent=True) or {}
    account_name = body.get("account_name")
    if not account_name:
        return jsonify(error="'account_name' is required"), 400

    try:
        result = contact_finder.find_contacts(
            account_name,
            domain=body.get("domain"),
            target_titles=body.get("target_titles"),
            limit=body.get("limit", 3),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
