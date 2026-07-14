"""
Thin HTTP wrapper around enrich_pipeline.py so it can run as a Render Web
Service instead of a Cron Job. The pipeline itself is unchanged and still
runs once per call, exits, done -- this server just stays alive between
calls (which is what a Web Service is required to do) and only triggers
a pipeline run when something explicitly requests it via POST /run.

X-API-Key auth on /run is required once SERVICE_API_KEY is set, since this
endpoint costs real money per call (Anthropic + Exa) and creates a real
Gmail draft -- leaving it open would let anyone who finds the URL trigger
that repeatedly.
"""
import os

from flask import Flask, jsonify, request

import enrich_pipeline

app = Flask(__name__)

API_KEY = os.getenv("SERVICE_API_KEY")


@app.get("/")
def health():
    return jsonify(status="ok")


@app.post("/run")
def run_pipeline():
    if API_KEY and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="unauthorized"), 401

    try:
        enrich_pipeline.main()
        return jsonify(status="done")
    except SystemExit as e:
        # enrich_pipeline.get_new_account() raises SystemExit when there's
        # no 'new' account to process -- that's a normal outcome, not a crash.
        return jsonify(status="no accounts to process", detail=str(e))
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
