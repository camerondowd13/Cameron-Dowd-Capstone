"""Shared Exa search + null-normalization helpers used by AccountFinder and
AccountResearcher."""
from exa_py import Exa

_NULLISH = {"null", "none", "n/a", "unknown", ""}


def clean_nullish(value):
    """Normalize model output so 'null'-as-string collapses to real None."""
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def run_exa_search(exa: Exa, query: str, include_domains=None) -> str:
    kwargs = dict(type="neural", num_results=5, contents={"text": {"maxCharacters": 800}})
    if include_domains:
        kwargs["include_domains"] = include_domains
    else:
        # PRD hard constraint: no scraping platforms whose ToS prohibits it (LinkedIn).
        kwargs["exclude_domains"] = ["linkedin.com"]
    response = exa.search(query, **kwargs)
    if not response.results:
        return "No results found."
    return "\n\n".join(
        f"- {r.title} ({r.url})\n  {(r.text or '').strip()}"
        for r in response.results
    )
