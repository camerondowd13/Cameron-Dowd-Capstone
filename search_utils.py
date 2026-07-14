"""Shared Exa search + null-normalization helpers used by AccountFinder and
AccountResearcher."""
from exa_py import Exa

_NULLISH = {"null", "none", "n/a", "unknown", ""}


def clean_nullish(value):
    """Normalize model output so 'null'-as-string collapses to real None."""
    if isinstance(value, str) and value.strip().lower() in _NULLISH:
        return None
    return value


def strip_linkedin(url):
    """PRD hard constraint: no LinkedIn. Excluding it from Exa's search
    results isn't enough on its own -- the model can still pick up a
    linkedin.com URL embedded in another page's text (e.g. a bio page
    linking to someone's profile) and cite it as a source. Enforce the
    constraint in code on every URL that leaves this codebase, not just
    at the search-request level."""
    if url and "linkedin.com" in url.lower():
        return None
    return url


def strip_linkedin_from_list(urls):
    return [u for u in (urls or []) if strip_linkedin(u) is not None]


def run_exa_search(exa: Exa, query: str, include_domains=None, num_results: int = 5, seen_urls=None) -> str:
    """seen_urls: optional set that gets filled with every URL Exa actually
    returned, so a caller can later verify a model-cited source_url was a
    real search result and not just plausible-sounding text -- grounding
    the "never invent a URL" instruction in code instead of trusting it."""
    kwargs = dict(type="neural", num_results=num_results, contents={"text": {"maxCharacters": 800}})
    if include_domains:
        kwargs["include_domains"] = include_domains
    else:
        # PRD hard constraint: no scraping platforms whose ToS prohibits it (LinkedIn).
        kwargs["exclude_domains"] = ["linkedin.com"]
    response = exa.search(query, **kwargs)
    if not response.results:
        return "No results found."
    if seen_urls is not None:
        seen_urls.update(r.url for r in response.results)
    return "\n\n".join(
        f"- {r.title} ({r.url})\n  {(r.text or '').strip()}"
        for r in response.results
    )
