"""
Unit tests for account_finder.py's deterministic logic, plus mocked tests
for _discover_via_apollo's branching (SIC filter, dedup, error-vs-
disqualification handling) -- no real API calls, so these run free and in
milliseconds. See test_account_finder.py::test_matches_industry_sic_case_insensitive
for a regression test on the exact bug that silently zeroed out every
candidate in production on 2026-07-16 (industry passed capitalized,
INDUSTRY_SIC_PREFIXES keyed lowercase, .get() without lowercasing returned
() every time).
"""
import account_finder


# ---- _matches_industry_sic ----

def test_matches_industry_sic_real_construction_company():
    org = {"sic_codes": ["1542"]}  # General Building Contractors
    assert account_finder._matches_industry_sic(org, "construction") is True


def test_matches_industry_sic_case_insensitive():
    # Regression test: the site's dropdown and find_accounts() both pass
    # industry capitalized ("Construction"), but INDUSTRY_SIC_PREFIXES keys
    # are lowercase. A bare dict .get() without lowercasing silently
    # returned () here, making every single candidate fail this check
    # regardless of its real SIC codes -- confirmed in production testing.
    org = {"sic_codes": ["1542"]}
    assert account_finder._matches_industry_sic(org, "Construction") is True
    assert account_finder._matches_industry_sic(org, "CONSTRUCTION") is True


def test_matches_industry_sic_rejects_wrong_industry():
    # Real example from testing: a staffing agency that surfaced via
    # Apollo's loose keyword-tag match on "construction" despite not
    # actually being a construction company.
    org = {"sic_codes": ["7361"]}  # Employment agencies
    assert account_finder._matches_industry_sic(org, "construction") is False


def test_matches_industry_sic_handles_missing_or_null_codes():
    assert account_finder._matches_industry_sic({}, "construction") is False
    assert account_finder._matches_industry_sic({"sic_codes": None}, "construction") is False
    assert account_finder._matches_industry_sic({"sic_codes": [None, "1542"]}, "construction") is True


def test_matches_industry_sic_unknown_industry_never_matches():
    org = {"sic_codes": ["1542"]}
    assert account_finder._matches_industry_sic(org, "not-a-real-industry") is False


# ---- _domain ----

def test_domain_strips_scheme_and_www():
    assert account_finder._domain("https://www.example.com/contact") == "example.com"


def test_domain_handles_bare_domain_no_scheme():
    # Supabase stores websites without a scheme (e.g. "company.com"), unlike
    # Exa's full-URL format -- urlparse treats a schemeless string as a path,
    # not a netloc, without the "//" prefix this function adds.
    assert account_finder._domain("example.com") == "example.com"


def test_domain_empty_or_none_returns_empty_string():
    assert account_finder._domain("") == ""
    assert account_finder._domain(None) == ""


# ---- _within_size_range ----

def test_within_size_range_inside_bounds():
    assert account_finder._within_size_range({"employee_count": 500}, 50, 1000) is True


def test_within_size_range_above_max_fails_regardless_of_trigger():
    assert account_finder._within_size_range(
        {"employee_count": 1001, "buying_trigger": "real trigger"}, 50, 1000
    ) is False


def test_within_size_range_below_min_requires_trigger():
    assert account_finder._within_size_range({"employee_count": 10}, 50, 1000) is False
    assert account_finder._within_size_range(
        {"employee_count": 10, "buying_trigger": "just raised funding"}, 50, 1000
    ) is True


def test_within_size_range_unknown_headcount_kept():
    # Discovery-stage leniency (different from icp.meets_size's stricter
    # confirmation-stage gate) -- Exa often can't find exact headcount for
    # smaller/private companies.
    assert account_finder._within_size_range({"employee_count": None}, 50, 1000) is True


# ---- _discover_via_apollo, mocked (no real API calls) ----

def _mock_org(name, sic_codes, website=None):
    return {"name": name, "sic_codes": sic_codes, "website_url": website or f"https://{name.lower().replace(' ', '')}.com"}


def test_discover_via_apollo_passes_target_titles_to_contact_finder(monkeypatch):
    # Regression test: _discover_via_apollo originally had no target_titles
    # parameter at all -- a user picking "CTO" on the site had it silently
    # dropped, since this (now-primary) discovery path never passed it to
    # contact_finder.find_contacts(), which then searched all 5 default
    # titles and surfaced whichever ranked highest (usually Founder/CEO).
    orgs = [_mock_org("Real Construction Co", ["1542"])]
    monkeypatch.setattr(account_finder.apollo_client, "search_organizations", lambda *a, **k: orgs)
    monkeypatch.setattr(
        account_finder.account_researcher, "research_account",
        lambda name, domain=None: {"meets_icp": True, "employee_count": 100, "buying_triggers": "trigger", "sources": []},
    )

    received_titles = []

    def fake_find_contacts(name, domain=None, target_titles=None, **kwargs):
        received_titles.append(target_titles)
        return {"contacts": [], "general_office": {"phone": "555-0000", "email": "info@test.com"}}

    monkeypatch.setattr(account_finder.contact_finder, "find_contacts", fake_find_contacts)

    account_finder._discover_via_apollo(
        "Texas", "construction", 50, 1000, limit=1,
        known_names=set(), known_domains=set(),
        target_titles=["CTO"],
    )

    assert received_titles == [["CTO"]]


def test_discover_via_apollo_filters_by_sic_and_dedup(monkeypatch):
    orgs = [
        _mock_org("Real Construction Co", ["1542"]),
        _mock_org("Staffing Agency Inc", ["7361"]),  # wrong industry, should be dropped
        _mock_org("Already Known Co", ["1542"], website="https://known.com"),
    ]
    monkeypatch.setattr(account_finder.apollo_client, "search_organizations", lambda *a, **k: orgs)
    monkeypatch.setattr(
        account_finder.account_researcher, "research_account",
        lambda name, domain=None: {"meets_icp": True, "employee_count": 100, "buying_triggers": "test trigger", "sources": ["https://source.com"]},
    )
    monkeypatch.setattr(
        account_finder.contact_finder, "find_contacts",
        lambda name, domain=None, **k: {"contacts": [{"name": "Jane Doe", "title": "CFO", "email": "jane@test.com", "phone": "555-1234"}], "general_office": None},
    )

    events = []
    result = account_finder._discover_via_apollo(
        "Texas", "construction", 50, 1000, limit=3,
        known_names={"already known co"}, known_domains=set(),
        events=events,
    )

    names = [c["name"] for c in result]
    assert "Real Construction Co" in names
    assert "Staffing Agency Inc" not in names  # dropped by SIC filter
    assert "Already Known Co" not in names  # dropped by dedup


def test_discover_via_apollo_distinguishes_error_from_disqualification(monkeypatch):
    # This is the exact distinction that mattered in production: an API
    # failure (infrastructure problem) must never look identical to a
    # genuine "doesn't meet ICP" disqualification in the emitted trace.
    orgs = [
        _mock_org("Errors Out Co", ["1542"]),
        _mock_org("Genuinely Disqualified Co", ["1542"]),
    ]
    monkeypatch.setattr(account_finder.apollo_client, "search_organizations", lambda *a, **k: orgs)

    def fake_research(name, domain=None):
        if name == "Errors Out Co":
            raise RuntimeError("API usage limit reached")
        return {"meets_icp": False, "employee_count": 5, "industry": "not construction", "buying_triggers": None}

    monkeypatch.setattr(account_finder.account_researcher, "research_account", fake_research)

    events = []
    result = account_finder._discover_via_apollo(
        "Texas", "construction", 50, 1000, limit=3,
        known_names=set(), known_domains=set(),
        events=events,
    )

    assert result == []
    research_events = {e["company"]: e["status"] for e in events if e["stage"] == "research"}
    assert research_events["Errors Out Co"] == "error"
    assert research_events["Genuinely Disqualified Co"] == "disqualified"


def test_discover_via_apollo_stops_once_limit_reached(monkeypatch):
    orgs = [_mock_org(f"Company {i}", ["1542"]) for i in range(10)]
    monkeypatch.setattr(account_finder.apollo_client, "search_organizations", lambda *a, **k: orgs)
    monkeypatch.setattr(
        account_finder.account_researcher, "research_account",
        lambda name, domain=None: {"meets_icp": True, "employee_count": 100, "buying_triggers": "trigger", "sources": []},
    )
    monkeypatch.setattr(
        account_finder.contact_finder, "find_contacts",
        lambda name, domain=None, **k: {"contacts": [], "general_office": {"phone": "555-0000", "email": "info@test.com"}},
    )

    result = account_finder._discover_via_apollo(
        "Texas", "construction", 50, 1000, limit=2,
        known_names=set(), known_domains=set(),
    )

    assert len(result) == 2  # never pads beyond limit, never processes all 10


def test_discover_via_apollo_returns_empty_without_industry():
    # Apollo's org search needs a keyword to search for -- can't discover
    # via this path with industry=None.
    result = account_finder._discover_via_apollo(
        "Texas", None, 50, 1000, limit=3, known_names=set(), known_domains=set(),
    )
    assert result == []
