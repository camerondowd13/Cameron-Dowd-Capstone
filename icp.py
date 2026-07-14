"""Cameron's ICP definition (PRD Appendix A), shared by AccountFinder and
AccountResearcher so both tools stay in sync on what actually qualifies."""

MIN_SIZE = 20
MAX_SIZE = 399

# Invoice-volume proxy: these verticals typically run 200+ invoices/month,
# the actual (not directly queryable) target.
VALID_INDUSTRIES = ["construction", "manufacturing", "healthcare"]

# PRD Appendix A, open item: "the trailing clause is open-ended" (title list
# isn't closed — "or others involved in purchasing AP automation software").
# Treat as guidance for search focus, not a hard filter.
TARGET_TITLES = ["CEO", "CFO", "CTO", "AP Manager", "Director of Finance"]


def meets_size(employee_count: int | None, has_trigger: bool) -> bool:
    """PRD size rule: 20-399 employees, or under 20 if a real buying trigger exists.
    Unknown headcount always fails here — this is the confirmation gate (PRD's
    'no partial credit' list-inclusion rule), not the lenient discovery stage."""
    if employee_count is None:
        return False
    if MIN_SIZE <= employee_count <= MAX_SIZE:
        return True
    return employee_count < MIN_SIZE and has_trigger
