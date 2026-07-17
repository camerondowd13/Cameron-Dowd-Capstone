"""Unit tests for icp.py's pure ICP-qualification logic."""
from icp import MAX_SIZE, MIN_SIZE, meets_size


def test_meets_size_within_range():
    assert meets_size(MIN_SIZE, has_trigger=False) is True
    assert meets_size(MAX_SIZE, has_trigger=False) is True
    assert meets_size((MIN_SIZE + MAX_SIZE) // 2, has_trigger=False) is True


def test_meets_size_above_max_always_fails():
    assert meets_size(MAX_SIZE + 1, has_trigger=False) is False
    assert meets_size(MAX_SIZE + 1, has_trigger=True) is False


def test_meets_size_below_min_requires_trigger():
    assert meets_size(MIN_SIZE - 1, has_trigger=False) is False
    assert meets_size(MIN_SIZE - 1, has_trigger=True) is True


def test_meets_size_unknown_headcount_always_fails():
    # No partial credit -- this is the confirmation gate, not discovery.
    assert meets_size(None, has_trigger=True) is False
    assert meets_size(None, has_trigger=False) is False
