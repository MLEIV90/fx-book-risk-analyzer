"""Tests for the app-layer data helpers (retry logic)."""
import pytest
from fxrisk.data import MarketDataError
import app_helpers


def test_retries_recover_from_transient_error():
    """A transient (non-MarketDataError) failure should be retried, then succeed."""
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient blip")
        return "ok"
    result = app_helpers._with_retries(flaky, attempts=3, base_delay=0.0)
    assert result == "ok"
    assert calls["n"] == 3


def test_market_data_error_not_retried():
    """A genuine MarketDataError means data is absent -> raise immediately."""
    calls = {"n": 0}
    def absent():
        calls["n"] += 1
        raise MarketDataError("no data")
    with pytest.raises(MarketDataError):
        app_helpers._with_retries(absent, attempts=3, base_delay=0.0)
    assert calls["n"] == 1          # not retried


def test_retries_exhausted_raises_market_data_error():
    """After all attempts fail, a MarketDataError is raised."""
    def always_fail():
        raise TimeoutError("down")
    with pytest.raises(MarketDataError):
        app_helpers._with_retries(always_fail, attempts=2, base_delay=0.0)
