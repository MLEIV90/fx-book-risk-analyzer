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


def _snap(pair, spot):
    from fxrisk.market import MarketSnapshot
    b, q = pair.split("/")
    return MarketSnapshot(pair=pair, base_ccy=b, quote_ccy=q, tenor_years=0.25,
                          spot=spot, r_base=0.03, r_quote=0.045,
                          vol_historical=0.08, vol_garch=None)


def test_book_notional_usd_quote_usd_fast_path():
    """A plain quote-USD book sums notional*spot directly."""
    from fxrisk.book import Position, Book
    p1 = Position("EUR/USD", True, 1_000_000, 90, 1.08, id="a")
    p2 = Position("GBP/USD", False, 500_000, 90, 1.27, id="b")
    book = Book([p1, p2])
    snapshots = {"a": _snap("EUR/USD", 1.08), "b": _snap("GBP/USD", 1.27)}
    total = app_helpers.book_notional_usd(book, snapshots)
    assert total == pytest.approx(1_000_000 * 1.08 + 500_000 * 1.27)


def test_book_notional_usd_converts_non_usd_quote():
    """A EUR/GBP position's notional (in GBP) is converted to USD via the
    book's own GBP/USD spot, not summed as if it were already USD."""
    from fxrisk.book import Position, Book
    eurusd, gbpusd = 1.08, 1.27
    eurgbp = eurusd / gbpusd
    p1 = Position("GBP/USD", True, 500_000, 90, gbpusd, id="a")
    p2 = Position("EUR/GBP", True, 300_000, 90, eurgbp, id="b")
    book = Book([p1, p2])
    snapshots = {"a": _snap("GBP/USD", gbpusd), "b": _snap("EUR/GBP", eurgbp)}
    total = app_helpers.book_notional_usd(book, snapshots)
    expected = 500_000 * gbpusd + 300_000 * eurgbp * gbpusd
    assert total == pytest.approx(expected)


def test_book_notional_usd_raises_when_conversion_missing():
    """A EUR/GBP-only book with no GBP/USD spot anywhere must fail loud."""
    from fxrisk.book import Position, Book
    eurgbp = 0.85
    p1 = Position("EUR/GBP", True, 300_000, 90, eurgbp, id="a")
    book = Book([p1])
    snapshots = {"a": _snap("EUR/GBP", eurgbp)}
    with pytest.raises(ValueError, match="GBP/USD"):
        app_helpers.book_notional_usd(book, snapshots)
