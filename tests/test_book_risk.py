"""Tests for the book-level rate / liquidity / stress layer. Pure logic."""
import numpy as np

from fxrisk.book import Position, Book
from fxrisk.market import MarketSnapshot
from fxrisk.book_risk import dv01_book, liquidity_book, stress_book


def _snap(pair, pid, spot=1.08, r_base=0.03, r_quote=0.045, tenor=0.25):
    b, q = pair.split("/")
    return MarketSnapshot(pair=pair, base_ccy=b, quote_ccy=q, tenor_years=tenor,
                          spot=spot, r_base=r_base, r_quote=r_quote,
                          vol_historical=0.08, vol_garch=0.09)


def _book_and_snaps():
    book = Book()
    p1 = Position("EUR/USD", True, 1_000_000, 90, 1.10, id="p1")
    p2 = Position("GBP/USD", False, 500_000, 90, 1.27, id="p2")
    book.add(p1)
    book.add(p2)
    snaps = {"p1": _snap("EUR/USD", "p1"),
             "p2": _snap("GBP/USD", "p2", spot=1.27)}
    return book, snaps


def test_dv01_aggregates_by_currency():
    book, snaps = _book_and_snaps()
    by_ccy, total = dv01_book(book, snaps)
    assert "EUR" in by_ccy and "USD" in by_ccy and "GBP" in by_ccy
    assert abs(total - sum(by_ccy.values())) < 1e-9


def test_liquidity_positive_for_nonempty_book():
    book, snaps = _book_and_snaps()
    buf = liquidity_book(book, snaps, confidence=0.99)
    assert buf > 0


def test_liquidity_zero_for_empty_book():
    assert liquidity_book(Book(), {}, 0.99) == 0.0


def test_stress_applies_scenarios_and_ratios():
    book, snaps = _book_and_snaps()
    results = stress_book(book, snaps, var_reference=100_000.0)
    # All standard scenarios present.
    assert len(results) == len(__import__("fxrisk.risk", fromlist=["STRESS_SCENARIOS"]).STRESS_SCENARIOS)
    for name, r in results.items():
        assert "pnl" in r and "x_var" in r


def test_stress_zero_move_for_uncovered_pair():
    # A pair not present in any scenario contributes zero stress P&L.
    book = Book([Position("CAD/USD", True, 1_000_000, 90, 0.73, id="c1")])
    snaps = {"c1": _snap("CAD/USD", "c1", spot=0.73)}
    results = stress_book(book, snaps, var_reference=50_000.0)
    for r in results.values():
        assert r["pnl"] == 0.0


def test_dv01_by_tenor_buckets():
    """Key-rate DV01 should bucket positions by tenor and sum to the net total."""
    from fxrisk.book_risk import dv01_book_by_tenor, dv01_book
    book, snaps = _book_and_snaps()
    buckets = dv01_book_by_tenor(book, snaps)
    assert set(buckets) == {"0-3m", "3-6m", "6-12m", "12m+"}
    # Sum of buckets should equal the net total DV01 across the book.
    _, total = dv01_book(book, snaps)
    assert abs(sum(buckets.values()) - total) < 1e-6
