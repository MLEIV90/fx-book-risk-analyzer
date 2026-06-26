"""
Tests for the valuation/report layer.

Pure logic only: we inject MarketSnapshots, so no network is needed. The live
fetch (value_book / book_sensitivity) is validated by running the app.
"""
from fxrisk.book import Position, Book
from fxrisk.market import MarketSnapshot
from fxrisk.book_analytics import (
    value_position_from_snapshot, build_report,
)


def _snap(pair, spot, r_base, r_quote, tenor=0.25):
    b, q = pair.split("/")
    return MarketSnapshot(pair=pair, base_ccy=b, quote_ccy=q, tenor_years=tenor,
                          spot=spot, r_base=r_base, r_quote=r_quote,
                          vol_historical=0.08, vol_garch=0.09,
                          notes=["EUR: Flat curve example note"])


def test_position_mtm_zero_when_struck_at_market():
    # If the strike equals the current market forward, MtM ~ 0.
    snap = _snap("EUR/USD", 1.08, 0.03, 0.045)
    strike = snap.forward()
    pos = Position("EUR/USD", True, 1_000_000, 90, strike)
    val = value_position_from_snapshot(pos, snap)
    assert abs(val.mtm_quote) < 1e-6


def test_position_mtm_positive_when_market_moves_in_favour():
    snap = _snap("EUR/USD", 1.08, 0.03, 0.045)
    # Long base struck BELOW the current forward -> in the money.
    pos = Position("EUR/USD", True, 1_000_000, 90, snap.forward() - 0.01)
    val = value_position_from_snapshot(pos, snap)
    assert val.mtm_quote > 0


def test_report_totals_and_composition():
    snap = _snap("EUR/USD", 1.08, 0.03, 0.045)
    fwd = snap.forward()
    book = Book()
    p_win = Position("EUR/USD", True, 1_000_000, 90, fwd - 0.01, id="win")
    p_lose = Position("EUR/USD", True, 1_000_000, 90, fwd + 0.01, id="lose")
    book.add(p_win)
    book.add(p_lose)
    vals = [value_position_from_snapshot(p, snap) for p in book]
    report = build_report(vals, book)

    assert report.gains_usd > 0
    assert report.losses_usd < 0
    # total = gains + losses
    assert abs(report.total_mtm_usd - (report.gains_usd + report.losses_usd)) < 1e-6
    # concentration shares sum to 100%
    assert abs(sum(s for _, _, s in report.concentration) - 100.0) < 1e-6


def test_report_carries_data_flags():
    snap = _snap("EUR/USD", 1.08, 0.03, 0.045)
    pos = Position("EUR/USD", True, 1_000_000, 90, snap.forward())
    report = build_report([value_position_from_snapshot(pos, snap)], Book([pos]))
    assert any("Flat curve" in f for f in report.data_flags)
