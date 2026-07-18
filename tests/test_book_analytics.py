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


def test_build_report_converts_non_usd_quote_mtm_to_usd():
    """
    A EUR/GBP position's MtM (in GBP) must be converted to USD via the book's
    own GBP/USD spot before summing -- not added directly as if it were USD
    (the pre-H2/Phase-3B-era bug this test guards against).
    """
    eurusd_snap = _snap("EUR/USD", 1.10, 0.03, 0.045)
    gbpusd_snap = _snap("GBP/USD", 1.30, 0.045, 0.045)
    eurgbp_snap = _snap("EUR/GBP", 1.10 / 1.30, 0.03, 0.045)

    p_eurusd = Position("EUR/USD", True, 1_000_000, 90, eurusd_snap.forward() - 0.01, id="a")
    p_eurgbp = Position("EUR/GBP", True, 1_000_000, 90, eurgbp_snap.forward() - 0.01, id="b")
    p_gbpusd = Position("GBP/USD", True, 1_000_000, 90, gbpusd_snap.forward(), id="c")
    book = Book([p_eurusd, p_eurgbp, p_gbpusd])

    val_eurusd = value_position_from_snapshot(p_eurusd, eurusd_snap)
    val_eurgbp = value_position_from_snapshot(p_eurgbp, eurgbp_snap)
    val_gbpusd = value_position_from_snapshot(p_gbpusd, gbpusd_snap)
    report = build_report([val_eurusd, val_eurgbp, val_gbpusd], book)

    naive_total = val_eurusd.mtm_quote + val_eurgbp.mtm_quote + val_gbpusd.mtm_quote
    expected_total = (val_eurusd.mtm_quote + val_eurgbp.mtm_quote * gbpusd_snap.spot
                      + val_gbpusd.mtm_quote)
    assert abs(report.total_mtm_usd - expected_total) < 1e-6
    assert abs(report.total_mtm_usd - naive_total) > 1e-6   # must differ from the bug


def test_build_report_raises_when_conversion_spot_missing():
    """A EUR/GBP-only book, with no GBP/USD spot anywhere, cannot be priced in
    USD -- must fail loud rather than silently treat GBP as USD."""
    import pytest
    eurgbp_snap = _snap("EUR/GBP", 0.85, 0.03, 0.045)
    pos = Position("EUR/GBP", True, 1_000_000, 90, eurgbp_snap.forward() - 0.01)
    with pytest.raises(ValueError, match="GBP/USD"):
        build_report([value_position_from_snapshot(pos, eurgbp_snap)], Book([pos]))


def test_shift_snapshot_scales_spot_only():
    """_shift_snapshot must scale spot by the shock and leave rates/vols intact."""
    from fxrisk.book_analytics import _shift_snapshot
    from fxrisk.market import MarketSnapshot
    snap = MarketSnapshot(
        pair="EUR/USD", base_ccy="EUR", quote_ccy="USD", tenor_years=0.25,
        spot=1.1000, r_base=0.021, r_quote=0.039,
        vol_historical=0.08, vol_garch=0.09)
    down = _shift_snapshot(snap, -5)
    up = _shift_snapshot(snap, +10)
    assert abs(down.spot - 1.1000 * 0.95) < 1e-12
    assert abs(up.spot - 1.1000 * 1.10) < 1e-12
    # Everything else unchanged.
    assert down.r_base == snap.r_base and down.r_quote == snap.r_quote
    assert down.vol_garch == snap.vol_garch
    assert down.tenor_years == snap.tenor_years


def test_shift_snapshot_zero_shock_is_identity():
    """A 0% shock returns the same spot."""
    from fxrisk.book_analytics import _shift_snapshot
    from fxrisk.market import MarketSnapshot
    snap = MarketSnapshot(
        pair="GBP/USD", base_ccy="GBP", quote_ccy="USD", tenor_years=0.5,
        spot=1.2700, r_base=0.045, r_quote=0.039,
        vol_historical=0.10, vol_garch=None)
    assert abs(_shift_snapshot(snap, 0).spot - 1.2700) < 1e-12
