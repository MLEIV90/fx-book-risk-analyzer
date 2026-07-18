"""
Tests for the market snapshot assembly.

The network parts (yfinance, FRED) are validated by running the app; here we
test the pure logic of the MarketSnapshot object (e.g. it prices a forward
correctly from its own fields).
"""
from fxrisk.market import MarketSnapshot
from fxrisk.forwards import forward_rate


def test_snapshot_prices_forward_consistently():
    snap = MarketSnapshot(
        pair="EUR/USD", base_ccy="EUR", quote_ccy="USD", tenor_years=0.25,
        spot=1.08, r_base=0.03, r_quote=0.045,
        vol_historical=0.08, vol_garch=0.09,
    )
    expected = forward_rate(1.08, 0.03, 0.045, 0.25)
    assert abs(snap.forward() - expected) < 1e-12
    assert round(snap.forward(), 5) == 1.08402


def test_snapshot_forward_above_spot_when_quote_yields_more():
    snap = MarketSnapshot(
        pair="EUR/USD", base_ccy="EUR", quote_ccy="USD", tenor_years=0.5,
        spot=1.10, r_base=0.02, r_quote=0.05,
        vol_historical=0.07, vol_garch=None,
    )
    assert snap.forward() > snap.spot


def test_snapshot_tau_quote_falls_back_to_tenor_years():
    """A snapshot built without tenor_years_quote (e.g. legacy/test code)
    must behave exactly as before H2: tau_quote falls back to tenor_years."""
    snap = MarketSnapshot(
        pair="EUR/USD", base_ccy="EUR", quote_ccy="USD", tenor_years=0.25,
        spot=1.08, r_base=0.03, r_quote=0.045,
        vol_historical=0.08, vol_garch=0.09,
    )
    assert snap.tau_quote == snap.tenor_years


def test_snapshot_gbp_quote_uses_its_own_tau_for_forward():
    """
    A GBP/USD snapshot with base and quote tau computed on their own
    day-count basis (H2) must price a DIFFERENT forward than one that
    (incorrectly) reused the base's tau for both legs.
    """
    from fxrisk.forwards import year_fraction
    days = 90
    tau_gbp = year_fraction(days, "GBP")
    tau_usd = year_fraction(days, "USD")
    assert tau_gbp != tau_usd

    snap_correct = MarketSnapshot(
        pair="GBP/USD", base_ccy="GBP", quote_ccy="USD", tenor_years=tau_gbp,
        spot=1.27, r_base=0.05, r_quote=0.045,
        vol_historical=0.08, vol_garch=None, tenor_years_quote=tau_usd,
    )
    snap_naive = MarketSnapshot(
        pair="GBP/USD", base_ccy="GBP", quote_ccy="USD", tenor_years=tau_gbp,
        spot=1.27, r_base=0.05, r_quote=0.045,
        vol_historical=0.08, vol_garch=None,               # no tenor_years_quote
    )
    assert snap_correct.tau_quote == tau_usd
    assert snap_naive.tau_quote == tau_gbp                  # old fallback behaviour
    assert snap_correct.forward() != snap_naive.forward()