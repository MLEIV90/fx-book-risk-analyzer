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