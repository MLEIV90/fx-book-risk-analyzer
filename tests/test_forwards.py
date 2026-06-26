"""Tests for the forwards engine. They verify the pricing returns expected values."""
from fxrisk.forwards import (
    year_fraction, forward_rate, forward_points,
    client_rate_with_spread, provider_revenue, forward_mtm,
)

# Reference case: EUR/USD, spot 1.08, EUR 3%, USD 4.5%, 3 months.
SPOT, R_EUR, R_USD = 1.08, 0.030, 0.045
TAU = year_fraction(90)


def test_year_fraction():
    assert year_fraction(90) == 0.25            # 90 days = a quarter of a year


def test_forward_above_spot():
    # USD yields more than EUR, so the forward sits above spot.
    f = forward_rate(SPOT, R_EUR, R_USD, TAU)
    assert round(f, 5) == 1.08402


def test_forward_points_positive():
    f = forward_rate(SPOT, R_EUR, R_USD, TAU)
    assert forward_points(SPOT, f) > 0          # forward above spot -> positive pips


def test_client_pays_more_when_buying():
    # If the client buys the base, they pay the spread above the theoretical rate.
    fair = forward_rate(SPOT, R_EUR, R_USD, TAU)
    rate = client_rate_with_spread(fair, 20, client_buys_base=True)
    assert rate > fair


def test_provider_revenue():
    # 20 pips on 1,000,000 = 2,000 units of quote currency.
    assert provider_revenue(1_000_000, 20) == 2000.0


def test_mtm_zero_at_inception():
    # If the strike equals the fair forward, the initial MtM is ~0.
    fair = forward_rate(SPOT, R_EUR, R_USD, TAU)
    mtm = forward_mtm(1_000_000, fair, fair, R_USD, TAU)
    assert abs(mtm) < 1e-6