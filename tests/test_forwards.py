"""Tests for the forwards engine. They verify the pricing returns expected values."""
from fxrisk.forwards import (
    year_fraction, forward_rate, forward_points,
    client_rate_with_spread, provider_revenue, forward_mtm,
)

# Reference case: EUR/USD, spot 1.08, EUR 3%, USD 4.5%, 3 months.
SPOT, R_EUR, R_USD = 1.08, 0.030, 0.045
TAU = year_fraction(90)


def test_year_fraction():
    assert year_fraction(90) == 0.25            # 90 days = a quarter of a year, default ACT/360


def test_year_fraction_eur_usd_use_act360():
    """EUR and USD money markets use ACT/360 (H2)."""
    assert year_fraction(360, "EUR") == 1.0
    assert year_fraction(360, "USD") == 1.0
    assert year_fraction(90, "EUR") == 90 / 360


def test_year_fraction_gbp_uses_act365():
    """GBP money markets use ACT/365, not ACT/360 (H2)."""
    assert year_fraction(365, "GBP") == 1.0
    assert year_fraction(90, "GBP") == 90 / 365
    assert year_fraction(90, "GBP") != year_fraction(90, "EUR")


def test_forward_rate_tau_quote_defaults_to_tau():
    """When tau_quote is omitted, forward_rate uses tau for both legs
    (exact for pairs sharing a day-count basis, e.g. EUR/USD) -- unchanged
    from the pre-H2 single-tau behaviour."""
    f_single = forward_rate(SPOT, R_EUR, R_USD, TAU)
    f_explicit = forward_rate(SPOT, R_EUR, R_USD, TAU, tau_quote=TAU)
    assert f_single == f_explicit


def test_forward_rate_gbp_leg_uses_its_own_basis():
    """
    A GBP/USD forward must price with the GBP leg on ACT/365 and the USD leg
    on ACT/360 -- using ACT/360 for both (the pre-H2 bug) gives a different,
    wrong forward.
    """
    spot, r_gbp, r_usd, days = 1.27, 0.05, 0.045, 90
    tau_gbp = year_fraction(days, "GBP")     # base
    tau_usd = year_fraction(days, "USD")     # quote
    correct = forward_rate(spot, r_gbp, r_usd, tau_gbp, tau_usd)
    naive = forward_rate(spot, r_gbp, r_usd, year_fraction(days))  # old ACT/360-for-all bug
    assert correct != naive


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