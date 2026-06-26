"""Tests for the options engine (Garman-Kohlhagen)."""
import math

from fxrisk.forwards import year_fraction, forward_rate
from fxrisk.options import (
    norm_cdf, garman_kohlhagen, option_delta, option_vega,
)

# Reference case: EUR/USD, spot 1.08, EUR 3%, USD 4.5%, 1 month, vol 8%.
SPOT, R_EUR, R_USD, VOL = 1.08, 0.030, 0.045, 0.08
TAU = year_fraction(30)
NOTIONAL = 1_000_000

# At-the-money-forward strike (standard reference strike).
K_ATM = forward_rate(SPOT, R_EUR, R_USD, TAU)


def test_norm_cdf_basics():
    assert round(norm_cdf(0.0), 4) == 0.5       # N(0) = 0.5
    assert norm_cdf(5.0) > 0.999                 # far right tail ~ 1


def test_call_premium_positive():
    prem = garman_kohlhagen(SPOT, K_ATM, R_EUR, R_USD, VOL, TAU, is_call=True)
    assert prem > 0


def test_put_call_parity_atmf():
    # At the at-the-money-forward strike, call and put premiums are ~equal.
    call = garman_kohlhagen(SPOT, K_ATM, R_EUR, R_USD, VOL, TAU, is_call=True)
    put = garman_kohlhagen(SPOT, K_ATM, R_EUR, R_USD, VOL, TAU, is_call=False)
    assert abs(call - put) < 1e-4


def test_atmf_delta_near_half():
    # An at-the-money-forward call has delta close to 0.5.
    delta = option_delta(SPOT, K_ATM, R_EUR, R_USD, VOL, TAU, is_call=True)
    assert 0.45 < delta < 0.55


def test_vega_positive():
    assert option_vega(SPOT, K_ATM, R_EUR, R_USD, VOL, TAU) > 0


def test_higher_vol_higher_premium():
    # More volatility -> more expensive option.
    low = garman_kohlhagen(SPOT, K_ATM, R_EUR, R_USD, 0.05, TAU, is_call=True)
    high = garman_kohlhagen(SPOT, K_ATM, R_EUR, R_USD, 0.15, TAU, is_call=True)
    assert high > low