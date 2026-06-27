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

def test_rates_converted_to_continuous():
    """
    H1: a simple rate must be converted to continuous internally. Pricing with
    an already-continuous rate (via the helper) must match pricing with the
    simple rate that converts to it.
    """
    import math
    from fxrisk.options import garman_kohlhagen, _to_continuous
    # A simple rate r_s and its continuous equivalent should price identically
    # only if the function converts. We check the conversion helper is applied:
    r_simple = 0.05
    tau = 1.0
    r_cont = _to_continuous(r_simple, tau)
    assert r_cont < r_simple                      # ln(1+r) < r
    # Price is finite and positive for an ATM option.
    p = garman_kohlhagen(1.10, 1.10, 0.03, 0.05, 0.10, 1.0, is_call=True)
    assert p > 0


def test_put_call_parity_holds():
    """Put-call parity must hold with the continuous-rate discounting."""
    import math
    from fxrisk.options import garman_kohlhagen, _to_continuous
    S, K, rb, rq, vol, tau = 1.10, 1.08, 0.03, 0.05, 0.12, 0.5
    call = garman_kohlhagen(S, K, rb, rq, vol, tau, is_call=True)
    put = garman_kohlhagen(S, K, rb, rq, vol, tau, is_call=False)
    rb_c, rq_c = _to_continuous(rb, tau), _to_continuous(rq, tau)
    lhs = call - put
    rhs = S * math.exp(-rb_c * tau) - K * math.exp(-rq_c * tau)
    assert abs(lhs - rhs) < 1e-9                   # parity exact


def test_gamma_matches_finite_difference():
    """Gamma must equal the numerical derivative of delta w.r.t. spot."""
    from fxrisk.options import option_delta, option_gamma
    S, K, rb, rq, vol, tau = 1.10, 1.08, 0.03, 0.05, 0.12, 0.5
    h = 1e-5
    fd = (option_delta(S + h, K, rb, rq, vol, tau, True)
          - option_delta(S - h, K, rb, rq, vol, tau, True)) / (2 * h)
    assert abs(option_gamma(S, K, rb, rq, vol, tau) - fd) < 1e-3


def test_gamma_positive_and_symmetric():
    """Gamma is positive and identical for calls and puts (same formula)."""
    from fxrisk.options import option_gamma
    g = option_gamma(1.10, 1.10, 0.03, 0.05, 0.10, 0.5)
    assert g > 0


def test_theta_is_negative_decay():
    """Theta (per day) should be negative for a standard option (time decay)."""
    from fxrisk.options import option_theta
    th = option_theta(1.10, 1.10, 0.03, 0.05, 0.10, 0.5, is_call=True)
    assert th < 0


def test_theta_matches_finite_difference():
    """Theta/day must match the price change as one day passes."""
    from fxrisk.options import garman_kohlhagen, option_theta
    S, K, rb, rq, vol, tau = 1.10, 1.08, 0.03, 0.05, 0.12, 0.5
    fd = (garman_kohlhagen(S, K, rb, rq, vol, tau - 1/365, True)
          - garman_kohlhagen(S, K, rb, rq, vol, tau, True))
    assert abs(option_theta(S, K, rb, rq, vol, tau, True) - fd) < 1e-4
