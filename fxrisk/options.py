"""
fxrisk.options
==============
FX option pricing via the Garman-Kohlhagen model (Black-Scholes adapted to FX).

Same conventions as fxrisk.forwards:
- Quote-per-base notation: EUR/USD = 1.08.
- Annual SIMPLE rates here are converted to continuous compounding internally,
  because Garman-Kohlhagen is a continuous-time model.
- Volatility is the annualised volatility of the pair, in decimal (0.08 = 8%).

Unlike a forward, an option has a PREMIUM, because it is a right, not an
obligation. The premium is the discounted expected value of a contingent payoff
-- conceptually, an insurance pure premium.

All functions are pure.
"""
from __future__ import annotations

import math


def norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function N(x).
    Uses the error function, which is exact up to floating-point precision.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """Standard normal probability density function n(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot: float, strike: float, r_base: float, r_quote: float,
           vol: float, tau: float) -> tuple[float, float]:
    """
    Helper: the d1 and d2 terms of Garman-Kohlhagen.

    d1 = [ln(S/K) + (r_quote - r_base + 0.5*vol^2) * tau] / (vol * sqrt(tau))
    d2 = d1 - vol * sqrt(tau)
    """
    vsqrt = vol * math.sqrt(tau)
    d1 = (math.log(spot / strike) + (r_quote - r_base + 0.5 * vol * vol) * tau) / vsqrt
    d2 = d1 - vsqrt
    return d1, d2


def _to_continuous(r_simple: float, tau: float) -> float:
    """
    Convert an annual SIMPLE rate to its continuously-compounded equivalent for
    the given tenor: r_c = ln(1 + r_simple * tau) / tau.

    Garman-Kohlhagen is a continuous-time model, while the curves (and forwards)
    use simple rates. Converting here keeps the option consistent with the rest
    of the engine and with this module's stated convention.
    """
    return math.log(1.0 + r_simple * tau) / tau


def garman_kohlhagen(spot: float, strike: float, r_base: float, r_quote: float,
                     vol: float, tau: float, is_call: bool = True) -> float:
    """
    Garman-Kohlhagen price of an FX option, per 1 unit of base currency.

    call = S * exp(-r_base*tau) * N(d1) - K * exp(-r_quote*tau) * N(d2)
    put  = K * exp(-r_quote*tau) * N(-d2) - S * exp(-r_base*tau) * N(-d1)

    Input rates are annual SIMPLE rates (as elsewhere in the engine); they are
    converted to continuous compounding internally. Multiply by the notional
    (in base) to get the total premium in quote currency.

    Edge cases (V1/V2): inputs are validated, and the degenerate limit where
    vol*sqrt(tau) -> 0 (at expiry, or zero volatility) returns the discounted
    intrinsic value instead of dividing by zero.
    """
    if not all(math.isfinite(x) for x in (spot, strike, r_base, r_quote, vol, tau)):
        raise ValueError("garman_kohlhagen received a non-finite input.")
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if vol < 0 or tau < 0:
        raise ValueError("Volatility and tenor must be non-negative.")

    rb_c = _to_continuous(r_base, tau) if tau > 0 else 0.0
    rq_c = _to_continuous(r_quote, tau) if tau > 0 else 0.0
    disc_base = math.exp(-rb_c * tau)
    disc_quote = math.exp(-rq_c * tau)

    # Degenerate limit: no time value left -> discounted intrinsic value.
    if vol * math.sqrt(tau) < 1e-12:
        fwd_intrinsic = spot * disc_base - strike * disc_quote
        if is_call:
            return max(fwd_intrinsic, 0.0)
        return max(-fwd_intrinsic, 0.0)

    d1, d2 = _d1_d2(spot, strike, rb_c, rq_c, vol, tau)
    if is_call:
        return spot * disc_base * norm_cdf(d1) - strike * disc_quote * norm_cdf(d2)
    return strike * disc_quote * norm_cdf(-d2) - spot * disc_base * norm_cdf(-d1)


def option_delta(spot: float, strike: float, r_base: float, r_quote: float,
                 vol: float, tau: float, is_call: bool = True) -> float:
    """
    Spot delta: how much the option value moves per unit move in spot.
    For hedging, buy delta * notional of the base currency.

    delta_call =  exp(-r_base*tau) * N(d1)
    delta_put  = -exp(-r_base*tau) * N(-d1)
    """
    d1, _ = _d1_d2(spot, strike, _to_continuous(r_base, tau),
                   _to_continuous(r_quote, tau), vol, tau)
    disc_base = math.exp(-_to_continuous(r_base, tau) * tau)
    if is_call:
        return disc_base * norm_cdf(d1)
    return -disc_base * norm_cdf(-d1)


def option_vega(spot: float, strike: float, r_base: float, r_quote: float,
                vol: float, tau: float) -> float:
    """
    Vega: sensitivity of the premium to volatility, per 1.00 (100 vol points)
    change in volatility. Divide by 100 for the change per 1 vol point.

    vega = S * exp(-r_base*tau) * sqrt(tau) * n(d1)

    Vega is the same for calls and puts.
    """
    d1, _ = _d1_d2(spot, strike, _to_continuous(r_base, tau),
                   _to_continuous(r_quote, tau), vol, tau)
    return spot * math.exp(-_to_continuous(r_base, tau) * tau) * math.sqrt(tau) * norm_pdf(d1)

def option_gamma(spot: float, strike: float, r_base: float, r_quote: float,
                 vol: float, tau: float) -> float:
    """
    Gamma: the rate of change of delta per unit move in spot -- the curvature
    (non-linearity) of the option. The same for calls and puts.

    gamma = exp(-r_base*tau) * n(d1) / (S * vol * sqrt(tau))

    Gamma is why an option's risk cannot be captured by delta alone: as spot
    moves, the delta itself changes, and gamma measures how fast.
    """
    rb_c = _to_continuous(r_base, tau)
    rq_c = _to_continuous(r_quote, tau)
    d1, _ = _d1_d2(spot, strike, rb_c, rq_c, vol, tau)
    return math.exp(-rb_c * tau) * norm_pdf(d1) / (spot * vol * math.sqrt(tau))


def option_theta(spot: float, strike: float, r_base: float, r_quote: float,
                 vol: float, tau: float, is_call: bool = True) -> float:
    """
    Theta: the change in the option's value as one day passes (time decay),
    holding everything else fixed. Returned PER CALENDAR DAY (the annual theta
    divided by 365), and is typically NEGATIVE -- an option loses value as it
    approaches expiry.

    Full Garman-Kohlhagen theta (annual), then divided by 365 for per-day.
    """
    rb_c = _to_continuous(r_base, tau)
    rq_c = _to_continuous(r_quote, tau)
    d1, d2 = _d1_d2(spot, strike, rb_c, rq_c, vol, tau)
    disc_base = math.exp(-rb_c * tau)
    disc_quote = math.exp(-rq_c * tau)
    term1 = -spot * disc_base * norm_pdf(d1) * vol / (2.0 * math.sqrt(tau))
    if is_call:
        annual = (term1
                  + rb_c * spot * disc_base * norm_cdf(d1)
                  - rq_c * strike * disc_quote * norm_cdf(d2))
    else:
        annual = (term1
                  - rb_c * spot * disc_base * norm_cdf(-d1)
                  + rq_c * strike * disc_quote * norm_cdf(-d2))
    return annual / 365.0
