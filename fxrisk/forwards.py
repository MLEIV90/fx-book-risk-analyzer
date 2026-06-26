"""
fxrisk.forwards
===============
FX forward pricing via Covered Interest Rate Parity (CIP).

Conventions:
- Quote-per-base notation: EUR/USD = 1.08  ->  1 EUR = 1.08 USD.
      base  = currency being bought/sold (EUR).
      quote = currency the price is measured in (USD).
- Annual SIMPLE rates, in decimal (0.045 = 4.5%).
  (Note: fxrisk.options uses continuous compounding instead, as Garman-Kohlhagen
  is a continuous-time model. Each convention is standard in its own context;
  on the short tenors used here the numerical difference is small.)
- ACT/360 day-count (money-market FX standard).
- 1 pip = 0.0001 for most pairs.

All functions are pure: they take numbers and return numbers,
with no interface or external-data dependencies.
"""
from __future__ import annotations

DAYS_BASIS: int = 360   # day-count basis (ACT/360)
PIP: float = 1e-4       # size of 1 pip


def year_fraction(days: int, basis: int = DAYS_BASIS) -> float:
    """Convert a tenor in days to a year fraction. E.g. 90 days -> 0.25."""
    return days / basis


def forward_rate(spot: float, r_base: float, r_quote: float, tau: float) -> float:
    """
    Theoretical forward via Covered Interest Rate Parity (CIP).

    F = S * (1 + r_quote * tau) / (1 + r_base * tau)

    It is the spot adjusted by the interest-rate differential, not a forecast.
    """
    return spot * (1.0 + r_quote * tau) / (1.0 + r_base * tau)


def forward_points(spot: float, fwd: float, pip: float = PIP) -> float:
    """Forward points = (F - S) expressed in pips. How far the forward sits from spot."""
    return (fwd - spot) / pip


def client_rate_with_spread(fair_fwd: float, spread_pips: float,
                            client_buys_base: bool, pip: float = PIP) -> float:
    """
    Rate the provider quotes to the client = theoretical forward +/- spread.

    The provider's margin is NOT a premium: it is the spread over the
    theoretical rate. If the client BUYS the base currency, they are quoted
    slightly higher (fair + spread).
    """
    s = spread_pips * pip
    return fair_fwd + s if client_buys_base else fair_fwd - s


def provider_revenue(notional_base: float, spread_pips: float, pip: float = PIP) -> float:
    """
    Provider's revenue (in QUOTE currency) from the spread = notional * spread.
    It scales with deal size, not with the risk taken on.
    """
    return notional_base * spread_pips * pip


def forward_mtm(notional_base: float, strike: float, fair_fwd_now: float,
                r_quote: float, tau_remaining: float, long_base: bool = True) -> float:
    """
    Mark-to-market (MtM) of a forward already struck at 'strike', in QUOTE currency.

    V = N_base * (F_now - K) * DF_quote
    DF_quote = 1 / (1 + r_quote * tau_remaining)

    Discounted at the QUOTE rate because the P&L (F_now - K) is in quote currency.
    """
    df_quote = 1.0 / (1.0 + r_quote * tau_remaining)
    v = notional_base * (fair_fwd_now - strike) * df_quote
    return v if long_base else -v