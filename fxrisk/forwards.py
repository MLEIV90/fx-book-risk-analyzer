"""
fxrisk.forwards
===============
FX forward pricing via Covered Interest Rate Parity (CIP).

Conventions:
- Quote-per-base notation: EUR/USD = 1.08  ->  1 EUR = 1.08 USD.
      base  = currency being bought/sold (EUR).
      quote = currency the price is measured in (USD).
- Annual SIMPLE rates, in decimal (0.045 = 4.5%).
- Day-count is CURRENCY-AWARE (H2): ACT/360 for EUR and USD (the Eurocurrency
  money-market convention), ACT/365 for GBP (the sterling money-market
  convention) -- see DAY_COUNT_BASIS / year_fraction. This is a real
  convention difference, not house style: using 360 on a GBP leg introduces a
  small but real pricing error, since r_GBP is itself quoted on an ACT/365
  basis.
- 1 pip = 0.0001 for most pairs.

All functions are pure: they take numbers and return numbers,
with no interface or external-data dependencies.
"""
from __future__ import annotations

# Money-market day-count basis, per currency (H2). ACT/360 is the
# Eurocurrency convention (EUR, USD); GBP money markets use ACT/365.
DAY_COUNT_BASIS: dict[str, int] = {
    "EUR": 360,
    "USD": 360,
    "GBP": 365,
}
DEFAULT_DAYS_BASIS: int = 360   # ACT/360, used when a currency isn't listed
PIP: float = 1e-4               # size of 1 pip


def day_count_basis(currency: str) -> int:
    """Money-market day-count basis for `currency` (see DAY_COUNT_BASIS)."""
    return DAY_COUNT_BASIS.get(currency, DEFAULT_DAYS_BASIS)


def year_fraction(days: int, currency: str | None = None,
                  basis: int | None = None) -> float:
    """
    Convert a tenor in days to a year fraction, on the correct money-market
    day-count basis for `currency`: ACT/360 for EUR and USD, ACT/365 for GBP
    (see DAY_COUNT_BASIS). `currency=None` uses the ACT/360 default. Pass
    `basis` directly to override either.
    """
    if basis is None:
        basis = day_count_basis(currency) if currency else DEFAULT_DAYS_BASIS
    return days / basis


def forward_rate(spot: float, r_base: float, r_quote: float, tau: float,
                 tau_quote: float | None = None) -> float:
    """
    Theoretical forward via Covered Interest Rate Parity (CIP).

    F = S * (1 + r_quote * tau_quote) / (1 + r_base * tau_base)

    `tau` is the BASE currency's year fraction (its own day-count basis --
    see `year_fraction`). `tau_quote` is the QUOTE currency's own year
    fraction; it defaults to `tau` when omitted, which is exact whenever base
    and quote share a day-count convention (e.g. EUR/USD, both ACT/360).
    Pass `tau_quote` explicitly whenever they differ (H2) -- any pair
    involving GBP (ACT/365) against EUR or USD (ACT/360).

    It is the spot adjusted by the interest-rate differential, not a forecast.
    """
    tq = tau_quote if tau_quote is not None else tau
    return spot * (1.0 + r_quote * tq) / (1.0 + r_base * tau)


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

    Discounted at the QUOTE rate because the P&L (F_now - K) is in quote
    currency -- so `tau_remaining` MUST be computed on the QUOTE currency's
    OWN day-count basis (H2: ACT/360 for EUR/USD, ACT/365 for GBP; see
    `year_fraction`), not a blanket basis borrowed from the base currency.
    """
    df_quote = 1.0 / (1.0 + r_quote * tau_remaining)
    v = notional_base * (fair_fwd_now - strike) * df_quote
    return v if long_base else -v