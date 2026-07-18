"""
fxrisk.market
=============
Assembles a complete, REAL market snapshot for a pair and tenor, by combining
the three data sources we built:

- spot           <- yfinance (observed)
- base/quote rate<- FRED curves, interpolated to the tenor (observed)
- volatility     <- historical and GARCH(1,1)-t from the price history (derived)

This is the glue layer: it turns the separate modules into a single object the
pricing and risk code can consume, with NO user-typed market parameters. Every
field carries its source. Network failures propagate as clear errors (the app
catches them); nothing is invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fxrisk.forwards import forward_rate, year_fraction
from fxrisk.data import fetch_spot_history, to_returns, TRIANGULATED_PAIRS
from fxrisk.curves import rate_for_tenor, supported_currencies
from fxrisk.garch import historical_vol, fit_garch


@dataclass
class MarketSnapshot:
    """
    A fully-sourced picture of the market for one pair at one tenor.

    Day-count (H2): `tenor_years` is the BASE currency's own year fraction;
    `tenor_years_quote` is the QUOTE currency's (defaults to `tenor_years`
    when not set, e.g. for snapshots built directly in tests -- exact
    whenever base and quote share a day-count basis). Use `tau_quote` to read
    the effective quote-side tau; MtM discounting (r_quote) must use it, not
    `tenor_years`, whenever the pair's legs differ in convention (any pair
    involving GBP against EUR or USD -- see fxrisk.forwards.DAY_COUNT_BASIS).
    """
    pair: str
    base_ccy: str
    quote_ccy: str
    tenor_years: float
    spot: float
    r_base: float
    r_quote: float
    vol_historical: float
    vol_garch: float | None
    notes: list[str] = field(default_factory=list)
    sources: dict = field(default_factory=dict)
    tenor_years_quote: float | None = None

    @property
    def tau_quote(self) -> float:
        """The QUOTE currency's own year fraction (falls back to
        `tenor_years` when `tenor_years_quote` isn't set)."""
        return self.tenor_years_quote if self.tenor_years_quote is not None else self.tenor_years

    def forward(self) -> float:
        """Theoretical forward from this snapshot (Covered Interest Rate Parity)."""
        return forward_rate(self.spot, self.r_base, self.r_quote,
                            self.tenor_years, self.tau_quote)


def get_market_snapshot(pair: str, tenor_days: int,
                        use_garch: bool = True,
                        history_period: str = "2y") -> MarketSnapshot:
    """
    Build a real market snapshot for `pair` at `tenor_days`.

    Steps, all from real sources:
      1. spot  = latest close from yfinance.
      2. rates = FRED curve for each currency, interpolated to the tenor.
      3. vol   = historical (and GARCH) from the return history.
    Raises if a currency has no configured rate curve, or if data is unavailable.
    """
    base_ccy, quote_ccy = pair.split("/")
    for ccy in (base_ccy, quote_ccy):
        if ccy not in supported_currencies():
            raise ValueError(f"No rate curve available for {ccy}.")

    # H2: day-count is currency-aware -- ACT/360 for EUR/USD, ACT/365 for GBP.
    # Base and quote can therefore have slightly different year fractions for
    # the SAME tenor_days (e.g. any GBP leg vs a EUR or USD leg).
    tenor_years = year_fraction(tenor_days, base_ccy)
    tenor_years_quote = year_fraction(tenor_days, quote_ccy)
    notes: list[str] = []
    sources: dict = {}

    # 1. Spot + history (yfinance directly, or triangulated for a cross like
    #    EUR/GBP -- see fxrisk.data.TRIANGULATED_PAIRS).
    prices = fetch_spot_history([pair], period=history_period)
    spot = float(prices[pair].iloc[-1])
    if pair in TRIANGULATED_PAIRS:
        leg_a, leg_b = TRIANGULATED_PAIRS[pair]
        sources["spot"] = f"Triangulated: {leg_a} / {leg_b} (yfinance latest close)"
        notes.append(
            f"{pair} is not directly quoted here; its spot and history are "
            f"DERIVED by triangulating {leg_a} and {leg_b} under CIP-consistent "
            "quote-per-base convention, then the forward applies CIP on top of "
            "that triangulated spot. It inherits both legs' data quality.")
    else:
        sources["spot"] = "yfinance (latest close)"

    # 2. Rates per currency from FRED, at this tenor -- each curve read at
    #    its OWN currency's year fraction (H2).
    r_base, curve_b = rate_for_tenor(base_ccy, tenor_years)
    r_quote, curve_q = rate_for_tenor(quote_ccy, tenor_years_quote)
    sources["rates"] = f"Rate curves ({', '.join(curve_b.sources + curve_q.sources)})"
    if curve_b.notes:
        notes.append(f"{base_ccy}: {curve_b.notes}")
    if curve_q.notes:
        notes.append(f"{quote_ccy}: {curve_q.notes}")

    # 3. Volatility from the return history.
    returns = to_returns(prices)[pair].to_numpy()
    vol_hist = historical_vol(returns)
    vol_garch = None
    if use_garch:
        try:
            vol_garch = fit_garch(returns, asymmetric=False, dist="t").current_vol_annual
            sources["volatility"] = "historical + GARCH(1,1)-t (from yfinance history)"
        except Exception as exc:
            notes.append(f"GARCH fit failed, using historical vol only: {exc}")
            sources["volatility"] = "historical (GARCH unavailable)"
    else:
        sources["volatility"] = "historical (from yfinance history)"

    return MarketSnapshot(
        pair=pair, base_ccy=base_ccy, quote_ccy=quote_ccy,
        tenor_years=tenor_years, spot=spot, r_base=r_base, r_quote=r_quote,
        vol_historical=vol_hist, vol_garch=vol_garch,
        notes=notes, sources=sources, tenor_years_quote=tenor_years_quote,
    )