"""
fxrisk.option_book
==================
A separate book for FX options, managed the way an options desk actually manages
one: by AGGREGATE GREEKS (total delta, gamma, vega, theta), not by a linear VaR.

Why a separate book, and why greeks instead of VaR:
- Forwards are linear, so a covariance VaR captures their risk exactly. Options
  are non-linear (gamma), so the same linear VaR would MISSTATE their risk. We
  therefore do NOT put options through the forward VaR.
- An options desk reads its risk from the greek profile: net delta (directional
  exposure), gamma (how fast delta moves), vega (volatility exposure), theta
  (daily time decay). That is the correct, honest lens for an option book.
- A full-revaluation VaR (re-pricing every option in every Monte Carlo scenario)
  is the natural next step to get an aggregate option VaR that respects the
  non-linearity; it is documented as future work, deliberately not approximated
  with a linear shortcut here.

This module is PURE: greeks are computed from fxrisk.options given the terms
stored on each position. Market data (spot, rates, vol) is supplied by the layer
above, exactly as for the forward book.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from fxrisk.options import (garman_kohlhagen, option_delta, option_gamma,
                            option_vega, option_theta)


@dataclass
class OptionPosition:
    """A single FX option held in the option book."""
    pair: str                       # e.g. "EUR/USD"
    is_call: bool                   # True = call, False = put
    notional_base: float            # size in base-currency units
    strike: float                   # strike rate
    tenor_days: int                 # days to expiry (original)
    vol: float                      # volatility used to price (GARCH/historical)
    premium_unit: float = 0.0       # premium per unit of base, at booking
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = ""

    @property
    def base_ccy(self) -> str:
        return self.pair.split("/")[0]

    @property
    def quote_ccy(self) -> str:
        return self.pair.split("/")[1]

    @property
    def kind(self) -> str:
        return "Call" if self.is_call else "Put"


class OptionBook:
    """A collection of option positions, managed by aggregate greeks."""

    def __init__(self, positions: list[OptionPosition] | None = None) -> None:
        self._positions: list[OptionPosition] = list(positions or [])

    def __iter__(self):
        return iter(self._positions)

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def is_empty(self) -> bool:
        return len(self._positions) == 0

    def add(self, position: OptionPosition) -> None:
        self._positions.append(position)

    def remove(self, position_id: str) -> None:
        self._positions = [p for p in self._positions if p.id != position_id]

    def clear(self) -> None:
        self._positions = []

    def pairs(self) -> list[str]:
        return sorted({p.pair for p in self._positions})


def option_book_greeks(book: OptionBook, spots: dict[str, float],
                       rates: dict[str, tuple[float, float]]) -> dict:
    """
    Aggregate greek profile of the option book -- the desk's risk summary.

    spots: current spot per pair, e.g. {"EUR/USD": 1.0850}.
    rates: (r_base, r_quote) per pair, e.g. {"EUR/USD": (0.021, 0.039)}.

    Greeks are scaled by each position's notional and summed. Delta/gamma carry a
    sign for puts (a put's delta is negative); gamma is the same sign for calls
    and puts but is signed here by position direction for aggregation. Returns
    totals plus a per-position breakdown.

    The aggregate value is the sum of current option values (mark-to-model with
    the stored vol) times notional -- the book's premium value, not a VaR.
    """
    totals = {"value": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    rows = []
    for p in book:
        spot = spots[p.pair]
        r_base, r_quote = rates[p.pair]
        tau = p.tenor_days / 365.0
        value_unit = garman_kohlhagen(spot, p.strike, r_base, r_quote, p.vol, tau,
                                      p.is_call)
        delta = option_delta(spot, p.strike, r_base, r_quote, p.vol, tau, p.is_call)
        gamma = option_gamma(spot, p.strike, r_base, r_quote, p.vol, tau)
        vega = option_vega(spot, p.strike, r_base, r_quote, p.vol, tau)
        theta = option_theta(spot, p.strike, r_base, r_quote, p.vol, tau, p.is_call)

        n = p.notional_base
        totals["value"] += value_unit * n
        totals["delta"] += delta * n
        totals["gamma"] += gamma * n
        totals["vega"] += vega * n
        totals["theta"] += theta * n
        rows.append({
            "id": p.id, "pair": p.pair, "kind": p.kind, "strike": p.strike,
            "notional": n, "value": value_unit * n, "delta": delta * n,
            "gamma": gamma * n, "vega": vega * n, "theta": theta * n,
        })
    return {"totals": totals, "positions": rows}
