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

import numpy as np

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


def option_book_var(book: OptionBook, spots: dict[str, float],
                    rates: dict[str, tuple[float, float]],
                    returns: dict[str, np.ndarray],
                    confidence: float = 0.99, n_sims: int = 20_000,
                    horizon_days: int = 1, seed: int | None = 42) -> dict:
    """
    Full-revaluation VaR of the option book (the correct, non-linear method).

    Unlike a linear (delta-equivalent) VaR, this RE-PRICES every option with
    Garman-Kohlhagen under each simulated spot scenario, so it captures the
    curvature (gamma) that makes options non-linear. Steps:

      1. Estimate each pair's daily spot volatility from `returns`.
      2. Simulate correlated 1-day spot shocks (normal, covariance from returns).
      3. For each scenario, shock each pair's spot, re-price every option, and
         sum the change in book value -> a P&L distribution.
      4. VaR = the loss at the (1 - confidence) percentile.

    Scope (declared): only SPOT is shocked. Volatility and rates are held fixed,
    so this is a spot (delta+gamma) VaR, not a vega VaR. Aggregating volatility
    risk would be the next step. Horizon is `horizon_days` (sqrt-time scaled).

    Returns the full-reval VaR, the linear delta-equivalent VaR for comparison,
    and the gamma effect (their difference) which quantifies the non-linearity.
    """
    pairs = book.pairs()
    if not pairs:
        return {"var_full_reval": 0.0, "var_delta_equiv": 0.0,
                "gamma_effect": 0.0, "confidence": confidence}

    # Covariance of daily returns across the pairs present in the book.
    ret_matrix = np.column_stack([np.asarray(returns[p], dtype=float) for p in pairs])
    if ret_matrix.ndim == 1:
        ret_matrix = ret_matrix.reshape(-1, 1)
    cov = np.atleast_2d(np.cov(ret_matrix, rowvar=False))

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(cov)
    # Simulated daily RETURNS per pair (zero drift), scaled to the horizon.
    z = rng.standard_normal((n_sims, len(pairs)))
    sim_returns = (z @ chol.T) * np.sqrt(horizon_days)

    # Base book value and per-option terms.
    base_value = 0.0
    # Pre-compute per-option references once.
    opts = []
    for p in book:
        tau = p.tenor_days / 365.0
        rb, rq = rates[p.pair]
        s0 = spots[p.pair]
        v0 = garman_kohlhagen(s0, p.strike, rb, rq, p.vol, tau, p.is_call)
        d0 = option_delta(s0, p.strike, rb, rq, p.vol, tau, p.is_call)
        base_value += v0 * p.notional_base
        opts.append((p, tau, rb, rq, s0, v0, d0))

    pair_idx = {pr: i for i, pr in enumerate(pairs)}

    # Full revaluation: P&L per scenario = sum over options of (revalued - base).
    pnl_full = np.zeros(n_sims)
    pnl_delta = np.zeros(n_sims)
    for (p, tau, rb, rq, s0, v0, d0) in opts:
        r_sim = sim_returns[:, pair_idx[p.pair]]
        shocked_spot = s0 * (1.0 + r_sim)
        # Re-price the option at each shocked spot (vectorised over scenarios).
        revalued = np.array([
            garman_kohlhagen(s, p.strike, rb, rq, p.vol, tau, p.is_call)
            for s in shocked_spot])
        pnl_full += (revalued - v0) * p.notional_base
        # Linear (delta-equivalent) approximation for comparison.
        pnl_delta += d0 * (shocked_spot - s0) * p.notional_base

    var_full = -np.percentile(pnl_full, (1.0 - confidence) * 100.0)
    var_delta = -np.percentile(pnl_delta, (1.0 - confidence) * 100.0)
    return {
        "var_full_reval": float(var_full),
        "var_delta_equiv": float(var_delta),
        "gamma_effect": float(var_full - var_delta),
        "base_value": float(base_value),
        "confidence": confidence,
        "horizon_days": horizon_days,
    }
