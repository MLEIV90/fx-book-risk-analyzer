"""
fxrisk.book_risk
================
Interest-rate risk (DV01), liquidity (variation margin) and stress testing,
aggregated over the whole book. These apply the tested primitives in
fxrisk.risk to every position and sum to the book level.

Together with fxrisk.book_analytics (valuation) and fxrisk.portfolio_risk
(VaR/ES/Kupiec/attribution), this completes the risk function over a real book.

Declared scope:
- Stress applies each scenario's % move to each pair by its exposure; pairs not
  covered by a scenario get a zero shock (declared, never invented).
- Liquidity uses the constant-volatility variation-margin simulation (a real
  desk would use implied vol or GARCH) -- the simplification is stated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fxrisk.risk import dv01_forward, simulate_liquidity_need, STRESS_SCENARIOS


@dataclass
class BookRiskReport:
    """Aggregated rate, liquidity and stress figures for the book."""
    dv01_by_currency: dict            # currency curve -> DV01
    dv01_total: float                 # sum across curves
    liquidity_buffer: float           # cash buffer at confidence
    stress_results: dict              # scenario -> {pnl, is_loss, loss_x_var}


def dv01_book(book, snapshots: dict) -> tuple[dict, float]:
    """
    Aggregate DV01 per currency curve across the book.

    snapshots: {position_id: MarketSnapshot}. Each forward's DV01 is computed by
    bump-and-reprice (from fxrisk.risk) and accumulated onto its base and quote
    currency curves. The provider's direction flips the sign.
    """
    by_ccy: dict[str, float] = {}
    for p in book:
        snap = snapshots[p.id]
        dv = dv01_forward(p.notional_base, p.strike, snap.spot,
                          snap.r_base, snap.r_quote, snap.tenor_years)
        sign = 1.0 if p.long_base else -1.0
        by_ccy[p.base_ccy] = by_ccy.get(p.base_ccy, 0.0) + sign * dv["dv01_base"]
        by_ccy[p.quote_ccy] = by_ccy.get(p.quote_ccy, 0.0) + sign * dv["dv01_quote"]
    total = sum(by_ccy.values())
    return by_ccy, total


def liquidity_book(book, snapshots: dict, confidence: float = 0.99,
                   returns: np.ndarray = None, positions: np.ndarray = None) -> float:
    """
    Aggregate liquidity buffer: the variation-margin cash the book may have to
    post over the margin horizon at the chosen confidence.

    Preferred method (when `returns` and `positions` are supplied): use the REAL
    volatility of the book's aggregate daily P&L, which correctly accounts for
    netting and correlation between pairs. The margin need is then the worst
    cumulative adverse move over the horizon.

    Fallback (no returns): the largest position's daily vol as a conservative
    proxy. Declared simplification, kept for backward compatibility.
    """
    if book.is_empty:
        return 0.0

    # Margin horizon: the shortest tenor in the book, capped at 90 days.
    horizon = min(90, min((max(int(p.tenor_days), 1) for p in book), default=60))

    if returns is not None and positions is not None:
        # Real aggregate book P&L volatility (accounts for netting/correlation).
        pnl = np.asarray(returns) @ np.asarray(positions)
        daily_pnl_vol = float(np.std(pnl))
        from scipy.stats import norm
        z = norm.ppf(confidence)
        # Worst cumulative adverse move over the horizon (sqrt-of-time).
        return z * daily_pnl_vol * np.sqrt(horizon)

    # Fallback: conservative single-vol proxy on total notional.
    total_notional_quote = 0.0
    vols = []
    for p in book:
        snap = snapshots[p.id]
        total_notional_quote += p.notional_base * snap.spot
        vols.append(snap.vol_historical / np.sqrt(252))
    daily_vol = max(vols)
    out = simulate_liquidity_need(total_notional_quote, daily_vol,
                                  horizon_days=horizon, confidence=confidence)
    return out["liquidity_buffer"]


def stress_book(book, snapshots: dict, var_reference: float) -> dict:
    """
    Apply each historical stress scenario to the whole book.

    For each scenario, P&L = sum over positions of exposure_quote * pair_move
    (positive = gain, negative = loss). Pairs absent from a scenario get a zero
    move (declared).

    Reports, per scenario:
      - pnl:      signed profit/loss in USD (negative = loss).
      - is_loss:  True if the scenario produces a loss.
      - loss_x_var: for a LOSS, how many times the reference VaR the loss is
                    (a positive multiple); 0.0 for a gain. This is only meaningful
                    for losses -- a gain is not 'tail risk', so it is not expressed
                    as a VaR multiple.
    """
    results: dict[str, dict] = {}
    for name, scenario in STRESS_SCENARIOS.items():
        pnl = 0.0
        for p in book:
            snap = snapshots[p.id]
            move = scenario.get(p.pair, 0.0)               # 0 if not covered
            sign = 1.0 if p.long_base else -1.0
            exposure_quote = sign * p.notional_base * snap.spot
            pnl += exposure_quote * move
        is_loss = pnl < 0
        if is_loss and var_reference > 0:
            loss_x_var = -pnl / var_reference               # positive multiple
        else:
            loss_x_var = 0.0
        results[name] = {"pnl": pnl, "is_loss": is_loss, "loss_x_var": loss_x_var}
    return results


def dv01_book_by_tenor(book, snapshots: dict) -> dict:
    """
    B4: key-rate DV01 -- DV01 grouped by tenor bucket, not just by currency.

    A single parallel-bump DV01 hides WHERE on the curve the rate risk sits. A
    book of mostly 90-day forwards has its risk at the short end; a book with
    2-year forwards at the long end. Bucketing the per-position DV01 by tenor
    shows the curve exposure profile -- the input a desk needs to hedge with the
    right instruments. Buckets: 0-3m, 3-6m, 6-12m, 12m+.
    """
    buckets = {"0-3m": 0.0, "3-6m": 0.0, "6-12m": 0.0, "12m+": 0.0}
    for p in book:
        snap = snapshots[p.id]
        dv = dv01_forward(p.notional_base, p.strike, snap.spot,
                          snap.r_base, snap.r_quote, snap.tenor_years)
        sign = 1.0 if p.long_base else -1.0
        total_dv = sign * dv["dv01_net"]
        d = p.tenor_days
        if d <= 90:
            buckets["0-3m"] += total_dv
        elif d <= 180:
            buckets["3-6m"] += total_dv
        elif d <= 365:
            buckets["6-12m"] += total_dv
        else:
            buckets["12m+"] += total_dv
    return buckets
