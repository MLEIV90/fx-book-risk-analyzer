"""
fxrisk.risk
===========
Market-risk metrics for an FX book: DV01, Value at Risk (VaR),
Expected Shortfall (ES), a liquidity (variation-margin) simulation, and
historical stress testing.

Design choice: the risk functions receive RETURNS and POSITIONS as inputs;
they do not fetch or simulate market data themselves. This keeps the engine
pure and testable -- where the data comes from is another layer's job.

Sign convention for P&L:
- A loss is a NEGATIVE number.
- VaR and ES are reported as POSITIVE loss magnitudes (the desk convention).

All functions are pure.
"""
from __future__ import annotations

import numpy as np

from fxrisk.forwards import forward_rate, forward_mtm


# --------------------------------------------------------------------------
# DV01 -- interest-rate sensitivity
# --------------------------------------------------------------------------
def dv01_forward(notional_base: float, strike: float, spot: float,
                 r_base: float, r_quote: float, tau: float,
                 bump_bp: float = 1.0) -> dict[str, float]:
    """
    DV01 of a single forward, by bump-and-reprice, per currency curve.

    DV01 = MtM(rate + 1bp) - MtM(rate now), done separately for each curve.
    Reporting per curve matters: a forward's two DV01s offset on a parallel
    move but not when a single curve moves -- the rate risk lives in the
    DIFFERENTIAL, which a single net number would hide.
    """
    bump = bump_bp * 1e-4  # 1 basis point = 0.0001

    fwd_now = forward_rate(spot, r_base, r_quote, tau)
    base_mtm = forward_mtm(notional_base, strike, fwd_now, r_quote, tau)

    fwd_b = forward_rate(spot, r_base + bump, r_quote, tau)
    mtm_b = forward_mtm(notional_base, strike, fwd_b, r_quote, tau)
    dv01_base = mtm_b - base_mtm

    fwd_q = forward_rate(spot, r_base, r_quote + bump, tau)
    mtm_q = forward_mtm(notional_base, strike, fwd_q, r_quote + bump, tau)
    dv01_quote = mtm_q - base_mtm

    return {
        "dv01_base": dv01_base,
        "dv01_quote": dv01_quote,
        "dv01_net": dv01_base + dv01_quote,
    }


# --------------------------------------------------------------------------
# VaR / ES -- three methods
# --------------------------------------------------------------------------
def _pnl_vector(returns: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Turn a matrix of returns into a vector of portfolio P&L.

    returns:   shape (n_scenarios, n_assets), each row one scenario.
    positions: shape (n_assets,), signed exposures in quote currency.
    P&L per scenario = sum over assets of (return * position).
    """
    return returns @ positions


def var_parametric(returns: np.ndarray, positions: np.ndarray,
                   confidence: float = 0.99) -> float:
    """
    Parametric (variance-covariance) VaR.

    Assumes returns are jointly normal. Builds the portfolio standard deviation
    from the covariance matrix, then scales by the normal z-score.
    Fast and smooth, but UNDERESTIMATES the tail when returns are fat-tailed.
    """
    from scipy.stats import norm

    cov = np.atleast_2d(np.cov(returns, rowvar=False))   # 1 factor -> 1x1
    port_var = float(positions @ cov @ positions)
    port_sd = np.sqrt(port_var)
    z = norm.ppf(confidence)
    return z * port_sd


def var_historical(returns: np.ndarray, positions: np.ndarray,
                   confidence: float = 0.99) -> float:
    """
    Historical VaR: no distribution assumption. Reprice the book on each
    historical scenario, then read the empirical percentile of the loss side.
    """
    pnl = _pnl_vector(returns, positions)
    return -np.percentile(pnl, (1.0 - confidence) * 100.0)


def var_montecarlo(returns: np.ndarray, positions: np.ndarray,
                   confidence: float = 0.99, n_sims: int = 100_000,
                   seed: int | None = 42) -> float:
    """
    Monte Carlo VaR: estimate the covariance, draw correlated normal returns via
    a Cholesky factor, reprice, and take the percentile.

    With a normal engine this lands close to the parametric VaR; the value of MC
    is that the engine can be swapped (e.g. Student-t) to model fat tails.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    mean = returns.mean(axis=0)
    cov = np.atleast_2d(np.cov(returns, rowvar=False))

    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_sims, len(positions)))
    sim_returns = mean + z @ chol.T
    pnl = _pnl_vector(sim_returns, positions)
    return -np.percentile(pnl, (1.0 - confidence) * 100.0)


def expected_shortfall(returns: np.ndarray, positions: np.ndarray,
                       confidence: float = 0.99) -> float:
    """
    Expected Shortfall (ES): the AVERAGE loss in the worst tail beyond the VaR.
    Tells you how deep the bad day is, not just where it starts.
    """
    pnl = _pnl_vector(returns, positions)
    cutoff = np.percentile(pnl, (1.0 - confidence) * 100.0)
    tail = pnl[pnl <= cutoff]
    return -tail.mean()


# --------------------------------------------------------------------------
# Liquidity -- variation-margin cash simulation
# --------------------------------------------------------------------------
def simulate_liquidity_need(notional_quote: float, daily_vol: float,
                            horizon_days: int = 60, n_sims: int = 10_000,
                            confidence: float = 0.99,
                            seed: int | None = 42) -> dict[str, float]:
    """
    Simulate the cash a provider may have to post as variation margin over time.

    A position hedged with a bank is marked daily: if it moves against the
    provider, cash is posted; if it moves in favour, cash is returned. Even a
    book that nets to zero P&L at maturity can need large cash in between.

    Method (same spirit as VaR, applied to CASH instead of P&L):
      1. Simulate daily moves of the hedge value (normal, constant vol -- a
         simplification; a real desk would use implied vol or GARCH).
      2. Accumulate the running margin balance along each path.
      3. Take the WORST drawdown (deepest cash outflow) of each path.
      4. Report a high percentile across paths -- the liquidity buffer.
    """
    rng = np.random.default_rng(seed)
    daily_moves = rng.standard_normal((n_sims, horizon_days)) * notional_quote * daily_vol
    cumulative = np.cumsum(daily_moves, axis=1)
    worst_drawdown = cumulative.min(axis=1)
    buffer = -np.percentile(worst_drawdown, (1.0 - confidence) * 100.0)
    avg_worst = -worst_drawdown.mean()
    return {"liquidity_buffer": buffer, "avg_worst_drawdown": avg_worst}


# --------------------------------------------------------------------------
# Stress testing -- apply real event shocks to today's book
# --------------------------------------------------------------------------
# Approximate moves of the pair during real events (quote-per-base % change).
STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "Brexit referendum (Jun 2016)": {"EUR/USD": -0.02, "GBP/USD": -0.08},
    "COVID crash (Mar 2020)":       {"EUR/USD": -0.03, "GBP/USD": -0.06},
    "SNB de-peg (Jan 2015)":        {"EUR/USD": -0.02, "GBP/USD": -0.03},
}


def stress_pnl(positions_by_pair: dict[str, float],
               scenario: dict[str, float]) -> float:
    """
    P&L of today's book under a stress scenario.

    positions_by_pair: signed quote-currency exposure per pair.
    scenario:          % move per pair.
    Stress P&L = sum over pairs of (exposure * % move). Only the percentage move
    travels from the past; it is applied to TODAY's positions, so the result is
    already in today's money -- no inflation adjustment needed.
    """
    total = 0.0
    for pair, exposure in positions_by_pair.items():
        total += exposure * scenario.get(pair, 0.0)
    return total
