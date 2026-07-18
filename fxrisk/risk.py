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

Horizon convention (A2): two horizons coexist and must not be mixed.
- VaR / ES operate on DAILY returns and are 1-day figures by construction.
  A 10-day figure is obtained by scaling with sqrt(10) (see portfolio_risk).
- historical_vol (fxrisk.garch) is ANNUALISED (x sqrt(252)) for reporting only;
  it is NOT the quantity fed into the daily VaR. Do not annualise the VaR inputs.

All functions are pure.
"""
from __future__ import annotations

import numpy as np

from fxrisk.forwards import forward_rate, forward_mtm


class MarketInputError(ValueError):
    """Raised when a market input fails a basic sanity check."""


def validate_market_inputs(spot: float, r_base: float, r_quote: float,
                           tau: float) -> None:
    """
    B3: sanity-check market inputs before any pricing/risk calculation.

    Catches obviously wrong data (negative spot, absurd rates, non-positive
    tenor) so the engine fails loudly instead of producing a meaningless number.
    Bounds are deliberately wide -- this catches data errors, not market views.
    """
    if not np.isfinite(spot) or spot <= 0:
        raise MarketInputError(f"Spot must be positive and finite, got {spot}.")
    for name, r in (("r_base", r_base), ("r_quote", r_quote)):
        if not np.isfinite(r) or not (-0.05 <= r <= 0.50):
            raise MarketInputError(
                f"{name}={r} is outside a plausible range (-5% to 50%).")
    if not np.isfinite(tau) or tau <= 0:
        raise MarketInputError(f"Tenor (tau) must be positive, got {tau}.")


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
    Fast and smooth, but UNDERESTIMATES the tail when returns are fat-tailed --
    FX returns empirically are, which is exactly why the Student-t and
    historical methods are also reported side by side (see NOTES.md).
    """
    from scipy.stats import norm

    returns = np.atleast_2d(returns)
    if returns.shape[0] < 2:
        # V3: a single observation has no estimable covariance. Return 0 rather
        # than emitting a divide-by-zero RuntimeWarning from np.cov(ddof=1).
        return 0.0
    cov = np.atleast_2d(np.cov(returns, rowvar=False))   # 1 factor -> 1x1
    port_var = float(positions @ cov @ positions)
    port_sd = np.sqrt(max(port_var, 0.0))
    z = norm.ppf(confidence)
    return z * port_sd


def var_historical(returns: np.ndarray, positions: np.ndarray,
                   confidence: float = 0.99) -> float:
    """
    Historical VaR: no distribution assumption. Reprice the book on each
    historical scenario, then read the empirical percentile of the loss side.

    Applicability (see NOTES.md): no distributional assumption, but validity
    rests on the TRAILING WINDOW being representative of the risk horizon,
    and the estimate is sensitive to WINDOW LENGTH (short = reactive but
    noisy; long = smooth but stale).
    """
    pnl = _pnl_vector(returns, positions)
    # VaR is a non-negative loss magnitude. If the loss-side percentile is itself
    # a gain (e.g. a strongly trending sample), the floor is zero: there is no
    # loss at this confidence. This is the standard convention.
    return max(-np.percentile(pnl, (1.0 - confidence) * 100.0), 0.0)


def var_montecarlo(returns: np.ndarray, positions: np.ndarray,
                   confidence: float = 0.99, n_sims: int = 100_000,
                   seed: int | None = 42, zero_drift: bool = True) -> float:
    """
    Monte Carlo VaR: estimate the covariance, draw correlated normal returns via
    a Cholesky factor, reprice, and take the percentile.

    With a normal engine this lands close to the parametric VaR; the value of MC
    is that the engine can be swapped (e.g. Student-t) to model fat tails.

    zero_drift (default True): for a 1-day VaR the historical mean return is
    statistical noise and biases the tail; standard practice sets the drift to
    zero. Set False only to study the effect of including the drift.

    Applicability (see NOTES.md): assumes the FITTED data-generating process
    (here, a multivariate normal on the sample covariance) is a good model of
    returns -- with a normal engine it inherits the same fat-tail
    understatement as the parametric VaR -- and carries simulation (sampling)
    error from a finite number of draws; `var_montecarlo_stability` reports
    that seed-to-seed error rather than assuming it away.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    mean = np.zeros(returns.shape[1]) if zero_drift else returns.mean(axis=0)
    cov = np.atleast_2d(np.cov(returns, rowvar=False))

    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_sims, len(positions)))
    sim_returns = mean + z @ chol.T
    pnl = _pnl_vector(sim_returns, positions)
    return -np.percentile(pnl, (1.0 - confidence) * 100.0)


def var_montecarlo_stability(returns: np.ndarray, positions: np.ndarray,
                             confidence: float = 0.99, n_sims: int = 50_000,
                             n_runs: int = 20) -> dict[str, float]:
    """
    A5: quantify the simulation (sampling) error of the Monte Carlo VaR.

    Re-runs the MC VaR with different seeds and reports the mean and the spread.
    A validator wants to know the VaR is stable, not a single lucky draw.
    """
    vals = [var_montecarlo(returns, positions, confidence, n_sims=n_sims,
                           seed=s) for s in range(n_runs)]
    vals = np.array(vals)
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "rel_spread": float((vals.max() - vals.min()) / vals.mean()),
    }


def expected_shortfall(returns: np.ndarray, positions: np.ndarray,
                       confidence: float = 0.99) -> float:
    """
    Expected Shortfall (ES): the AVERAGE loss in the worst tail beyond the VaR.
    Tells you how deep the bad day is, not just where it starts.

    Stability note: at high confidence the tail may contain very few points
    (e.g. ~1% of N). With fewer than ~10 tail observations the ES is unstable;
    a warning is emitted so the caller knows the estimate is noisy.
    """
    import warnings
    pnl = _pnl_vector(returns, positions)
    cutoff = np.percentile(pnl, (1.0 - confidence) * 100.0)
    tail = pnl[pnl <= cutoff]
    if len(tail) < 10:
        warnings.warn(
            f"Expected Shortfall tail has only {len(tail)} observation(s) at "
            f"{confidence:.0%} confidence; the estimate is statistically unstable. "
            f"Use a longer history or a lower confidence.", stacklevel=2)
    return -tail.mean()


# --------------------------------------------------------------------------
# Liquidity -- variation-margin cash simulation
# --------------------------------------------------------------------------
def simulate_liquidity_need(notional_quote: float, daily_vol: float,
                            horizon_days: int = 60, n_sims: int = 10_000,
                            confidence: float = 0.99,
                            seed: int | None = 42,
                            stress_multiplier: float = 1.5) -> dict[str, float]:
    """
    Simulate the cash a provider may have to post as variation margin over time.

    A position hedged with a bank is marked daily: if it moves against the
    provider, cash is posted; if it moves in favour, cash is returned. Even a
    book that nets to zero P&L at maturity can need large cash in between.

    Method (same spirit as VaR, applied to CASH instead of P&L):
      1. Simulate daily moves of the hedge value (normal innovations).
      2. Accumulate the running margin balance along each path.
      3. Take the WORST drawdown (deepest cash outflow) of each path.
      4. Report a high percentile across paths -- the liquidity buffer.

    A4: liquidity matters most precisely when volatility spikes, so a constant
    daily_vol understates the buffer in a crisis. We therefore report TWO
    figures: a base buffer at daily_vol, and a STRESSED buffer at
    daily_vol * stress_multiplier (default 1.5x). The stressed figure is the one
    a treasury function should hold.
    """
    rng = np.random.default_rng(seed)

    def _buffer(vol: float) -> tuple[float, float]:
        moves = rng.standard_normal((n_sims, horizon_days)) * notional_quote * vol
        cumulative = np.cumsum(moves, axis=1)
        worst = cumulative.min(axis=1)
        return (-np.percentile(worst, (1.0 - confidence) * 100.0),
                -worst.mean())

    base_buffer, avg_worst = _buffer(daily_vol)
    stressed_buffer, _ = _buffer(daily_vol * stress_multiplier)
    return {
        "liquidity_buffer": base_buffer,
        "stressed_buffer": stressed_buffer,
        "avg_worst_drawdown": avg_worst,
        "stress_multiplier": stress_multiplier,
    }


# --------------------------------------------------------------------------
# Stress testing -- apply real event shocks to today's book
# --------------------------------------------------------------------------
# Approximate moves of the pair during real events (quote-per-base % change).
# A3 (traceability): each figure is the approximate peak-to-trough spot move of
# the pair around the event window below. These are documented reference moves,
# not live-recomputed; a production model would recompute them from the dated
# price series. Sources: public spot history (e.g. ECB/Fed reference rates) for
# the windows shown.
# Stress scenarios: peak-to-trough spot moves during real crises.
# COMPUTED (not hand-typed) by scripts/compute_stress_scenarios.py from real
# yfinance price history of each window. Re-run that script to recalibrate.
# Each value is the worst peak-to-trough return of the pair inside the event
# window. Sources/windows are documented in STRESS_SCENARIO_SOURCES below.
STRESS_SCENARIO_SOURCES: dict[str, str] = {
    "Brexit referendum (Jun 2016)":
        "Peak-to-trough spot move, 22 Jun - 8 Jul 2016 (referendum result). "
        "GBP/USD -12.6%, EUR/USD -2.8%.",
    "COVID crash (Mar 2020)":
        "Peak-to-trough move, 20 Feb - 23 Mar 2020 (liquidity crisis). "
        "GBP/USD -12.3%, EUR/USD -6.5%.",
    "UK mini-budget (Sep 2022)":
        "Peak-to-trough move, 22-30 Sep 2022 (the 'mini-budget' that drove "
        "GBP/USD toward an all-time low). GBP/USD -4.8%, EUR/USD -2.5%.",
}

STRESS_SCENARIOS: dict[str, dict[str, float]] = {
    "Brexit referendum (Jun 2016)": {"EUR/USD": -0.0276, "GBP/USD": -0.126},
    "COVID crash (Mar 2020)":       {"EUR/USD": -0.065, "GBP/USD": -0.1225},
    "UK mini-budget (Sep 2022)":    {"EUR/USD": -0.0249, "GBP/USD": -0.0476},
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
