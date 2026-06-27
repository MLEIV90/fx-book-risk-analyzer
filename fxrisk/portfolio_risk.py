"""
fxrisk.portfolio_risk
=====================
Market-risk measurement over the whole book: VaR (three methods), Expected
Shortfall, Kupiec backtesting and risk attribution.

This layer maps the book onto market factors and reuses the tested primitives in
fxrisk.risk. It is the answer to the committee's two questions: "how much can we
lose?" (VaR / ES) and "is that number trustworthy?" (Kupiec), plus "where does
the risk come from?" (attribution).

Scope, declared: risk is measured on the forward positions' spot exposure (the
delta of each forward to its pair). Options, if any, are valued separately and
do NOT enter the portfolio VaR in this version.

Pure functions where possible: VaR/ES/Kupiec/attribution take returns and
positions as arrays and can be unit-tested without the network.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fxrisk.risk import (
    var_parametric, var_historical, var_montecarlo, expected_shortfall,
)


@dataclass
class VaRReport:
    """
    Committee-ready market-risk summary for the book.

    Scope (declared): this is a SPOT VaR. It measures the risk of exchange-rate
    moves, treating each forward by its spot (delta) exposure. Interest-rate risk
    is measured separately via DV01 (see fxrisk.book_risk); the two are NOT added
    into a single number, which would require a joint spot-rate covariance matrix
    (out of scope). All VaR figures are 1-day unless the 10-day fields are used.
    """
    confidence: float
    var_parametric: float
    var_historical: float
    var_montecarlo: float
    expected_shortfall: float
    factors: list[str]                       # the pairs driving the risk
    risk_contribution: dict                   # pair -> % of portfolio variance
    diversification_benefit: float            # 1 - portfolio VaR / sum standalone
    standalone_var: dict                      # pair -> standalone VaR
    horizon_days: int = 1                     # horizon of the figures above

    # 10-day regulatory horizon, scaled by the square-root-of-time rule.
    # VaR_10d = VaR_1d * sqrt(10). Assumes i.i.d. returns -- the Basel
    # convention, and a declared simplification.
    @property
    def var_historical_10d(self) -> float:
        return self.var_historical * np.sqrt(10)

    @property
    def expected_shortfall_10d(self) -> float:
        return self.expected_shortfall * np.sqrt(10)


@dataclass
class KupiecResult:
    """Outcome of a Kupiec proportion-of-failures backtest."""
    observations: int
    exceptions: int
    expected_exceptions: float
    failure_rate: float
    lr_statistic: float
    p_value: float
    passed: bool                              # True if the model is not rejected


def _factor_positions(book, spots: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """
    Map the book to per-pair spot exposures (quote-currency notionals).

    A forward long N base of a pair has spot exposure ~ N * spot in quote terms.
    Positions on the same pair are netted. `spots` provides the current spot per
    pair. Returns (ordered pairs, positions vector aligned to those pairs).
    """
    # All pair exposures are summed in quote-currency terms into one covariance
    # and one VaR. That is only valid when every pair shares the same quote
    # currency (USD here). A non-USD-quoted pair (e.g. EUR/JPY) would mix P&L in
    # different currencies without conversion, silently corrupting the VaR. We
    # fail loudly rather than return a wrong number -- supporting mixed quote
    # currencies would require converting each leg to a common numeraire first.
    non_usd = sorted({p.pair for p in book if p.quote_ccy != "USD"})
    if non_usd:
        raise ValueError(
            "Portfolio VaR currently assumes all pairs are quote-USD; found "
            f"non-USD-quoted pair(s): {', '.join(non_usd)}. Converting each pair's "
            "P&L to a common numeraire is out of scope in this version.")

    exposure: dict[str, float] = {}
    for p in book:
        sign = 1.0 if p.long_base else -1.0
        exposure[p.pair] = exposure.get(p.pair, 0.0) + sign * p.notional_base * spots[p.pair]
    pairs = sorted(exposure.keys())
    positions = np.array([exposure[pair] for pair in pairs])
    return pairs, positions


def portfolio_var(returns: np.ndarray, positions: np.ndarray,
                  confidence: float = 0.99, factors: list[str] | None = None
                  ) -> VaRReport:
    """
    Full VaR report from a returns matrix and a positions vector.

    returns:   (n_days, n_factors) historical returns, columns aligned to factors.
    positions: (n_factors,) signed quote-currency exposures.
    """
    factors = factors or [f"f{i}" for i in range(len(positions))]

    # Ensure returns is 2D (n_days, n_factors) so a single-factor book works.
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)

    vp = var_parametric(returns, positions, confidence)
    vh = var_historical(returns, positions, confidence)
    vm = var_montecarlo(returns, positions, confidence, n_sims=50_000)
    es = expected_shortfall(returns, positions, confidence)

    # Risk attribution via the covariance: contribution of each factor to the
    # portfolio variance (component contributions sum to total variance).
    # Note: this attribution is variance-based (it assumes the parametric/normal
    # view), while the headline VaR may be the historical one. The two agree
    # exactly only under normality; the attribution is read as an approximate
    # decomposition of where the risk concentrates.
    cov = np.cov(returns, rowvar=False)
    cov = np.atleast_2d(cov)                         # 1 factor -> 1x1, not scalar
    port_var = float(positions @ cov @ positions)
    marginal = cov @ positions                       # d(variance)/d(position)
    contrib = positions * marginal                   # component contributions
    contribution = {factors[i]: float(contrib[i] / port_var * 100.0)
                    for i in range(len(factors))} if port_var > 0 else {}

    # Standalone VaR per factor and diversification benefit.
    from scipy.stats import norm
    z = norm.ppf(confidence)
    sigma = returns.std(axis=0)
    standalone = {factors[i]: float(z * abs(positions[i]) * sigma[i])
                  for i in range(len(factors))}
    sum_standalone = sum(standalone.values()) or 1.0
    diversification = 1.0 - vp / sum_standalone

    return VaRReport(
        confidence=confidence, var_parametric=vp, var_historical=vh,
        var_montecarlo=vm, expected_shortfall=es, factors=factors,
        risk_contribution=contribution, diversification_benefit=diversification,
        standalone_var=standalone,
    )


def kupiec_backtest(pnl: np.ndarray, var_series: np.ndarray,
                    confidence: float = 0.99) -> KupiecResult:
    """
    Kupiec proportion-of-failures (POF) test.

    Counts how often the realised loss exceeded the VaR and tests whether that
    frequency is statistically consistent with (1 - confidence). A model that is
    not rejected (p_value above 5%) is considered adequate.

    pnl:        realised P&L per period (losses negative).
    var_series: the VaR (positive loss magnitude) for each period.
    """
    pnl = np.asarray(pnl, dtype=float)
    var_series = np.asarray(var_series, dtype=float)
    n = len(pnl)
    exceptions = int(np.sum(-pnl > var_series))      # loss beyond VaR
    p = 1.0 - confidence
    expected = n * p
    pi = exceptions / n if n > 0 else 0.0

    # Likelihood-ratio statistic for the POF test.
    if exceptions == 0:
        lr = -2.0 * (n * np.log(1 - p))
    elif exceptions == n:
        lr = -2.0 * (n * np.log(p))
    else:
        lr = -2.0 * (
            (n - exceptions) * np.log(1 - p) + exceptions * np.log(p)
            - (n - exceptions) * np.log(1 - pi) - exceptions * np.log(pi)
        )

    from scipy.stats import chi2
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return KupiecResult(
        observations=n, exceptions=exceptions, expected_exceptions=float(expected),
        failure_rate=float(pi), lr_statistic=float(lr), p_value=p_value,
        passed=bool(p_value > 0.05),
    )


def rolling_backtest(returns: np.ndarray, positions: np.ndarray,
                     confidence: float = 0.99, window: int = 250) -> KupiecResult:
    """
    Proper rolling (out-of-sample) backtest of the historical VaR.

    Unlike kupiec_backtest with a constant VaR, this re-estimates the VaR each
    day from a trailing window and tests it against the NEXT day's realised P&L
    -- exactly how a VaR model is validated in production. For each day t beyond
    the initial window, the VaR is computed from days [t-window, t) and compared
    to the P&L on day t. The exception count then feeds the same Kupiec POF test.

    returns:   (n_days, n_factors) historical returns.
    positions: (n_factors,) exposures.
    window:    trailing window length used to estimate each day's VaR (e.g. 250).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n_days = returns.shape[0]
    if n_days <= window + 1:
        raise ValueError("Not enough history for a rolling backtest with this window.")

    pnl_series = returns @ positions          # realised daily P&L
    exceptions = 0
    tested = 0
    for t in range(window, n_days):
        train = returns[t - window:t]         # data strictly before day t
        var_t = var_historical(train, positions, confidence)
        if -pnl_series[t] > var_t:            # next-day loss beyond the VaR
            exceptions += 1
        tested += 1

    # Kupiec POF test on the out-of-sample exceptions.
    p = 1.0 - confidence
    expected = tested * p
    pi = exceptions / tested if tested > 0 else 0.0
    if exceptions == 0:
        lr = -2.0 * (tested * np.log(1 - p))
    elif exceptions == tested:
        lr = -2.0 * (tested * np.log(p))
    else:
        lr = -2.0 * (
            (tested - exceptions) * np.log(1 - p) + exceptions * np.log(p)
            - (tested - exceptions) * np.log(1 - pi) - exceptions * np.log(pi)
        )
    from scipy.stats import chi2
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return KupiecResult(
        observations=tested, exceptions=exceptions, expected_exceptions=float(expected),
        failure_rate=float(pi), lr_statistic=float(lr), p_value=p_value,
        passed=bool(p_value > 0.05),
    )


def stressed_var(returns: np.ndarray, positions: np.ndarray,
                 confidence: float = 0.99, window: int = 250) -> dict:
    """
    Stressed VaR: the historical VaR recalibrated to the most volatile window in
    the available history (the Basel approach uses a stress period such as 2008).

    Here we locate the trailing window of length `window` with the highest
    portfolio P&L volatility -- the worst stress period present in the data --
    and compute the historical VaR over it. This answers 'how much would we lose
    if markets behaved like their worst observed regime', rather than like the
    recent (calm) past.

    Returns the stressed VaR, the normal-period VaR, and the ratio between them.
    Declared limit: the stress period is the worst in the *available* free
    history (~2 years); a full implementation would use a fixed crisis window
    (e.g. 2008-2009) from a longer dataset.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n_days = returns.shape[0]
    pnl = returns @ positions

    if n_days <= window:
        stressed = var_historical(returns, positions, confidence)
    else:
        # Find the window with the highest P&L volatility.
        worst_vol, worst_start = -1.0, 0
        for start in range(0, n_days - window + 1, 5):     # step 5 for speed
            vol = float(np.std(pnl[start:start + window]))
            if vol > worst_vol:
                worst_vol, worst_start = vol, start
        stress_returns = returns[worst_start:worst_start + window]
        stressed = var_historical(stress_returns, positions, confidence)

    normal = var_historical(returns, positions, confidence)
    ratio = stressed / normal if normal > 0 else float("nan")
    return {"stressed_var": stressed, "normal_var": normal, "ratio": ratio}


def var_ewma(returns: np.ndarray, positions: np.ndarray,
             confidence: float = 0.99, lam: float = 0.94) -> float:
    """
    B2: parametric VaR using an EWMA (RiskMetrics) covariance.

    A plain sample covariance weights a two-year-old day the same as yesterday.
    EWMA weights recent observations more (decay lambda, 0.94 is the RiskMetrics
    daily standard), so the VaR reacts faster to the current regime.
    """
    from scipy.stats import norm
    returns = np.asarray(returns, dtype=float)
    if returns.ndim == 1:
        returns = returns.reshape(-1, 1)
    n = returns.shape[0]
    weights = lam ** np.arange(n - 1, -1, -1)            # most recent = highest
    weights /= weights.sum()
    demeaned = returns - np.average(returns, axis=0, weights=weights)
    cov = (demeaned * weights[:, None]).T @ demeaned     # weighted covariance
    cov = np.atleast_2d(cov)
    port_sd = np.sqrt(float(positions @ cov @ positions))
    return norm.ppf(confidence) * port_sd


def var_student_t(returns: np.ndarray, positions: np.ndarray,
                  confidence: float = 0.99) -> float:
    """
    B1: parametric VaR with a Student-t quantile to capture fat tails.

    The normal VaR underestimates extreme losses because FX returns are
    fat-tailed. This fits the degrees of freedom of the portfolio P&L and uses
    the t-quantile instead of the normal z-score, giving a heavier tail.
    """
    from scipy.stats import t as student_t
    pnl = (np.asarray(returns, dtype=float).reshape(len(returns), -1) @ positions)
    mu, sigma = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
    # Fit degrees of freedom; guard against too-few points.
    try:
        nu, _, _ = student_t.fit(pnl, floc=mu, fscale=sigma)
        nu = max(nu, 3.0)                                # keep variance finite
    except Exception:
        nu = 5.0
    q = student_t.ppf(1.0 - confidence, df=nu)           # negative tail quantile
    return -(mu + sigma * q)


def christoffersen_independence(pnl: np.ndarray, var_series: np.ndarray
                                ) -> dict[str, float]:
    """
    B5: Christoffersen test of INDEPENDENCE of VaR exceptions.

    Kupiec checks how MANY exceptions occur; Christoffersen checks whether they
    CLUSTER (an exception today making one tomorrow more likely), which signals a
    model that does not react to changing risk. Tests the transition
    probabilities of the exception indicator with a likelihood-ratio statistic.
    Returns the LR statistic, its p-value (chi-square, 1 df), and whether
    independence is NOT rejected (p > 0.05 = good, no clustering).
    """
    from scipy.stats import chi2
    pnl = np.asarray(pnl, dtype=float)
    var_series = np.asarray(var_series, dtype=float)
    hits = (-pnl > var_series).astype(int)              # 1 = exception

    # Count transitions between consecutive days.
    n00 = n01 = n10 = n11 = 0
    for prev, cur in zip(hits[:-1], hits[1:]):
        if prev == 0 and cur == 0: n00 += 1
        elif prev == 0 and cur == 1: n01 += 1
        elif prev == 1 and cur == 0: n10 += 1
        else: n11 += 1

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    # Likelihood ratio for independence.
    def _safe_log(x): return np.log(x) if x > 0 else 0.0
    ll_uncond = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_cond = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
               + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))
    lr = -2.0 * (ll_uncond - ll_cond)
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    return {
        "lr_statistic": float(lr), "p_value": p_value,
        "independent": bool(p_value > 0.05),
        "clustering_ratio_pi11_vs_pi01": float(pi11 / pi01) if pi01 > 0 else float("nan"),
    }
