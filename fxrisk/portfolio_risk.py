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
delta of each forward to its pair).

Numeraire: every pair's exposure is expressed in a single common numeraire
(USD) before aggregating, so pairs that don't share a quote currency (e.g.
EUR/USD and EUR/GBP together) can sit in the same VaR. Quote-USD pairs need no
conversion; a non-USD-quoted pair is converted at the CURRENT spot of its
quote currency against USD (e.g. EUR/GBP via GBP/USD) -- a declared quanto-
style approximation, see `_factor_positions`.

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
    """
    Outcome of a Kupiec proportion-of-failures backtest.

    Two p-values are reported side by side (H1, statistical rigour):
    - p_value: the classical ASYMPTOTIC chi-squared (1 df) p-value. Its
      validity relies on the exception count being large enough for the
      chi-squared approximation to hold -- often NOT the case here (a 99%
      VaR over a ~2-year/250-day window typically has single-digit
      exceptions).
    - p_value_mc: an EXACT Monte Carlo p-value (Dufour 2006), simulated under
      the same null with the same LR statistic, valid regardless of sample
      size.
    `passed` is decided on p_value_mc (the authoritative one). Use
    `mc_agrees_with_asymptotic` to see whether the asymptotic shortcut would
    have given the same accept/reject call -- when it doesn't, trust the MC
    value, especially with few exceptions.

    Applicability (see NOTES.md): even with an exact p-value, a test with few
    exceptions has LOW POWER -- it cannot reliably tell a correctly-calibrated
    model from a moderately miscalibrated one. A "pass" here is WEAK evidence
    of correctness (the data didn't contradict the model), not proof of it.
    """
    observations: int
    exceptions: int
    expected_exceptions: float
    failure_rate: float
    lr_statistic: float
    p_value: float                            # asymptotic (chi-squared, 1 df)
    p_value_mc: float                         # Monte Carlo, exact by construction
    passed: bool                              # True if not rejected, by p_value_mc

    @property
    def mc_agrees_with_asymptotic(self) -> bool:
        """True if the asymptotic and Monte Carlo p-values agree on accept/reject at 5%."""
        return (self.p_value > 0.05) == (self.p_value_mc > 0.05)


def _factor_positions(book, spots: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """
    Map the book to per-pair spot exposures, expressed in a single common
    numeraire (USD), so the portfolio VaR can aggregate pairs that don't all
    share the same quote currency (e.g. a book of EUR/USD, GBP/USD, EUR/GBP).

    A forward long N base of a pair has spot exposure ~ N * spot, in the
    pair's own QUOTE currency.

    Fast path (unchanged from the quote-USD-only version): if a pair is
    quote-USD, that exposure IS already in USD, and is used as-is -- no
    conversion, no extra data needed.

    Common-numeraire path: a pair quoted in a non-USD currency (e.g. EUR/GBP,
    quote GBP) has its exposure -- naturally denominated in GBP -- converted
    to USD by multiplying by the CURRENT spot of "{quote}/USD" (here GBP/USD).
    `spots` must then also carry that conversion rate; in a book that already
    holds the quote currency's own USD pair (e.g. GBP/USD, as in a book of
    {EUR/USD, GBP/USD, EUR/GBP}), it comes for free from that position's own
    spot. If it isn't available, the pair genuinely cannot be priced in USD
    with the data at hand, and this fails loud rather than guessing --
    supporting a quote currency with NO USD conversion available (e.g. a
    EUR/JPY position with no JPY/USD spot) is still out of scope.

    Quanto approximation (declared, not hidden): the USD conversion uses the
    CURRENT "{quote}/USD" spot, held FIXED, rather than jointly modelling how
    that conversion rate itself co-moves with the position's own P&L. E.g. for
    EUR/GBP, this ignores the covariance between EUR/GBP and GBP/USD -- both
    move into the one USD P&L number, but only through today's fixed GBP/USD
    level, not their joint distribution. This is the standard quanto-style
    simplification and is acceptable for a spot/delta VaR at this scope, but
    it means the USD risk of a cross position is not a fully currency-hedged
    number.

    Positions on the same pair are netted. `spots` provides the current spot
    for every pair in the book, plus "{ccy}/USD" for every non-USD quote
    currency present. Returns (ordered pairs, USD positions vector aligned to
    those pairs).
    """
    exposure_quote: dict[str, float] = {}
    for p in book:
        sign = 1.0 if p.long_base else -1.0
        exposure_quote[p.pair] = (exposure_quote.get(p.pair, 0.0)
                                  + sign * p.notional_base * spots[p.pair])

    pairs = sorted(exposure_quote.keys())
    positions = np.empty(len(pairs))
    for i, pair in enumerate(pairs):
        quote_ccy = pair.split("/")[1]
        exp = exposure_quote[pair]
        if quote_ccy == "USD":
            positions[i] = exp                            # fast path: already USD
            continue
        conv_pair = f"{quote_ccy}/USD"
        if conv_pair not in spots:
            raise ValueError(
                f"Cannot express {pair}'s exposure in USD: no spot available "
                f"for '{conv_pair}'. Portfolio VaR needs a USD conversion rate "
                "for every non-USD quote currency in the book -- either the "
                f"book also holds a {conv_pair} position (its spot is reused), "
                "or that conversion rate must be supplied separately.")
        positions[i] = exp * spots[conv_pair]             # common-numeraire path
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
    sigma = returns.std(axis=0, ddof=1)   # ddof=1 to match np.cov (consistency)
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


def _kupiec_lr_stat(exceptions: int, n: int, p: float) -> float:
    """
    Kupiec proportion-of-failures (POF) likelihood-ratio statistic.

    Shared by the OBSERVED statistic and by Monte Carlo simulated draws under
    the null (see `_kupiec_mc_pvalue`), so both go through the exact same
    formula -- the point of a Monte Carlo p-value is to sidestep the
    asymptotic chi-squared approximation, not to introduce a second, possibly
    inconsistent, implementation of the statistic itself.
    """
    pi = exceptions / n if n > 0 else 0.0
    if exceptions == 0:
        return -2.0 * (n * np.log(1 - p))
    elif exceptions == n:
        return -2.0 * (n * np.log(p))
    return -2.0 * (
        (n - exceptions) * np.log(1 - p) + exceptions * np.log(p)
        - (n - exceptions) * np.log(1 - pi) - exceptions * np.log(pi)
    )


def _kupiec_mc_pvalue(lr_obs: float, n: int, p: float,
                      n_sims: int = 10_000, seed: int = 12345) -> float:
    """
    Exact Monte Carlo p-value for the Kupiec LR statistic (Dufour 2006).

    The null here is UNCONDITIONAL COVERAGE: exceptions occur i.i.d. at the
    THEORETICAL rate p = 1 - confidence, regardless of clustering. Simulates
    n_sims breach-indicator series of length n, i.i.d. Bernoulli(p), computes
    the SAME LR statistic (`_kupiec_lr_stat`) on each, and returns the
    (r+1)/(N+1)-corrected proportion of simulated statistics >= the observed
    one. Exact by construction: unlike the chi-squared asymptotic p-value, it
    does not rely on a large-sample approximation, which is unreliable when
    the exception count is small -- often single digits for a 99% VaR over a
    ~2-year/250-day window.
    """
    rng = np.random.default_rng(seed)
    exceptions_sim = rng.binomial(1, p, size=(n_sims, n)).sum(axis=1)
    lr_sims = np.array([_kupiec_lr_stat(int(e), n, p) for e in exceptions_sim])
    r = int(np.sum(lr_sims >= lr_obs))
    return (r + 1) / (n_sims + 1)


def kupiec_backtest(pnl: np.ndarray, var_series: np.ndarray,
                    confidence: float = 0.99, n_sims: int = 10_000,
                    seed: int = 12345) -> KupiecResult:
    """
    Kupiec proportion-of-failures (POF) test.

    Counts how often the realised loss exceeded the VaR and tests whether that
    frequency is statistically consistent with (1 - confidence). Reports both
    the classical asymptotic (chi-squared) p-value and an exact Monte Carlo
    p-value (see `_kupiec_mc_pvalue`); `passed` is decided on the Monte Carlo
    one. See KupiecResult and NOTES.md for the applicability conditions of
    this test.

    pnl:        realised P&L per period (losses negative).
    var_series: the VaR (positive loss magnitude) for each period.
    n_sims:     Monte Carlo draws for the exact p-value (default 10,000).
    seed:       fixed seed, for reproducibility.
    """
    pnl = np.asarray(pnl, dtype=float)
    var_series = np.asarray(var_series, dtype=float)
    n = len(pnl)
    exceptions = int(np.sum(-pnl > var_series))      # loss beyond VaR
    p = 1.0 - confidence
    expected = n * p
    pi = exceptions / n if n > 0 else 0.0

    lr = _kupiec_lr_stat(exceptions, n, p)

    from scipy.stats import chi2
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    p_value_mc = _kupiec_mc_pvalue(lr, n, p, n_sims=n_sims, seed=seed)
    return KupiecResult(
        observations=n, exceptions=exceptions, expected_exceptions=float(expected),
        failure_rate=float(pi), lr_statistic=float(lr),
        p_value=p_value, p_value_mc=p_value_mc,
        passed=bool(p_value_mc > 0.05),
    )


def rolling_backtest(returns: np.ndarray, positions: np.ndarray,
                     confidence: float = 0.99, window: int = 250,
                     n_sims: int = 10_000, seed: int = 12345) -> KupiecResult:
    """
    Proper rolling (out-of-sample) backtest of the historical VaR.

    Unlike kupiec_backtest with a constant VaR, this re-estimates the VaR each
    day from a trailing window and tests it against the NEXT day's realised P&L
    -- exactly how a VaR model is validated in production. For each day t beyond
    the initial window, the VaR is computed from days [t-window, t) and compared
    to the P&L on day t. The exception count then feeds the same Kupiec POF test
    (asymptotic AND Monte Carlo p-values, see `kupiec_backtest`).

    Estimation-risk caveat (declared, not corrected): this backtest treats
    each day's rolling VaR as if it were the TRUE model, but it is itself
    estimated from a finite trailing window. Standard Kupiec/MC backtests do
    not correct for that extra estimation uncertainty, so a 'pass' here is
    evidence the ESTIMATED VaR process performed adequately out-of-sample, not
    a guarantee about the underlying true model.

    returns:   (n_days, n_factors) historical returns.
    positions: (n_factors,) exposures.
    window:    trailing window length used to estimate each day's VaR (e.g. 250).
    n_sims:    Monte Carlo draws for the exact p-value (default 10,000).
    seed:      fixed seed, for reproducibility.
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

    # Kupiec POF test on the out-of-sample exceptions (same code path as
    # kupiec_backtest: _kupiec_lr_stat / _kupiec_mc_pvalue).
    p = 1.0 - confidence
    expected = tested * p
    pi = exceptions / tested if tested > 0 else 0.0
    lr = _kupiec_lr_stat(exceptions, tested, p)

    from scipy.stats import chi2
    p_value = float(1.0 - chi2.cdf(lr, df=1))
    p_value_mc = _kupiec_mc_pvalue(lr, tested, p, n_sims=n_sims, seed=seed)
    return KupiecResult(
        observations=tested, exceptions=exceptions, expected_exceptions=float(expected),
        failure_rate=float(pi), lr_statistic=float(lr),
        p_value=p_value, p_value_mc=p_value_mc,
        passed=bool(p_value_mc > 0.05),
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
    # The stressed VaR is, by definition, the loss in the WORST regime; it can
    # never be milder than the all-sample VaR. With homogeneous data the most
    # volatile window can, by chance, give a slightly lower figure -- an artefact.
    # We floor the stressed VaR at the normal VaR so it is never misleadingly
    # smaller than the everyday number.
    stressed = max(stressed, normal)
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


def _fit_student_t(pnl: np.ndarray) -> tuple[float, float, float]:
    """
    MLE-fit a Student-t (nu, loc, scale) to portfolio P&L, all three parameters
    free.

    A tempting shortcut is to pin loc/scale to the sample mean/std
    (`student_t.fit(pnl, floc=mu, fscale=sigma)`) and only fit `nu`. That is
    wrong: a Student-t with scale `s` has standard deviation
    `s * sqrt(nu / (nu - 2))`, not `s`. Forcing scale = sample std makes the
    optimizer compensate by inflating `nu` -- on data drawn from a true
    nu=5 t-distribution, that shortcut recovers nu≈16.5, badly biased toward
    thin tails. The bias then compounds: `student_t.ppf(df=nu)` returns the
    *raw* t-quantile, whose variance is `nu/(nu-2)` (not 1), so multiplying it
    by sigma double-applies the tail-thickness adjustment and further
    overstates the quantile. The two errors happen to roughly cancel in the
    final VaR number, but that is luck, not design, and the fitted `nu` is
    unusable on its own. Fitting nu/loc/scale jointly avoids both.
    """
    from scipy.stats import t as student_t
    nu, loc, scale = student_t.fit(pnl)
    nu = max(nu, 2.1)                                     # keep variance finite
    return nu, loc, scale


def var_student_t(returns: np.ndarray, positions: np.ndarray,
                  confidence: float = 0.99) -> float:
    """
    B1: parametric VaR with a Student-t quantile to capture fat tails.

    The normal VaR underestimates extreme losses because FX returns are
    fat-tailed. This fits a Student-t distribution to the portfolio P&L, with
    degrees of freedom, location and scale all free (see `_fit_student_t` for
    why a naive floc/fscale fit is wrong), and reads the VaR quantile directly
    off the fitted distribution.
    """
    from scipy.stats import t as student_t
    pnl = (np.asarray(returns, dtype=float).reshape(len(returns), -1) @ positions)
    try:
        nu, loc, scale = _fit_student_t(pnl)
        var = -student_t.ppf(1.0 - confidence, df=nu, loc=loc, scale=scale)
    except Exception:
        # Declared fallback: if the MLE fit fails to converge (e.g. too few
        # points, or constant/degenerate data), fall back to a normal VaR on
        # the sample mean/std. Thinner-tailed than a proper Student-t, but
        # always well-defined.
        from scipy.stats import norm
        mu, sigma = float(np.mean(pnl)), float(np.std(pnl, ddof=1))
        var = -(mu + sigma * norm.ppf(1.0 - confidence))
    return max(var, 0.0)                                  # VaR is non-negative


def _safe_log(x: float) -> float:
    return np.log(x) if x > 0 else 0.0


def _christoffersen_lr_stat(hits: np.ndarray) -> tuple[float, float, float, float]:
    """
    Christoffersen independence likelihood-ratio statistic, computed from a
    0/1 hit-indicator series (transition counts n00/n01/n10/n11 between
    consecutive days, then LR of the conditional vs unconditional exception
    model). Shared by the OBSERVED statistic and by Monte Carlo simulated
    draws under the null (see `christoffersen_independence`), so both go
    through the exact same formula.

    Returns (lr, pi01, pi11, pi): the LR statistic and the three transition
    probabilities (P(exception | no exception yesterday), P(exception |
    exception yesterday), and the unconditional exception rate).
    """
    hits = np.asarray(hits)
    prev, cur = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total if total > 0 else 0.0

    ll_uncond = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_cond = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
               + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))
    lr = -2.0 * (ll_uncond - ll_cond)
    return float(lr), pi01, pi11, pi


def christoffersen_independence(pnl: np.ndarray, var_series: np.ndarray,
                                n_sims: int = 10_000, seed: int = 12345
                                ) -> dict[str, float]:
    """
    B5: Christoffersen test of INDEPENDENCE of VaR exceptions.

    Kupiec checks how MANY exceptions occur; Christoffersen checks whether they
    CLUSTER (an exception today making one tomorrow more likely), which signals a
    model that does not react to changing risk. Tests the transition
    probabilities of the exception indicator with a likelihood-ratio statistic.
    Reports both the asymptotic (chi-squared) and an exact Monte Carlo
    p-value; `independent` is decided on the Monte Carlo one.

    Applicability (see NOTES.md): this is a FIRST-ORDER MARKOV test -- it only
    checks whether an exception today predicts one tomorrow. It is blind to
    higher-order or longer-range clustering (e.g. exceptions bunching within
    a week without consecutive-day repeats). A "pass" rules out simple
    day-to-day clustering, not every form of it.

    Null-specification note (H1): the null here is ONLY that exceptions are
    serially independent -- their RATE is unrestricted (it is not also being
    tested against the theoretical VaR miss rate p; that is Kupiec's job).
    The Monte Carlo simulation therefore draws i.i.d. Bernoulli at the
    OBSERVED breach rate pi_hat, NOT at the theoretical p = 1 - confidence.
    Simulating at p would silently test a different, joint (coverage +
    independence) null and misstate this test's own p-value.
    """
    from scipy.stats import chi2
    pnl = np.asarray(pnl, dtype=float)
    var_series = np.asarray(var_series, dtype=float)
    hits = (-pnl > var_series).astype(int)              # 1 = exception
    n = len(hits)

    lr, pi01, pi11, pi = _christoffersen_lr_stat(hits)
    p_value = float(1.0 - chi2.cdf(lr, df=1))

    pi_hat = float(np.mean(hits)) if n > 0 else 0.0
    rng = np.random.default_rng(seed)
    sims = rng.binomial(1, pi_hat, size=(n_sims, n))
    lr_sims = np.array([_christoffersen_lr_stat(row)[0] for row in sims])
    r = int(np.sum(lr_sims >= lr))
    p_value_mc = (r + 1) / (n_sims + 1)

    return {
        "lr_statistic": float(lr), "p_value": p_value, "p_value_mc": p_value_mc,
        "independent": bool(p_value_mc > 0.05),
        "mc_agrees_with_asymptotic": (p_value > 0.05) == (p_value_mc > 0.05),
        "clustering_ratio_pi11_vs_pi01": float(pi11 / pi01) if pi01 > 0 else float("nan"),
    }
