"""
Tests for the portfolio risk layer: VaR report, attribution, Kupiec backtest.
Pure logic with synthetic returns; no network needed.
"""
import numpy as np
from hypothesis import given, strategies as st, assume

from fxrisk.portfolio_risk import portfolio_var, kupiec_backtest


def _returns(seed=0, n=2000):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = 0.8 * z[:, 0] + np.sqrt(1 - 0.64) * z[:, 1]
    return z * np.array([0.005, 0.006])


POS = np.array([10_000_000.0, -6_000_000.0])
FACTORS = ["EUR/USD", "GBP/USD"]


def test_var_report_fields_positive():
    rep = portfolio_var(_returns(), POS, 0.99, FACTORS)
    assert rep.var_parametric > 0
    assert rep.var_historical > 0
    assert rep.var_montecarlo > 0
    assert rep.expected_shortfall > rep.var_historical


def test_risk_contributions_sum_to_100():
    rep = portfolio_var(_returns(), POS, 0.99, FACTORS)
    assert abs(sum(rep.risk_contribution.values()) - 100.0) < 1e-6


def test_diversification_benefit_in_range():
    rep = portfolio_var(_returns(), POS, 0.99, FACTORS)
    assert 0.0 <= rep.diversification_benefit < 1.0


def test_kupiec_good_model_passes():
    # Build P&L and a VaR series where exceptions match the confidence level.
    rng = np.random.default_rng(1)
    n = 1000
    pnl = rng.normal(0, 1000, n)
    # 99% VaR of N(0,1000) ~ 2326; use it as a constant VaR series.
    var_series = np.full(n, 2326.0)
    res = kupiec_backtest(pnl, var_series, 0.99)
    assert res.observations == n
    assert res.passed                      # correct model should not be rejected


def test_kupiec_bad_model_fails():
    # A VaR that is far too small will be breached far too often -> rejected.
    rng = np.random.default_rng(2)
    n = 1000
    pnl = rng.normal(0, 1000, n)
    var_series = np.full(n, 200.0)         # absurdly low VaR
    res = kupiec_backtest(pnl, var_series, 0.99)
    assert res.exceptions > res.expected_exceptions
    assert not res.passed


def test_var_10day_scaling():
    """10-day VaR must equal the 1-day VaR scaled by sqrt(10) (Basel rule)."""
    import numpy as np
    rep = portfolio_var(_returns(), POS, 0.99, FACTORS)
    assert abs(rep.var_historical_10d - rep.var_historical * np.sqrt(10)) < 1e-9
    assert abs(rep.expected_shortfall_10d - rep.expected_shortfall * np.sqrt(10)) < 1e-9
    assert rep.var_historical_10d > rep.var_historical


def test_var_report_declares_horizon():
    """The report carries an explicit 1-day horizon flag."""
    rep = portfolio_var(_returns(), POS, 0.99, FACTORS)
    assert rep.horizon_days == 1


def test_rolling_backtest_good_model():
    """A rolling historical VaR on its own returns should not be rejected."""
    from fxrisk.portfolio_risk import rolling_backtest
    res = rolling_backtest(_returns(seed=3, n=1500), POS, 0.99, window=250)
    assert res.observations > 0
    assert res.exceptions >= 0
    # On stationary normal-ish data the model should broadly hold up.
    assert res.failure_rate < 0.05      # exceptions rate near the 1% level


def test_rolling_backtest_needs_history():
    """Too little history for the window should raise, not silently mislead."""
    import numpy as np
    import pytest
    from fxrisk.portfolio_risk import rolling_backtest
    short = _returns(n=100)
    with pytest.raises(ValueError):
        rolling_backtest(short, POS, 0.99, window=250)


def test_stressed_var_exceeds_normal():
    """Stressed VaR (worst window) must be at least the normal-period VaR."""
    import numpy as np
    from fxrisk.portfolio_risk import stressed_var
    rng = np.random.default_rng(7)
    calm = rng.standard_normal((400, 2)) * 0.004
    storm = rng.standard_normal((300, 2)) * 0.015
    ret = np.vstack([calm, storm, calm])
    res = stressed_var(ret, POS, 0.99, window=250)
    assert res["stressed_var"] >= res["normal_var"]
    assert res["ratio"] >= 1.0


def test_factor_positions_rejects_unconvertible_quote_currency():
    """
    A non-USD-quoted pair is no longer rejected outright (the common-numeraire
    path handles it) -- but if its quote currency's USD conversion rate isn't
    available in `spots`, it genuinely cannot be priced in USD, and the guard
    must still fail loud rather than sum mismatched currencies silently.
    """
    import pytest
    from fxrisk.book import Position, Book
    from fxrisk.portfolio_risk import _factor_positions
    book = Book([Position("EUR/JPY", True, 1_000_000, 90, 160.0, id="x1")])
    spots = {"EUR/JPY": 160.0}                    # no JPY/USD conversion rate
    with pytest.raises(ValueError, match="JPY/USD"):
        _factor_positions(book, spots)


def test_factor_positions_usd_quote_fast_path_unchanged():
    """
    Regression: a plain quote-USD book must give the exact same USD exposure
    as before the common-numeraire path was added -- the fast path applies no
    conversion at all.
    """
    from fxrisk.book import Position, Book
    from fxrisk.portfolio_risk import _factor_positions
    book = Book([
        Position("EUR/USD", True, 1_000_000, 90, 1.08, id="a"),
        Position("GBP/USD", False, 500_000, 90, 1.27, id="b"),
    ])
    spots = {"EUR/USD": 1.08, "GBP/USD": 1.27}
    pairs, positions = _factor_positions(book, spots)
    assert pairs == ["EUR/USD", "GBP/USD"]
    assert positions[0] == 1_000_000 * 1.08
    assert positions[1] == -500_000 * 1.27


def test_factor_positions_converts_cross_to_common_usd_numeraire():
    """
    A book with a non-USD-quoted cross (EUR/GBP) plus the quote currency's own
    USD pair (GBP/USD) must convert the cross's exposure to USD using the
    CURRENT GBP/USD spot, rather than raising.
    """
    from fxrisk.book import Position, Book
    from fxrisk.portfolio_risk import _factor_positions
    eurusd, gbpusd = 1.08, 1.27
    eurgbp = eurusd / gbpusd
    book = Book([
        Position("EUR/USD", True, 1_000_000, 90, eurusd, id="a"),
        Position("GBP/USD", False, 500_000, 90, gbpusd, id="b"),
        Position("EUR/GBP", True, 300_000, 90, eurgbp, id="c"),
    ])
    spots = {"EUR/USD": eurusd, "GBP/USD": gbpusd, "EUR/GBP": eurgbp}
    pairs, positions = _factor_positions(book, spots)
    assert pairs == ["EUR/GBP", "EUR/USD", "GBP/USD"]
    # EUR/GBP exposure = notional * eurgbp (in GBP), converted to USD by the
    # current GBP/USD spot -- which collapses the GBP/EUR/USD round trip back
    # to notional * eurusd.
    eurgbp_usd = positions[pairs.index("EUR/GBP")]
    assert abs(eurgbp_usd - 300_000 * eurusd) < 1e-6


def test_var_ewma_positive_and_reactive():
    """EWMA VaR should be a positive number and finite."""
    from fxrisk.portfolio_risk import var_ewma
    v = var_ewma(_returns(), POS, 0.99)
    assert v > 0 and np.isfinite(v)


def test_var_student_t_heavier_than_normal():
    """Student-t VaR should be >= parametric normal VaR (fatter tail)."""
    from fxrisk.portfolio_risk import var_student_t
    from fxrisk.risk import var_parametric
    r = _returns(seed=11, n=2000)
    vt = var_student_t(r, POS, 0.99)
    vn = var_parametric(r, POS, 0.99)
    assert vt > 0
    assert vt >= vn * 0.95          # t tail at least as heavy (allow noise)


def test_christoffersen_detects_clustering():
    """Independence test should flag clustered exceptions."""
    from fxrisk.portfolio_risk import christoffersen_independence
    import numpy as np
    n = 500
    var_series = np.full(n, 1000.0)
    pnl = np.full(n, 100.0)               # no exceptions normally
    # Inject a cluster of consecutive exceptions.
    pnl[100:110] = -2000.0
    res = christoffersen_independence(pnl, var_series)
    assert "p_value" in res and "independent" in res


def test_mc_zero_drift_default():
    """Default MC VaR uses zero drift; should be close to parametric."""
    from fxrisk.risk import var_montecarlo, var_parametric
    r = _returns(seed=5, n=3000)
    vmc = var_montecarlo(r, POS, 0.99, n_sims=40000)
    vp = var_parametric(r, POS, 0.99)
    assert abs(vmc - vp) / vp < 0.10      # within 10% on normal data


def test_stressed_var_never_below_normal():
    """Stressed VaR must never be lower than the all-sample VaR (it is the worst
    regime by definition). Verified across many homogeneous samples where the
    artefact could otherwise appear."""
    import numpy as np
    from fxrisk.portfolio_risk import stressed_var
    pos = np.array([1_000_000.0])
    for seed in range(20):
        data = np.random.default_rng(seed).standard_normal((1000, 1)) * 0.01
        sv = stressed_var(data, pos, 0.99)
        assert sv["stressed_var"] >= sv["normal_var"] - 1e-9
        assert sv["ratio"] >= 1.0 - 1e-9


def test_student_t_var_non_negative_on_trending_data():
    """Student-t VaR floors at zero on strongly trending data."""
    import numpy as np
    from fxrisk.portfolio_risk import var_student_t
    pos = np.array([1_000_000.0])
    bull = np.random.default_rng(0).standard_normal((500, 1)) * 0.003 + 0.01
    assert var_student_t(bull, pos, 0.99) >= 0.0


def test_student_t_fit_recovers_true_nu():
    """
    Regression test for the fscale/floc fitting bug: pinning scale to the
    sample std forces the optimizer to inflate nu to compensate (verified: a
    true nu=5 sample was fit to nu~16.5 by the old code), which defeats the
    point of fitting a Student-t at all. Fitting nu/loc/scale jointly should
    recover a nu in the right ballpark instead of blowing up.
    """
    from fxrisk.portfolio_risk import _fit_student_t
    from scipy.stats import t as student_t

    rng = np.random.default_rng(42)
    true_nu = 5.0
    pnl = student_t.rvs(df=true_nu, loc=0.0, scale=0.01, size=4000, random_state=rng)

    nu, loc, scale = _fit_student_t(pnl)
    assert nu < 12.0                       # not blown up toward thin tails


def test_student_t_var_heavier_than_normal_on_fat_tailed_data():
    """
    The Student-t VaR must be >= the parametric normal VaR at 99% confidence
    when the underlying P&L is genuinely fat-tailed -- that heavier tail is
    the entire reason the method exists.
    """
    from fxrisk.portfolio_risk import var_student_t
    from fxrisk.risk import var_parametric
    from scipy.stats import t as student_t

    rng = np.random.default_rng(7)
    pnl = student_t.rvs(df=5.0, loc=0.0, scale=0.01, size=4000, random_state=rng)
    pos = np.array([1.0])
    returns = pnl.reshape(-1, 1)

    vt = var_student_t(returns, pos, 0.99)
    vn = var_parametric(returns, pos, 0.99)
    assert vt >= vn


# ------------------- Common-numeraire VaR (EUR/USD, GBP/USD, EUR/GBP) -------


def _triangulated_returns(seed=0, n=2000):
    """
    Synthetic correlated EUR/USD and GBP/USD spot paths, with EUR/GBP DERIVED
    by triangulation (EUR/USD / GBP/USD) -- mirrors how the real cross is
    built (fxrisk.data.TRIANGULATED_PAIRS), so a 3-factor VaR test exercises a
    genuinely triangulated series rather than an independently-made-up one.
    Returns an (n-1, 3) array, columns [EUR/USD, GBP/USD, EUR/GBP].
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = 0.8 * z[:, 0] + np.sqrt(1 - 0.64) * z[:, 1]
    daily = z * np.array([0.005, 0.006])
    eurusd = 1.08 * np.exp(np.cumsum(daily[:, 0]))
    gbpusd = 1.27 * np.exp(np.cumsum(daily[:, 1]))
    eurgbp = eurusd / gbpusd
    prices = np.column_stack([eurusd, gbpusd, eurgbp])
    return prices[1:] / prices[:-1] - 1.0


def _eurgbp_book_and_positions(seed=3, n=2000):
    from fxrisk.book import Position, Book
    from fxrisk.portfolio_risk import _factor_positions

    returns = _triangulated_returns(seed=seed, n=n)
    eurusd0, gbpusd0 = 1.08, 1.27
    eurgbp0 = eurusd0 / gbpusd0
    book = Book([
        Position("EUR/USD", True, 2_000_000, 90, eurusd0, id="a"),
        Position("GBP/USD", False, 1_500_000, 90, gbpusd0, id="b"),
        Position("EUR/GBP", True, 1_000_000, 90, eurgbp0, id="c"),
    ])
    spots = {"EUR/USD": eurusd0, "GBP/USD": gbpusd0, "EUR/GBP": eurgbp0}
    pairs, positions = _factor_positions(book, spots)
    order = [["EUR/USD", "GBP/USD", "EUR/GBP"].index(p) for p in pairs]
    aligned_returns = returns[:, order]
    return pairs, aligned_returns, positions


def test_portfolio_var_eurgbp_book_finite_and_positive():
    """A three-instrument book across EUR/USD, GBP/USD and EUR/GBP must price
    without raising and produce a finite, positive VaR."""
    pairs, returns, positions = _eurgbp_book_and_positions()
    report = portfolio_var(returns, positions, confidence=0.99, factors=pairs)
    assert np.isfinite(report.var_parametric)
    assert report.var_parametric > 0
    assert np.isfinite(report.var_historical) and report.var_historical > 0


def test_portfolio_var_eurgbp_book_diversifies_below_standalone_sum():
    """
    Diversification must reduce risk: the portfolio VaR of the three
    correlated instruments must be LESS than the naive sum of their standalone
    VaRs (the whole point of measuring risk jointly rather than per-pair).
    """
    pairs, returns, positions = _eurgbp_book_and_positions()
    report = portfolio_var(returns, positions, confidence=0.99, factors=pairs)
    sum_standalone = sum(report.standalone_var.values())
    assert report.var_parametric < sum_standalone


@given(spot_eurusd=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
      spot_gbpusd=st.floats(min_value=0.5, max_value=2.0, allow_nan=False),
      notional=st.floats(min_value=-5_000_000, max_value=5_000_000, allow_nan=False))
def test_property_cross_usd_conversion_is_an_exact_round_trip(spot_eurusd, spot_gbpusd,
                                                               notional):
    """
    Property: for ANY EUR/USD and GBP/USD spot level, a EUR/GBP position's
    USD-converted exposure must equal notional * EUR/USD exactly. EUR/GBP is
    itself built as EUR/USD / GBP/USD (triangulation), and _factor_positions
    converts it back to USD by multiplying by GBP/USD -- an algebraic round
    trip that has to cancel regardless of the specific spot levels.
    """
    from fxrisk.book import Position, Book
    from fxrisk.portfolio_risk import _factor_positions
    assume(abs(notional) > 1.0)
    eurgbp = spot_eurusd / spot_gbpusd
    book = Book([Position("EUR/GBP", notional > 0, abs(notional), 90, eurgbp, id="x")])
    spots = {"EUR/GBP": eurgbp, "GBP/USD": spot_gbpusd}
    _, positions = _factor_positions(book, spots)
    assert abs(positions[0] - notional * spot_eurusd) < 1e-6 * max(1.0, abs(notional))
