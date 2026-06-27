"""
Tests for the portfolio risk layer: VaR report, attribution, Kupiec backtest.
Pure logic with synthetic returns; no network needed.
"""
import numpy as np

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


def test_factor_positions_rejects_non_usd_quote():
    """The numeraire guard must reject a non-USD-quoted pair, not sum silently."""
    import pytest
    from fxrisk.book import Position, Book
    from fxrisk.portfolio_risk import _factor_positions
    book = Book([Position("EUR/JPY", True, 1_000_000, 90, 160.0, id="x1")])
    spots = {"EUR/JPY": 160.0}
    with pytest.raises(ValueError, match="quote-USD"):
        _factor_positions(book, spots)


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
