"""Tests for the risk engine: DV01, VaR (3 methods), ES, liquidity, stress."""
import numpy as np

from fxrisk.forwards import year_fraction, forward_rate
from fxrisk.risk import (
    dv01_forward, var_parametric, var_historical, var_montecarlo,
    expected_shortfall, simulate_liquidity_need, stress_pnl, STRESS_SCENARIOS,
)

SPOT, R_EUR, R_USD = 1.08, 0.030, 0.045
TAU = year_fraction(90)


# ----- DV01 -----
def test_dv01_returns_three_keys():
    fair = forward_rate(SPOT, R_EUR, R_USD, TAU)
    dv = dv01_forward(1_000_000, fair, SPOT, R_EUR, R_USD, TAU)
    assert set(dv) == {"dv01_base", "dv01_quote", "dv01_net"}


def test_dv01_base_and_quote_opposite_sign():
    fair = forward_rate(SPOT, R_EUR, R_USD, TAU)
    dv = dv01_forward(1_000_000, fair, SPOT, R_EUR, R_USD, TAU)
    assert dv["dv01_base"] * dv["dv01_quote"] < 0


# ----- VaR / ES fixtures -----
def _sample_returns(seed: int = 0, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = 0.8 * z[:, 0] + np.sqrt(1 - 0.8**2) * z[:, 1]
    return z * np.array([0.0044, 0.0050])


POSITIONS = np.array([-10_800_000.0, 16_510_000.0])


def test_var_methods_positive():
    r = _sample_returns()
    assert var_parametric(r, POSITIONS) > 0
    assert var_historical(r, POSITIONS) > 0
    assert var_montecarlo(r, POSITIONS, n_sims=20_000) > 0


def test_var_methods_close():
    r = _sample_returns()
    vp = var_parametric(r, POSITIONS)
    vh = var_historical(r, POSITIONS)
    vm = var_montecarlo(r, POSITIONS, n_sims=50_000)
    assert abs(vp - vh) / vp < 0.15
    assert abs(vp - vm) / vp < 0.15


def test_es_greater_than_var():
    r = _sample_returns()
    es = expected_shortfall(r, POSITIONS)
    vh = var_historical(r, POSITIONS)
    assert es > vh


# ----- Liquidity -----
def test_liquidity_returns_two_keys():
    out = simulate_liquidity_need(1_000_000, 0.005, horizon_days=60, n_sims=2000)
    assert {"liquidity_buffer", "avg_worst_drawdown", "stressed_buffer",
            "stress_multiplier"} <= set(out)
    # The stressed buffer must exceed the base buffer (higher vol -> more cash).
    assert out["stressed_buffer"] > out["liquidity_buffer"]


def test_liquidity_buffer_positive_and_largest():
    out = simulate_liquidity_need(1_000_000, 0.005, horizon_days=60, n_sims=5000)
    assert out["liquidity_buffer"] > 0
    assert out["liquidity_buffer"] >= out["avg_worst_drawdown"]


# ----- Stress -----
def test_stress_pnl_uses_positions_and_moves():
    positions = {"EUR/USD": 1_000_000.0, "GBP/USD": -500_000.0}
    scenario = {"EUR/USD": -0.02, "GBP/USD": -0.08}
    # 1,000,000*(-0.02) + (-500,000)*(-0.08) = -20,000 + 40,000 = 20,000
    assert stress_pnl(positions, scenario) == 20_000.0


def test_stress_scenarios_present():
    assert len(STRESS_SCENARIOS) >= 3
    for sc in STRESS_SCENARIOS.values():
        assert "EUR/USD" in sc and "GBP/USD" in sc

def test_var_parametric_single_point_no_warning():
    """V3: VaR with a single observation returns 0 without a RuntimeWarning."""
    import warnings
    import numpy as np
    from fxrisk.risk import var_parametric
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = var_parametric(np.array([[0.01]]), np.array([1e6]), 0.99)
    assert v == 0.0


def test_var_non_negative_on_trending_data():
    """VaR must never be negative, even on strongly positive-trending data where
    the loss-side percentile is itself a gain (it floors at zero)."""
    import numpy as np
    from fxrisk.risk import var_historical
    pos = np.array([1_000_000.0])
    bull = np.random.default_rng(0).standard_normal((500, 1)) * 0.003 + 0.01
    assert var_historical(bull, pos, 0.99) >= 0.0
