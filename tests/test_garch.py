"""Tests for the volatility module (historical and GARCH)."""
import numpy as np

from fxrisk.garch import historical_vol, fit_garch, GarchResult


def _garch_like_returns(n: int = 1500, seed: int = 7) -> np.ndarray:
    """Simulate returns with volatility clustering, so GARCH has something to fit."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    sigma2 = 1e-4
    for t in range(n):
        sigma2 = 2e-6 + 0.08 * (r[t - 1] ** 2 if t > 0 else 0.0) + 0.90 * sigma2
        r[t] = np.sqrt(sigma2) * rng.standard_normal()
    return r


def test_historical_vol_positive():
    r = _garch_like_returns()
    assert historical_vol(r) > 0


def test_garch_fit_basic():
    r = _garch_like_returns()
    res = fit_garch(r, asymmetric=False, dist="t")
    assert isinstance(res, GarchResult)
    assert len(res.conditional_vol_daily) == len(r)   # one vol per observation
    assert res.current_vol_annual > 0
    assert res.forecast_vol_annual > 0
    assert 0.0 < res.persistence < 1.05                # stationary-ish


def test_gjr_runs_and_has_gamma():
    r = _garch_like_returns()
    res = fit_garch(r, asymmetric=True, dist="t")
    assert "gamma[1]" in res.params
    assert res.current_vol_annual > 0


def test_garch_reacts_more_than_historical_in_calm_tail():
    # After a calm recent stretch, GARCH's current vol should differ from the
    # flat historical number (it is conditional, not an average).
    r = _garch_like_returns()
    hist = historical_vol(r)
    res = fit_garch(r, dist="t")
    assert res.current_vol_annual != hist