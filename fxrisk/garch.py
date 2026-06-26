"""
fxrisk.garch
============
Volatility estimation for FX returns, at two levels of sophistication.

- historical_vol: the simple annualised standard deviation of returns. One
  number for the whole window; it cannot react to changing regimes.
- fit_garch: a CONDITIONAL volatility model. GARCH(1,1) captures volatility
  clustering (a violent day tends to be followed by another), so it reflects
  today's volatility given recent history. Student-t innovations capture the
  fat tails of FX returns. The asymmetric variant (GJR-GARCH) adds a leverage
  term, so down-moves can raise volatility more than up-moves.

Industry-standard estimation via the `arch` library.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS: int = 252


def historical_vol(returns: np.ndarray, annualize: int = TRADING_DAYS) -> float:
    """Simple annualised volatility: standard deviation of returns x sqrt(252)."""
    return float(np.std(np.asarray(returns, dtype=float), ddof=1) * np.sqrt(annualize))


@dataclass
class GarchResult:
    """Container for a fitted GARCH model and the figures a desk reads from it."""
    model: str
    conditional_vol_daily: np.ndarray   # daily conditional vol, one per observation
    current_vol_annual: float           # latest conditional vol, annualised
    forecast_vol_annual: float          # next-step forecast, annualised
    persistence: float                  # alpha + beta (+ 0.5*gamma for GJR)
    params: dict
    log_likelihood: float


def fit_garch(returns: np.ndarray, asymmetric: bool = False, dist: str = "t",
              annualize: int = TRADING_DAYS, horizon: int = 1) -> GarchResult:
    """
    Fit GARCH(1,1) (or GJR-GARCH(1,1,1) if asymmetric=True) with the chosen
    innovation distribution ('t' for Student-t, 'normal' otherwise).

    Returns a GarchResult with the conditional volatility series, the current
    and one-step-ahead annualised volatility, and the persistence.
    """
    from arch import arch_model

    r = np.asarray(returns, dtype=float)
    scaled = r * 100.0  # the arch library is numerically happier on percent scale

    o = 1 if asymmetric else 0
    am = arch_model(scaled, mean="Constant", vol="GARCH", p=1, o=o, q=1, dist=dist)
    res = am.fit(disp="off")

    cond_vol_daily = res.conditional_volatility / 100.0
    current_vol_annual = float(cond_vol_daily[-1] * np.sqrt(annualize))

    fc = res.forecast(horizon=horizon, reindex=False)
    fc_var_pct2 = float(fc.variance.values[-1, -1])         # in percent^2
    forecast_vol_annual = float(np.sqrt(fc_var_pct2) / 100.0 * np.sqrt(annualize))

    params = {k: float(v) for k, v in res.params.to_dict().items()}
    alpha = params.get("alpha[1]", 0.0)
    beta = params.get("beta[1]", 0.0)
    gamma = params.get("gamma[1]", 0.0)
    persistence = alpha + beta + 0.5 * gamma

    name = f"GJR-GARCH(1,1,1)-{dist}" if asymmetric else f"GARCH(1,1)-{dist}"
    return GarchResult(
        model=name,
        conditional_vol_daily=cond_vol_daily,
        current_vol_annual=current_vol_annual,
        forecast_vol_annual=forecast_vol_annual,
        persistence=float(persistence),
        params=params,
        log_likelihood=float(res.loglikelihood),
    )