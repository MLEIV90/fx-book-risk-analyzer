"""
fxrisk.data
===========
Market data access for the FX toolkit.

This is the ONLY module that talks to the outside world. Isolating it means
that if the data source changes or fails, nothing else in the engine breaks.

Data source: yfinance (free) -- chosen for a reproducible demo. In production
this layer would be backed by a professional feed such as Bloomberg or
Refinitiv/LSEG, which is the industry standard for real-time market data.

Design choice on failure: the live fetch FAILS LOUDLY. If real data cannot be
retrieved, it raises MarketDataError rather than silently returning synthetic
numbers -- showing synthetic prices as if they were real would be misleading.
The synthetic generator below exists ONLY to make the test suite independent of
the network; it is never used as a stand-in for live data in the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Yahoo Finance tickers for a couple of common FX pairs.
DEFAULT_TICKERS: dict[str, str] = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}


class MarketDataError(RuntimeError):
    """Raised when live market data cannot be retrieved."""


def fetch_spot_history(pairs: list[str] | None = None,
                       period: str = "1y") -> pd.DataFrame:
    """
    Download daily spot history for the given pairs.

    Returns a DataFrame of closing prices, one column per pair.
    Raises MarketDataError if the data cannot be retrieved -- the caller (the
    app) is expected to catch it and show a clear "data unavailable" message,
    NOT to fall back to synthetic data.
    """
    pairs = pairs or list(DEFAULT_TICKERS.keys())
    try:
        import yfinance as yf

        tickers = [DEFAULT_TICKERS[p] for p in pairs]
        raw = yf.download(tickers, period=period, progress=False)["Close"]
        if raw is None or raw.empty:
            raise MarketDataError("Live market data is currently unavailable.")
        raw.columns = pairs
        return raw.dropna()
    except MarketDataError:
        raise
    except Exception as exc:  # network down, API change, bad ticker, etc.
        raise MarketDataError(
            "Live market data could not be retrieved. Please try again later."
        ) from exc


def synthetic_spot_history(pairs: list[str], n_days: int = 252,
                           seed: int = 42) -> pd.DataFrame:
    """
    Generate plausible synthetic spot paths.

    FOR TESTING ONLY -- this keeps the test suite independent of the network.
    It is intentionally NOT wired into fetch_spot_history, so the live app never
    shows synthetic numbers in place of real market data.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)

    # Correlated daily returns (corr ~ 0.8 between the two pairs).
    corr = np.full((n, n), 0.8)
    np.fill_diagonal(corr, 1.0)
    vols = np.full(n, 0.005)
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov)

    z = rng.standard_normal((n_days, n))
    daily_returns = z @ chol.T

    start = np.array([1.08, 1.27])[:n] if n <= 2 else np.full(n, 1.10)
    prices = start * np.exp(np.cumsum(daily_returns, axis=0))
    idx = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")
    return pd.DataFrame(prices, columns=pairs, index=idx)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns from a price DataFrame."""
    return prices.pct_change().dropna()


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of the returns (the shape of how pairs move together)."""
    return returns.corr()


def rolling_correlation(returns: pd.DataFrame, pair_a: str, pair_b: str,
                        window: int = 60) -> pd.Series:
    """
    Rolling correlation between two pairs over time.
    Shows that correlation is not constant -- it drifts and spikes in stress.
    """
    return returns[pair_a].rolling(window).corr(returns[pair_b]).dropna()