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

# Crosses not directly ticked in this app's scope. Derived (not observed) by
# triangulating two DIRECT, USD-quoted legs under the standard quote-per-base
# identity:
#   EUR/GBP (GBP per 1 EUR) = EUR/USD (USD per 1 EUR) / GBP/USD (USD per 1 GBP)
# Both legs are real yfinance data; only the ratio is derived, so EUR/GBP
# inherits both legs' data quality (and, downstream, the flat-GBP-curve
# limitation when it is used to price a forward -- see fxrisk.market).
TRIANGULATED_PAIRS: dict[str, tuple[str, str]] = {
    "EUR/GBP": ("EUR/USD", "GBP/USD"),
}


class MarketDataError(RuntimeError):
    """Raised when live market data cannot be retrieved."""


def _fetch_direct(tickers_pairs: list[str], period: str) -> pd.DataFrame:
    """
    Download daily close history for pairs with a direct yfinance ticker in
    DEFAULT_TICKERS. Returns a DataFrame with one column per pair, labelled by
    pair name. Internal building block of `fetch_spot_history`.
    """
    import yfinance as yf

    tickers = [DEFAULT_TICKERS[p] for p in tickers_pairs]
    close = yf.download(tickers, period=period, progress=False)["Close"]
    if close is None or len(close) == 0:
        raise MarketDataError("Live market data is currently unavailable.")

    # yfinance returns a different shape for one ticker vs many:
    # - many tickers -> DataFrame with one column per ticker
    # - one ticker   -> a Series (or single-column frame) without our labels
    # Normalise to a DataFrame whose columns are the requested pairs, in order.
    if isinstance(close, pd.Series):
        raw = close.to_frame()
        raw.columns = tickers_pairs
    elif len(tickers_pairs) == 1:
        raw = close.iloc[:, [0]].copy()
        raw.columns = tickers_pairs
    else:
        # Reorder columns to match the requested pair order, then relabel.
        raw = close[tickers].copy()
        raw.columns = tickers_pairs
    return raw


def fetch_spot_history(pairs: list[str] | None = None,
                       period: str = "2y") -> pd.DataFrame:
    """
    Download daily spot history for the given pairs.

    Two kinds of pair:
    - Direct (DEFAULT_TICKERS): a real yfinance ticker, fetched as observed.
    - Triangulated crosses (TRIANGULATED_PAIRS, e.g. EUR/GBP): derived by
      dividing two directly-fetched USD legs. The cross series is DERIVED,
      not directly observed, and inherits both legs' data quality.

    Returns a DataFrame of prices, one column per requested pair, in the
    requested order. Raises MarketDataError if the data cannot be retrieved
    -- the caller (the app) is expected to catch it and show a clear "data
    unavailable" message, NOT to fall back to synthetic data.
    """
    pairs = pairs or list(DEFAULT_TICKERS.keys())
    unknown = [p for p in pairs if p not in DEFAULT_TICKERS and p not in TRIANGULATED_PAIRS]
    if unknown:
        raise MarketDataError(f"No data source configured for: {', '.join(unknown)}.")

    direct_requested = [p for p in pairs if p in DEFAULT_TICKERS]
    triangulated_requested = [p for p in pairs if p in TRIANGULATED_PAIRS]
    # Every leg that must actually be downloaded: the direct pairs asked for,
    # plus each triangulated pair's two USD legs (deduplicated, order-stable).
    legs_needed = list(dict.fromkeys(
        direct_requested
        + [leg for p in triangulated_requested for leg in TRIANGULATED_PAIRS[p]]))

    try:
        raw = _fetch_direct(legs_needed, period)

        out = pd.DataFrame(index=raw.index)
        for p in direct_requested:
            out[p] = raw[p]
        for p in triangulated_requested:
            base_leg, quote_leg = TRIANGULATED_PAIRS[p]
            out[p] = raw[base_leg] / raw[quote_leg]

        out = out[pairs].dropna()
        if out.empty:
            raise MarketDataError("Live market data is currently unavailable.")
        return out
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
    # Build a date index that always matches the number of rows exactly.
    n_rows = prices.shape[0]
    idx = pd.bdate_range(start="2020-01-01", periods=n_rows)
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