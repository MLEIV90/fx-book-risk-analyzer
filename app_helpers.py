"""
app_helpers
===========
Thin glue used by the Streamlit app to assemble, from a Book, the inputs the
risk engine needs (snapshots, returns matrix, factor positions). Kept out of
app.py so the interface stays readable.

Data layer (B2/B3): market data is network-bound, so this module wraps the pure
engine fetchers with:
- Caching (Streamlit @st.cache_data, 10-minute TTL) so the same data is not
  re-downloaded on every interaction or across tabs.
- Retries with backoff so a transient network blip does not fail the request.
The engine (fxrisk/) stays pure and stateless; all caching/retry lives here.
"""
from __future__ import annotations

import time
import numpy as np

from fxrisk.data import fetch_spot_history, to_returns, MarketDataError
from fxrisk.market import get_market_snapshot

try:
    import streamlit as st
    _CACHE = st.cache_data(ttl=600, show_spinner=False)   # 10-minute TTL
except Exception:                                          # tests / no-streamlit
    def _CACHE(func):
        return func

# Data freshness window, surfaced in the UI.
CACHE_TTL_SECONDS = 600


def _with_retries(fn, *args, attempts: int = 3, base_delay: float = 0.6, **kwargs):
    """
    Call fn with retries on transient failures. Re-raises MarketDataError as-is
    on the final attempt; retries everything else (network blips, timeouts).
    """
    last_exc = None
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except MarketDataError:
            raise                                          # data genuinely absent
        except Exception as exc:                           # transient -> retry
            last_exc = exc
            if i < attempts - 1:
                time.sleep(base_delay * (i + 1))
    raise MarketDataError(
        "Live market data could not be retrieved after several attempts. "
        "Please check your connection and try again."
    ) from last_exc


@_CACHE
def cached_snapshot(pair: str, tenor_days: int):
    """Cached, retried market snapshot for one pair/tenor."""
    return _with_retries(get_market_snapshot, pair, tenor_days)


@_CACHE
def cached_spot_history(pairs: tuple[str, ...], period: str = "2y"):
    """Cached, retried spot history. `pairs` is a tuple so it is hashable."""
    return _with_retries(fetch_spot_history, list(pairs), period=period)


def snapshots_for_book(book) -> dict:
    """One real market snapshot per position id (cached per pair/tenor)."""
    return {p.id: cached_snapshot(p.pair, p.tenor_days) for p in book}


def factor_setup(book, snapshots: dict, history_period: str = "2y"):
    """
    Build the inputs for portfolio VaR:
    - pairs: distinct pairs in the book (the risk factors), sorted.
    - returns: (n_days, n_pairs) real returns matrix aligned to pairs.
    - positions: (n_pairs,) net spot exposure in quote currency per pair.
    """
    pairs = book.pairs()

    # Same numeraire guard as portfolio_risk: all pairs must be quote-USD, or the
    # returns and exposures would be summed across currencies without conversion.
    non_usd = sorted({p.pair for p in book if p.quote_ccy != "USD"})
    if non_usd:
        raise ValueError(
            "Portfolio VaR currently assumes all pairs are quote-USD; found "
            f"non-USD-quoted pair(s): {', '.join(non_usd)}.")

    prices = cached_spot_history(tuple(pairs), period=history_period)
    rets_df = to_returns(prices)
    returns = rets_df[pairs].to_numpy()

    # Net spot exposure per pair (quote-currency), provider sign.
    spots = {p.pair: snapshots[p.id].spot for p in book}
    exposure = {pr: 0.0 for pr in pairs}
    for p in book:
        sign = 1.0 if p.long_base else -1.0
        exposure[p.pair] += sign * p.notional_base * spots[p.pair]
    positions = np.array([exposure[pr] for pr in pairs])
    return pairs, returns, positions
