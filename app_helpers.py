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

from fxrisk.data import fetch_spot_history, to_returns, MarketDataError
from fxrisk.market import get_market_snapshot

# Data freshness window, surfaced in the UI.
CACHE_TTL_SECONDS = 600


def _cache(func):
    """
    Apply Streamlit's data cache if Streamlit is importable, else return the
    function unchanged (so the module also works in tests / without Streamlit).
    Defined as a plain decorator applied per-function -- the robust pattern that
    Streamlit Cloud always supports.
    """
    try:
        import streamlit as st
        return st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)(func)
    except Exception:
        return func


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


@_cache
def cached_snapshot(pair: str, tenor_days: int):
    """Cached, retried market snapshot for one pair/tenor."""
    return _with_retries(get_market_snapshot, pair, tenor_days)


@_cache
def cached_spot_history(pairs: tuple, period: str = "2y"):
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
    - positions: (n_pairs,) net exposure per pair, in a common USD numeraire
      (see fxrisk.portfolio_risk._factor_positions for the conversion and its
      declared quanto-style approximation for non-USD-quoted pairs).
    """
    from fxrisk.portfolio_risk import _factor_positions

    pairs = book.pairs()
    prices = cached_spot_history(tuple(pairs), period=history_period)
    rets_df = to_returns(prices)
    returns = rets_df[pairs].to_numpy()

    spots = {p.pair: snapshots[p.id].spot for p in book}
    _, positions = _factor_positions(book, spots)
    return pairs, returns, positions


def book_notional_usd(book, snapshots: dict) -> float:
    """
    Total book notional (gross, absolute per position), in a common USD
    numeraire -- same conversion approach as `factor_setup`/
    `fxrisk.portfolio_risk._factor_positions`: a position's notional * spot
    is already USD when quote-USD; otherwise converted via the current spot
    of quote/USD (requires that pair's spot to also be in `snapshots`, e.g.
    the book also holding that quote currency's own USD pair).
    """
    spots = {p.pair: snapshots[p.id].spot for p in book}
    total = 0.0
    for p in book:
        amt = abs(p.notional_base) * spots[p.pair]
        if p.quote_ccy != "USD":
            conv_pair = f"{p.quote_ccy}/USD"
            if conv_pair not in spots:
                raise ValueError(
                    f"Cannot express {p.pair}'s notional in USD: no spot "
                    f"available for '{conv_pair}'.")
            amt *= spots[conv_pair]
        total += amt
    return total
