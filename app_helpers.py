"""
app_helpers
===========
Thin glue used by the Streamlit app to assemble, from a Book, the inputs the
risk engine needs (snapshots, returns matrix, factor positions). Kept out of
app.py so the interface stays readable. Network-bound; cached in the app.
"""
from __future__ import annotations

import numpy as np

from fxrisk.data import fetch_spot_history, to_returns
from fxrisk.market import get_market_snapshot


def snapshots_for_book(book) -> dict:
    """One real market snapshot per position id."""
    return {p.id: get_market_snapshot(p.pair, p.tenor_days) for p in book}


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

    prices = fetch_spot_history(pairs, period=history_period)
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
