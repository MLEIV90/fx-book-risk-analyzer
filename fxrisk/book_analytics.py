"""
fxrisk.book_analytics
=====================
Valuation layer: turns the pure book into a committee-ready mark-to-market
report. This layer DOES touch the market (via fxrisk.market); the pure book
structure stays in fxrisk.book.

What it produces, for a risk committee:
- MtM per position and the book total (in USD; see note below).
- Profit/loss composition (how much of the book is in vs out of the money).
- Attribution and concentration (each position's share of the total).
- Net exposure per currency, alongside its valuation.
- Book sensitivity to a spot shock (the bridge to VaR, in plain terms).
- Data-quality flags (each valuation carries its source notes).

Common-currency note: the book MtM is summed in USD. Positions quoted in USD
(e.g. EUR/USD, GBP/USD) sum directly. A non-USD-quoted pair would need a spot
conversion -- declared and NOT implemented in v1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from fxrisk.forwards import forward_mtm
from fxrisk.market import get_market_snapshot, MarketSnapshot
from fxrisk.book import Book, Position


@dataclass
class PositionValuation:
    """The valuation of a single position, with its data-quality notes."""
    position: Position
    market_forward: float
    mtm_quote: float                 # MtM in the position's quote currency
    quote_ccy: str
    notes: list[str] = field(default_factory=list)


@dataclass
class BookReport:
    """A committee-ready snapshot of the book's value and its composition."""
    valuations: list[PositionValuation]
    total_mtm_usd: float
    gains_usd: float                 # sum of positive MtMs
    losses_usd: float                # sum of negative MtMs
    net_exposure: dict               # net cash flow per currency (from the book)
    concentration: list[tuple]       # (position_id, mtm_usd, share_pct) desc
    data_flags: list[str]            # aggregated data-quality notes


def value_position_from_snapshot(position: Position,
                                 snapshot: MarketSnapshot) -> PositionValuation:
    """
    Pure valuation: given a position and a market snapshot, compute its MtM.
    Separated from the fetch so it can be unit-tested without the network.
    """
    market_fwd = snapshot.forward()
    mtm = forward_mtm(
        notional_base=position.notional_base,
        strike=position.strike,
        fair_fwd_now=market_fwd,
        r_quote=snapshot.r_quote,
        tau_remaining=snapshot.tenor_years,
        long_base=position.long_base,
    )
    return PositionValuation(
        position=position,
        market_forward=market_fwd,
        mtm_quote=mtm,
        quote_ccy=position.quote_ccy,
        notes=list(snapshot.notes),
    )


def build_report(valuations: list[PositionValuation], book: Book) -> BookReport:
    """
    Pure assembly of the committee report from per-position valuations.
    Assumes quote currencies sum in USD (declared simplification).
    """
    total = sum(v.mtm_quote for v in valuations)
    gains = sum(v.mtm_quote for v in valuations if v.mtm_quote > 0)
    losses = sum(v.mtm_quote for v in valuations if v.mtm_quote < 0)

    # Concentration: each position's share of gross MtM, descending.
    gross = sum(abs(v.mtm_quote) for v in valuations) or 1.0
    concentration = sorted(
        [(v.position.id, v.mtm_quote, abs(v.mtm_quote) / gross * 100.0)
         for v in valuations],
        key=lambda x: abs(x[1]), reverse=True,
    )

    flags = sorted({n for v in valuations for n in v.notes})

    return BookReport(
        valuations=valuations,
        total_mtm_usd=total,
        gains_usd=gains,
        losses_usd=losses,
        net_exposure=book.net_exposure_by_currency(),
        concentration=concentration,
        data_flags=flags,
    )


def value_book(book: Book) -> BookReport:
    """
    Fetch a real snapshot for each position and build the committee report.
    Network-bound (yfinance / FRED / ECB); raises if data is unavailable.
    """
    valuations = []
    for p in book:
        snap = get_market_snapshot(p.pair, p.tenor_days)
        valuations.append(value_position_from_snapshot(p, snap))
    return build_report(valuations, book)


def book_sensitivity(book: Book, shocks_pct: tuple[float, ...] = (-5, -1, 1, 5)
                     ) -> dict[float, float]:
    """
    Book MtM under spot shocks -- the bridge to VaR, in committee language.
    For each shock, re-fetch is avoided: we re-value using a shifted snapshot
    built from one fetch per position. Returns {shock_pct: total_mtm_usd}.
    """
    base_snaps = {p.id: get_market_snapshot(p.pair, p.tenor_days) for p in book}
    out: dict[float, float] = {}
    for shock in shocks_pct:
        total = 0.0
        for p in book:
            snap = base_snaps[p.id]
            shifted = MarketSnapshot(
                pair=snap.pair, base_ccy=snap.base_ccy, quote_ccy=snap.quote_ccy,
                tenor_years=snap.tenor_years, spot=snap.spot * (1 + shock / 100.0),
                r_base=snap.r_base, r_quote=snap.r_quote,
                vol_historical=snap.vol_historical, vol_garch=snap.vol_garch,
            )
            total += value_position_from_snapshot(p, shifted).mtm_quote
        out[shock] = total
    return out
