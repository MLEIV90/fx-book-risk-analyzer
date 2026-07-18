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

Common-numeraire note: the book MtM is summed in USD, via the same approach
as fxrisk.portfolio_risk._factor_positions -- positions quoted in USD (e.g.
EUR/USD, GBP/USD) sum directly; a non-USD-quoted position (e.g. EUR/GBP) is
converted at the CURRENT spot of its quote currency against USD (e.g. via
GBP/USD), a declared quanto-style approximation, requiring the book to also
hold that quote currency's own USD pair. See `_mtm_usd`.
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
    spot: float                      # the position's own pair spot, at valuation
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

    Time convention (declared): the snapshot is built with the position's
    ORIGINAL tenor, so the book is valued as a snapshot taken AT BOOKING -- the
    model does not age positions (no theta / passage of time since the trade
    date). This is appropriate for the tool's use (a book defined 'as of today')
    but means it is a static picture, not a day-by-day re-aging of each trade.
    """
    market_fwd = snapshot.forward()
    mtm = forward_mtm(
        notional_base=position.notional_base,
        strike=position.strike,
        fair_fwd_now=market_fwd,
        r_quote=snapshot.r_quote,
        # H2: discounted at r_quote, so tau_remaining must be the QUOTE
        # currency's own day-count basis, not the base's.
        tau_remaining=snapshot.tau_quote,
        long_base=position.long_base,
    )
    return PositionValuation(
        position=position,
        market_forward=market_fwd,
        mtm_quote=mtm,
        quote_ccy=position.quote_ccy,
        spot=snapshot.spot,
        notes=list(snapshot.notes),
    )


def _mtm_usd(mtm_quote: float, quote_ccy: str, spots: dict[str, float]) -> float:
    """
    Convert a QUOTE-currency MtM to a common USD numeraire -- the same
    approach as fxrisk.portfolio_risk._factor_positions: already USD when
    `quote_ccy` is USD; otherwise converted via the CURRENT spot of
    `quote_ccy`/USD, a declared quanto-style approximation (ignores the
    covariance between that conversion rate and the position's own P&L).
    `spots` must carry that conversion rate (e.g. the book also holding that
    quote currency's own USD pair, whose spot is reused); otherwise this
    fails loud rather than summing mismatched currencies.
    """
    if quote_ccy == "USD":
        return mtm_quote
    conv_pair = f"{quote_ccy}/USD"
    if conv_pair not in spots:
        raise ValueError(
            f"Cannot express a {quote_ccy}-quoted MtM in USD: no spot "
            f"available for '{conv_pair}'. Add a {conv_pair} position to the "
            "book, or value this pair separately.")
    return mtm_quote * spots[conv_pair]


def build_report(valuations: list[PositionValuation], book: Book) -> BookReport:
    """
    Pure assembly of the committee report from per-position valuations.
    MtMs are converted to a common USD numeraire before aggregating -- see
    `_mtm_usd` for the conversion and its declared approximation.
    """
    spots = {v.position.pair: v.spot for v in valuations}
    mtm_usd = [_mtm_usd(v.mtm_quote, v.quote_ccy, spots) for v in valuations]

    total = sum(mtm_usd)
    gains = sum(m for m in mtm_usd if m > 0)
    losses = sum(m for m in mtm_usd if m < 0)

    # Concentration: each position's share of gross MtM, descending.
    gross = sum(abs(m) for m in mtm_usd) or 1.0
    concentration = sorted(
        [(v.position.id, m, abs(m) / gross * 100.0)
         for v, m in zip(valuations, mtm_usd)],
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


def _shift_snapshot(snap: MarketSnapshot, shock_pct: float) -> MarketSnapshot:
    """
    Pure helper: return a copy of the snapshot with spot shocked by shock_pct
    (e.g. -5 => spot * 0.95). Rates and vols are held fixed. Extracted so the
    shift logic can be unit-tested without any network fetch.
    """
    return MarketSnapshot(
        pair=snap.pair, base_ccy=snap.base_ccy, quote_ccy=snap.quote_ccy,
        tenor_years=snap.tenor_years, spot=snap.spot * (1 + shock_pct / 100.0),
        r_base=snap.r_base, r_quote=snap.r_quote,
        vol_historical=snap.vol_historical, vol_garch=snap.vol_garch,
        tenor_years_quote=snap.tenor_years_quote,
    )


def book_sensitivity(book: Book, shocks_pct: tuple[float, ...] = (-5, -1, 1, 5)
                     ) -> dict[float, float]:
    """
    Book MtM under spot shocks -- the bridge to VaR, in committee language.
    For each shock, re-fetch is avoided: we re-value using a shifted snapshot
    built from one fetch per position. Returns {shock_pct: total_mtm_usd},
    each total converted to a common USD numeraire (see `_mtm_usd`).
    """
    base_snaps = {p.id: get_market_snapshot(p.pair, p.tenor_days) for p in book}
    out: dict[float, float] = {}
    for shock in shocks_pct:
        total = 0.0
        spots = {p.pair: base_snaps[p.id].spot * (1 + shock / 100.0) for p in book}
        for p in book:
            shifted = _shift_snapshot(base_snaps[p.id], shock)
            val = value_position_from_snapshot(p, shifted)
            total += _mtm_usd(val.mtm_quote, val.quote_ccy, spots)
        out[shock] = total
    return out
