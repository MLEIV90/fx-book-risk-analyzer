"""
FX Book Risk Analyzer
=====================
Desk-side tool to build an FX forward book and analyse its risk, from the
provider's perspective. Real market data (yfinance spot, FRED/ECB rate curves,
GARCH volatility); pure tested engine in `fxrisk`.

Screens: Libro (book) | Valuación (MtM) | Riesgo (VaR/backtest/DV01/stress) |
Control (limits & status) — four flat tabs, no nested sub-tabs. Explanatory
text and declared limitations live in collapsed expanders above the tabs.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fxrisk.curves import supported_currencies
from fxrisk.market import get_market_snapshot
from fxrisk.forwards import client_rate_with_spread, forward_points
from fxrisk.book import Position, Book
from fxrisk.data import MarketDataError
from fxrisk.book_analytics import value_book, book_sensitivity
from fxrisk.portfolio_risk import (
    portfolio_var, kupiec_backtest, rolling_backtest, stressed_var,
    var_student_t, var_ewma, christoffersen_independence)
from fxrisk.book_risk import (dv01_book, liquidity_book, stress_book,
                              dv01_book_by_tenor)
from fxrisk.limits import LimitsConfig, check_limits
from app_helpers import snapshots_for_book, factor_setup, book_notional_usd

st.set_page_config(page_title="FX Book Risk Analyzer", layout="wide",
                   initial_sidebar_state="expanded")

INK, MUTED, INDIGO, AMBER, RED, GREEN, GRID = (
    "#E6E9EF", "#8A93A0", "#7C8CFF", "#E0A23C", "#E06B6B", "#4FB286", "#2A2F3A")

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

      :root{
        --bg:#0C0E12; --panel:#13161D; --panel2:#171B23;
        --line:#242A35; --line2:#2E3542;
        --ink:#E8ECF3; --muted:#8A93A3; --faint:#5C6678;
        --accent:#6E8BFF; --accent-soft:rgba(110,139,255,0.12);
        --ok:#46B98A; --warn:#E0A23C; --bad:#E0675F;
      }
      html, body, [class*="css"] { font-family:'Inter',sans-serif; color:var(--ink); }
      .stApp { background:var(--bg); }

      /* header */
      .eyebrow   { font-size:0.72rem; letter-spacing:0.18em; text-transform:uppercase;
                   color:var(--accent); font-weight:600; }
      .app-title { font-size:1.75rem; font-weight:700; margin:0.1rem 0 0; letter-spacing:-0.01em; }
      .app-sub   { color:var(--muted); margin-top:0.25rem; font-size:0.95rem; }
      .stamp     { color:var(--faint); font-size:0.78rem; font-family:'JetBrains Mono',monospace; }

      /* metric cards */
      .cardrow { display:flex; gap:0.7rem; flex-wrap:wrap; margin:0.3rem 0 0.4rem; }
      .card { flex:1 1 0; min-width:150px; background:var(--panel);
              border:1px solid var(--line); border-radius:10px; padding:0.85rem 1rem; }
      .card .label { font-size:0.72rem; color:var(--muted); text-transform:uppercase;
                     letter-spacing:0.06em; margin-bottom:0.35rem; }
      .card .value { font-family:'JetBrains Mono',monospace; font-size:1.5rem;
                     font-weight:600; color:var(--ink); line-height:1.1; }
      .card .sub   { font-size:0.74rem; color:var(--faint); margin-top:0.3rem;
                     font-family:'JetBrains Mono',monospace; }
      .card.accent { border-color:var(--line2); box-shadow:inset 3px 0 0 var(--accent); }
      .card.ok     { box-shadow:inset 3px 0 0 var(--ok); }
      .card.warn   { box-shadow:inset 3px 0 0 var(--warn); }
      .card.bad    { box-shadow:inset 3px 0 0 var(--bad); }
      .card .value.pos { color:var(--ok); }
      .card .value.neg { color:var(--bad); }

      /* section titles */
      .sect { font-size:1.05rem; font-weight:600; margin:1.1rem 0 0.2rem;
              padding-bottom:0.3rem; border-bottom:1px solid var(--line); }

      .interp { background:var(--accent-soft); border-left:3px solid var(--accent);
                padding:0.75rem 1rem; border-radius:0 6px 6px 0; font-size:0.9rem;
                color:#C9D2E6; margin:0.5rem 0; }
      .step   { background:var(--panel); border:1px solid var(--line);
                padding:0.95rem 1.1rem; border-radius:10px; margin-bottom:0.55rem; }
      .stepnum{ display:inline-block; width:1.55rem; height:1.55rem; line-height:1.55rem;
                text-align:center; border-radius:8px; background:var(--accent); color:#0C0E12;
                font-weight:700; margin-right:0.55rem; font-family:'JetBrains Mono',monospace; }

      .pill { display:inline-block; padding:0.12rem 0.6rem; border-radius:999px;
              font-size:0.78rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
      .pill.ok   { background:rgba(70,185,138,0.15); color:var(--ok); }
      .pill.warn { background:rgba(224,162,60,0.15); color:var(--warn); }
      .pill.bad  { background:rgba(224,103,95,0.15); color:var(--bad); }

      .stTabs [data-baseweb="tab-list"]{ gap:0.3rem; }
      .stTabs [data-baseweb="tab"]{ font-weight:500; }
      [data-testid="stMetricValue"]{ font-family:'JetBrains Mono',monospace; }
      [data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:8px; }
      section[data-testid="stSidebar"]{ background:var(--panel); border-right:1px solid var(--line); }
    </style>
    """, unsafe_allow_html=True)


def cards(items):
    """Render a row of metric cards. items: list of dicts(label,value,sub,kind,sign)."""
    html = '<div class="cardrow">'
    for it in items:
        kind = it.get("kind", "")
        sign = it.get("sign", "")
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        html += (f'<div class="card {kind}"><div class="label">{it["label"]}</div>'
                 f'<div class="value {sign}">{it["value"]}</div>{sub}</div>')
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def sect(title):
    st.markdown(f'<div class="sect">{title}</div>', unsafe_allow_html=True)

PLOT_BG = "rgba(0,0,0,0)"
PLOT_INK, PLOT_MUTED, PLOT_ACCENT, PLOT_AMBER, PLOT_RED, PLOT_GRID = (
    "#E8ECF3", "#8A93A3", "#6E8BFF", "#E0A23C", "#E0675F", "#242A35")


def _plotly_layout(fig, height=300):
    fig.update_layout(
        height=height, paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_MUTED, family="Inter, sans-serif", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=PLOT_MUTED)))
    fig.update_xaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID,
                     linecolor=PLOT_GRID)
    fig.update_yaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID,
                     linecolor=PLOT_GRID)
    return fig

# --- session state ---------------------------------------------------------
if "book" not in st.session_state:
    st.session_state.book = Book()
book: Book = st.session_state.book

SUPPORTED = supported_currencies()
PAIRS = [p for p in ("EUR/USD", "GBP/USD", "EUR/GBP")
        if all(c in SUPPORTED for c in p.split("/"))]

# Tooltip strings (plain-language help for both technical and business users).
HELP = {
    "var_limit": "Maximum 1-day loss the desk is allowed to risk, in USD. "
                 "If the book's VaR exceeds it, the limit is breached. Set 0 to ignore.",
    "exp_limit": "Maximum net exposure allowed in any single currency, in that "
                 "currency's units. 'Per ccy' means the check runs separately for "
                 "EUR, USD, GBP, etc. Set 0 to ignore.",
    "confidence": "How extreme a loss the VaR describes. 99% means 'the loss not "
                  "exceeded on 99% of days' — i.e. a 1-in-100 bad day.",
    "spread": "The margin the provider adds over the fair forward rate. It is the "
              "provider's revenue and the client's cost of certainty.",
    "tenor": "Days until the forward settles. The rate used is read from each "
             "currency's curve at this tenor.",
    "notional": "Size of the trade, in units of the base currency.",
    "total_mtm": "What the whole book is worth today at current market rates.",
    "var": "Value at Risk: the loss not exceeded at the chosen confidence over 1 day.",
    "es": "Expected Shortfall: the average loss on the worst days beyond the VaR.",
    "dv01": "Change in book value for a 1 basis-point (0.01%) move in interest rates.",
    "liq": "Cash the desk may need to post as daily margin in a bad scenario.",
    "div": "How much lower the portfolio VaR is than the sum of each position's "
           "standalone VaR — the benefit of holding several imperfectly-correlated pairs.",
}


def glossary():
    with st.expander("📖 Glossary — key terms in one line"):
        st.markdown(
            "- **Spot** — the current exchange rate.\n"
            "- **Forward** — a rate locked today for a future date (spot adjusted for "
            "the interest-rate difference, not a forecast).\n"
            "- **Notional** — the size of a trade, in the base currency.\n"
            "- **MtM (Mark-to-Market)** — what a position is worth today at current rates.\n"
            "- **VaR (Value at Risk)** — the loss not exceeded at a given confidence "
            "over a horizon (here, 1 day).\n"
            "- **ES (Expected Shortfall)** — the average loss in the tail beyond the VaR.\n"
            "- **DV01** — value change per 1 basis-point move in interest rates.\n"
            "- **Net exposure** — the net amount of each currency the book is long or short.\n"
            "- **Diversification benefit** — risk reduction from holding imperfectly "
            "correlated positions.\n"
            "- **Kupiec test** — a statistical check that the VaR is reliable "
            "(were breaches as frequent as the model implies?). Reported with both an "
            "asymptotic and an exact Monte Carlo p-value (see NOTES.md).\n"
            "- **Stress test** — applying a real past crisis's moves to today's book.")


def _id_to_num() -> dict:
    return {p.id: f"#{i+1:03d}" for i, p in enumerate(book)}


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="eyebrow">FX Risk Desk</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">FX Book Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Build an FX forward book, then analyse its risk — '
            'valuation, VaR, limits, liquidity and stress. Real market data.</div>',
            unsafe_allow_html=True)
st.caption(f"Market data fetched live · session started "
           f"{datetime.now().strftime('%d %b %Y, %H:%M')}")
st.write("")

# --------------------------------------------------------------------------
# About / limitations — available above the tabs, not a navigation step.
# --------------------------------------------------------------------------
with st.expander("ℹ️ About this tool — what it does & how to use it", expanded=False):
    st.markdown(
        "This is a risk-desk tool for an FX provider — a firm that sells currency "
        "hedges (forwards) to clients. When a client hedges, the provider takes the "
        "**opposite position**; the risk the client transfers becomes the provider's "
        "to manage. This tool builds the provider's **book** and measures its risk.")

    cestA, cestB = st.columns(2)
    with cestA:
        st.markdown("**For a risk/quant reader**")
        st.caption("VaR by five methods, Expected Shortfall, Kupiec + Christoffersen "
                   "backtesting (asymptotic and exact Monte Carlo p-values), risk "
                   "attribution, DV01, liquidity and historical stress — over a real "
                   "multi-currency book, with limit control.")
    with cestB:
        st.markdown("**For a business reader**")
        st.caption("How much the book is worth, how much it could lose on a bad day, "
                   "whether that is within approved limits, and how it behaves in a "
                   "crisis — all in plain numbers.")

    st.divider()
    st.markdown("**How to use it — three steps**")
    st.markdown(
        '<div class="step"><span class="stepnum">1</span><b>Build the book.</b> '
        'In <b>Libro</b>, click <i>Load example</i> or add a trade. Each trade is '
        'priced at live market rates and the provider\'s mirror position is '
        'recorded.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step"><span class="stepnum">2</span><b>Read the analysis.</b> '
        '<b>Valuación</b> shows what the book is worth; <b>Riesgo</b> shows VaR, '
        'Expected Shortfall, the model backtest, rate/liquidity risk and stress — '
        'all on one screen.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step"><span class="stepnum">3</span><b>Set limits.</b> '
        'In the sidebar, set a VaR or exposure limit. <b>Control</b> then shows '
        'green/red whether the book is within the limit — turning measurement into '
        'control.</div>', unsafe_allow_html=True)

    st.caption("Data sources: spot from yfinance (EUR/GBP triangulated from EUR/USD and "
               "GBP/USD); rate curves from FRED (USD, GBP) and the ECB (EUR); volatility "
               "via GARCH(1,1)-t. Limitations are declared below. Educational tool — "
               "not investment advice.")

with st.expander("⚠️ Scope & limitations", expanded=False):
    st.markdown(
        '<div class="interp"><b>What this tool is.</b> A demonstration of FX risk '
        'methodology — pricing, VaR, stress, backtesting — built on real but free, '
        'delayed data, for EUR/USD, GBP/USD and EUR/GBP. It is rigorous <i>within</i> a '
        'declared scope. <b>What it is not.</b> A production trading or risk '
        'system: those use real-time institutional data, full curves across 130+ '
        'currencies, implied volatility, and validated infrastructure '
        '(Bloomberg, Murex, etc.). The limits below are deliberate and declared — '
        'knowing a model\'s boundaries is part of using it responsibly.</div>',
        unsafe_allow_html=True)

    st.markdown("##### 1 · Data")
    st.markdown(
        "- **Free, delayed data.** Spot from yfinance and rates from FRED/ECB are "
        "free and lightly delayed, not the real-time institutional feeds a desk "
        "uses. Fine for demonstration, not for live trading.\n"
        "- **Three pairs.** EUR/USD and GBP/USD are directly quoted; EUR/GBP is "
        "triangulated from the other two (see fxrisk.data.TRIANGULATED_PAIRS). The "
        "scope was limited on purpose to keep the model focused and defensible.\n"
        "- **GBP curve is a single point.** The GBP rate is one 3-month interbank "
        "rate treated as a flat curve. A full curve would come from the Bank of "
        "England; not wired in.")

    st.markdown("##### 2 · Valuation")
    st.markdown(
        "- **Static snapshot, no ageing.** Positions are valued with their "
        "original tenor — the book is a picture 'as of booking'. There is no "
        "theta/passage-of-time re-ageing of each trade.\n"
        "- **Flat curve extrapolation.** Outside the 3M–2Y anchor range the rate "
        "is held flat (a position with under ~90 days left uses the 3M rate).\n"
        "- **Compounding conventions.** Forwards use simple rates (the CIP "
        "convention), with a currency-aware day-count: ACT/360 for EUR and "
        "USD (Eurocurrency money-market convention), ACT/365 for GBP "
        "(sterling convention) — each leg uses its own currency's basis.\n"
        "- **Common-numeraire MtM.** A non-USD-quoted position's MtM (e.g. EUR/GBP, "
        "in GBP) is converted to USD via the current GBP/USD spot — a declared "
        "quanto-style approximation, requiring the book to also hold that quote "
        "currency's own USD pair.")

    st.markdown("##### 3 · Market risk (VaR)")
    st.markdown(
        "- **Spot VaR.** The VaR measures exchange-rate risk only; interest-rate "
        "risk is reported separately as DV01. They are not combined into one "
        "number.\n"
        "- **Common-numeraire VaR.** Every pair's exposure is converted to USD "
        "before aggregating, so a non-USD-quoted cross (e.g. EUR/GBP) can share "
        "a VaR with quote-USD pairs; the conversion uses the CURRENT USD spot "
        "of the quote currency, a declared quanto-style approximation. A pair "
        "with no USD conversion rate available still fails loud rather than "
        "being summed across currencies incorrectly.\n"
        "- **Normality of the parametric VaR.** The parametric VaR assumes normal "
        "returns. This is mitigated by also reporting historical, EWMA and "
        "Student-t VaR, and by Kupiec + Christoffersen backtests.\n"
        "- **Backtest statistical assumptions.** Asymptotic chi-squared p-values are "
        "unreliable with few exceptions, so an exact Monte Carlo p-value is also "
        "reported; even so, a 'pass' with few exceptions is weak evidence, and "
        "Christoffersen's independence test is blind to non-adjacent clustering. "
        "Full detail in NOTES.md.\n"
        "- **Fixed stress scenarios.** Stress moves are computed from real crisis "
        "windows (Brexit, COVID, UK mini-budget) but are then fixed; they are not "
        "re-derived live. The script that computes them is in the repo for "
        "traceability.")

    st.markdown("##### 4 · Overall")
    st.markdown(
        "- **Demonstration tool.** Educational/illustrative, not investment "
        "advice and not a sellable product.\n"
        "- **No institutional infrastructure.** No real-time data, no 130+ "
        "currency coverage, none of the security, compliance and validation "
        "layers a production system carries.\n"
        "- **Forwards only.** This version covers the provider's forward book. "
        "FX options (Garman-Kohlhagen pricing, Greeks, full-revaluation VaR) "
        "are out of scope here and are being developed as their own tool.")

    st.caption("These limitations are stated so the numbers are read for what they "
               "are: a rigorous demonstration within a clearly bounded scope.")

glossary()
st.write("")


# --------------------------------------------------------------------------
# Sidebar: global risk-limit inputs (persistent across all screens)
# --------------------------------------------------------------------------
if not PAIRS:
    st.error("No supported pairs available.")
    st.stop()

with st.sidebar:
    st.header("Risk limits")
    st.caption("Limits turn risk measurement into risk control. Leave 0 to ignore a "
               "limit. A suggested value appears once the book is valued (in "
               "**Control**).")
    var_limit = st.number_input("VaR limit (USD)", value=0, step=50_000, format="%d",
                                help=HELP["var_limit"])
    exp_limit = st.number_input("Net exposure limit per currency", value=0,
                                step=500_000, format="%d", help=HELP["exp_limit"])
    confidence = st.select_slider("Confidence level", [0.95, 0.975, 0.99], value=0.99,
                                  help=HELP["confidence"])


# --------------------------------------------------------------------------
# Navigation: FOUR flat screens, no nested tabs.
# --------------------------------------------------------------------------
tab_book, tab_val, tab_risk, tab_control = st.tabs(
    ["Libro", "Valuación", "Riesgo", "Control"])

# ============================ 1. LIBRO (Book) ==============================
# NOTE: this tab's button handlers can mutate `book` (add/remove/clear/load
# example) on this very run. It MUST render, and any such mutation MUST be
# applied, before the shared computation block below reads `book` -- otherwise
# a just-added position would be missing from `snapshots` (a stale-state bug).
with tab_book:
    st.subheader("Book actions")
    st.caption("Load a sample book or clear it, or build an individual trade below.")
    cc1, cc2 = st.columns(2)
    if cc1.button("Load example", use_container_width=True,
                  help="Load a 3-position sample book to explore the tool."):
        try:
            with st.spinner("Building example book at live rates..."):
                example = Book()
                for pr, side, notl, ten, spr in [
                    ("EUR/USD", True, 2_000_000, 90, 18),
                    ("EUR/USD", False, 1_200_000, 180, 22),
                    ("GBP/USD", True, 800_000, 120, 25),
                ]:
                    s = get_market_snapshot(pr, ten)
                    rate = client_rate_with_spread(s.forward(), spr, side)
                    example.add(Position(pr, not side, float(notl), ten, float(rate),
                                         label="Example"))
                st.session_state.book = example
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load example: {exc}")
    if cc2.button("Clear book", use_container_width=True):
        st.session_state.book = Book()
        st.rerun()

    st.divider()
    st.subheader("Forward — build & price")
    st.caption("Define a client trade; the provider books the mirror at live "
               "market rates. The position is added to the book.")
    fpair = st.selectbox("Currency pair", PAIRS, key="fwd_pair",
                         help="EUR/USD and GBP/USD are directly quoted; EUR/GBP is "
                              "triangulated from the other two.")
    fbase, fquote = fpair.split("/")
    fside = st.radio("Client wants to", [f"Buy {fbase}", f"Sell {fbase}"],
                     key="fwd_side",
                     help="An importer buys the base currency; an exporter sells it.")
    fbuys = fside.startswith("Buy")
    fnotional = st.number_input(f"Notional ({fbase})", value=1_000_000, step=100_000,
                                format="%d", min_value=1, key="fwd_notional",
                                help=HELP["notional"])
    ftenor = st.slider("Tenor (days)", 7, 730, 90, key="fwd_tenor", help=HELP["tenor"])
    fspread = st.number_input("Provider spread (pips)", value=20.0, step=1.0,
                              key="fwd_spread", help=HELP["spread"])

    if st.button("Add forward to book", type="primary"):
        try:
            with st.spinner("Pricing at live market rates..."):
                snap = get_market_snapshot(fpair, ftenor)
                client_rate = client_rate_with_spread(snap.forward(), fspread, fbuys)
                pos = Position(fpair, not fbuys, float(fnotional), int(ftenor),
                               float(client_rate),
                               label=f"Client {'buys' if fbuys else 'sells'} {fbase}")
                book.add(pos)
            st.success(f"Booked: provider {pos.side.lower()} {fnotional:,.0f} "
                       f"{fbase} @ {client_rate:.4f}. Theoretical forward "
                       f"{snap.forward():.4f}; spread {fspread:.0f} pips.")
        except MarketDataError:
            st.error("Live market data is unavailable right now. Please try again "
                     "in a moment — the tool does not use synthetic prices.")
        except Exception as exc:
            st.error(f"Could not book the trade: {exc}")

    st.divider()
    st.subheader("The book")
    st.caption("Each row is a position the provider holds (the mirror of a client "
               "trade). 'Strike' is the rate quoted to the client.")
    if book.is_empty:
        st.info("The book is empty. Add a trade above, or click **Load example**.")
    else:
        id_to_num = _id_to_num()
        st.dataframe(
            [{"#": id_to_num[p.id], "Pair": p.pair, "Provider side": p.side,
              "Notional (base)": f"{p.notional_base:,.0f}",
              "Tenor (days)": p.tenor_days, "Strike": f"{p.strike:.4f}",
              "Note": p.label} for p in book],
            hide_index=True, use_container_width=True)
        num_to_id = {v: k for k, v in id_to_num.items()}
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox("Remove a position", ["--"] + list(num_to_id),
                            help="Pick a position number to remove it from the book.")
        c2.write(""); c2.write("")
        if c2.button("Remove") and pick != "--":
            book.remove(num_to_id[pick]); st.rerun()

        st.divider()
        st.subheader("Exposure by currency")
        st.caption("Net = the amount of each currency the book is long (+) or short (−) "
                   "after offsetting positions. This is what the desk actually carries.")
        net = book.net_exposure_by_currency()
        gross = book.gross_exposure_by_currency()
        cols = st.columns(len(net))
        for col, (ccy, amt) in zip(cols, sorted(net.items())):
            col.metric(f"Net {ccy}", f"{amt:,.0f}",
                       help=f"Gross (ignoring netting): {gross.get(ccy, 0):,.0f} {ccy}")

        _items = sorted(net.items())
        _ccys = [c for c, _ in _items]
        _amts = [a for _, a in _items]
        _bar_colors = [PLOT_ACCENT if a >= 0 else PLOT_AMBER for a in _amts]
        fig_exp = go.Figure(go.Bar(
            x=_amts, y=_ccys, orientation="h", marker_color=_bar_colors,
            text=[f"{a:,.0f}" for a in _amts], textposition="auto",
            hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
        fig_exp.add_vline(x=0, line_color=PLOT_GRID)
        fig_exp.update_xaxes(title_text="Net exposure (currency units)")
        st.plotly_chart(_plotly_layout(fig_exp, height=max(160, 60 * len(_ccys))),
                        use_container_width=True, config={"displayModeBar": False})
        st.caption("Long (positive) and short (negative) are directions, not "
                   "good or bad — this is simply the net FX position the desk carries.")

        st.divider()
        with st.expander("Client view — what each trade looked like to the client"):
            st.caption("The derived, secondary side: the mirror of each provider "
                       "position, as the client who hedged would see it.")
            st.dataframe(
                [{"#": id_to_num[p.id], "Pair": p.pair,
                  "Client": ("Buys " + p.base_ccy) if (not p.long_base)
                            else ("Sells " + p.base_ccy),
                  "Notional": f"{p.notional_base:,.0f}",
                  "Rate quoted": f"{p.strike:.4f}",
                  "Tenor (days)": p.tenor_days} for p in book],
                hide_index=True, use_container_width=True)
            st.caption("The rate quoted already includes the provider's spread — the "
                       "provider's revenue and the client's cost of certainty.")

# --------------------------------------------------------------------------
# Shared heavy computation (once per run), used by Valuación / Riesgo / Control.
# Runs AFTER the Libro tab above, so it sees any add/remove/clear/load-example
# mutation made on this same run.
# --------------------------------------------------------------------------
report = snapshots = pairs = returns = positions = var_report = None
vt = ve = notional_usd = None
governing_var = governing_var_label = None
compute_error = None
if not book.is_empty:
    try:
        with st.spinner("Fetching spot & rates, estimating volatility (GARCH), "
                        "valuing the book and computing risk..."):
            report = value_book(book)
            snapshots = snapshots_for_book(book)
            pairs, returns, positions = factor_setup(book, snapshots)
            var_report = portfolio_var(returns, positions, confidence, pairs)
            notional_usd = book_notional_usd(book, snapshots)
            try:
                vt = var_student_t(returns, positions, confidence)
                ve = var_ewma(returns, positions, confidence)
            except Exception:
                vt = ve = None

            # Governing VaR (H-review): the number limits/status are checked
            # against. Neither the plain historical nor the plain parametric
            # figure is, on its own, the conservative choice -- so we use the
            # MAX of historical and Student-t, the two fat-tail-aware methods.
            # That is the standard prudent choice for limit monitoring, and it
            # is the SAME number and SAME label used everywhere a limit or
            # status check is shown (Control screen, check_limits calls),
            # so the figure is never ambiguous between screens.
            if vt is not None:
                governing_var = max(var_report.var_historical, vt)
                governing_var_label = "max(historical, Student-t)"
            else:
                governing_var = var_report.var_historical
                governing_var_label = "historical (Student-t unavailable)"
    except MarketDataError:
        compute_error = ("Live market data is unavailable right now. The tool does not "
                         "show synthetic prices — please try again in a moment.")
    except Exception as exc:
        compute_error = f"Analysis could not be completed: {exc}"


def _needs_book_guard() -> bool:
    """
    Render a guard message if this screen's content can't be shown yet
    (empty book, or a data/compute error). Returns True if the caller should
    skip its own content this run.
    """
    if book.is_empty:
        st.info("The book is empty. Build a trade in **Libro**, or load the example "
                "book there.")
        return True
    if compute_error:
        st.error(compute_error)
        return True
    return False


# ============================ 2. VALUACIÓN ==================================
with tab_val:
    if _needs_book_guard():
        pass
    else:
        st.subheader("Book valuation (Mark-to-Market)")
        st.caption("What the book is worth today at current market rates, and where "
                   "that value sits.")
        cards([
            {"label": "Total MtM (USD)", "value": f"{report.total_mtm_usd:,.0f}",
             "kind": "accent", "sign": "pos" if report.total_mtm_usd >= 0 else "neg"},
            {"label": "Gains (USD)", "value": f"{report.gains_usd:,.0f}",
             "kind": "ok", "sign": "pos"},
            {"label": "Losses (USD)", "value": f"{report.losses_usd:,.0f}",
             "kind": "bad", "sign": "neg" if report.losses_usd < 0 else ""},
        ])
        st.markdown(
            f'<div class="interp">The book is worth <b>{report.total_mtm_usd:,.0f} USD</b> '
            f'at current market. Gains and losses show the composition behind the net '
            f'figure — a small net can hide large offsetting positions.</div>',
            unsafe_allow_html=True)
        st.write("")

        st.markdown("##### Forward pricing detail")
        st.caption("Strike vs the fair (theoretical) forward, in forward points, and "
                   "the resulting MtM — per position, in its own quote currency.")
        id_to_num = _id_to_num()
        st.dataframe(
            [{"#": id_to_num.get(v.position.id, v.position.id),
              "Pair": v.position.pair,
              "Strike": f"{v.position.strike:.4f}",
              "Fair forward": f"{v.market_forward:.4f}",
              "Forward pts": f"{forward_points(v.spot, v.market_forward):+.1f}",
              f"MtM ({v.quote_ccy})": f"{v.mtm_quote:,.0f}"}
             for v in report.valuations],
            hide_index=True, use_container_width=True)

        st.markdown("##### Rate curves used")
        st.caption("The base/quote rate read from each currency's curve, at each "
                   "pair's tenor — what drives the fair forward away from spot.")
        rate_rows = []
        seen_pairs = set()
        for v in report.valuations:
            if v.position.pair in seen_pairs:
                continue
            seen_pairs.add(v.position.pair)
            snap = snapshots[v.position.id]
            rate_rows.append({
                "Pair": v.position.pair, "Spot": f"{snap.spot:.4f}",
                f"r_{snap.base_ccy}": f"{snap.r_base:.3%}",
                f"r_{snap.quote_ccy}": f"{snap.r_quote:.3%}",
                "Tenor (yrs)": f"{snap.tenor_years:.3f}",
            })
        st.dataframe(rate_rows, hide_index=True, use_container_width=True)

        st.markdown("##### Concentration (share of gross MtM)")
        st.caption("How much of the book's value sits in each position. A high share in "
                   "one row means the book leans heavily on a single trade.")
        st.dataframe(
            [{"#": id_to_num.get(pid, pid), "MtM (USD)": f"{mtm:,.0f}",
              "Share": f"{share:.1f}%"} for pid, mtm, share in report.concentration],
            hide_index=True, use_container_width=True)

        st.markdown("##### Sensitivity to a spot shock")
        st.caption("How the book's value changes if every spot rate moves by the shown "
                   "amount — a quick feel for directional risk before the formal VaR.")
        sens = book_sensitivity(book)
        scols = st.columns(len(sens))
        for col, (shock, mtm) in zip(scols, sorted(sens.items())):
            col.metric(f"{shock:+.0f}% spot", f"{mtm:,.0f}")

        if report.data_flags:
            st.markdown("##### Data-quality notes")
            for f in report.data_flags:
                st.caption(f"• {f}")

# ============================ 3. RIESGO ====================================
with tab_risk:
    if _needs_book_guard():
        pass
    else:
        book_notional = notional_usd or 1.0
        cfg = LimitsConfig(var_limit=var_limit or None, net_exposure_limit=exp_limit or None)

        st.subheader("Risk")
        st.caption("Everything the book could lose, whether the model is trustworthy, "
                   "its interest-rate and liquidity sensitivity, and how it behaves in "
                   "a historical crisis — all on one screen.")

        # ==================== (a) VaR & Expected Shortfall ====================
        st.markdown("### VaR & Expected Shortfall")
        cards([
            {"label": "VaR · parametric", "value": f"{var_report.var_parametric:,.0f}",
             "sub": f"{var_report.var_parametric / book_notional:.2%} of book"},
            {"label": "VaR · historical", "value": f"{var_report.var_historical:,.0f}",
             "sub": f"{var_report.var_historical / book_notional:.2%} of book",
             "kind": "accent"},
            {"label": "VaR · Monte Carlo", "value": f"{var_report.var_montecarlo:,.0f}",
             "sub": f"{var_report.var_montecarlo / book_notional:.2%} of book"},
            {"label": "Expected Shortfall", "value": f"{var_report.expected_shortfall:,.0f}",
             "sub": f"{var_report.expected_shortfall / book_notional:.2%} of book",
             "kind": "warn"},
        ])

        if vt is not None and ve is not None:
            cards([
                {"label": "VaR · Student-t", "value": f"{vt:,.0f}",
                 "sub": "fat-tailed", "kind": "accent"},
                {"label": "VaR · EWMA", "value": f"{ve:,.0f}",
                 "sub": "regime-weighted (λ=0.94)"},
            ])
            st.caption("Student-t captures fat tails the normal VaR misses; EWMA weights "
                       "recent days more, so it reacts faster to the current regime. Both "
                       "are validation-grade refinements over the plain normal VaR.")

            st.markdown("##### VaR methods compared")
            methods = {
                "Parametric": var_report.var_parametric,
                "Historical": var_report.var_historical,
                "Monte Carlo": var_report.var_montecarlo,
                "EWMA": ve,
                "Student-t": vt,
            }
            order = sorted(methods, key=methods.get)
            vals = [methods[m] for m in order]
            colors = [PLOT_ACCENT if m == "Student-t" else PLOT_MUTED for m in order]
            fig_cmp = go.Figure(go.Bar(
                x=vals, y=order, orientation="h", marker_color=colors,
                text=[f"{v:,.0f}" for v in vals], textposition="auto",
                hovertemplate="%{y}: %{x:,.0f} USD<extra></extra>"))
            fig_cmp.update_xaxes(title_text="1-day VaR (USD)")
            st.plotly_chart(_plotly_layout(fig_cmp, height=260),
                            use_container_width=True, config={"displayModeBar": False})
            spread = (max(vals) - min(vals)) / min(vals) if min(vals) > 0 else 0
            _highest = order[-1]
            st.markdown(
                f'<div class="interp">The five methods span <b>{min(vals):,.0f}</b> to '
                f'<b>{max(vals):,.0f}</b> USD ({spread:.0%} apart), with <b>{_highest}</b> '
                f'the most conservative on this book. They disagree by '
                f'design: <b>Parametric</b> assumes a normal distribution; '
                f'<b>Historical</b> makes no distributional assumption; <b>Monte Carlo</b> '
                f'simulates from the covariance; <b>EWMA</b> weights recent days more, so '
                f'it tracks the current regime; <b>Student-t</b> models fat tails and tends '
                f'to be among the most conservative. A wide spread signals fat tails or a '
                f'shifting regime — the normal VaR alone would understate the risk.</div>',
                unsafe_allow_html=True)

        st.caption(f"1-day horizon at {confidence:.1%} confidence. Book notional ≈ "
                   f"{book_notional:,.0f} USD. The methods should broadly agree; "
                   f"differences reveal how fat-tailed the data is.")
        st.markdown(
            '<div class="interp"><b>Scope:</b> this is a <b>spot VaR</b> — it measures '
            'exchange-rate risk only. Interest-rate risk is reported separately as DV01 '
            '(below); the two are not added into one number, which would need a joint '
            'spot-rate covariance model.</div>', unsafe_allow_html=True)

        st.markdown("##### 10-day VaR (regulatory horizon)")
        h1, h2 = st.columns(2)
        h1.metric("VaR historical · 10-day", f"{var_report.var_historical_10d:,.0f}",
                  help="1-day VaR scaled by √10 (Basel square-root-of-time rule).")
        h2.metric("Expected Shortfall · 10-day", f"{var_report.expected_shortfall_10d:,.0f}")
        st.caption("Basel requires a 10-day horizon (the assumed time to unwind positions "
                   "under stress). Scaled by √10, which assumes i.i.d. returns — a declared "
                   "convention.")

        st.markdown("##### P&L distribution")
        st.caption("Each bar is a day's simulated profit/loss. The lines mark the VaR "
                   "and ES — losses to their left are the bad tail.")
        pnl = returns @ positions
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=pnl, nbinsx=50, marker_color=PLOT_ACCENT,
                                   opacity=0.65, name="Daily P&L",
                                   hovertemplate="P&L: %{x:,.0f}<br>Days: %{y}<extra></extra>"))
        fig.add_vline(x=-var_report.var_historical, line_color=PLOT_AMBER, line_width=2,
                      annotation_text=f"VaR {confidence:.0%}", annotation_position="top")
        fig.add_vline(x=-var_report.expected_shortfall, line_color=PLOT_RED,
                      line_width=2, line_dash="dash",
                      annotation_text="ES", annotation_position="top left")
        fig.update_xaxes(title_text="Daily P&L (USD)")
        fig.update_yaxes(title_text="Days")
        st.plotly_chart(_plotly_layout(fig), use_container_width=True,
                        config={"displayModeBar": False})

        st.markdown("##### Risk contribution by factor")
        st.caption("How much of the total risk each currency pair is responsible for. "
                   "This is where the risk comes from.")
        rc = var_report.risk_contribution
        fig2 = go.Figure(go.Bar(
            x=list(rc.values()), y=list(rc.keys()), orientation="h",
            marker_color=PLOT_ACCENT,
            hovertemplate="%{y}: %{x:.1f}% of variance<extra></extra>"))
        fig2.update_xaxes(title_text="% of portfolio variance")
        st.plotly_chart(_plotly_layout(fig2), use_container_width=True,
                        config={"displayModeBar": False})
        _div = var_report.diversification_benefit
        st.metric("Diversification benefit" if _div >= 0 else "Concentration penalty",
                  f"{_div:.0%}", help=HELP["div"])

        if len(pairs) > 1:
            st.markdown("##### Correlation between factors")
            st.caption("How closely the pairs move together. High correlation means little "
                       "diversification — which explains the benefit figure above.")
            corr = np.corrcoef(returns, rowvar=False)
            st.dataframe(
                {pairs[j]: {pairs[i]: round(float(corr[i, j]), 2)
                            for i in range(len(pairs))} for j in range(len(pairs))},
                use_container_width=True)

        st.divider()

        # ==================== (b) Backtesting ====================
        st.markdown("### Backtesting")
        st.caption("The proper validation: the VaR is re-estimated each day from a trailing "
                   "window and tested against the NEXT day's loss — how a model is checked "
                   "in production, not with a constant VaR.")
        try:
            kup = rolling_backtest(returns, positions, confidence, window=250)
            method_note = "Rolling 250-day window, re-estimated daily. "
        except ValueError:
            pnl_bt = returns @ positions
            kup = kupiec_backtest(pnl_bt, np.full(len(pnl_bt), var_report.var_historical),
                                  confidence)
            method_note = "Constant-VaR test (history too short for a rolling window). "
        k1, k2, k3 = st.columns(3)
        k1.metric("Days tested", f"{kup.observations}",
                  help="Out-of-sample days the model was checked on.")
        k2.metric("Exceptions", f"{kup.exceptions}",
                  help=f"Days the loss exceeded the VaR. Expected ~{kup.expected_exceptions:.0f}.")
        k3.metric("Model", "PASS" if kup.passed else "REVIEW",
                  help=f"Exact Monte Carlo p-value {kup.p_value_mc:.2f} (authoritative); "
                       f"asymptotic chi-squared {kup.p_value:.2f}. Above 0.05 = not rejected.")
        kup_agree = ("the two agree — the asymptotic approximation was adequate here"
                    if kup.mc_agrees_with_asymptotic else
                    "they DISAGREE — with this few exceptions the asymptotic chi-squared "
                    "approximation is unreliable, so the Monte Carlo value is authoritative")
        st.markdown(
            f'<div class="interp">{method_note}The VaR was breached <b>{kup.exceptions}</b> '
            f'times in {kup.observations} tested days (expected ~{kup.expected_exceptions:.0f}). '
            f'The Kupiec proportion-of-failures test {"does not reject" if kup.passed else "rejects"} '
            f'the model — exact Monte Carlo p-value <b>{kup.p_value_mc:.2f}</b> '
            f'(asymptotic chi-squared: {kup.p_value:.2f}); {kup_agree}.</div>',
            unsafe_allow_html=True)

        pnl_full = returns @ positions
        chr_res = christoffersen_independence(
            pnl_full, np.full(len(pnl_full), var_report.var_historical))
        chr_agree = ("agrees with the asymptotic approximation"
                    if chr_res["mc_agrees_with_asymptotic"] else
                    "disagrees with the asymptotic approximation — the Monte Carlo value is authoritative")
        st.markdown(
            f'<div class="interp"><b>Independence (Christoffersen):</b> '
            f'{"exceptions are not clustered — good" if chr_res["independent"] else "exceptions cluster — the model may be slow to react"} '
            f'— exact Monte Carlo p-value <b>{chr_res["p_value_mc"]:.2f}</b> '
            f'(asymptotic: {chr_res["p_value"]:.2f}, {chr_agree}). Kupiec checks how many breaches occur; '
            f'this checks whether they bunch together, which a count alone would miss.'
            f'</div>', unsafe_allow_html=True)
        st.caption("See NOTES.md for each test's applicability conditions (asymptotic "
                   "validity, low power with few exceptions, first-order-Markov scope).")

        st.divider()

        # ==================== (c) DV01 & liquidity ====================
        st.markdown("### DV01 & liquidity")
        st.caption("Sensitivity to interest rates, and the cash the book may tie up as "
                   "variation margin.")
        dv_by_ccy, dv_total = dv01_book(book, snapshots)
        st.markdown("##### Interest-rate risk (DV01 by curve)")
        dcols = st.columns(len(dv_by_ccy) + 1)
        for col, (ccy, dv) in zip(dcols, sorted(dv_by_ccy.items())):
            col.metric(f"DV01 {ccy}", f"{dv:,.2f}", help=HELP["dv01"])
        dcols[-1].metric("DV01 total", f"{dv_total:,.2f}",
                         help="Net across curves — small, because the two legs offset. "
                              "The real rate risk is in the differential.")
        st.caption(f"DV01 total ≈ {dv_total:,.2f} is NOT 'no rate risk' — it is near zero "
                   f"because the forward's two legs largely offset under a PARALLEL move "
                   f"of all curves together. The actual rate risk lives in the "
                   f"DIFFERENTIAL between curves (the per-currency figures above) and in "
                   f"the key-rate buckets below, where the curves are free to move "
                   f"independently.")

        st.markdown("##### Key-rate DV01 (by tenor bucket)")
        kr = dv01_book_by_tenor(book, snapshots)
        kcols = st.columns(len(kr))
        for col, (bucket, dv) in zip(kcols, kr.items()):
            col.metric(f"DV01 {bucket}", f"{dv:,.2f}")
        st.caption("Where on the curve the rate risk sits. A single parallel-bump DV01 "
                   "hides this; bucketing by tenor shows what to hedge and with which "
                   "instruments.")

        st.markdown("##### Liquidity (variation margin)")
        liq = liquidity_book(book, snapshots, confidence, returns=returns, positions=positions)
        st.metric("Liquidity buffer (USD)", f"{liq:,.0f}", help=HELP["liq"])
        st.markdown(
            '<div class="interp">A book hedged in profit-and-loss can still need cash, '
            'because the bank charges margin daily while the client settles only at '
            'maturity. This is the buffer Treasury should keep available. '
            '<i>(Based on the book\'s real aggregate P&L volatility, scaled over the '
            'margin horizon.)</i></div>', unsafe_allow_html=True)

        st.divider()

        # ==================== (d) Stress ====================
        st.markdown("### Stress")
        st.caption("Historical crisis scenarios applied to today's book, and the VaR "
                   "recalibrated to the worst observed period.")
        st.markdown("##### Historical stress scenarios")
        st.caption("Each row applies the real market moves of a past crisis to today's "
                   "book. A positive P&L is a gain, a negative one a loss. 'Loss × VaR' "
                   "shows, for a loss, how many times a normal-day VaR it is.")
        stress = stress_book(book, snapshots, var_report.var_historical)
        st.dataframe(
            [{"Scenario": name,
              "P&L (USD)": f"{r['pnl']:,.0f}",
              "Outcome": "Loss" if r["is_loss"] else "Gain",
              "Loss × VaR": (f"{r['loss_x_var']:.1f}×" if r["is_loss"] else "—")}
             for name, r in stress.items()],
            hide_index=True, use_container_width=True)

        losses = {n: r for n, r in stress.items() if r["is_loss"]}
        if losses:
            worst = min(losses.items(), key=lambda kv: kv[1]["pnl"])
            st.markdown(
                f'<div class="interp">Worst loss: <b>{worst[0]}</b>, '
                f'<b>{worst[1]["pnl"]:,.0f} USD</b> on today\'s book '
                f'(~{worst[1]["loss_x_var"]:.1f}× the VaR). This is the tail a normal-day '
                f'VaR does not capture.</div>', unsafe_allow_html=True)
        else:
            best = max(stress.items(), key=lambda kv: kv[1]["pnl"])
            st.markdown(
                f'<div class="interp">None of these historical crises produces a loss on '
                f'today\'s book — it is positioned to <b>gain</b> from them (the largest '
                f'gain, <b>{best[1]["pnl"]:,.0f} USD</b>, is in the {best[0]} scenario). '
                f'That is itself informative: the book sits on the favourable side of '
                f'these moves. A book with the opposite positioning would show losses '
                f'here.</div>', unsafe_allow_html=True)

        st.markdown("##### Stressed VaR (worst rolling window)")
        st.caption("The VaR recalibrated to the most volatile period in the available "
                   "history — 'how much would we lose if markets behaved like their worst "
                   "observed regime', as Basel requires.")
        try:
            sv = stressed_var(returns, positions, confidence)
            sv1, sv2, sv3 = st.columns(3)
            sv1.metric("Normal-period VaR", f"{sv['normal_var']:,.0f}")
            sv2.metric("Stressed VaR", f"{sv['stressed_var']:,.0f}")
            sv3.metric("Stress multiplier",
                       f"{sv['ratio']:.2f}×" if sv['ratio'] == sv['ratio'] else "n/a",
                       help="How much higher the stressed VaR is than the normal one.")
            st.caption("Declared limit: the stress window is the worst in ~2 years of free "
                       "history; a full implementation would fix a crisis window (e.g. 2008).")
        except ValueError:
            st.caption("Not enough history to identify a stress window for this book "
                       "(needs a longer return series than is currently available).")
        except Exception as exc:
            st.caption(f"Stressed VaR could not be computed: {exc}")

        st.divider()

        # ==================== Export ====================
        st.markdown("##### Export")
        st.caption("Download a formatted Excel risk report (summary, positions, risk "
                   "detail) — the kind of file a treasury desk would circulate.")
        try:
            from report_export import build_excel_report
            from fxrisk.book_risk import (dv01_book as _dv01, liquidity_book as _liq,
                                           stress_book as _stress,
                                           dv01_book_by_tenor as _krd)
            from fxrisk.book_analytics import value_position_from_snapshot
            # Same governing VaR as the Control screen's limit check (see above).
            lim_for_export = check_limits(cfg, var_value=governing_var,
                                          net_exposure=book.net_exposure_by_currency())
            _rows = []
            for p in book:
                snap = snapshots[p.id]
                mtm = value_position_from_snapshot(p, snap).mtm_quote
                _rows.append({
                    "pair": p.pair,
                    "side": ("Provider sells " + p.base_ccy) if (not p.long_base)
                            else ("Provider buys " + p.base_ccy),
                    "notional": p.notional_base, "rate": p.strike,
                    "tenor": p.tenor_days, "mtm": mtm})
            _summary = {
                "notional": book_notional, "book_value": report.total_mtm_usd,
                "confidence": confidence,
                "var_parametric": var_report.var_parametric,
                "var_historical": var_report.var_historical,
                "var_montecarlo": var_report.var_montecarlo,
                "var_ewma": ve, "var_student_t": vt,
                "expected_shortfall": var_report.expected_shortfall,
                "limits": [{"label": c.name, "status": c.status,
                            "ok": not c.breached} for c in lim_for_export.checks],
            }
            _dvc, _ = _dv01(book, snapshots)
            _detail = {
                "dv01_by_ccy": _dvc,
                "dv01_by_tenor": _krd(book, snapshots),
                "liquidity": _liq(book, snapshots, confidence, returns=returns,
                                  positions=positions),
                "stress": {n: r["pnl"] for n, r in
                           _stress(book, snapshots, var_report.var_historical).items()},
            }
            xlsx_bytes = build_excel_report(book_rows=_rows, summary=_summary,
                                            risk_detail=_detail)
            st.download_button(
                "Download Excel risk report", data=xlsx_bytes,
                file_name="fx_book_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as exc:
            st.caption(f"Report export unavailable: {exc}")

# ============================ 4. CONTROL ====================================
with tab_control:
    st.subheader("Book status")
    if book.is_empty:
        st.caption("No book loaded.")
    else:
        n_pos = len(book)
        st.metric("Positions", f"{n_pos} forward" + ("s" if n_pos != 1 else ""))
        _net = book.net_exposure_by_currency()
        st.caption("Net exposure: " + " · ".join(
            f"{ccy} {amt:,.0f}" for ccy, amt in sorted(_net.items())))

    st.divider()

    if _needs_book_guard():
        pass
    else:
        cfg = LimitsConfig(var_limit=var_limit or None, net_exposure_limit=exp_limit or None)
        lim = check_limits(cfg, var_value=governing_var,
                           net_exposure=book.net_exposure_by_currency())

        # Traffic-light status.
        breached = [c for c in lim.checks if c.breached]
        near = [c for c in lim.checks if not c.breached and c.utilisation >= 80]
        if breached:
            light, label, msg = (PLOT_RED, "BREACH",
                f"{len(breached)} limit(s) breached — action required.")
        elif near:
            light, label, msg = (PLOT_AMBER, "WATCH",
                f"{len(near)} limit(s) above 80% utilisation — monitor closely.")
        elif lim.checks:
            light, label, msg = ("#46B98A", "OK", "All limits within bounds.")
        else:
            light, label, msg = (PLOT_MUTED, "NO LIMITS",
                "No risk limits set — define them in the sidebar to enable monitoring.")

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.8rem;'
            f'padding:0.9rem 1.1rem;border-radius:0.6rem;'
            f'background:{light}22;border-left:5px solid {light};margin-bottom:1rem;">'
            f'<span style="width:0.9rem;height:0.9rem;border-radius:50%;'
            f'background:{light};display:inline-block;"></span>'
            f'<b style="color:{light};font-size:1.05rem;">{label}</b>'
            f'<span style="color:#8A93A3;">· {msg}</span></div>',
            unsafe_allow_html=True)

        cards([
            {"label": "Book notional (USD)", "value": f"{(notional_usd or 0):,.0f}"},
            {"label": "Book value / MtM (USD)",
             "value": f"{report.total_mtm_usd:,.0f}",
             "sign": "pos" if report.total_mtm_usd >= 0 else "neg"},
            {"label": f"VaR · 1-day ({confidence:.0%}) · governing",
             "value": f"{governing_var:,.0f}", "kind": "accent",
             "sub": governing_var_label},
            {"label": "Expected Shortfall",
             "value": f"{var_report.expected_shortfall:,.0f}", "kind": "warn"},
        ])
        _all_var_methods = [v for v in (var_report.var_parametric, var_report.var_historical,
                                        var_report.var_montecarlo, vt, ve) if v is not None]
        st.caption(
            f"The governing VaR — used for this status and for the limit check below — "
            f"is **{governing_var_label}**: the more conservative of the two fat-tail-"
            f"aware methods (Riesgo shows all five side by side, currently spanning "
            f"{min(_all_var_methods):,.0f} to {max(_all_var_methods):,.0f}). Taking the "
            f"max of historical and Student-t is the standard prudent choice for limit "
            f"monitoring — it doesn't pick whichever number is smallest.")

        st.divider()
        st.subheader("Limit control")
        if not var_limit and not exp_limit:
            suggested_var = governing_var * 2
            st.caption(f"No limits set. Tip: a common starting VaR limit is about 2× the "
                       f"current governing VaR (~{suggested_var:,.0f} USD). Set it in the "
                       f"sidebar to see the green/red check here.")
        for c in lim.checks:
            cls = "bad" if c.breached else ("warn" if c.utilisation >= 80 else "ok")
            st.markdown(
                f'<div style="margin:0.3rem 0;"><span class="pill {cls}">{c.status}</span> '
                f'&nbsp; <b>{c.name}</b> &nbsp; '
                f'<span style="font-family:JetBrains Mono,monospace; color:#8A93A3;">'
                f'{c.current:,.0f} / {c.limit:,.0f} · {c.utilisation:.0f}% used</span></div>',
                unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "FX Book Risk Analyzer · Data: yfinance (spot), FRED & ECB (rate curves), "
    "GARCH(1,1)-t (volatility) · Market data is cached for 10 minutes · "
    "Educational/demonstration tool, not investment advice · Assumptions and "
    "data-quality limits are declared above and in NOTES.md.")
