"""
FX Book Risk Analyzer
=====================
Desk-side tool to build an FX forward book and analyse its risk, from the
provider's perspective. Real market data (yfinance spot, FRED/ECB rate curves,
GARCH volatility); pure tested engine in `fxrisk`.

Screens: Overview | Book | Valuation | Market risk | Rate/Liquidity/Stress | Client.
The interface is self-explanatory: every input and metric carries a tooltip, and
the Overview tab guides a first-time user step by step.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fxrisk.curves import supported_currencies
from fxrisk.market import get_market_snapshot, MarketSnapshot
from fxrisk.forwards import client_rate_with_spread
from fxrisk.book import Position, Book
from fxrisk.data import MarketDataError
from fxrisk.book_analytics import value_book, book_sensitivity
from fxrisk.portfolio_risk import (
    portfolio_var, kupiec_backtest, rolling_backtest, stressed_var,
    var_student_t, var_ewma, christoffersen_independence)
from fxrisk.book_risk import (dv01_book, liquidity_book, stress_book,
                              dv01_book_by_tenor)
from fxrisk.limits import LimitsConfig, check_limits
from app_helpers import snapshots_for_book, factor_setup

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
PAIRS = [f"{b}/USD" for b in ("EUR", "GBP") if b in SUPPORTED]

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
    with st.expander("Glossary — key terms in one line"):
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
            "(were breaches as frequent as the model implies?).\n"
            "- **Stress test** — applying a real past crisis's moves to today's book.")


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
# Sidebar: trade entry + limits
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Book actions")
    st.caption("Load a sample book or clear it. Build individual trades in the "
               "Instruments → Forward tab.")
    if not PAIRS:
        st.error("No supported pairs available.")
        st.stop()

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
    st.header("Risk limits")
    st.caption("Limits turn risk measurement into risk control. Leave 0 to ignore a "
               "limit. A suggested value appears once the book is valued.")
    var_limit = st.number_input("VaR limit (USD)", value=0, step=50_000, format="%d",
                                help=HELP["var_limit"])
    exp_limit = st.number_input("Net exposure limit per currency", value=0,
                                step=500_000, format="%d", help=HELP["exp_limit"])
    confidence = st.select_slider("Confidence level", [0.95, 0.975, 0.99], value=0.99,
                                  help=HELP["confidence"])


# --------------------------------------------------------------------------
# Navigation: two zones -- Instruments (price/explore) and Book & Risk (manage)
# --------------------------------------------------------------------------
tab_home, tab_instruments, tab_bookrisk = st.tabs(
    ["Overview", "Instruments", "Book & Risk"])

# ============================ 0. OVERVIEW ================================
with tab_home:
    st.subheader("What this tool does")
    st.markdown(
        "This is a risk-desk tool for an FX provider — a firm that sells currency "
        "hedges (forwards) to clients. When a client hedges, the provider takes the "
        "**opposite position**; the risk the client transfers becomes the provider's "
        "to manage. This tool builds the provider's **book** and measures its risk.")

    cestA, cestB = st.columns(2)
    with cestA:
        st.markdown("**For a risk/quant reader**")
        st.caption("VaR by three methods, Expected Shortfall, Kupiec backtesting, "
                   "risk attribution, DV01, liquidity and historical stress — over a "
                   "real multi-currency book, with limit control.")
    with cestB:
        st.markdown("**For a business reader**")
        st.caption("How much the book is worth, how much it could lose on a bad day, "
                   "whether that is within approved limits, and how it behaves in a "
                   "crisis — all in plain numbers.")

    st.divider()
    st.subheader("How to use it — three steps")
    st.markdown(
        '<div class="step"><span class="stepnum">1</span><b>Build the book.</b> '
        'Add a trade in <b>Instruments → Forward</b> (or click <i>Load example</i> '
        'in the sidebar). Each trade is priced at live market rates and the '
        'provider\'s mirror position is recorded. See it under <b>Book & Risk → '
        'Book</b>.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="step"><span class="stepnum">2</span><b>Read the analysis.</b> '
        'Under <b>Book & Risk</b>: <b>Valuation</b> shows what the book is worth; '
        '<b>Market risk</b> shows VaR, Expected Shortfall and the model backtest; '
        '<b>Rates & Liquidity</b> shows interest-rate, cash and crisis risk.</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="step"><span class="stepnum">3</span><b>Set limits.</b> '
        'In the sidebar, set a VaR or exposure limit. The Market risk tab then shows '
        'green/red whether the book is within the limit — turning measurement into '
        'control.</div>', unsafe_allow_html=True)

    st.divider()
    cest1, cest2 = st.columns([1, 2])
    with cest1:
        if st.button("Load example book to start", type="primary",
                     use_container_width=True):
            try:
                with st.spinner("Building example book..."):
                    example = Book()
                    for pr, side, notl, ten, spr in [
                        ("EUR/USD", True, 2_000_000, 90, 18),
                        ("EUR/USD", False, 1_200_000, 180, 22),
                        ("GBP/USD", True, 800_000, 120, 25),
                    ]:
                        s = get_market_snapshot(pr, ten)
                        rate = client_rate_with_spread(s.forward(), spr, side)
                        example.add(Position(pr, not side, float(notl), ten,
                                             float(rate), label="Example"))
                    st.session_state.book = example
                st.rerun()
            except Exception as exc:
                st.error(f"Could not load example: {exc}")
    with cest2:
        st.caption("Data sources: spot from yfinance; rate curves from FRED (USD) and "
                   "the ECB (EUR); volatility via GARCH(1,1)-t. Limitations are declared "
                   "in each section. Educational tool — not investment advice.")
    glossary()

# Define the sub-tabs inside each zone. The existing `with tab_*:` blocks below
# attach to these, so the two-zone nesting works without re-indenting content.
with tab_instruments:
    st.caption("Price and explore a single instrument before putting it in the "
               "book. Forwards are live; options arrive next.")
    sub_fwd, sub_opt = st.tabs(["Forward", "Option"])

with tab_bookrisk:
    st.caption("The book holds every instrument; the analysis below is for the "
               "whole book.")
    tab_book, tab_val, tab_mkt, tab_rls, tab_client = st.tabs(
        ["Book", "Valuation", "Market risk", "Rates & Liquidity", "Client"])

# ===================== INSTRUMENTS · FORWARD =============================
with sub_fwd:
    st.subheader("Forward — build & price")
    st.caption("Define a client trade; the provider books the mirror at live "
               "market rates. The position is added to the book.")
    fpair = st.selectbox("Currency pair", PAIRS, key="fwd_pair",
                         help="EUR/USD and GBP/USD (base USD).")
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

# ===================== INSTRUMENTS · OPTION (placeholder) ================
with sub_opt:
    st.subheader("Option — call & put")
    st.info("Coming next: option pricing (Garman-Kohlhagen), the Greeks "
            "(delta, gamma, vega, theta), payoff and sensitivity charts. The "
            "pricing engine already exists in `fxrisk.options`; this screen will "
            "expose it.")

# ============================ 1. BOOK =====================================
with tab_book:
    st.subheader("The book")
    st.caption("Each row is a position the provider holds (the mirror of a client "
               "trade). 'Strike' is the rate quoted to the client.")
    if book.is_empty:
        st.info("The book is empty. Build a trade in Instruments → Forward, or load "
                "the example book from the sidebar.")
    else:
        id_to_num = {p.id: f"#{i+1:03d}" for i, p in enumerate(book)}
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

# Empty-book guard for analysis tabs.
if book.is_empty:
    for t, msg in [(tab_val, "value the book"), (tab_mkt, "measure market risk"),
                   (tab_rls, "see rate, liquidity and stress"),
                   (tab_client, "see the client view")]:
        with t:
            st.info(f"Add positions first to {msg}. Build a forward in Instruments → "
                    f"Forward, or load the example book.")
    st.stop()

# Shared heavy computations.
try:
    with st.spinner("Fetching spot & rates, estimating volatility (GARCH), "
                    "valuing the book and computing risk..."):
        report = value_book(book)
        snapshots = snapshots_for_book(book)
        pairs, returns, positions = factor_setup(book, snapshots)
        var_report = portfolio_var(returns, positions, confidence, pairs)
except MarketDataError:
    st.error("Live market data is unavailable right now. The tool does not show "
             "synthetic prices — please try again in a moment.")
    st.stop()
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.stop()

book_notional_usd = sum(abs(p.notional_base) * snapshots[p.id].spot for p in book)

# ============================ 2. VALUATION ================================
with tab_val:
    st.subheader("Book valuation (Mark-to-Market)")
    st.caption("What the book is worth today at current market rates, and where that "
               "value sits.")
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

    st.markdown("##### Concentration (share of gross MtM)")
    st.caption("How much of the book's value sits in each position. A high share in "
               "one row means the book leans heavily on a single trade.")
    id_to_num = {p.id: f"#{i+1:03d}" for i, p in enumerate(book)}
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

# ============================ 3. MARKET RISK =============================
with tab_mkt:
    st.subheader("Market risk")
    st.caption("How much the book could lose on a bad day, whether that number is "
               "trustworthy, and where the risk comes from.")

    cards([
        {"label": "VaR · parametric", "value": f"{var_report.var_parametric:,.0f}",
         "sub": f"{var_report.var_parametric / book_notional_usd:.2%} of book"},
        {"label": "VaR · historical", "value": f"{var_report.var_historical:,.0f}",
         "sub": f"{var_report.var_historical / book_notional_usd:.2%} of book",
         "kind": "accent"},
        {"label": "VaR · Monte Carlo", "value": f"{var_report.var_montecarlo:,.0f}",
         "sub": f"{var_report.var_montecarlo / book_notional_usd:.2%} of book"},
        {"label": "Expected Shortfall", "value": f"{var_report.expected_shortfall:,.0f}",
         "sub": f"{var_report.expected_shortfall / book_notional_usd:.2%} of book",
         "kind": "warn"},
    ])

    # Fat-tail and regime-aware VaR (audit improvements B1, B2).
    try:
        vt = var_student_t(returns, positions, confidence)
        ve = var_ewma(returns, positions, confidence)
        cards([
            {"label": "VaR · Student-t", "value": f"{vt:,.0f}",
             "sub": "fat-tailed", "kind": "accent"},
            {"label": "VaR · EWMA", "value": f"{ve:,.0f}",
             "sub": "regime-weighted (λ=0.94)"},
        ])
        st.caption("Student-t captures fat tails the normal VaR misses; EWMA weights "
                   "recent days more, so it reacts faster to the current regime. Both "
                   "are validation-grade refinements over the plain normal VaR.")
    except Exception:
        pass
    st.caption(f"1-day horizon at {confidence:.1%} confidence. Book notional ≈ "
               f"{book_notional_usd:,.0f} USD. The three methods should broadly agree; "
               f"differences reveal how fat-tailed the data is.")
    st.markdown(
        '<div class="interp"><b>Scope:</b> this is a <b>spot VaR</b> — it measures '
        'exchange-rate risk only. Interest-rate risk is reported separately as DV01 '
        '(Rate / Liquidity / Stress tab); the two are not added into one number, '
        'which would need a joint spot-rate covariance model.</div>',
        unsafe_allow_html=True)

    # 10-day regulatory horizon.
    st.markdown("##### 10-day VaR (regulatory horizon)")
    h1, h2 = st.columns(2)
    h1.metric("VaR historical · 10-day", f"{var_report.var_historical_10d:,.0f}",
              help="1-day VaR scaled by √10 (Basel square-root-of-time rule).")
    h2.metric("Expected Shortfall · 10-day", f"{var_report.expected_shortfall_10d:,.0f}")
    st.caption("Basel requires a 10-day horizon (the assumed time to unwind positions "
               "under stress). Scaled by √10, which assumes i.i.d. returns — a declared "
               "convention.")

    # Limits with a suggested default shown to the user.
    suggested_var = var_report.var_historical * 2
    st.markdown("##### Limit control")
    if not var_limit and not exp_limit:
        st.caption(f"No limits set. Tip: a common starting VaR limit is about 2× the "
                   f"current VaR (~{suggested_var:,.0f} USD). Set it in the sidebar to "
                   f"see the green/red check here.")
    cfg = LimitsConfig(var_limit=var_limit or None,
                       net_exposure_limit=exp_limit or None)
    lim = check_limits(cfg, var_value=var_report.var_historical,
                       net_exposure=book.net_exposure_by_currency())
    for c in lim.checks:
        cls = "bad" if c.breached else ("warn" if c.utilisation >= 80 else "ok")
        st.markdown(
            f'<div style="margin:0.3rem 0;"><span class="pill {cls}">{c.status}</span> '
            f'&nbsp; <b>{c.name}</b> &nbsp; '
            f'<span style="font-family:JetBrains Mono,monospace; color:#8A93A3;">'
            f'{c.current:,.0f} / {c.limit:,.0f} · {c.utilisation:.0f}% used</span></div>',
            unsafe_allow_html=True)

    st.write("")
    g1, g2 = st.columns(2)
    with g1:
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
    with g2:
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

    st.metric("Diversification benefit",
              f"{var_report.diversification_benefit:.0%}", help=HELP["div"])

    if len(pairs) > 1:
        st.markdown("##### Correlation between factors")
        st.caption("How closely the pairs move together. High correlation means little "
                   "diversification — which explains the benefit figure above.")
        corr = np.corrcoef(returns, rowvar=False)
        st.dataframe(
            {pairs[j]: {pairs[i]: round(float(corr[i, j]), 2)
                        for i in range(len(pairs))} for j in range(len(pairs))},
            use_container_width=True)

    st.markdown("##### Stressed VaR")
    st.caption("The VaR recalibrated to the most volatile period in the available "
               "history — 'how much would we lose if markets behaved like their worst "
               "observed regime', as Basel requires.")
    try:
        sv = stressed_var(returns, positions, confidence)
        sv1, sv2, sv3 = st.columns(3)
        sv1.metric("Normal-period VaR", f"{sv['normal_var']:,.0f}")
        sv2.metric("Stressed VaR", f"{sv['stressed_var']:,.0f}")
        sv3.metric("Stress multiplier", f"{sv['ratio']:.2f}×",
                   help="How much higher the stressed VaR is than the normal one.")
        st.caption("Declared limit: the stress window is the worst in ~2 years of free "
                   "history; a full implementation would fix a crisis window (e.g. 2008).")
    except Exception:
        st.caption("Not enough history to compute a stressed VaR for this book.")

    st.markdown("##### Backtesting (rolling, out-of-sample)")
    st.caption("The proper validation: the VaR is re-estimated each day from a trailing "
               "window and tested against the NEXT day's loss — how a model is checked "
               "in production, not with a constant VaR.")
    try:
        kup = rolling_backtest(returns, positions, confidence, window=250)
        method_note = ("Rolling 250-day window, re-estimated daily. ")
    except ValueError:
        # Fall back to the simple constant-VaR test if history is too short.
        pnl = returns @ positions
        kup = kupiec_backtest(pnl, np.full(len(pnl), var_report.var_historical),
                              confidence)
        method_note = ("Constant-VaR test (history too short for a rolling window). ")
    k1, k2, k3 = st.columns(3)
    k1.metric("Days tested", f"{kup.observations}",
              help="Out-of-sample days the model was checked on.")
    k2.metric("Exceptions", f"{kup.exceptions}",
              help=f"Days the loss exceeded the VaR. Expected ~{kup.expected_exceptions:.0f}.")
    k3.metric("Model", "PASS" if kup.passed else "REVIEW",
              help=f"Kupiec p-value {kup.p_value:.2f}. Above 0.05 = not rejected.")
    st.markdown(
        f'<div class="interp">{method_note}The VaR was breached <b>{kup.exceptions}</b> '
        f'times in {kup.observations} tested days (expected ~{kup.expected_exceptions:.0f}). '
        f'The Kupiec proportion-of-failures test {"does not reject" if kup.passed else "rejects"} '
        f'the model (p-value {kup.p_value:.2f}).</div>', unsafe_allow_html=True)

    # Christoffersen independence test (audit improvement B5).
    pnl_full = returns @ positions
    chr_res = christoffersen_independence(
        pnl_full, np.full(len(pnl_full), var_report.var_historical))
    st.markdown(
        f'<div class="interp"><b>Independence (Christoffersen):</b> '
        f'{"exceptions are not clustered — good" if chr_res["independent"] else "exceptions cluster — the model may be slow to react"} '
        f'(p-value {chr_res["p_value"]:.2f}). Kupiec checks how many breaches occur; '
        f'this checks whether they bunch together, which a count alone would miss.'
        f'</div>', unsafe_allow_html=True)

# ==================== 4. RATE / LIQUIDITY / STRESS ======================
with tab_rls:
    st.subheader("Rate risk, liquidity and stress")
    st.caption("Three risks beyond market VaR: sensitivity to interest rates, the cash "
               "the book may tie up, and how it behaves in a historical crisis.")

    dv_by_ccy, dv_total = dv01_book(book, snapshots)
    st.markdown("##### Interest-rate risk (DV01 by curve)")
    dcols = st.columns(len(dv_by_ccy) + 1)
    for col, (ccy, dv) in zip(dcols, sorted(dv_by_ccy.items())):
        col.metric(f"DV01 {ccy}", f"{dv:,.2f}", help=HELP["dv01"])
    dcols[-1].metric("DV01 total", f"{dv_total:,.2f}",
                     help="Net across curves — small, because the two legs offset. "
                          "The real rate risk is in the differential.")
    st.caption("Value change per 1bp move, per currency curve. The legs of a forward "
               "partly offset, so the rate risk lives in the differential between curves.")

    # Key-rate DV01 by tenor bucket (audit improvement B4).
    st.markdown("##### Key-rate DV01 (by tenor bucket)")
    kr = dv01_book_by_tenor(book, snapshots)
    kcols = st.columns(len(kr))
    for col, (bucket, dv) in zip(kcols, kr.items()):
        col.metric(f"DV01 {bucket}", f"{dv:,.2f}")
    st.caption("Where on the curve the rate risk sits. A single parallel-bump DV01 "
               "hides this; bucketing by tenor shows what to hedge and with which "
               "instruments.")

    st.divider()
    st.markdown("##### Liquidity (variation margin)")
    liq = liquidity_book(book, snapshots, confidence, returns=returns,
                         positions=positions)
    st.metric("Liquidity buffer (USD)", f"{liq:,.0f}", help=HELP["liq"])
    st.markdown(
        '<div class="interp">A book hedged in profit-and-loss can still need cash, '
        'because the bank charges margin daily while the client settles only at '
        'maturity. This is the buffer Treasury should keep available. '
        '<i>(Based on the book\'s real aggregate P&L volatility, scaled over the '
        'margin horizon.)</i></div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("##### Stress testing")
    st.caption("Each row applies the real market moves of a past crisis to today's "
               "book. 'x VaR' shows how many times bigger the loss is than a normal-day VaR.")
    stress = stress_book(book, snapshots, var_report.var_historical)
    st.dataframe(
        [{"Scenario": name, "P&L (USD)": f"{r['pnl']:,.0f}",
          "x VaR": f"{r['x_var']:.1f}x"} for name, r in stress.items()],
        hide_index=True, use_container_width=True)
    worst = min(stress.items(), key=lambda kv: kv[1]["pnl"])
    st.markdown(
        f'<div class="interp">Worst scenario: <b>{worst[0]}</b>, '
        f'<b>{worst[1]["pnl"]:,.0f} USD</b> on today\'s book '
        f'(~{worst[1]["x_var"]:.1f}× the VaR). This is the tail a normal-day VaR does '
        f'not capture.</div>', unsafe_allow_html=True)

# ============================ 5. CLIENT =================================
with tab_client:
    st.subheader("Client view")
    st.caption("The derived, secondary side: what each booked trade looked like to the "
               "client who hedged.")
    id_to_num = {p.id: f"#{i+1:03d}" for i, p in enumerate(book)}
    st.dataframe(
        [{"#": id_to_num[p.id], "Pair": p.pair,
          "Client": ("Buys " + p.base_ccy) if (not p.long_base) else ("Sells " + p.base_ccy),
          "Notional": f"{p.notional_base:,.0f}", "Rate quoted": f"{p.strike:.4f}",
          "Tenor (days)": p.tenor_days} for p in book],
        hide_index=True, use_container_width=True)
    st.caption("The rate quoted already includes the provider's spread — the provider's "
               "revenue and the client's cost of certainty.")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "FX Book Risk Analyzer · Data: yfinance (spot), FRED & ECB (rate curves), "
    "GARCH(1,1)-t (volatility) · Educational/demonstration tool, not investment "
    "advice · Assumptions and data-quality limits are declared in each section.")
