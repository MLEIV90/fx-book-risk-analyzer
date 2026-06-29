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
from fxrisk.market import get_market_snapshot
from fxrisk.forwards import client_rate_with_spread
from fxrisk.options import (garman_kohlhagen, option_delta, option_gamma,
                            option_vega, option_theta)
from fxrisk.option_book import (OptionPosition, OptionBook, option_book_greeks,
                                option_book_var)
from fxrisk.book import Position, Book
from fxrisk.data import MarketDataError, to_returns
from fxrisk.book_analytics import value_book, book_sensitivity
from fxrisk.portfolio_risk import (
    portfolio_var, kupiec_backtest, rolling_backtest, stressed_var,
    var_student_t, var_ewma, christoffersen_independence)
from fxrisk.book_risk import (dv01_book, liquidity_book, stress_book,
                              dv01_book_by_tenor)
from fxrisk.limits import LimitsConfig, check_limits
from app_helpers import (snapshots_for_book, factor_setup, cached_snapshot,
                         cached_spot_history)

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
if "option_book" not in st.session_state:
    st.session_state.option_book = OptionBook()
option_book: OptionBook = st.session_state.option_book

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

    # H4: persistent at-a-glance status, visible on every screen.
    st.divider()
    st.header("Book status")
    if book.is_empty:
        st.caption("No book loaded.")
    else:
        n_pos = len(book)
        n_opt = len(option_book)
        st.metric("Positions", f"{n_pos} forward" + ("s" if n_pos != 1 else "")
                  + (f" · {n_opt} option" + ("s" if n_opt != 1 else "") if n_opt else ""))
        _net = book.net_exposure_by_currency()
        st.caption("Net exposure: " + " · ".join(
            f"{ccy} {amt:,.0f}" for ccy, amt in sorted(_net.items())))
        st.caption("Full risk numbers in **Dashboard** and **Book & Risk**.")


# --------------------------------------------------------------------------
# Navigation: two zones -- Instruments (price/explore) and Book & Risk (manage)
# --------------------------------------------------------------------------
tab_home, tab_dash, tab_instruments, tab_bookrisk, tab_limits = st.tabs(
    ["Overview", "Dashboard", "Instruments", "Book & Risk", "Limitations"])

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

# ============================ DASHBOARD =================================
with tab_dash:
    st.subheader("Risk dashboard")
    if book.is_empty:
        st.info("No book loaded yet. Build a trade in **Instruments → Forward** or "
                "click **Load example** in the sidebar — the executive summary of "
                "your book's risk will appear here.")
    else:
        try:
            with st.spinner("Summarising book risk..."):
                d_snaps = snapshots_for_book(book)
                d_pairs, d_returns, d_positions = factor_setup(book, d_snaps)
                d_var = portfolio_var(d_returns, d_positions, confidence, d_pairs)
                d_report = value_book(book)
                d_notional = sum(abs(p.notional_base) * d_snaps[p.id].spot for p in book)
                d_net = book.net_exposure_by_currency()
                d_cfg = LimitsConfig(var_limit=var_limit or None,
                                     net_exposure_limit=exp_limit or None)
                d_lim = check_limits(d_cfg, var_value=d_var.var_historical,
                                     net_exposure=d_net)

            # Traffic-light status from the limit checks.
            breached = [c for c in d_lim.checks if c.breached]
            near = [c for c in d_lim.checks if not c.breached and c.utilisation >= 80]
            if breached:
                light, label, msg = (PLOT_RED, "BREACH",
                    f"{len(breached)} limit(s) breached — action required.")
            elif near:
                light, label, msg = (PLOT_AMBER, "WATCH",
                    f"{len(near)} limit(s) above 80% utilisation — monitor closely.")
            elif d_lim.checks:
                light, label, msg = ("#46B98A", "OK",
                    "All limits within bounds.")
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

            # Key metrics at a glance.
            cards([
                {"label": "Book notional (USD)", "value": f"{d_notional:,.0f}"},
                {"label": "Book value / MtM (USD)",
                 "value": f"{d_report.total_mtm_usd:,.0f}",
                 "sign": "pos" if d_report.total_mtm_usd >= 0 else "neg"},
                {"label": f"VaR · 1-day ({confidence:.0%})",
                 "value": f"{d_var.var_historical:,.0f}", "kind": "accent"},
                {"label": "Expected Shortfall",
                 "value": f"{d_var.expected_shortfall:,.0f}", "kind": "warn"},
            ])

            # Limit utilisation bars.
            if d_lim.checks:
                st.markdown("##### Limit utilisation")
                for c in d_lim.checks:
                    cls = "bad" if c.breached else ("warn" if c.utilisation >= 80 else "ok")
                    st.markdown(
                        f'<div style="margin:0.3rem 0;"><span class="pill {cls}">'
                        f'{c.status}</span> &nbsp; <b>{c.name}</b> &nbsp; '
                        f'<span style="font-family:JetBrains Mono,monospace;color:#8A93A3;">'
                        f'{c.current:,.0f} / {c.limit:,.0f} · {c.utilisation:.0f}% used'
                        f'</span></div>', unsafe_allow_html=True)

            # Net exposure by currency.
            st.markdown("##### Net exposure by currency")
            _items = sorted(d_net.items())
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

            st.caption("This is the executive summary. See **Book & Risk** for the "
                       "full VaR methods, backtests, rate/liquidity and stress detail.")
        except MarketDataError:
            st.error("Live market data is unavailable right now. Please try again.")
        except Exception as exc:
            st.error(f"Could not build the dashboard: {exc}")

# Define the sub-tabs inside each zone. The existing `with tab_*:` blocks below
# attach to these, so the two-zone nesting works without re-indenting content.
with tab_instruments:
    st.caption("Price and explore a single instrument before putting it in the "
               "book. Forwards are live; options arrive next.")
    sub_fwd, sub_opt = st.tabs(["Forward", "Option"])

with tab_bookrisk:
    st.caption("The book holds every instrument; the analysis below is for the "
               "whole book.")
    tab_book, tab_val, tab_mkt, tab_rls = st.tabs(
        ["Book", "Valuation", "Market risk", "Rates & Liquidity"])

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
    st.subheader("Option — call & put (Garman-Kohlhagen)")
    st.markdown(
        '<div class="interp"><b>Theoretical price, not a market price.</b> This '
        'screen prices the option with <b>GARCH/historical volatility</b>, not the '
        '<b>implied volatility</b> the market quotes. The real market price would '
        'use implied vol; this is an indicative model price. (See Limitations.)</div>',
        unsafe_allow_html=True)

    oc1, oc2, oc3 = st.columns(3)
    opair = oc1.selectbox("Currency pair", PAIRS, key="opt_pair")
    otype = oc2.radio("Type", ["Call", "Put"], key="opt_type", horizontal=True)
    is_call = otype == "Call"
    otenor = oc3.slider("Tenor (days)", 7, 730, 90, key="opt_tenor")
    obase, oquote = opair.split("/")
    onotional = oc1.number_input(f"Notional ({obase})", value=1_000_000, step=100_000,
                                 format="%d", min_value=1, key="opt_notional")

    if st.button("Price option", type="primary", key="opt_price_btn"):
        try:
            with st.spinner("Fetching spot, rates and volatility..."):
                snap = cached_snapshot(opair, otenor)
                tau = snap.tenor_years
                # GARCH vol when available, else historical (declared fallback).
                vol = snap.vol_garch if snap.vol_garch is not None else snap.vol_historical
                vol_label = "GARCH" if snap.vol_garch is not None else "historical"
                # Default strike = at-the-money forward, but let the user adjust.
                st.session_state["opt_snap"] = {
                    "spot": snap.spot, "fwd": snap.forward(), "r_base": snap.r_base,
                    "r_quote": snap.r_quote, "vol": vol, "tau": tau, "pair": opair,
                    "vol_label": vol_label, "tenor_days": otenor}
        except MarketDataError:
            st.error("Live market data is unavailable right now. Please try again.")
        except Exception as exc:
            st.error(f"Could not price the option: {exc}")

    snapd = st.session_state.get("opt_snap")
    if snapd and snapd["pair"] == opair:
        spot, fwd = snapd["spot"], snapd["fwd"]
        rb, rq, vol, tau = (snapd["r_base"], snapd["r_quote"], snapd["vol"], snapd["tau"])

        # Strike control, defaulting to the at-the-money forward.
        strike = st.number_input("Strike", value=round(float(fwd), 4),
                                 step=0.0010, format="%.4f", key="opt_strike",
                                 help="Default is the at-the-money forward. "
                                      "Move it to see in/out-of-the-money behaviour.")

        # Price and primary Greeks.
        premium_unit = garman_kohlhagen(spot, strike, rb, rq, vol, tau, is_call)
        premium_total = premium_unit * onotional
        delta = option_delta(spot, strike, rb, rq, vol, tau, is_call)
        vega = option_vega(spot, strike, rb, rq, vol, tau)

        st.markdown("##### Price")
        cards([
            {"label": f"Premium (total {oquote})", "value": f"{premium_total:,.0f}",
             "kind": "accent"},
            {"label": "Premium (per unit)", "value": f"{premium_unit:.5f}"},
            {"label": "Spot", "value": f"{spot:.4f}"},
            {"label": "Forward", "value": f"{fwd:.4f}"},
        ])
        st.caption(f"{snapd.get('vol_label','GARCH')} volatility used: {vol:.1%} annual "
                   f"· tenor {tau*365:.0f} days · {otype} struck at {strike:.4f}.")

        st.markdown("##### Primary risk (Greeks)")
        cards([
            {"label": "Delta", "value": f"{delta:.4f}",
             "sub": "hedge ratio (base per option)", "kind": "accent"},
            {"label": "Vega", "value": f"{vega/100:.5f}",
             "sub": "per 1 vol point"},
        ])
        st.caption("Delta: how much the option moves per unit of spot — the amount "
                   "of base currency to hedge. Vega: sensitivity to volatility.")

        with st.expander("Full Greeks & put-call parity check"):
            gamma = option_gamma(spot, strike, rb, rq, vol, tau)
            theta = option_theta(spot, strike, rb, rq, vol, tau, is_call)
            cards([
                {"label": "Gamma", "value": f"{gamma:.4f}",
                 "sub": "delta change per unit spot"},
                {"label": "Theta / day", "value": f"{theta:.6f}",
                 "sub": "time decay (per calendar day)"},
            ])
            # Put-call parity validation: C - P = S*DF_base - K*DF_quote.
            call = garman_kohlhagen(spot, strike, rb, rq, vol, tau, True)
            put = garman_kohlhagen(spot, strike, rb, rq, vol, tau, False)
            import math as _m
            rb_c = _m.log(1 + rb * tau) / tau
            rq_c = _m.log(1 + rq * tau) / tau
            parity_rhs = spot * _m.exp(-rb_c * tau) - strike * _m.exp(-rq_c * tau)
            st.caption(f"Put-call parity: C − P = {call - put:.6f}, "
                       f"S·DF − K·DF = {parity_rhs:.6f} — "
                       f"{'consistent ✓' if abs((call-put)-parity_rhs) < 1e-6 else 'mismatch'}. "
                       f"A passing parity check confirms the pricer is internally consistent.")

        # ---- charts ----
        import numpy as _np
        grid = _np.linspace(spot * 0.85, spot * 1.15, 60)

        st.markdown("##### Payoff at expiry")
        st.caption("Profit/loss per unit of base at maturity, depending on where spot "
                   "settles. The premium paid is the maximum loss.")
        if is_call:
            payoff = _np.maximum(grid - strike, 0.0) - premium_unit
        else:
            payoff = _np.maximum(strike - grid, 0.0) - premium_unit
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=grid, y=payoff, mode="lines",
                                   line=dict(color=PLOT_ACCENT, width=2),
                                   name="Payoff",
                                   hovertemplate="Spot %{x:.4f}<br>P&L %{y:.5f}<extra></extra>"))
        fig_p.add_hline(y=0, line_color=PLOT_GRID)
        fig_p.add_vline(x=strike, line_color=PLOT_AMBER, line_dash="dash",
                        annotation_text="Strike")
        fig_p.update_xaxes(title_text="Spot at expiry")
        fig_p.update_yaxes(title_text="P&L per unit")
        st.plotly_chart(_plotly_layout(fig_p), use_container_width=True,
                        config={"displayModeBar": False})

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("##### Value vs spot (today)")
            st.caption("How the premium changes if spot moves today — smooth, unlike "
                       "the kinked payoff, because time value remains.")
            vals = [garman_kohlhagen(s, strike, rb, rq, vol, tau, is_call) for s in grid]
            fig_v = go.Figure(go.Scatter(x=grid, y=vals, mode="lines",
                              line=dict(color=PLOT_ACCENT, width=2),
                              hovertemplate="Spot %{x:.4f}<br>Value %{y:.5f}<extra></extra>"))
            fig_v.add_vline(x=spot, line_color=PLOT_AMBER, line_dash="dash",
                            annotation_text="Spot now")
            fig_v.update_xaxes(title_text="Spot today")
            fig_v.update_yaxes(title_text="Option value")
            st.plotly_chart(_plotly_layout(fig_v), use_container_width=True,
                            config={"displayModeBar": False})
        with gc2:
            st.markdown("##### Delta & gamma vs spot")
            st.caption("Delta changes as spot moves (that change is gamma) — the "
                       "non-linearity that makes options different from forwards.")
            deltas = [option_delta(s, strike, rb, rq, vol, tau, is_call) for s in grid]
            gammas = [option_gamma(s, strike, rb, rq, vol, tau) for s in grid]
            fig_g = go.Figure()
            fig_g.add_trace(go.Scatter(x=grid, y=deltas, mode="lines", name="Delta",
                                       line=dict(color=PLOT_ACCENT, width=2),
                                       hovertemplate="Spot %{x:.4f}<br>Delta %{y:.4f}<extra></extra>"))
            fig_g.add_trace(go.Scatter(x=grid, y=gammas, mode="lines", name="Gamma",
                                       line=dict(color=PLOT_AMBER, width=2), yaxis="y2",
                                       hovertemplate="Spot %{x:.4f}<br>Gamma %{y:.4f}<extra></extra>"))
            fig_g.update_layout(yaxis2=dict(overlaying="y", side="right",
                                            showgrid=False, title="Gamma"))
            fig_g.update_xaxes(title_text="Spot")
            fig_g.update_yaxes(title_text="Delta")
            st.plotly_chart(_plotly_layout(fig_g), use_container_width=True,
                            config={"displayModeBar": False})

        # ---- add to the (separate) option book ----
        st.divider()
        if st.button("Add to option book", key="opt_add_btn"):
            option_book.add(OptionPosition(
                pair=opair, is_call=is_call, notional_base=float(onotional),
                strike=float(strike), tenor_days=int(snapd["tenor_days"]), vol=float(vol),
                premium_unit=float(premium_unit),
                label=f"{otype} {opair} @ {strike:.4f}"))
            st.success(f"Added to option book: {otype} {onotional:,.0f} {obase} "
                       f"@ {strike:.4f}.")
    else:
        st.caption("Choose the option's terms and click **Price option** to see its "
                   "premium, Greeks and payoff.")

    # ---- the option book, managed by aggregate greeks ----
    st.divider()
    st.markdown("##### Option book (managed by aggregate greeks)")
    st.markdown(
        '<div class="interp">Options are kept in their <b>own book</b>, separate '
        'from the forward book. An option book is managed by its <b>greek '
        'profile</b> (net delta, gamma, vega, theta), not by a linear VaR — a '
        'covariance VaR would misstate non-linear risk. A full-revaluation option '
        'VaR is the correct next step and is documented as future work.</div>',
        unsafe_allow_html=True)

    if option_book.is_empty:
        st.caption("The option book is empty. Price an option above and click "
                   "**Add to option book**.")
    else:
        try:
            with st.spinner("Valuing option book at live rates..."):
                ob_spots, ob_rates = {}, {}
                for pr in option_book.pairs():
                    s = cached_snapshot(pr, 90)
                    ob_spots[pr] = s.spot
                    ob_rates[pr] = (s.r_base, s.r_quote)
                greeks = option_book_greeks(option_book, ob_spots, ob_rates)
            tot = greeks["totals"]
            cards([
                {"label": "Book value", "value": f"{tot['value']:,.0f}",
                 "sub": "sum of option values", "kind": "accent"},
                {"label": "Net delta", "value": f"{tot['delta']:,.0f}",
                 "sub": "base-ccy directional exposure"},
                {"label": "Net gamma", "value": f"{tot['gamma']:,.0f}"},
                {"label": "Net vega", "value": f"{tot['vega']/100:,.0f}",
                 "sub": "per 1 vol point"},
                {"label": "Net theta/day", "value": f"{tot['theta']:,.0f}",
                 "sub": "daily time decay", "kind": "warn"},
            ])
            rows = [{
                "Pair": r["pair"], "Type": r["kind"], "Strike": f"{r['strike']:.4f}",
                "Notional": f"{r['notional']:,.0f}", "Value": f"{r['value']:,.0f}",
                "Delta": f"{r['delta']:,.0f}", "Gamma": f"{r['gamma']:,.0f}",
                "Vega": f"{r['vega']/100:,.0f}", "Theta/day": f"{r['theta']:,.0f}",
            } for r in greeks["positions"]]
            st.dataframe(rows, use_container_width=True, hide_index=True)
            if st.button("Clear option book", key="opt_clear_btn"):
                option_book.clear()
                st.rerun()
            st.caption("Net delta is the directional FX exposure (hedge it with a "
                       "spot/forward). Net gamma shows how fast that delta moves. "
                       "Net vega is exposure to volatility. Net theta is the daily "
                       "bleed from time decay.")

            # A2: full-revaluation VaR of the option book (non-linear).
            st.markdown("##### Option book VaR (full revaluation)")
            try:
                ob_returns = {}
                hist = cached_spot_history(tuple(option_book.pairs()), period="2y")
                hist_ret = to_returns(hist)
                for pr in option_book.pairs():
                    ob_returns[pr] = hist_ret[pr].to_numpy()
                vres = option_book_var(option_book, ob_spots, ob_rates, ob_returns,
                                       confidence=confidence, n_sims=20000)
                cards([
                    {"label": f"VaR · full reval ({confidence:.0%})",
                     "value": f"{vres['var_full_reval']:,.0f}",
                     "sub": "re-prices every option", "kind": "accent"},
                    {"label": "VaR · delta-equivalent",
                     "value": f"{vres['var_delta_equiv']:,.0f}",
                     "sub": "linear approximation"},
                    {"label": "Gamma effect",
                     "value": f"{vres['gamma_effect']:+,.0f}",
                     "sub": "non-linearity captured", "kind": "warn"},
                ])
                st.markdown(
                    '<div class="interp"><b>Full revaluation</b> re-prices each option '
                    'with Garman-Kohlhagen under thousands of simulated spot scenarios, '
                    'so it captures the <b>gamma</b> (curvature) that a linear '
                    '<b>delta-equivalent</b> VaR misses. The <b>gamma effect</b> is the '
                    'gap between them: for long options it is usually negative (positive '
                    'gamma cushions losses), so the linear VaR overstates the risk. '
                    'Scope: only spot is shocked (a spot/delta-gamma VaR); volatility '
                    'risk (vega) is held fixed and would be the next step.</div>',
                    unsafe_allow_html=True)
            except MarketDataError:
                st.caption("Market data unavailable to compute the option book VaR.")
            except Exception as exc:
                st.caption(f"Could not compute option book VaR: {exc}")
        except MarketDataError:
            st.error("Live market data unavailable to value the option book.")
        except Exception as exc:
            st.error(f"Could not value the option book: {exc}")

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

# Empty-book guard for analysis tabs.
if book.is_empty:
    for t, msg in [(tab_val, "value the book"), (tab_mkt, "measure market risk"),
                   (tab_rls, "see rate, liquidity and stress")]:
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
book_notional_usd = book_notional_usd or 1.0   # guard the VaR-as-%-of-book divisor

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

    # --- Shared computations (done once, used across the sub-tabs below) ---
    try:
        vt = var_student_t(returns, positions, confidence)
        ve = var_ewma(returns, positions, confidence)
    except Exception:
        vt = ve = None
    suggested_var = var_report.var_historical * 2
    cfg = LimitsConfig(var_limit=var_limit or None,
                       net_exposure_limit=exp_limit or None)
    lim = check_limits(cfg, var_value=var_report.var_historical,
                       net_exposure=book.net_exposure_by_currency())

    mkt_var, mkt_bt, mkt_stress = st.tabs(
        ["VaR & methods", "Backtesting", "Stress & contribution"])

    # ============== Sub-tab 1: VaR & methods ==============
    with mkt_var:
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

            # A3: side-by-side comparison of all five VaR methods.
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
                x=vals, y=order, orientation="h",
                marker_color=colors,
                text=[f"{v:,.0f}" for v in vals], textposition="auto",
                hovertemplate="%{y}: %{x:,.0f} USD<extra></extra>"))
            fig_cmp.update_xaxes(title_text="1-day VaR (USD)")
            st.plotly_chart(_plotly_layout(fig_cmp, height=260),
                            use_container_width=True, config={"displayModeBar": False})
            spread = (max(vals) - min(vals)) / min(vals) if min(vals) > 0 else 0
            _highest = order[-1]   # the method that is actually largest here
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

        # P&L distribution.
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

    # ============== Sub-tab 2: Backtesting ==============
    with mkt_bt:
        st.markdown("##### Backtesting (rolling, out-of-sample)")
        st.caption("The proper validation: the VaR is re-estimated each day from a trailing "
                   "window and tested against the NEXT day's loss — how a model is checked "
                   "in production, not with a constant VaR.")
        try:
            kup = rolling_backtest(returns, positions, confidence, window=250)
            method_note = ("Rolling 250-day window, re-estimated daily. ")
        except ValueError:
            pnl_bt = returns @ positions
            kup = kupiec_backtest(pnl_bt, np.full(len(pnl_bt), var_report.var_historical),
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

        pnl_full = returns @ positions
        chr_res = christoffersen_independence(
            pnl_full, np.full(len(pnl_full), var_report.var_historical))
        st.markdown(
            f'<div class="interp"><b>Independence (Christoffersen):</b> '
            f'{"exceptions are not clustered — good" if chr_res["independent"] else "exceptions cluster — the model may be slow to react"} '
            f'(p-value {chr_res["p_value"]:.2f}). Kupiec checks how many breaches occur; '
            f'this checks whether they bunch together, which a count alone would miss.'
            f'</div>', unsafe_allow_html=True)

    # ============== Sub-tab 3: Stress & contribution ==============
    with mkt_stress:
        st.markdown("##### Limit control")
        if not var_limit and not exp_limit:
            st.caption(f"No limits set. Tip: a common starting VaR limit is about 2× the "
                       f"current VaR (~{suggested_var:,.0f} USD). Set it in the sidebar to "
                       f"see the green/red check here.")
        for c in lim.checks:
            cls = "bad" if c.breached else ("warn" if c.utilisation >= 80 else "ok")
            st.markdown(
                f'<div style="margin:0.3rem 0;"><span class="pill {cls}">{c.status}</span> '
                f'&nbsp; <b>{c.name}</b> &nbsp; '
                f'<span style="font-family:JetBrains Mono,monospace; color:#8A93A3;">'
                f'{c.current:,.0f} / {c.limit:,.0f} · {c.utilisation:.0f}% used</span></div>',
                unsafe_allow_html=True)

        st.write("")
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

        st.markdown("##### Stressed VaR")
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

        # C1: downloadable professional Excel risk report.
        st.divider()
        st.markdown("##### Export")
        st.caption("Download a formatted Excel risk report (summary, positions, risk "
                   "detail) — the kind of file a treasury desk would circulate.")
        try:
            from report_export import build_excel_report
            from fxrisk.book_risk import (dv01_book as _dv01, liquidity_book as _liq,
                                           stress_book as _stress,
                                           dv01_book_by_tenor as _krd)
            from fxrisk.book_analytics import value_position_from_snapshot
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
                "notional": book_notional_usd, "book_value": report.total_mtm_usd,
                "confidence": confidence,
                "var_parametric": var_report.var_parametric,
                "var_historical": var_report.var_historical,
                "var_montecarlo": var_report.var_montecarlo,
                "var_ewma": ve, "var_student_t": vt,
                "expected_shortfall": var_report.expected_shortfall,
                "limits": [{"label": c.name, "status": c.status,
                            "ok": not c.breached} for c in lim.checks],
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
        worst = min(losses.items(), key=lambda kv: kv[1]["pnl"])   # most negative
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

# ============================ 5. CLIENT =================================
# ============================ LIMITATIONS ================================
with tab_limits:
    st.subheader("Limitations & scope")
    st.markdown(
        '<div class="interp"><b>What this tool is.</b> A demonstration of FX risk '
        'methodology — pricing, VaR, stress, greeks — built on real but free, '
        'delayed data, for EUR/USD and GBP/USD. It is rigorous <i>within</i> a '
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
        "- **Two pairs only.** EUR/USD and GBP/USD, both quoted in USD. The scope "
        "was limited on purpose to keep the model focused and defensible.\n"
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
        "convention); options convert to continuous compounding internally "
        "(Garman-Kohlhagen is a continuous-time model). On short tenors the "
        "numerical difference is small.")

    st.markdown("##### 3 · Market risk (VaR)")
    st.markdown(
        "- **Spot VaR.** The VaR measures exchange-rate risk only; interest-rate "
        "risk is reported separately as DV01. They are not combined into one "
        "number.\n"
        "- **Quote-USD assumption.** The portfolio VaR assumes every pair is "
        "quoted in USD; a non-USD-quoted pair is rejected (fail-loud) rather than "
        "summed across currencies incorrectly.\n"
        "- **Normality of the parametric VaR.** The parametric VaR assumes normal "
        "returns. This is mitigated by also reporting historical, EWMA and "
        "Student-t VaR, and by Kupiec + Christoffersen backtests.\n"
        "- **Fixed stress scenarios.** Stress moves are computed from real crisis "
        "windows (Brexit, COVID, UK mini-budget) but are then fixed; they are not "
        "re-derived live. The script that computes them is in the repo for "
        "traceability.")

    st.markdown("##### 4 · Options")
    st.markdown(
        "- **GARCH volatility, not implied.** Options are priced with "
        "GARCH/historical volatility, **not** the implied volatility the market "
        "quotes. The result is an indicative **model price**, not a market price — "
        "this is the single most important option limitation.\n"
        "- **Option book by greeks, not VaR.** The option book is managed by its "
        "aggregate greek profile (delta, gamma, vega, theta), not a linear VaR, "
        "because a covariance VaR would misstate non-linear risk. A "
        "full-revaluation option VaR is the correct next step and is left as "
        "documented future work.")

    st.markdown("##### 5 · Overall")
    st.markdown(
        "- **Demonstration tool.** Educational/illustrative, not investment "
        "advice and not a sellable product.\n"
        "- **No institutional infrastructure.** No real-time data, no 130+ "
        "currency coverage, no implied-vol surface, none of the security, "
        "compliance and validation layers a production system carries.")

    st.caption("These limitations are stated so the numbers are read for what they "
               "are: a rigorous demonstration within a clearly bounded scope.")

# --------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------
st.divider()
st.caption(
    "FX Book Risk Analyzer · Data: yfinance (spot), FRED & ECB (rate curves), "
    "GARCH(1,1)-t (volatility) · Market data is cached for 10 minutes · "
    "Educational/demonstration tool, not investment advice · Assumptions and "
    "data-quality limits are declared in each section.")
