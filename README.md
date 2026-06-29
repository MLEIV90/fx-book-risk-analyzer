# FX Book Risk Analyzer

A desk-side tool that builds a book of FX **forwards and options** and measures its risk from the **provider's** point of view — valuation, Value at Risk, interest-rate risk, liquidity, stress testing and limit control. Built on **real market data**, with a pure, fully-tested calculation engine and a trading-desk interface.

**▶ Live app:** https://fx-book-risk-analyzer-jdyzbrg4acezrkqfhcyrdv.streamlit.app/

---

## The idea: provider-first

When a client hedges a currency exposure, the FX provider takes the **mirror** position and inherits the market risk. If the client buys the base currency, the provider is short it; if the client sells, the provider is long. This tool builds the **provider's book** and measures the risk that the provider — not the client — ends up carrying.

A direct consequence: because a provider is typically short the currencies its importer clients buy, the book often **gains** in a crisis where those currencies fall. The tool reflects this honestly rather than assuming every shock is a loss.

---

## What it does

**Instruments**
- **Forwards** priced by Covered Interest Rate Parity, with mark-to-market, client spread and provider revenue.
- **Options** priced with **Garman-Kohlhagen**, full Greeks (delta, gamma, vega, theta) and a put-call parity check.

**Valuation**
- Mark-to-market of the whole book, gain/loss composition and position concentration.

**Market risk**
- **Value at Risk by five methods** — parametric, historical, Monte Carlo, **EWMA** (λ=0.94) and **Student-t** — shown side by side.
- **Expected Shortfall**, risk attribution per pair, diversification benefit, and the **10-day** regulatory VaR.
- **Stressed VaR** calibrated to the most volatile period in the data.
- **Backtesting** — rolling, out-of-sample **Kupiec** (coverage) and **Christoffersen** (independence) tests.
- **Full-revaluation VaR** for the option book, which re-prices every option in each simulated scenario to capture **gamma** — not just a delta-equivalent approximation.

**Rate / Liquidity / Stress**
- **DV01** by currency and by tenor bucket (key-rate), variation-margin liquidity buffer, and historical stress scenarios (Brexit 2016, COVID 2020, UK mini-budget 2022).

**Control & reporting**
- VaR and net-exposure **limits** with green/amber/red status, an executive **dashboard**, and a formatted **Excel risk report** export.

---

## Architecture

Two layers that never mix:

- **`fxrisk/`** — the pure calculation engine (13 modules, no UI, no global state). Takes numbers, returns numbers. Covered by **121 automated tests**, including property-based tests (Hypothesis).
- **`app.py`** + helpers — the Streamlit interface (five screens: Overview, Dashboard, Instruments, Book & Risk, Limitations), plus caching/retry helpers and the Excel report generator.

Keeping the engine pure means the mathematics can be tested without the interface, a presentation bug can never corrupt a calculation, and the engine is reusable from any front-end.

---

## Data sources

All data is **real** and from official, free sources:

- **Spot & history** — yfinance (EUR/USD, GBP/USD; 2-year daily history).
- **Rate curves** — FRED (USD Treasuries) and the ECB (EUR yield curve), at 3M/6M/1Y/2Y; GBP uses a single 3-month point, **declared as a flat curve**.
- **Volatility** — historical and **GARCH(1,1)-t**, preferring GARCH when available.

Every figure is **observed**, **derived** or **assumed**, and the tool always declares which. If live data cannot be retrieved, it **fails clearly** rather than showing synthetic numbers.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Testing

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## Validation

The numbers are validated at two levels:

- **121 automated tests** covering pricing, the five VaR methods, sign and edge-case invariants, and option Greeks — including **property-based tests** that check invariants (e.g. put-call parity) over thousands of random inputs.
- A **ten-layer audit** across correctness, robustness and experience: pricing verified against analytical references (put-call parity to machine precision; the three core VaRs within 1% of their analytical values), presentation logic checked so on-screen labels match what each number means, edge cases hardened (VaR floored at zero, stressed VaR floored at the normal VaR), and conventions unified (ACT/360 throughout).

A full technical and user manual is included in the repository: **`FX_Book_Risk_Analyzer_Manual.docx`** — it documents every module, formula, screen and calculation in detail.

---

## Limitations (declared by design)

This is an educational / demonstration tool, not investment advice.

- **Spot VaR**, not rate VaR — interest-rate risk is reported separately as DV01, not combined into one number.
- **USD-quoted pairs only** — the portfolio VaR sums exposures in USD; a non-USD-quoted pair fails loud rather than returning a wrong number.
- **Free, delayed data** — suitable for a reproducible demo; a production build would use a professional feed.
- **Flat GBP curve** and a stress window taken from ~2 years of free history.
- **No implied volatility** (no reliable free source for these pairs) and **no NDFs** (used for non-convertible emerging currencies, outside the EUR/GBP/USD scope) — both investigated and deliberately left out, explained in the manual.

Assumptions are declared throughout the app and the manual.
