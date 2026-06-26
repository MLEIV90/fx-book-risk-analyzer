# FX Book Risk Analyzer

A desk-side tool to build an FX forward book and analyse its risk from the
provider's perspective — valuation, Value at Risk, limits, interest-rate risk,
liquidity and stress testing. Built on **real market data** with a pure,
fully-tested calculation engine.

## What it does

When a client hedges a currency exposure, the FX provider takes the opposite
position. This tool builds the provider's **book** and measures the resulting
risk:

- **Valuation** — mark-to-market of the book, gain/loss composition, concentration.
- **Market risk** — VaR by three methods (parametric, historical, Monte Carlo),
  Expected Shortfall, risk attribution, diversification, 10-day regulatory VaR,
  Stressed VaR, and a rolling out-of-sample backtest (Kupiec).
- **Rate / Liquidity / Stress** — DV01 by curve, variation-margin liquidity buffer,
  and historical stress scenarios (Brexit, COVID, SNB de-peg).
- **Limit control** — VaR and net-exposure limits with green/amber/red status.

## Architecture

- `fxrisk/` — the pure calculation engine (13 modules, no UI, no live data).
  Covered by an automated test suite (`pytest`).
- `app.py` — the Streamlit interface (six screens).
- `app_helpers.py` — glue between the book and the engine.

## Data sources

- **Spot & history** — yfinance
- **Rate curves** — FRED (USD Treasuries) and the ECB (EUR yield curve);
  other currencies use a single 3-month point (declared as a flat curve).
- **Volatility** — historical and GARCH(1,1)-t.

Every figure is **observed**, **derived**, or **assumed**, and the tool always
declares which. If live data cannot be retrieved, it fails clearly rather than
showing synthetic numbers.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Testing

```bash
pytest
```

## Limitations

This is an educational/demonstration tool, not investment advice. It uses free,
delayed data; covers a few major pairs; uses simplified rate curves for some
currencies; and does not include implied volatility, options in the portfolio
VaR, or integration with trading/compliance systems. Assumptions are declared
throughout.
