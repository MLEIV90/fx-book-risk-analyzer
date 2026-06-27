"""
compute_stress_scenarios.py
===========================
One-off calibration script for the historical stress scenarios used by
fxrisk.risk.STRESS_SCENARIOS.

It is NOT part of the application. Run it by hand to (re)compute the real
peak-to-trough spot move of each pair during each crisis window, straight from
market data. Copy the printed values into STRESS_SCENARIOS, with a comment
pointing back to this script -- so the stress numbers are traceable and
reproducible, not hand-typed approximations.

Usage:
    python scripts/compute_stress_scenarios.py

Requires: yfinance (already in requirements.txt).
"""
from __future__ import annotations

import yfinance as yf

# Pairs in scope and their Yahoo tickers.
TICKERS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
}

# Each scenario as a dated window. The move is computed from real prices in the
# window, NOT typed by hand. Windows are the core days of each event.
SCENARIO_WINDOWS = {
    "Brexit referendum (Jun 2016)": ("2016-06-22", "2016-07-08"),
    "COVID crash (Mar 2020)":       ("2020-02-20", "2020-03-23"),
    "UK mini-budget (Sep 2022)":    ("2022-09-22", "2022-09-30"),
}


def peak_to_trough_move(ticker: str, start: str, end: str) -> float | None:
    """
    The worst peak-to-trough return of the pair inside the window: the largest
    drop from a running maximum to a subsequent low. Returns a negative decimal
    (e.g. -0.083 = -8.3%), or None if no data is available for the window.
    """
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data is None or data.empty:
        return None
    close = data["Close"].dropna()
    if close.empty:
        return None
    prices = close.to_numpy().flatten()
    # Largest drop from a running maximum to a later point.
    running_max = prices[0]
    worst = 0.0
    for p in prices:
        running_max = max(running_max, p)
        drop = p / running_max - 1.0
        worst = min(worst, drop)
    return float(worst)


def main() -> None:
    print("Computing stress scenarios from real market data...\n")
    result: dict[str, dict[str, float]] = {}
    for scenario, (start, end) in SCENARIO_WINDOWS.items():
        print(f"{scenario}  [{start} -> {end}]")
        result[scenario] = {}
        for pair, ticker in TICKERS.items():
            move = peak_to_trough_move(ticker, start, end)
            if move is None:
                print(f"  {pair}: NO DATA (declare unavailable, do not invent)")
            else:
                print(f"  {pair}: {move:.4f}  ({move*100:.2f}%)")
                result[scenario][pair] = round(move, 4)
        print()

    print("=" * 60)
    print("Copy this dict into fxrisk/risk.py as STRESS_SCENARIOS,")
    print("with a comment: 'computed by scripts/compute_stress_scenarios.py'.")
    print("=" * 60)
    print("STRESS_SCENARIOS = {")
    for scenario, moves in result.items():
        inner = ", ".join(f'"{p}": {m}' for p, m in moves.items())
        print(f'    "{scenario}": {{{inner}}},')
    print("}")


if __name__ == "__main__":
    main()
