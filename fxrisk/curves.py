"""
fxrisk.curves
=============
Real interest-rate curves per currency, from official free sources.

Sources, per currency:
- USD: FRED Treasury constant-maturity points (1M, 3M, 6M, 1Y). Daily, full curve.
- EUR: ECB yield curve (AAA euro-area government spot rates) at 3M, 6M, 9M, 1Y.
       Daily, full curve, from the ECB Data Portal.
- GBP, JPY, CHF, CAD, AUD: FRED 3-month interbank reference rate (single point,
       monthly) -> treated as a flat curve and SAID SO. A full curve for these
       would come from each central bank (BoE, etc.); not wired in v1.

Honesty: every rate is REAL. Where only one point exists, the curve is flat and
the limitation is recorded in `notes`. Nothing is invented; on failure we raise.
In production all curves would come from a professional feed (Bloomberg/Refinitiv).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import numpy as np

# --- FRED-based currencies: (tenor_years, series_id) -----------------------
FRED_SERIES: dict[str, list[tuple[float, str]]] = {
    # USD curve. Anchor points 3M, 6M, 1Y, 2Y -- aligned with EUR (the ECB
    # curve has no 1M point), so both curves are built the same way.
    # Intermediate tenors (e.g. 9M) are interpolated.
    "USD": [
        (3 / 12, "DGS3MO"), (6 / 12, "DGS6MO"), (1.0, "DGS1"), (2.0, "DGS2"),
    ],
    "GBP": [(3 / 12, "IR3TIB01GBM156N")],
    "JPY": [(3 / 12, "IR3TIB01JPM156N")],
    "CHF": [(3 / 12, "IR3TIB01CHM156N")],
    "CAD": [(3 / 12, "IR3TIB01CAM156N")],
    "AUD": [(3 / 12, "IR3TIB01AUM156N")],
}

# --- ECB yield curve points for EUR: (tenor_years, maturity_code) ----------
# Anchor tenors 3M, 6M, 1Y, 2Y -- the ECB curve starts at 3M (no 1M point).
# Same set as USD; intermediate tenors are interpolated.
ECB_EUR_POINTS: list[tuple[float, str]] = [
    (3 / 12, "SR_3M"), (6 / 12, "SR_6M"), (1.0, "SR_1Y"), (2.0, "SR_2Y"),
]
ECB_BASE_KEY = "B.U2.EUR.4F.G_N_A.SV_C_YM"   # AAA euro-area spot-rate yield curve
ECB_URL = "https://data-api.ecb.europa.eu/service/data/YC/{key}.{mat}"


class CurveError(RuntimeError):
    """Raised when a rate curve cannot be retrieved."""


@dataclass
class RateCurve:
    """A simple interpolated rate curve for one currency."""
    currency: str
    tenors: np.ndarray            # in years, ascending
    rates: np.ndarray             # annual, decimal (0.045 = 4.5%)
    is_flat: bool = False
    notes: str = ""
    sources: list[str] = field(default_factory=list)

    def rate_at(self, tenor_years: float) -> float:
        """Linearly interpolate (and flat-extrapolate) the rate at a tenor."""
        if len(self.tenors) == 1:
            return float(self.rates[0])
        return float(np.interp(tenor_years, self.tenors, self.rates))


def supported_currencies() -> list[str]:
    """Currencies for which a curve is configured."""
    return ["EUR"] + list(FRED_SERIES.keys())


# --------------------------------------------------------------------------
# FRED
# --------------------------------------------------------------------------
def _fetch_fred_latest(series_id: str) -> float:
    """Latest value of a FRED series, as a decimal rate."""
    from pandas_datareader import data as web

    df = web.DataReader(series_id, "fred")
    s = df[series_id].dropna()
    if s.empty:
        raise CurveError(f"FRED series '{series_id}' returned no data.")
    return float(s.iloc[-1]) / 100.0


def _build_fred_curve(currency: str) -> RateCurve:
    tenors, rates, sources, errors = [], [], [], []
    for tenor, series_id in FRED_SERIES[currency]:
        try:
            rates.append(_fetch_fred_latest(series_id))
            tenors.append(tenor)
            sources.append(series_id)
        except Exception as exc:
            errors.append(f"{series_id}: {exc}")
    if not tenors:
        raise CurveError(f"Could not retrieve any rate for {currency}. {errors}")
    order = np.argsort(tenors)
    t, r = np.array(tenors)[order], np.array(rates)[order]
    is_flat = len(t) == 1
    notes = (f"Flat curve: single FRED point for {currency}. Term structure not "
             f"modelled.") if is_flat else ""
    return RateCurve(currency, t, r, is_flat, notes, sources)


# --------------------------------------------------------------------------
# ECB (EUR full curve)
# --------------------------------------------------------------------------
def parse_ecb_csv(text: str) -> float:
    """
    Parse the ECB Data Portal CSV response and return the latest OBS_VALUE as a
    decimal rate. Separated out so it can be unit-tested without the network.
    """
    reader = csv.DictReader(io.StringIO(text))
    rows = [row for row in reader if row.get("OBS_VALUE", "").strip()]
    if not rows:
        raise CurveError("ECB response contained no observations.")
    return float(rows[-1]["OBS_VALUE"]) / 100.0


def _fetch_ecb_spot(maturity_code: str) -> float:
    """Fetch one ECB yield-curve spot rate (e.g. 'SR_6M') as a decimal."""
    import requests

    url = ECB_URL.format(key=ECB_BASE_KEY, mat=maturity_code)
    resp = requests.get(url, params={"lastNObservations": 1, "format": "csvdata"},
                        headers={"Accept": "text/csv"}, timeout=20)
    if resp.status_code != 200 or not resp.text.strip():
        raise CurveError(f"ECB request failed for {maturity_code} "
                         f"(HTTP {resp.status_code}).")
    return parse_ecb_csv(resp.text)


def _build_ecb_eur_curve() -> RateCurve:
    tenors, rates, sources, errors = [], [], [], []
    for tenor, mat in ECB_EUR_POINTS:
        try:
            rates.append(_fetch_ecb_spot(mat))
            tenors.append(tenor)
            sources.append(f"ECB:{mat}")
        except Exception as exc:
            errors.append(f"{mat}: {exc}")
    if not tenors:
        raise CurveError(f"Could not retrieve ECB EUR curve. {errors}")
    order = np.argsort(tenors)
    t, r = np.array(tenors)[order], np.array(rates)[order]
    notes = "" if len(t) > 1 else "Flat curve: only one ECB point retrieved."
    if errors:
        notes = (notes + " Some ECB points failed: " + "; ".join(errors)).strip()
    return RateCurve("EUR", t, r, len(t) == 1, notes, sources)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def get_rate_curve(currency: str) -> RateCurve:
    """Build a real rate curve for a currency from its official source."""
    if currency == "EUR":
        return _build_ecb_eur_curve()
    if currency in FRED_SERIES:
        return _build_fred_curve(currency)
    raise CurveError(f"No rate curve source configured for {currency}.")


def rate_for_tenor(currency: str, tenor_years: float) -> tuple[float, RateCurve]:
    """Convenience: interpolated rate at a tenor, plus the curve."""
    curve = get_rate_curve(currency)
    return curve.rate_at(tenor_years), curve