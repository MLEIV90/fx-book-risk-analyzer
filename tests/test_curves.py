"""
Tests for the rate-curve logic.

Pure logic only (interpolation + CSV parsing). The live FRED/ECB fetches are
validated by running them directly; tests never depend on the network.
"""
import numpy as np
import pytest

from fxrisk.curves import (
    RateCurve, CurveError, get_rate_curve, supported_currencies, parse_ecb_csv,
)


def test_interpolation_between_points():
    curve = RateCurve("USD", np.array([0.25, 1.0]), np.array([0.04, 0.05]))
    assert abs(curve.rate_at(0.625) - 0.045) < 1e-9


def test_flat_curve_returns_single_rate():
    curve = RateCurve("GBP", np.array([0.25]), np.array([0.038]), is_flat=True)
    assert curve.rate_at(0.1) == 0.038
    assert curve.rate_at(2.0) == 0.038


def test_extrapolation_is_flat_beyond_ends():
    curve = RateCurve("USD", np.array([0.25, 1.0]), np.array([0.04, 0.05]))
    assert curve.rate_at(5.0) == 0.05
    assert curve.rate_at(0.01) == 0.04


def test_unknown_currency_raises():
    assert "ARS" not in supported_currencies()
    with pytest.raises(CurveError):
        get_rate_curve("ARS")


def test_majors_are_configured():
    for ccy in ("USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"):
        assert ccy in supported_currencies()


def test_parse_ecb_csv_returns_decimal():
    # Minimal ECB-style CSV: the parser takes the last OBS_VALUE and /100.
    sample = (
        "KEY,TIME_PERIOD,OBS_VALUE\n"
        "YC...,2026-06-23,2.45\n"
        "YC...,2026-06-24,2.51\n"
    )
    assert abs(parse_ecb_csv(sample) - 0.0251) < 1e-12


def test_parse_ecb_csv_empty_raises():
    with pytest.raises(CurveError):
        parse_ecb_csv("KEY,TIME_PERIOD,OBS_VALUE\n")