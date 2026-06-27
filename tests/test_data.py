"""
Tests for the data layer.

These use the SYNTHETIC generator so the test suite never depends on the
network. The real fetch (fetch_spot_history) is exercised manually / in the app.
"""
import numpy as np

from fxrisk.data import (
    synthetic_spot_history, to_returns, correlation_matrix, rolling_correlation,
)

PAIRS = ["EUR/USD", "GBP/USD"]


def test_synthetic_shape():
    prices = synthetic_spot_history(PAIRS, n_days=252)
    assert prices.shape == (252, 2)
    assert list(prices.columns) == PAIRS


def test_returns_one_row_shorter():
    prices = synthetic_spot_history(PAIRS, n_days=100)
    rets = to_returns(prices)
    assert len(rets) == len(prices) - 1     # pct_change drops the first row


def test_correlation_in_range():
    prices = synthetic_spot_history(PAIRS, n_days=252)
    corr = correlation_matrix(to_returns(prices))
    off_diag = corr.iloc[0, 1]
    assert -1.0 <= off_diag <= 1.0
    assert corr.iloc[0, 0] == 1.0           # a series correlates 1 with itself


def test_rolling_correlation_runs():
    prices = synthetic_spot_history(PAIRS, n_days=252)
    rc = rolling_correlation(to_returns(prices), "EUR/USD", "GBP/USD", window=60)
    assert len(rc) > 0
    assert (rc.abs() <= 1.0 + 1e-9).all()

def test_to_returns_single_pair_shape():
    """
    H6: returns from a single-pair price frame must keep one column, not collapse.
    Guards the single-pair path that previously mismatched columns.
    """
    from fxrisk.data import synthetic_spot_history, to_returns
    prices = synthetic_spot_history(["EUR/USD"], n_days=300)
    assert prices.shape[1] == 1
    rets = to_returns(prices)
    assert rets.shape[1] == 1
    assert list(rets.columns) == ["EUR/USD"]
