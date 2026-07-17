"""
Property-based tests (Hypothesis).
==================================
Instead of checking fixed examples, these check INVARIANTS that must hold for
ANY valid input: Hypothesis generates hundreds of cases and actively searches
for one that breaks the property. This catches edge cases hand-written tests
miss, and documents the mathematical guarantees the model relies on.
"""
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from fxrisk.forwards import forward_rate, forward_mtm, year_fraction
from fxrisk.risk import var_parametric, var_historical, expected_shortfall

# Reusable strategies for sensible market inputs.
spots = st.floats(min_value=0.5, max_value=2.0)
rates = st.floats(min_value=-0.01, max_value=0.10)
taus = st.floats(min_value=0.02, max_value=2.0)


# ----------------------------- Forwards --------------------------------------
@given(spot=spots, rb=rates, rq=rates, tau=taus)
def test_forward_positive_when_spot_positive(spot, rb, rq, tau):
    """A CIP forward is always positive when spot and (1+r*tau) are positive."""
    assume(1 + rb * tau > 0)
    assert forward_rate(spot, rb, rq, tau) > 0


@given(spot=spots, rb=rates, tau=taus)
def test_forward_equals_spot_when_rates_equal(spot, rb, tau):
    """If base and quote rates are equal, the forward equals the spot."""
    f = forward_rate(spot, rb, rb, tau)
    assert abs(f - spot) < 1e-9


@given(spot=spots, rb=rates, rq=rates, tau=taus, notional=st.floats(1e3, 1e9))
def test_mtm_long_short_mirror(spot, rb, rq, tau, notional):
    """The MtM of a long forward is exactly minus the MtM of the short."""
    assume(1 + rb * tau > 0)
    f = forward_rate(spot, rb, rq, tau)
    long = forward_mtm(notional, spot, f, rq, tau, long_base=True)
    short = forward_mtm(notional, spot, f, rq, tau, long_base=False)
    assert abs(long + short) < 1e-6 * max(1.0, abs(long))


# ------------------------------- VaR -----------------------------------------
@given(seed=st.integers(0, 10_000))
@settings(max_examples=50, deadline=None)
def test_var_non_negative(seed):
    """VaR is a positive loss magnitude; never negative."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((400, 1)) * 0.01
    pos = np.array([1_000_000.0])
    assert var_parametric(r, pos, 0.99) >= 0
    assert var_historical(r, pos, 0.99) >= 0


@given(seed=st.integers(0, 10_000))
@settings(max_examples=50, deadline=None)
def test_var_monotonic_in_confidence(seed):
    """Higher confidence => higher (or equal) VaR."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((600, 1)) * 0.01
    pos = np.array([1_000_000.0])
    assert var_parametric(r, pos, 0.99) >= var_parametric(r, pos, 0.95) - 1e-6


@given(seed=st.integers(0, 10_000))
@settings(max_examples=50, deadline=None)
def test_es_at_least_var(seed):
    """Expected Shortfall is always >= VaR (it averages the worse tail)."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal((1500, 1)) * 0.01     # enough tail points at 99%
    pos = np.array([1_000_000.0])
    assert expected_shortfall(r, pos, 0.99) >= var_historical(r, pos, 0.99) - 1e-6
