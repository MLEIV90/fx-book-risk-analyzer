"""Tests for the option book and its aggregate greeks."""
import numpy as np
from fxrisk.option_book import (OptionPosition, OptionBook, option_book_greeks,
                                option_book_var)
from fxrisk.options import option_delta, option_gamma


def _sample_book():
    book = OptionBook()
    book.add(OptionPosition("EUR/USD", True, 1_000_000, 1.10, 90, 0.08,
                            premium_unit=0.01, label="call1"))
    book.add(OptionPosition("GBP/USD", False, 500_000, 1.27, 180, 0.10,
                            premium_unit=0.02, label="put1"))
    return book


SPOTS = {"EUR/USD": 1.0850, "GBP/USD": 1.2700}
RATES = {"EUR/USD": (0.021, 0.039), "GBP/USD": (0.045, 0.039)}


def test_add_and_len():
    book = _sample_book()
    assert len(book) == 2
    assert not book.is_empty
    assert book.pairs() == ["EUR/USD", "GBP/USD"]


def test_remove_and_clear():
    book = _sample_book()
    first_id = next(iter(book)).id
    book.remove(first_id)
    assert len(book) == 1
    book.clear()
    assert book.is_empty


def test_greeks_aggregate_keys():
    book = _sample_book()
    result = option_book_greeks(book, SPOTS, RATES)
    assert set(result["totals"]) == {"value", "delta", "gamma", "vega", "theta"}
    assert len(result["positions"]) == 2


def test_delta_scaled_by_notional():
    """Total delta must equal sum of per-option delta * notional."""
    from fxrisk.forwards import year_fraction
    book = _sample_book()
    result = option_book_greeks(book, SPOTS, RATES)
    manual = 0.0
    for p in book:
        tau = year_fraction(p.tenor_days)   # ACT/360, consistent with engine
        rb, rq = RATES[p.pair]
        manual += option_delta(SPOTS[p.pair], p.strike, rb, rq, p.vol, tau,
                               p.is_call) * p.notional_base
    assert abs(result["totals"]["delta"] - manual) < 1e-6


def test_call_delta_positive_put_delta_negative():
    """A call contributes positive delta, a put negative."""
    call_book = OptionBook([OptionPosition("EUR/USD", True, 1_000_000, 1.10, 90, 0.08)])
    put_book = OptionBook([OptionPosition("EUR/USD", False, 1_000_000, 1.10, 90, 0.08)])
    call_d = option_book_greeks(call_book, SPOTS, RATES)["totals"]["delta"]
    put_d = option_book_greeks(put_book, SPOTS, RATES)["totals"]["delta"]
    assert call_d > 0
    assert put_d < 0


def test_gamma_positive_for_long_options():
    """Long options (call or put) both have positive gamma."""
    book = _sample_book()
    result = option_book_greeks(book, SPOTS, RATES)
    # Each position is long, so per-position gamma*notional is positive; sum > 0.
    assert result["totals"]["gamma"] > 0


def test_theta_negative_time_decay():
    """A book of long options bleeds value over time: total theta negative."""
    book = _sample_book()
    result = option_book_greeks(book, SPOTS, RATES)
    assert result["totals"]["theta"] < 0


def _returns_dict(seed=0, n=500, vol=0.006):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {"EUR/USD": rng.standard_normal(n) * vol}


def test_full_reval_var_positive_and_bounded():
    """Full-revaluation VaR must be positive and below the notional."""
    book = OptionBook([OptionPosition("EUR/USD", True, 10_000_000, 1.0850, 30, 0.08)])
    r = option_book_var(book, {"EUR/USD": 1.0850}, {"EUR/USD": (0.021, 0.039)},
                        _returns_dict(), n_sims=10000)
    assert 0 < r["var_full_reval"] < 10_000_000


def test_full_reval_differs_from_delta_for_atm():
    """For an ATM option (high gamma), full-reval must differ from delta-equiv."""
    book = OptionBook([OptionPosition("EUR/USD", True, 10_000_000, 1.0850, 30, 0.08)])
    r = option_book_var(book, {"EUR/USD": 1.0850}, {"EUR/USD": (0.021, 0.039)},
                        _returns_dict(), n_sims=20000)
    # Non-trivial gamma effect: the two methods disagree.
    assert abs(r["gamma_effect"]) > 1.0
    assert abs(r["gamma_effect"]) / r["var_delta_equiv"] > 0.05


def test_full_reval_converges_to_delta_for_deep_itm():
    """For a deep ITM option (low gamma), full-reval ~ delta-equiv (near-linear)."""
    book = OptionBook([OptionPosition("EUR/USD", True, 10_000_000, 0.95, 30, 0.08)])
    r = option_book_var(book, {"EUR/USD": 1.0850}, {"EUR/USD": (0.021, 0.039)},
                        _returns_dict(), n_sims=20000)
    assert abs(r["gamma_effect"]) / r["var_delta_equiv"] < 0.05


def test_full_reval_empty_book_is_zero():
    """An empty option book has zero VaR."""
    r = option_book_var(OptionBook(), {}, {}, {}, n_sims=1000)
    assert r["var_full_reval"] == 0.0
