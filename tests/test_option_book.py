"""Tests for the option book and its aggregate greeks."""
import numpy as np
from fxrisk.option_book import OptionPosition, OptionBook, option_book_greeks
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
    book = _sample_book()
    result = option_book_greeks(book, SPOTS, RATES)
    manual = 0.0
    for p in book:
        tau = p.tenor_days / 365.0
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
