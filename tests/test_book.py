"""Tests for the book module: positions, aggregation, and persistence."""
from fxrisk.book import Position, Book


def _sample_book() -> Book:
    book = Book()
    # Provider long 1,000,000 EUR/USD at 1.10  -> +1,000,000 EUR, -1,100,000 USD
    book.add(Position("EUR/USD", long_base=True, notional_base=1_000_000,
                      tenor_days=90, strike=1.10, id="p1"))
    # Provider short 500,000 EUR/USD at 1.12   -> -500,000 EUR, +560,000 USD
    book.add(Position("EUR/USD", long_base=False, notional_base=500_000,
                      tenor_days=180, strike=1.12, id="p2"))
    return book


def test_position_legs_long():
    p = Position("EUR/USD", True, 1_000_000, 90, 1.10)
    legs = p.legs()
    assert legs["EUR"] == 1_000_000
    assert legs["USD"] == -1_100_000


def test_position_legs_short():
    p = Position("EUR/USD", False, 1_000_000, 90, 1.10)
    legs = p.legs()
    assert legs["EUR"] == -1_000_000
    assert legs["USD"] == 1_100_000


def test_net_exposure_nets_offsetting_positions():
    net = _sample_book().net_exposure_by_currency()
    assert net["EUR"] == 500_000             # 1,000,000 - 500,000
    assert net["USD"] == -1_100_000 + 560_000  # -540,000


def test_gross_exposure_ignores_netting():
    gross = _sample_book().gross_exposure_by_currency()
    assert gross["EUR"] == 1_500_000         # 1,000,000 + 500,000
    assert gross["USD"] == 1_100_000 + 560_000


def test_add_remove_get_len():
    book = _sample_book()
    assert len(book) == 2
    assert book.get("p1").pair == "EUR/USD"
    assert book.remove("p1") is True
    assert len(book) == 1
    assert book.remove("nope") is False      # removing a missing id is False


def test_currencies_and_pairs():
    book = _sample_book()
    book.add(Position("GBP/USD", True, 200_000, 30, 1.27, id="p3"))
    assert book.currencies() == ["EUR", "GBP", "USD"]
    assert book.pairs() == ["EUR/USD", "GBP/USD"]


def test_persistence_round_trip():
    book = _sample_book()
    restored = Book.from_dict(book.to_dict())
    assert len(restored) == 2
    assert restored.net_exposure_by_currency() == book.net_exposure_by_currency()
    assert restored.get("p1").strike == 1.10


def test_empty_book():
    book = Book()
    assert book.is_empty
    assert book.net_exposure_by_currency() == {}
