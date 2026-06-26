"""
fxrisk.book
===========
The provider's FX forward book: a collection of positions and the aggregated
exposure a risk desk manages.

Perspective: the book holds the PROVIDER's positions (the opposite of what each
client hedges). Each position is a forward with an agreed rate (the strike),
fixed when the trade is booked.

This module is PURE: it defines positions and aggregates exposures from numbers
already stored on them. It does NOT fetch market data or value positions -- that
is done by combining this with `fxrisk.market` in the layer above. Keeping it
pure makes the book testable in isolation.

Net exposure per currency = net maturity cash flow per currency. A forward where
the provider is long N base of BASE/QUOTE at rate K means: receive +N base, pay
-N*K quote. The book sums these legs across positions, per currency -- the
desk's aggregate FX exposure.
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict


@dataclass
class Position:
    """A single forward position held by the provider."""
    pair: str                       # e.g. "EUR/USD"
    long_base: bool                 # True = provider is long the base currency
    notional_base: float            # size in base-currency units
    tenor_days: int                 # days to maturity
    strike: float                   # agreed forward rate (set when booked)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = ""                 # optional human label

    @property
    def base_ccy(self) -> str:
        return self.pair.split("/")[0]

    @property
    def quote_ccy(self) -> str:
        return self.pair.split("/")[1]

    @property
    def side(self) -> str:
        return "Long" if self.long_base else "Short"

    def legs(self) -> dict[str, float]:
        """
        Maturity cash-flow legs of this position, per currency.
        Long base: +notional base, -notional*strike quote. Short: negated.
        """
        sign = 1.0 if self.long_base else -1.0
        base_amt = sign * self.notional_base
        quote_amt = -sign * self.notional_base * self.strike
        legs: dict[str, float] = defaultdict(float)
        legs[self.base_ccy] += base_amt
        legs[self.quote_ccy] += quote_amt
        return dict(legs)


@dataclass
class Book:
    """A collection of provider positions, with aggregated exposures."""
    positions: list[Position] = field(default_factory=list)

    # --- collection management ---
    def add(self, position: Position) -> None:
        self.positions.append(position)

    def remove(self, position_id: str) -> bool:
        """Remove a position by id. Returns True if one was removed."""
        before = len(self.positions)
        self.positions = [p for p in self.positions if p.id != position_id]
        return len(self.positions) < before

    def get(self, position_id: str) -> Position | None:
        return next((p for p in self.positions if p.id == position_id), None)

    def __len__(self) -> int:
        return len(self.positions)

    def __iter__(self):
        return iter(self.positions)

    @property
    def is_empty(self) -> bool:
        return len(self.positions) == 0

    # --- aggregation ---
    def net_exposure_by_currency(self) -> dict[str, float]:
        """Net maturity cash flow per currency across all positions."""
        net: dict[str, float] = defaultdict(float)
        for p in self.positions:
            for ccy, amt in p.legs().items():
                net[ccy] += amt
        return dict(net)

    def gross_exposure_by_currency(self) -> dict[str, float]:
        """Sum of ABSOLUTE legs per currency (ignores netting)."""
        gross: dict[str, float] = defaultdict(float)
        for p in self.positions:
            for ccy, amt in p.legs().items():
                gross[ccy] += abs(amt)
        return dict(gross)

    def currencies(self) -> list[str]:
        """All currencies touched by the book, sorted."""
        ccys = set()
        for p in self.positions:
            ccys.add(p.base_ccy)
            ccys.add(p.quote_ccy)
        return sorted(ccys)

    def pairs(self) -> list[str]:
        """Distinct pairs in the book, sorted."""
        return sorted({p.pair for p in self.positions})

    # --- persistence ---
    def to_dict(self) -> dict:
        return {"positions": [asdict(p) for p in self.positions]}

    @classmethod
    def from_dict(cls, data: dict) -> "Book":
        return cls(positions=[Position(**pd) for pd in data.get("positions", [])])
