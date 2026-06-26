"""
fxrisk.limits
=============
Risk-limit checks: the layer that turns risk MEASUREMENT into risk CONTROL.

A risk desk does not only compute VaR -- it controls the book against approved
limits and reports utilisation to the committee. This module compares the book's
risk figures against user-defined limits and returns a clear OK / BREACH status
with the utilisation percentage.

Pure module: it takes numbers (current values and limits) and returns statuses.
No market data, fully testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LimitCheck:
    """The result of checking one metric against its limit."""
    name: str
    current: float
    limit: float
    utilisation: float                # current / limit, as a percentage
    breached: bool
    detail: str = ""

    @property
    def status(self) -> str:
        return "BREACH" if self.breached else "OK"


@dataclass
class LimitsConfig:
    """Limits defined by the desk. None means the limit is not set."""
    var_limit: float | None = None
    net_exposure_limit: float | None = None        # per currency, absolute
    liquidity_limit: float | None = None           # available cash buffer
    warning_threshold: float = 80.0                # % utilisation -> warning


@dataclass
class LimitsReport:
    """All limit checks for the book, for the committee dashboard."""
    checks: list[LimitCheck] = field(default_factory=list)

    @property
    def any_breach(self) -> bool:
        return any(c.breached for c in self.checks)

    @property
    def any_warning(self) -> bool:
        return any(not c.breached and c.utilisation is not None
                   and c.utilisation >= 80.0 for c in self.checks)


def _check(name: str, current: float, limit: float | None,
           detail: str = "") -> LimitCheck | None:
    """Build a LimitCheck, or None if the limit is not set."""
    if limit is None:
        return None
    util = abs(current) / limit * 100.0 if limit != 0 else float("inf")
    return LimitCheck(name=name, current=current, limit=limit,
                      utilisation=util, breached=abs(current) > limit,
                      detail=detail)


def check_limits(config: LimitsConfig, var_value: float | None = None,
                 net_exposure: dict | None = None,
                 liquidity_need: float | None = None) -> LimitsReport:
    """
    Run all configured limit checks against the book's current risk figures.

    var_value:     current portfolio VaR.
    net_exposure:  {currency: net amount} from the book.
    liquidity_need:current liquidity buffer required.
    """
    checks: list[LimitCheck] = []

    if var_value is not None:
        c = _check("VaR", var_value, config.var_limit)
        if c:
            checks.append(c)

    if net_exposure is not None and config.net_exposure_limit is not None:
        # One check per currency; the worst (highest utilisation) leads.
        for ccy, amount in sorted(net_exposure.items()):
            c = _check(f"Net exposure {ccy}", amount, config.net_exposure_limit,
                       detail=ccy)
            if c:
                checks.append(c)

    if liquidity_need is not None:
        c = _check("Liquidity buffer", liquidity_need, config.liquidity_limit,
                   detail="required vs available cash")
        if c:
            checks.append(c)

    return LimitsReport(checks=checks)
