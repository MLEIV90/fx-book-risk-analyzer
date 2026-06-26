"""Tests for the risk-limit control layer. Pure logic."""
from fxrisk.limits import LimitsConfig, check_limits


def test_var_within_limit_is_ok():
    cfg = LimitsConfig(var_limit=100_000)
    rep = check_limits(cfg, var_value=80_000)
    assert len(rep.checks) == 1
    assert rep.checks[0].status == "OK"
    assert not rep.any_breach
    assert abs(rep.checks[0].utilisation - 80.0) < 1e-9


def test_var_breach_flagged():
    cfg = LimitsConfig(var_limit=100_000)
    rep = check_limits(cfg, var_value=120_000)
    assert rep.checks[0].breached
    assert rep.any_breach
    assert rep.checks[0].status == "BREACH"


def test_warning_zone_at_80pct():
    cfg = LimitsConfig(var_limit=100_000)
    rep = check_limits(cfg, var_value=85_000)
    assert not rep.any_breach
    assert rep.any_warning            # 85% utilisation -> warning, not breach


def test_unset_limit_produces_no_check():
    cfg = LimitsConfig()              # nothing set
    rep = check_limits(cfg, var_value=999_999)
    assert rep.checks == []


def test_net_exposure_per_currency():
    cfg = LimitsConfig(net_exposure_limit=1_000_000)
    net = {"EUR": 500_000, "USD": -1_500_000}   # USD breaches
    rep = check_limits(cfg, net_exposure=net)
    by_name = {c.name: c for c in rep.checks}
    assert by_name["Net exposure EUR"].status == "OK"
    assert by_name["Net exposure USD"].status == "BREACH"   # abs(-1.5M) > 1M


def test_liquidity_limit():
    cfg = LimitsConfig(liquidity_limit=200_000)
    rep = check_limits(cfg, liquidity_need=250_000)
    assert rep.checks[0].breached
