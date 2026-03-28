# tests/test_tax_policy.py
from datetime import date, timedelta
from taxopt.data_types import TaxLot
from taxopt.tax_policy import USCapitalGainsPolicy, LotMethod


def _long_lot(basis: float, acq: date) -> TaxLot:
    return TaxLot("AAPL", 10.0, basis, acq)


def _short_lot(basis: float, long_held_days: int | None = None) -> TaxLot:
    open_date = date(2023, 6, 1)
    meta: dict = {"short_open_date": open_date}
    if long_held_days is not None:
        meta["long_held_since"] = open_date - timedelta(days=long_held_days)
    return TaxLot("AAPL", -10.0, basis, open_date, metadata=meta)


# ---------------------------------------------------------------------------
# classify_gain: long positions
# ---------------------------------------------------------------------------

def test_long_short_term(us_policy):
    assert us_policy.classify_gain(_long_lot(100.0, date(2024, 1, 1)), date(2024, 6, 1), gain=50.0) == "short_term"


def test_long_long_term(us_policy):
    assert us_policy.classify_gain(_long_lot(100.0, date(2022, 1, 1)), date(2023, 6, 1), gain=50.0) == "long_term"


def test_long_exactly_threshold(us_policy):
    assert us_policy.classify_gain(_long_lot(100.0, date(2023, 1, 1)), date(2024, 1, 2), gain=50.0) == "long_term"


# ---------------------------------------------------------------------------
# classify_gain: short positions
# ---------------------------------------------------------------------------

def test_short_gain_always_short_term(us_policy):
    assert us_policy.classify_gain(_short_lot(100.0), date(2024, 6, 1), gain=200.0) == "short_term"


def test_short_loss_no_long_held_is_short_term(us_policy):
    assert us_policy.classify_gain(_short_lot(100.0), date(2024, 6, 1), gain=-200.0) == "short_term"


def test_short_loss_long_held_under_threshold_is_short_term(us_policy):
    assert us_policy.classify_gain(_short_lot(100.0, long_held_days=200), date(2024, 6, 1), gain=-200.0) == "short_term"


def test_short_loss_long_held_over_threshold_is_long_term(us_policy):
    assert us_policy.classify_gain(_short_lot(100.0, long_held_days=400), date(2024, 6, 1), gain=-200.0) == "long_term"


# ---------------------------------------------------------------------------
# netting_rules
# ---------------------------------------------------------------------------

def test_netting_st_loss_offsets_lt_gain(us_policy):
    result = us_policy.netting_rules({"short_term": -300.0, "long_term": 500.0})
    assert result["short_term"] == 0.0
    assert result["long_term"] == 200.0


def test_netting_lt_loss_offsets_st_gain(us_policy):
    result = us_policy.netting_rules({"short_term": 500.0, "long_term": -300.0})
    assert result["short_term"] == 200.0
    assert result["long_term"] == 0.0


def test_netting_both_gains_no_cross_netting(us_policy):
    result = us_policy.netting_rules({"short_term": 300.0, "long_term": 200.0})
    assert result["short_term"] == 300.0
    assert result["long_term"] == 200.0


def test_netting_both_losses_no_cross_netting(us_policy):
    result = us_policy.netting_rules({"short_term": -300.0, "long_term": -200.0})
    assert result["short_term"] == -300.0
    assert result["long_term"] == -200.0


def test_netting_st_loss_larger_than_lt_gain(us_policy):
    result = us_policy.netting_rules({"short_term": -600.0, "long_term": 400.0})
    assert result["short_term"] == -200.0
    assert result["long_term"] == 0.0


# ---------------------------------------------------------------------------
# Lot sorting via match_and_realize
# ---------------------------------------------------------------------------

def test_lot_method_fifo():
    policy = USCapitalGainsPolicy(lot_method=LotMethod.FIFO)
    lots = [
        TaxLot("X", 10.0, 90.0, date(2023, 6, 1)),
        TaxLot("X", 10.0, 80.0, date(2022, 1, 1)),
    ]
    _, remaining, _ = policy.match_and_realize("X", lots, 10.0, 100.0, date(2024, 1, 1))
    assert len(remaining) == 1
    assert remaining[0].cost_basis == 90.0   # older (2022) lot consumed first


def test_lot_method_max_loss():
    policy = USCapitalGainsPolicy(lot_method=LotMethod.MAX_LOSS)
    lots = [
        TaxLot("X", 10.0,  50.0, date(2022, 1, 1)),
        TaxLot("X", 10.0, 120.0, date(2023, 1, 1)),
    ]
    _, remaining, _ = policy.match_and_realize("X", lots, 10.0, 100.0, date(2024, 1, 1))
    assert remaining[0].cost_basis == 50.0   # highest basis consumed first
