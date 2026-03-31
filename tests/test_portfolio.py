# tests/test_portfolio.py
from datetime import date, timedelta
import pytest
from taxopt.data_types import TaxLot, LotClose, LongOpen, ShortOpen
from taxopt.portfolio import Portfolio, _MIN_ACTION_QTY
from taxopt.tax_policy import USCapitalGainsPolicy, NoTaxPolicy


def _portfolio_with_lot(asset: str, qty: float, basis: float, acq: date) -> Portfolio:
    p = Portfolio(cash=10_000.0)
    p.add_lot(TaxLot(asset, qty, basis, acq))
    return p


def _all_lots(p: Portfolio, asset: str) -> list[TaxLot]:
    return p.lots.get(asset, [])


# ---------------------------------------------------------------------------
# Buying
# ---------------------------------------------------------------------------

def test_buy_creates_lot(prices):
    p = Portfolio(cash=10_000.0)
    new_p, _ = p.apply_actions([LongOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    assert sum(l.quantity for l in new_p.lots["AAPL"]) == 10.0

def test_buy_debits_cash(prices):
    p = Portfolio(cash=10_000.0)
    new_p, _ = p.apply_actions([LongOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    assert new_p.cash == 10_000.0 - 10.0 * prices["AAPL"]

def test_dust_long_open_is_ignored(prices):
    p = Portfolio(cash=10_000.0)
    new_p, _ = p.apply_actions(
        [LongOpen("AAPL", _MIN_ACTION_QTY / 2)], prices, NoTaxPolicy(), date(2024, 1, 1)
    )
    assert "AAPL" not in new_p.lots
    assert new_p.cash == p.cash

def test_dust_short_open_is_ignored(prices):
    p = Portfolio(cash=10_000.0)
    new_p, _ = p.apply_actions(
        [ShortOpen("AAPL", _MIN_ACTION_QTY / 2)], prices, NoTaxPolicy(), date(2024, 1, 1)
    )
    assert "AAPL" not in new_p.lots
    assert new_p.cash == p.cash


# ---------------------------------------------------------------------------
# Selling
# ---------------------------------------------------------------------------

def test_sell_closes_lot(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    new_p, _ = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert "AAPL" not in new_p.lots

def test_sell_credits_cash(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    new_p, _ = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert new_p.cash == p.cash + 10.0 * prices["AAPL"]

def test_partial_sell_leaves_correct_quantity(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    new_p, _ = p.apply_actions(
        [LotClose(asset="AAPL", quantity=4.0, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert abs(sum(l.quantity for l in new_p.lots["AAPL"]) - 6.0) < 1e-9

def test_sell_more_than_held_raises(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 5.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    with pytest.raises(ValueError, match="Insufficient lots"):
        p.apply_actions(
            [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
            prices, us_policy, date(2024, 1, 1),
        )

def test_partial_sell_below_threshold_drops_lot(prices, us_policy):
    """Remainder below _MIN_ACTION_QTY must be fully closed, not kept."""
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    new_p, _ = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0 - _MIN_ACTION_QTY / 2, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert "AAPL" not in new_p.lots

def test_partial_sell_above_threshold_keeps_lot(prices, us_policy):
    """Remainder above _MIN_ACTION_QTY must survive."""
    keep = _MIN_ACTION_QTY * 2
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    new_p, _ = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0 - keep, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert abs(new_p.lots["AAPL"][0].quantity - keep) < 1e-9

# ---------------------------------------------------------------------------
# Tax report
# ---------------------------------------------------------------------------

def test_sell_produces_correct_gain(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    _, report = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert report.total_gain == 10.0 * (prices["AAPL"] - 100.0)

def test_long_term_gain_classified_correctly(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    _, report = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert "long_term" in report.totals_by_type
    assert report.totals_by_type.get("short_term", 0.0) == 0.0

def test_short_term_gain_classified_correctly(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2024, 1, 1))
    lot = p.lots["AAPL"][0]
    _, report = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, us_policy, date(2024, 3, 1),
    )
    assert report.totals_by_type.get("short_term", 0.0) == 10.0 * (prices["AAPL"] - 100.0)


# ---------------------------------------------------------------------------
# Portfolio value / NAV / invariants
# ---------------------------------------------------------------------------

def test_total_value(prices):
    p = Portfolio(cash=1_000.0)
    p.add_lot(TaxLot("AAPL", 10.0, 100.0, date(2022, 1, 1)))
    p.add_lot(TaxLot("MSFT", 5.0, 200.0, date(2022, 1, 1)))
    assert p.total_value(prices) == 1_000.0 + 10.0 * prices["AAPL"] + 5.0 * prices["MSFT"]

def test_buy_sell_same_price_cash_neutral(prices):
    p = Portfolio(cash=10_000.0)
    p2, _ = p.apply_actions([LongOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    lot = p2.lots["AAPL"][0]
    p3, _ = p2.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, NoTaxPolicy(), date(2024, 1, 1),
    )
    assert abs(p3.cash - 10_000.0) < 1e-9

def test_long_open_conserves_nav(prices):
    p = Portfolio(cash=10_000.0)
    new_p, _ = p.apply_actions([LongOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    assert abs(new_p.total_value(prices) - p.total_value(prices)) < 1e-6

def test_lot_close_long_conserves_nav(prices, us_policy):
    p = _portfolio_with_lot("AAPL", 10.0, 100.0, date(2022, 1, 1))
    lot = p.lots["AAPL"][0]
    new_p, _ = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, us_policy, date(2024, 6, 1),
    )
    assert abs(new_p.total_value(prices) - p.total_value(prices)) < 1e-6

def test_short_open_conserves_nav(prices):
    p = Portfolio(cash=10_000.0)
    new_p, _ = p.apply_actions([ShortOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    assert abs(new_p.total_value(prices) - p.total_value(prices)) < 1e-6

def test_short_cover_conserves_nav(prices):
    p = Portfolio(cash=20_000.0)
    p2, _ = p.apply_actions([ShortOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    lot = next(l for ls in p2.lots.values() for l in ls if l.quantity < 0)
    p3, _ = p2.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, NoTaxPolicy(), date(2024, 6, 1),
    )
    assert abs(p3.total_value(prices) - p2.total_value(prices)) < 1e-6

def test_round_trip_long_conserves_nav(prices):
    p = Portfolio(cash=10_000.0)
    p2, _ = p.apply_actions([LongOpen("AAPL", 10.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    lot = p2.lots["AAPL"][0]
    p3, _ = p2.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot)],
        prices, NoTaxPolicy(), date(2024, 1, 1),
    )
    assert abs(p3.cash - 10_000.0) < 1e-6
    assert abs(p3.total_value(prices) - 10_000.0) < 1e-6

def test_combined_actions_conserve_nav(prices):
    p = Portfolio(cash=50_000.0)
    p2, _ = p.apply_actions([LongOpen("AAPL", 100.0)], prices, NoTaxPolicy(), date(2024, 1, 1))
    lot = p2.lots["AAPL"][0]
    nav_before = p2.total_value(prices)
    p3, _ = p2.apply_actions([
        LotClose(asset="AAPL", quantity=50.0, lot_ref=lot),
        LongOpen("MSFT", 20.0),
        ShortOpen("GOOG", 30.0),
    ], prices, NoTaxPolicy(), date(2024, 1, 1))
    assert abs(p3.total_value(prices) - nav_before) < 1e-6

def test_lot_close_closes_correct_lot(prices, us_policy):
    today = date.today()
    p = Portfolio(cash=0.0)
    lot_a = TaxLot("AAPL", 10.0, 50.0, today - timedelta(days=400))
    lot_b = TaxLot("AAPL", 10.0, 160.0, today - timedelta(days=30))
    p.add_lot(lot_a)
    p.add_lot(lot_b)
    new_p, report = p.apply_actions(
        [LotClose(asset="AAPL", quantity=10.0, lot_ref=lot_b)],   # ← no lot_index
        prices, us_policy, today,
    )
    assert len(new_p.lots["AAPL"]) == 1
    assert new_p.lots["AAPL"][0].cost_basis == 50.0
    assert report.events[0].gain_type == "short_term"
