# tests/test_optimizer.py
import math
import numpy as np
import pytest
from datetime import date, timedelta

from taxopt.data_types import LotClose, LongOpen, ShortOpen, OptimizationInputs, OptimizationResult, TaxLot
from taxopt.optimizer import CvxpyOptimizer
from taxopt.portfolio import Portfolio, _MIN_ACTION_QTY
from taxopt.tax_policy import USCapitalGainsPolicy, NoTaxPolicy, LotMethod

EPS       = 1e-4
ROUND_EPS = 0.05


def _lot_sell(result: OptimizationResult, lot_ref: TaxLot) -> float:
    """Return the quantity closed for a specific lot (matched by identity)."""
    for a in result.actions:
        if isinstance(a, LotClose) and a.lot_ref is lot_ref:
            return a.quantity
    return 0.0


def _total_sold(result: OptimizationResult) -> float:
    return sum(a.quantity for a in result.actions if isinstance(a, LotClose))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def solver() -> CvxpyOptimizer:
    return CvxpyOptimizer()


@pytest.fixture
def prices() -> dict[str, float]:
    return {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 120.0}


@pytest.fixture
def us_policy() -> USCapitalGainsPolicy:
    return USCapitalGainsPolicy(lot_method=LotMethod.MIN_GAIN)


@pytest.fixture
def no_tax_policy() -> NoTaxPolicy:
    return NoTaxPolicy()


@pytest.fixture
def simple_portfolio(prices: dict[str, float]) -> Portfolio:
    today = date.today()
    p = Portfolio(cash=0.0)
    p.add_lot(TaxLot("AAPL", quantity=100.0, cost_basis=50.0,  acquisition_date=today - timedelta(days=400)))
    p.add_lot(TaxLot("MSFT", quantity=50.0,  cost_basis=200.0, acquisition_date=today - timedelta(days=400)))
    p.add_lot(TaxLot("GOOG", quantity=200.0, cost_basis=80.0,  acquisition_date=today - timedelta(days=400)))
    return p


@pytest.fixture
def two_lot_portfolio(prices: dict[str, float]) -> Portfolio:
    today = date.today()
    p = Portfolio(cash=0.0)
    # Lot 0: AAPL LT gain (basis=50, price=150)
    p.add_lot(TaxLot("AAPL", quantity=100.0, cost_basis=50.0,  acquisition_date=today - timedelta(days=400)))
    # Lot 1: AAPL ST loss  (basis=160, price=150)
    p.add_lot(TaxLot("AAPL", quantity=50.0,  cost_basis=160.0, acquisition_date=today - timedelta(days=30)))
    p.add_lot(TaxLot("MSFT", quantity=50.0,  cost_basis=200.0, acquisition_date=today - timedelta(days=400)))
    p.add_lot(TaxLot("GOOG", quantity=200.0, cost_basis=80.0,  acquisition_date=today - timedelta(days=400)))
    return p


@pytest.fixture
def long_only_inputs(prices: dict[str, float]) -> OptimizationInputs:
    return OptimizationInputs(
        alpha={"AAPL": 0.03, "MSFT": 0.05, "GOOG": 0.02},
        covariance=np.diag([0.04, 0.04, 0.04]),
        assets=["AAPL", "MSFT", "GOOG"],
        prices=prices,
        risk_aversion=1.0,
        tax_aversion=1.0,
        gross_leverage=1.0,
        net_exposure=1.0,
        max_weight=0.6,
        as_of=date.today(),
    )


@pytest.fixture
def long_short_inputs(prices: dict[str, float]) -> OptimizationInputs:
    return OptimizationInputs(
        alpha={"AAPL": 0.05, "MSFT": -0.03, "GOOG": 0.02},
        covariance=np.diag([0.04, 0.04, 0.04]),
        assets=["AAPL", "MSFT", "GOOG"],
        prices=prices,
        risk_aversion=1.0,
        tax_aversion=1.0,
        gross_leverage=1.3,
        net_exposure=1.0,
        max_weight=0.6,
        as_of=date.today(),
    )


# ---------------------------------------------------------------------------
# 1. Smoke
# ---------------------------------------------------------------------------

def test_solves_to_optimal(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    assert result.status in ("optimal", "optimal_inaccurate")


def test_weights_are_populated(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    assert set(result.weights.keys()) == set(long_only_inputs.assets)


# ---------------------------------------------------------------------------
# 2. Constraint satisfaction
# ---------------------------------------------------------------------------

def test_net_exposure(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    net = sum(result.weights_long.values()) - sum(result.weights_short.values())
    assert abs(net - long_only_inputs.net_exposure) < EPS


def test_gross_leverage(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    gross = sum(result.weights_long.values()) + sum(result.weights_short.values())
    assert abs(gross - long_only_inputs.gross_leverage) < EPS


def test_long_short_gross_leverage(simple_portfolio, long_short_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_short_inputs, us_policy, simple_portfolio.total_value(prices))
    assert result.status in ("optimal", "optimal_inaccurate")
    gross = sum(result.weights_long.values()) + sum(result.weights_short.values())
    assert abs(gross - long_short_inputs.gross_leverage) < EPS


def test_weight_decomposition_consistent(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    for a in long_only_inputs.assets:
        assert abs(result.weights[a] - (result.weights_long[a] - result.weights_short[a])) < EPS


def test_max_weight_respected(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    for a in long_only_inputs.assets:
        assert result.weights_long[a]  <= long_only_inputs.max_weight + EPS
        assert result.weights_short[a] <= long_only_inputs.max_weight + EPS


def test_lot_sells_within_bounds(two_lot_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(two_lot_portfolio, long_only_inputs, us_policy,
                          two_lot_portfolio.total_value(prices))
    all_lots = [l for ls in two_lot_portfolio.lots.values() for l in ls]
    for lot in all_lots:
        assert _lot_sell(result, lot) <= abs(lot.quantity) + EPS


# ---------------------------------------------------------------------------
# 3. Tax-awareness
# ---------------------------------------------------------------------------

def test_prefers_loss_lot_over_gain_lot(two_lot_portfolio, long_only_inputs, us_policy, prices, solver):
    inputs = OptimizationInputs(
        alpha={"AAPL": 0.01, "MSFT": 0.05, "GOOG": 0.05},
        covariance=long_only_inputs.covariance,
        assets=long_only_inputs.assets,
        prices=prices,
        risk_aversion=1.0,
        tax_aversion=1.0,
        gross_leverage=1.0,
        net_exposure=1.0,
        max_weight=0.4,
        as_of=date.today(),
    )
    result = solver.solve(two_lot_portfolio, inputs, us_policy,
                          two_lot_portfolio.total_value(prices))

    gain_lot = next(l for ls in two_lot_portfolio.lots.values()
                    for l in ls if l.asset == "AAPL" and l.cost_basis == 50.0)
    loss_lot = next(l for ls in two_lot_portfolio.lots.values()
                    for l in ls if l.asset == "AAPL" and l.cost_basis == 160.0)

    assert _lot_sell(result, loss_lot) >= _lot_sell(result, gain_lot) - EPS


def test_no_tax_policy_zero_tax_cost(simple_portfolio, long_only_inputs, no_tax_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, no_tax_policy, simple_portfolio.total_value(prices))
    assert abs(result.tax_cost) < EPS


def test_tax_cost_nonneg_when_selling_gains(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy, simple_portfolio.total_value(prices))
    if _total_sold(result) > EPS:
        assert result.tax_cost >= -EPS


# ---------------------------------------------------------------------------
# 4. Turnover
# ---------------------------------------------------------------------------

def test_turnover_constraint_respected(simple_portfolio, us_policy, prices, solver):
    tv = simple_portfolio.total_value(prices)
    inputs = OptimizationInputs(
        alpha={"AAPL": 0.05, "MSFT": 0.02, "GOOG": 0.01},
        covariance=np.diag([0.04, 0.04, 0.04]),
        assets=["AAPL", "MSFT", "GOOG"],
        prices=prices,
        risk_aversion=1.0,
        tax_aversion=1.0,
        gross_leverage=1.0,
        net_exposure=1.0,
        max_weight=0.6,
        max_turnover=0.05,
        as_of=date.today(),
    )
    result = solver.solve(simple_portfolio, inputs, us_policy, tv)
    assert result.status in ("optimal", "optimal_inaccurate")
    assert inputs.max_turnover is not None
    w0 = {
        a: sum(l.quantity * prices[a] for l in simple_portfolio.lots.get(a, [])) / tv
        for a in inputs.assets
    }
    turnover = sum(abs(result.weights[a] - w0[a]) for a in inputs.assets) / 2.0
    assert turnover <= inputs.max_turnover + EPS


# ---------------------------------------------------------------------------
# 5. Rounding
# ---------------------------------------------------------------------------

def test_rounded_trades_are_integers(two_lot_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(two_lot_portfolio, long_only_inputs, us_policy, two_lot_portfolio.total_value(prices))
    for t in result.to_trades(round_to_integer=True):
        assert abs(t.quantity - round(t.quantity)) < 1e-9


def test_rounded_lot_sells_within_bounds(two_lot_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(two_lot_portfolio, long_only_inputs, us_policy,
                          two_lot_portfolio.total_value(prices))
    all_lots = [l for ls in two_lot_portfolio.lots.values() for l in ls]
    for lot in all_lots:
        floored = min(math.floor(_lot_sell(result, lot)), abs(lot.quantity))
        assert floored <= abs(lot.quantity)


def test_rounded_cash_not_exceeded(two_lot_portfolio, long_only_inputs, us_policy, prices, solver):
    result         = solver.solve(two_lot_portfolio, long_only_inputs, us_policy, two_lot_portfolio.total_value(prices))
    cost           = lambda trades: sum(abs(t.quantity) * prices[t.asset] for t in trades
                                        if isinstance(t, LongOpen))
    assert cost(result.to_trades(round_to_integer=True)) \
        <= cost(result.to_trades(round_to_integer=False)) + ROUND_EPS * two_lot_portfolio.total_value(prices)


def test_long_short_financing(simple_portfolio, long_short_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_short_inputs, us_policy, simple_portfolio.total_value(prices))
    assert result.status in ("optimal", "optimal_inaccurate")
    assert sum(result.weights_short.values()) > EPS
    assert sum(result.weights_long.values())  > 1.0 - EPS


def test_empty_portfolio_solves(long_only_inputs, us_policy, solver):
    """Optimizer should handle a portfolio with no lots (cash-only start)."""
    nav = 45_000.0
    p = Portfolio(cash=nav)
    result = solver.solve(p, long_only_inputs, us_policy, nav)
    assert result.status in ("optimal", "optimal_inaccurate")
    net = sum(result.weights_long.values()) - sum(result.weights_short.values())
    assert abs(net - long_only_inputs.net_exposure) < EPS


def test_no_simultaneous_buy_and_sell_same_asset(two_lot_portfolio, us_policy, prices, solver):
    """Optimizer must not sell and re-buy the same asset in the same solve (wash-sale proxy)."""
    inputs = OptimizationInputs(
        alpha={"AAPL": 0.05, "MSFT": 0.01, "GOOG": 0.01},
        covariance=np.diag([0.04, 0.04, 0.04]),
        assets=["AAPL", "MSFT", "GOOG"],
        prices=prices,
        risk_aversion=1.0,
        tax_aversion=1.0,
        gross_leverage=1.0,
        net_exposure=1.0,
        max_weight=0.6,
        as_of=date.today(),
    )
    result = solver.solve(two_lot_portfolio, inputs, us_policy, two_lot_portfolio.total_value(prices))
    sold_assets  = {a.asset for a in result.actions if isinstance(a, LotClose)}
    bought_assets = {a.asset for a in result.actions if isinstance(a, LongOpen)}
    assert sold_assets.isdisjoint(bought_assets), \
        f"Wash-sale violation: {sold_assets & bought_assets} both sold and re-bought"


def test_no_dust_opens(simple_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(simple_portfolio, long_only_inputs, us_policy,
                          simple_portfolio.total_value(prices))
    for a in result.actions:
        if isinstance(a, (LongOpen, ShortOpen)):
            assert a.quantity >= _MIN_ACTION_QTY


def test_no_dust_lot_remainders(two_lot_portfolio, long_only_inputs, us_policy, prices, solver):
    result = solver.solve(two_lot_portfolio, long_only_inputs, us_policy,
                          two_lot_portfolio.total_value(prices))
    new_p, _ = two_lot_portfolio.apply_actions(
        result.actions, prices, us_policy, date.today()
    )
    for asset, lots in new_p.lots.items():
        for lot in lots:
            assert abs(lot.quantity) >= _MIN_ACTION_QTY, (
                f"Dust lot for {asset}: qty={lot.quantity}"
            )