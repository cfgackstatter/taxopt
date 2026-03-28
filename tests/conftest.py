# tests/conftest.py
import numpy as np
import pytest
from datetime import date

from taxopt.data_types import TaxLot, OptimizationInputs
from taxopt.optimizer import CvxpyOptimizer
from taxopt.portfolio import Portfolio
from taxopt.tax_policy import USCapitalGainsPolicy, NoTaxPolicy, LotMethod


ASSETS = ["AAPL", "MSFT", "GOOG"]
PRICES = {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 200.0}
TOTAL_VALUE = 10_000.0

# Simple diagonal covariance (uncorrelated, equal variance)
COV = np.diag([0.04, 0.04, 0.04])

# Mild alpha signal favouring AAPL
ALPHA = {"AAPL": 0.05, "MSFT": 0.02, "GOOG": 0.01}


@pytest.fixture
def prices() -> dict[str, float]:
    return PRICES.copy()


@pytest.fixture
def us_policy() -> USCapitalGainsPolicy:
    return USCapitalGainsPolicy(lot_method=LotMethod.MIN_GAIN)


@pytest.fixture
def no_tax_policy() -> NoTaxPolicy:
    return NoTaxPolicy()


@pytest.fixture
def long_only_inputs() -> OptimizationInputs:
    return OptimizationInputs(
        alpha=ALPHA,
        covariance=COV,
        assets=ASSETS,
        prices=PRICES,
        risk_aversion=1.0,
        gross_leverage=1.0,
        net_exposure=1.0,
        max_weight=0.6,
    )


@pytest.fixture
def long_short_inputs() -> OptimizationInputs:
    return OptimizationInputs(
        alpha=ALPHA,
        covariance=COV,
        assets=ASSETS,
        prices=PRICES,
        risk_aversion=1.0,
        gross_leverage=1.6,   # 130/30
        net_exposure=1.0,
        max_weight=0.6,
    )


@pytest.fixture
def simple_portfolio() -> Portfolio:
    """Three long lots, one per asset, all long-term (held > 1yr)."""
    p = Portfolio(cash=0.0)
    p.add_lot(TaxLot("AAPL", 20.0, 100.0, date(2022, 1, 1)))   # unrealized gain
    p.add_lot(TaxLot("MSFT", 10.0, 250.0, date(2022, 1, 1)))   # unrealized gain
    p.add_lot(TaxLot("GOOG", 15.0, 180.0, date(2022, 1, 1)))   # unrealized gain
    return p


@pytest.fixture
def two_lot_portfolio() -> Portfolio:
    """
    AAPL has two lots:
      - lot A: large LT gain  (basis=50,  acq 2022) -> high tax cost to sell
      - lot B: small ST loss  (basis=160, acq 2025) -> tax benefit to sell
    MSFT and GOOG have single neutral lots.
    """
    p = Portfolio(cash=0.0)
    p.add_lot(TaxLot("AAPL", 20.0,  50.0, date(2022, 1, 1)))   # big LT gain
    p.add_lot(TaxLot("AAPL", 10.0, 160.0, date(2025, 6, 1)))   # ST loss
    p.add_lot(TaxLot("MSFT", 10.0, 300.0, date(2022, 1, 1)))   # flat
    p.add_lot(TaxLot("GOOG", 15.0, 200.0, date(2022, 1, 1)))   # flat
    return p
