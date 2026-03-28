# taxopt

A Python library for **tax-aware portfolio optimization** using mixed-integer quadratic programming (MIQP). Designed for long/short equity strategies with lot-level tax tracking, wash-sale rule enforcement, and tax-loss harvesting.

---

## Overview

`taxopt` solves the portfolio rebalancing problem jointly with tax minimization:

```math
\max_w \ \alpha^\top w - \lambda \, w^\top \Sigma w - \gamma_{\text{tax}} \cdot \text{tax\_cost}
```

subject to leverage, net exposure, per-asset weight caps, turnover limits, and wash-sale constraints.

Key features:
- **Lot-level tax tracking** — each position is tracked as individual tax lots with acquisition date and cost basis
- **Wash-sale enforcement** — binary variables prevent simultaneous closing and opening of the same asset on the same side
- **Tax-loss harvesting** — the optimizer prefers to realize losses and defer gains when the tax cost is high
- **130/30 and market-neutral** — configurable gross leverage and net exposure support a range of long/short structures
- **Turnover relaxation** — automatically widens the turnover constraint if the problem is infeasible
- **US capital gains policy** — short-term vs long-term classification per §1221 and §1233, with configurable rates and lot selection methods

---

## Project Structure

```
taxopt/
├── __init__.py          # Public API
├── data_types.py        # Core dataclasses: TaxLot, OptimizationInputs/Result, PortfolioAction
├── optimizer.py         # CvxpyOptimizer: MIQP via CVXPY + SCIP
├── portfolio.py         # Portfolio: lot management, apply_actions
├── tax_policy.py        # USCapitalGainsPolicy, NoTaxPolicy, LotMethod
└── tax_report.py        # TaxReport, RealizedGain

notebooks/
└── 01_tax_optimization_demo.ipynb   # End-to-end backtest
```

---

## Installation

Requires Python 3.11+.

```bash
pip install cvxpy[SCIP] numpy scipy pandas yfinance plotly
```

Or clone the repo and install in editable mode:

```bash
git clone https://github.com/cfgackstatter/taxopt.git
cd taxopt
pip install -e ".[notebook]"
```

---

## Quick Start

```python
from datetime import date
from taxopt import (
    Portfolio, TaxLot,
    OptimizationInputs, CvxpyOptimizer,
    USCapitalGainsPolicy, LotMethod,
)
import numpy as np

# Build a portfolio with existing lots
portfolio = Portfolio(cash=10_000.0)
portfolio.add_lot(TaxLot("AAPL", quantity=10.0, cost_basis=150.0, acquisition_date=date(2022, 6, 1)))
portfolio.add_lot(TaxLot("MSFT", quantity=5.0,  cost_basis=280.0, acquisition_date=date(2023, 1, 15)))

# Define optimization inputs
inputs = OptimizationInputs(
    alpha={"AAPL": 0.05, "MSFT": 0.03, "GOOG": 0.07},
    covariance=np.diag([0.04, 0.04, 0.04]),
    assets=["AAPL", "MSFT", "GOOG"],
    prices={"AAPL": 190.0, "MSFT": 310.0, "GOOG": 140.0},
    risk_aversion=2.0,
    tax_aversion=1.0,
    gross_leverage=1.0,
    net_exposure=1.0,
    max_weight=0.50,
    max_turnover=0.20,
    as_of=date(2024, 1, 31),
)

policy = USCapitalGainsPolicy(lot_method=LotMethod.MIN_GAIN)
solver = CvxpyOptimizer(solver="SCIP", tax_aware=True)
nav    = portfolio.total_value(inputs.prices)

result = solver.solve(portfolio, inputs, policy, nav)

print(result.status)
print(result.weights)
print(result.realized_gain, result.tax_cost)

# Apply trades
new_portfolio, tax_report = portfolio.apply_actions(result.actions, inputs.prices, policy, inputs.as_of)
```

---

## Key Concepts

### Tax Lots

Each position is a `TaxLot(asset, quantity, cost_basis, acquisition_date)`. Negative quantity = short. Lots are closed individually; partial closes are supported.

### Lot Selection Methods

| `LotMethod`  | Description |
|---|---|
| `FIFO`       | Oldest lots first |
| `LIFO`       | Newest lots first |
| `MIN_GAIN`   | Smallest gain (or largest loss) first — default for TLH |
| `MAX_GAIN`   | Largest gain first |
| `MAX_LOSS`   | Highest cost basis first |

### Leverage Configurations

| `gross_leverage` | `net_exposure` | Long | Short | Strategy |
|---|---|---|---|---|
| 1.0 | 1.0 | 100% | 0%  | Long-only |
| 1.6 | 1.0 | 130% | 30% | 130/30 |
| 1.0 | 0.0 | 50%  | 50% | Market-neutral |
| 2.0 | 0.0 | 100% | 100%| Dollar-neutral levered |

### Tax Policy

`USCapitalGainsPolicy` implements:
- **Long positions**: ST if held < 366 days, LT otherwise
- **Short positions**: always ST on gains (§1233); losses follow §1233(d)
- **Netting**: ST losses offset LT gains before applying rates

Default rates: ST = 35%, LT = 20% (configurable).

---

## Optimizer Parameters

```python
CvxpyOptimizer(
    solver="SCIP",           # MIP solver (SCIP required for wash-sale binaries)
    verbose=False,           # Solver output
    tax_aware=True,          # Include tax cost in objective
    relax_turnover=True,     # Widen turnover if infeasible
    turnover_relax_step=0.05,        # +5pp per attempt
    turnover_relax_max_attempts=5,   # Up to +25pp total
    mip_gap=0.02,            # Accept 2% MIP gap for speed
)
```

---

## Notebook

`notebooks/01_tax_optimization_demo.ipynb` runs a monthly rebalancing backtest using a 130/30 structure with momentum alpha. It reports:

- Pre-tax and after-tax NAV over time
- Realized ST/LT gains per rebalance
- Cumulative tax alpha (% of NAV saved via harvesting)
- Final lot inspection table

---

## Requirements

- Python ≥ 3.11
- `cvxpy` with SCIP backend
- `numpy`, `scipy`
- `pandas`, `yfinance` (notebook only)
- `plotly` (notebook only)
