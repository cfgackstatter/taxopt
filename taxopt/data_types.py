# data_types.py
from __future__ import annotations
import math
from dataclasses import dataclass, field, replace
from datetime import date
from typing import TYPE_CHECKING, Mapping, Protocol, TypeAlias

import numpy as np

if TYPE_CHECKING:
    from .portfolio import Portfolio
    from .tax_policy import TaxPolicy

AssetId: TypeAlias = str

# ---------------------------------------------------------------------------
# Tax lots
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TaxLot:
    asset: AssetId
    quantity: float
    cost_basis: float
    acquisition_date: date
    metadata: Mapping[str, object] | None = None

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def with_quantity(self, quantity: float) -> TaxLot:
        return replace(self, quantity=quantity)

# ---------------------------------------------------------------------------
# Portfolio actions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LotClose:
    lot_index: int
    asset: AssetId
    quantity: float
    lot_ref: TaxLot

@dataclass(frozen=True)
class LongOpen:
    asset: AssetId
    quantity: float

@dataclass(frozen=True)
class ShortOpen:
    asset: AssetId
    quantity: float

PortfolioAction = LotClose | LongOpen | ShortOpen

# ---------------------------------------------------------------------------
# Optimizer I/O
# ---------------------------------------------------------------------------

@dataclass
class OptimizationInputs:
    alpha: dict[AssetId, float]
    covariance: np.ndarray
    assets: list[AssetId]
    prices: dict[AssetId, float]
    risk_aversion: float = 1.0
    tax_aversion: float = 1.0
    gross_leverage: float = 1.0
    net_exposure: float = 1.0
    max_weight: float = 0.10
    max_turnover: float | None = None
    as_of: date = field(default_factory=date.today)

@dataclass(frozen=True)
class OptimizationResult:
    weights: dict[AssetId, float]
    weights_long: dict[AssetId, float]
    weights_short: dict[AssetId, float]
    actions: list[PortfolioAction]
    realized_gain: float
    tax_cost: float
    status: str
    effective_turnover: float
    _prices: dict[AssetId, float] = field(default_factory=dict)

    def to_trades(self, round_to_integer: bool = False) -> list[PortfolioAction]:
        if not round_to_integer:
            return list(self.actions)
        out: list[PortfolioAction] = []
        for a in self.actions:
            if isinstance(a, LotClose):
                qty = min(math.floor(a.quantity), abs(a.lot_ref.quantity))
                if qty > 0:
                    out.append(replace(a, quantity=float(qty)))
            else:
                qty = math.floor(a.quantity)
                if qty > 0:
                    out.append(replace(a, quantity=float(qty)))
        return out

class Optimizer(Protocol):
    def solve(
        self,
        portfolio: Portfolio,
        inputs: OptimizationInputs,
        tax_policy: TaxPolicy,
        total_value: float,
    ) -> OptimizationResult: ...
