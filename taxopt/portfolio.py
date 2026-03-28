# portfolio.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from .data_types import AssetId, TaxLot, LotClose, LongOpen, ShortOpen, PortfolioAction
from .tax_report import RealizedGain, TaxReport


@dataclass
class Portfolio:
    lots: dict[AssetId, list[TaxLot]] = field(default_factory=dict)
    cash: float = 0.0

    def copy(self) -> Portfolio:
        return Portfolio(lots={a: list(ls) for a, ls in self.lots.items()}, cash=self.cash)

    def positions(self) -> dict[AssetId, float]:
        return {a: sum(l.quantity for l in ls) for a, ls in self.lots.items()}

    def total_value(self, prices: Mapping[AssetId, float]) -> float:
        return sum(l.quantity * prices[a] for a, ls in self.lots.items() for l in ls) + self.cash

    def add_lot(self, lot: TaxLot) -> None:
        self.lots.setdefault(lot.asset, []).append(lot)

    def apply_actions(
        self,
        actions: Sequence[PortfolioAction],
        prices: Mapping[AssetId, float],
        tax_policy: Any,
        as_of: date,
    ) -> tuple[Portfolio, TaxReport]:
        new_port = self.copy()
        realized_events: list[RealizedGain] = []

        for action in [a for a in actions if isinstance(a, LotClose)]:
            px       = prices[action.asset]
            lot      = action.lot_ref
            qty      = action.quantity          # units to close (positive)
            is_short = lot.quantity < 0
            sign     = -1.0 if is_short else 1.0

            if qty > abs(lot.quantity) + 1e-9:
                raise ValueError(f"Insufficient lots for {action.asset}: short by {qty - abs(lot.quantity):.6f}")

            proceeds = qty * px
            gain     = (proceeds - qty * lot.cost_basis) * sign
            gain_type = tax_policy.classify_gain(lot, as_of, gain)

            realized_events.append(RealizedGain(
                asset=action.asset, lot=lot, quantity=qty,
                proceeds=proceeds, cost=qty * lot.cost_basis,
                gain=gain, gain_type=gain_type,
            ))

            # Update the specific lot directly
            asset_lots = new_port.lots.get(action.asset, [])
            remaining_qty = abs(lot.quantity) - qty
            new_lots = [l for l in asset_lots if l is not lot]
            if remaining_qty > 1e-9:
                new_lots.append(lot.with_quantity(sign * remaining_qty))
            new_port.lots[action.asset] = new_lots

            # Cash: long close receives proceeds, short cover pays proceeds
            new_port.cash += -proceeds if is_short else proceeds

        for action in [a for a in actions if not isinstance(a, LotClose)]:
            px = prices[action.asset]
            if isinstance(action, LongOpen):
                new_port.add_lot(TaxLot(action.asset, action.quantity, px, as_of))
                new_port.cash -= action.quantity * px
            elif isinstance(action, ShortOpen):
                new_port.add_lot(TaxLot(action.asset, -action.quantity, px, as_of))
                new_port.cash += action.quantity * px

        return new_port, TaxReport.from_events(realized_events, tax_policy, as_of)

