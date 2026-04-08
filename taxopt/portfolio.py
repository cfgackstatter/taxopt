# portfolio.py
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping, Sequence

from .data_types import _MIN_ACTION_QTY, AssetId, TaxLot, LotClose, LongOpen, ShortOpen, PortfolioAction
from .tax_report import RealizedGain, TaxReport
from .tax_policy import TaxPolicy


@dataclass
class Portfolio:
    lots: dict[AssetId, list[TaxLot]] = field(default_factory=dict)
    cash: float = 0.0

    def __str__(self) -> str:
        lines = [f"Portfolio  cash=${self.cash:,.2f}  lots={sum(len(v) for v in self.lots.values())}"]
        lines.append(f"  {'Asset':<8} {'Qty':>10} {'Basis':>8} {'Acquired':<12} {'Side':<6}")
        lines.append("  " + "-" * 50)
        for asset, lots in sorted(self.lots.items()):
            for lot in lots:
                side = "SHORT" if lot.quantity < 0 else "LONG"
                lines.append(
                    f"  {asset:<8} {lot.quantity:>10.4f} {lot.cost_basis:>8.2f} "
                    f"{str(lot.acquisition_date):<12} {side:<6}"
                )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = ""
        for asset, lots in sorted(self.lots.items()):
            for lot in lots:
                side = "SHORT" if lot.quantity < 0 else "LONG"
                gain_color = "#c0392b" if lot.quantity < 0 else "#27ae60"
                rows += (
                    f"<tr><td>{asset}</td><td>{lot.quantity:.4f}</td>"
                    f"<td>${lot.cost_basis:.2f}</td><td>{lot.acquisition_date}</td>"
                    f"<td style='color:{gain_color}'>{side}</td></tr>"
                )
        return f"""
        <b>Portfolio</b> &nbsp; cash=<b>${self.cash:,.2f}</b> &nbsp;
        lots=<b>{sum(len(v) for v in self.lots.values())}</b><br>
        <table border='0' style='border-collapse:collapse;font-size:13px'>
        <thead><tr style='border-bottom:1px solid #ccc'>
            <th align='left'>Asset</th><th align='right'>Qty</th>
            <th align='right'>Basis</th><th>Acquired</th><th>Side</th>
        </tr></thead>
        <tbody>{rows}</tbody>
        </table>"""

    def copy(self) -> Portfolio:
        return Portfolio(lots={a: list(ls) for a, ls in self.lots.items()}, cash=self.cash)

    def positions(self) -> dict[AssetId, float]:
        return {a: sum(l.quantity for l in ls) for a, ls in self.lots.items()}

    def total_value(self, prices: Mapping[AssetId, float]) -> float:
        return sum(l.quantity * prices[a] for a, ls in self.lots.items() for l in ls) + self.cash

    def value_by_asset(self, prices: Mapping[AssetId, float]) -> dict[AssetId, float]:
        vals: dict[AssetId, float] = {}
        for asset, lots in self.lots.items():
            qty = sum(l.quantity for l in lots)
            if abs(qty) < 1e-9:
                continue
            vals[asset] = qty * prices[asset]
        return vals

    def weights(self, prices: Mapping[AssetId, float]) -> dict[AssetId, float]:
        vals = self.value_by_asset(prices)
        total = sum(vals.values())
        if total <= 0:
            return {a: 0.0 for a in vals}
        return {a: v / total for a, v in vals.items()}

    def add_lot(self, lot: TaxLot) -> None:
        self.lots.setdefault(lot.asset, []).append(lot)

    def apply_actions(
        self,
        actions: Sequence[PortfolioAction],
        prices: Mapping[AssetId, float],
        tax_policy: TaxPolicy,
        as_of: date,
    ) -> tuple[Portfolio, TaxReport]:
        new_port = self.copy()
        realized_events: list[RealizedGain] = []

        closes  = [a for a in actions if isinstance(a, LotClose)]
        opens   = [a for a in actions if not isinstance(a, LotClose)]

        for action in closes:
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
            if remaining_qty > _MIN_ACTION_QTY:
                new_lots.append(lot.with_quantity(sign * remaining_qty))
            new_port.lots[action.asset] = new_lots
            if not new_lots:
                del new_port.lots[action.asset]

            # Cash: long close receives proceeds, short cover pays proceeds
            new_port.cash += -proceeds if is_short else proceeds

        for action in opens:
            px = prices[action.asset]
            if isinstance(action, LongOpen):
                if action.quantity < _MIN_ACTION_QTY:
                    continue
                new_port.add_lot(TaxLot(action.asset, action.quantity, px, as_of))
                new_port.cash -= action.quantity * px
            elif isinstance(action, ShortOpen):
                if action.quantity < _MIN_ACTION_QTY:
                    continue
                new_port.add_lot(TaxLot(action.asset, -action.quantity, px, as_of))
                new_port.cash += action.quantity * px

        return new_port, TaxReport.from_events(realized_events, tax_policy, as_of)
