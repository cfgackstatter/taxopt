# tax_policy.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Callable, Protocol, cast, runtime_checkable

from .data_types import AssetId, TaxLot
from .tax_report import RealizedGain


@runtime_checkable
class TaxPolicy(Protocol):
    def classify_gain(self, lot: TaxLot, sale_date: date, gain: float) -> str: ...

    def match_and_realize(
        self,
        asset: AssetId,
        lots: list[TaxLot],
        sell_quantity: float,
        price: float,
        as_of: date,
    ) -> tuple[list[RealizedGain], list[TaxLot], float]: ...

    def netting_rules(self, totals_by_type: dict[str, float]) -> dict[str, float]: ...


class LotMethod(Enum):
    FIFO     = "fifo"
    LIFO     = "lifo"
    MIN_GAIN = "min_gain"
    MAX_GAIN = "max_gain"
    MAX_LOSS = "max_loss"


def _sort_lots(lots: list[TaxLot], method: LotMethod, price: float) -> list[TaxLot]:
    match method:
        case LotMethod.FIFO:
            return sorted(lots, key=lambda l: l.acquisition_date)
        case LotMethod.LIFO:
            return sorted(lots, key=lambda l: l.acquisition_date, reverse=True)
        case LotMethod.MIN_GAIN:
            return sorted(lots, key=lambda l: price - l.cost_basis)
        case LotMethod.MAX_GAIN:
            return sorted(lots, key=lambda l: price - l.cost_basis, reverse=True)
        case LotMethod.MAX_LOSS:
            return sorted(lots, key=lambda l: l.cost_basis, reverse=True)
        case _:
            raise ValueError(f"Unknown lot method: {method}")


def _consume_lots(
    asset: AssetId,
    sorted_lots: list[TaxLot],
    sell_quantity: float,
    price: float,
    as_of: date,
    classify: Callable[[TaxLot, date, float], str],   # fixed
) -> tuple[list[RealizedGain], list[TaxLot], float]:
    events: list[RealizedGain] = []
    remaining: list[TaxLot] = []
    qty_left = sell_quantity

    for lot in sorted_lots:
        if qty_left <= 0:
            remaining.append(lot)
            continue
        abs_lot_qty = abs(lot.quantity)
        consumed = min(abs_lot_qty, qty_left)
        sign = 1.0 if lot.quantity > 0 else -1.0

        proceeds = consumed * price
        cost     = consumed * lot.cost_basis
        gain     = (proceeds - cost) * sign

        events.append(RealizedGain(
            asset=asset, lot=lot, quantity=consumed,
            proceeds=proceeds, cost=cost, gain=gain,
            gain_type=classify(lot, as_of, gain),
        ))

        if abs_lot_qty > qty_left:
            remaining.append(lot.with_quantity(sign * (abs_lot_qty - consumed)))
        qty_left -= consumed

    if qty_left > 1e-6:
        raise ValueError(f"Insufficient lots for {asset}: short by {qty_left:.6f}")

    return events, remaining, sell_quantity * price


@dataclass
class USCapitalGainsPolicy:
    lt_threshold_days: int = 366
    st_rate: float = 0.35
    lt_rate: float = 0.20
    lot_method: LotMethod = LotMethod.FIFO

    def classify_gain(self, lot: TaxLot, sale_date: date, gain: float) -> str:
        if lot.quantity < 0:
            if gain >= 0:
                return "short_term"   # §1233: short sale gains always ST
            # §1233(d): loss is LT only if substantially identical long was held
            # > lt_threshold_days on the date the short was opened
            meta = lot.metadata or {}
            short_open_date = cast(date, meta.get("short_open_date", lot.acquisition_date))
            long_held_since = cast(date | None, meta.get("long_held_since"))
            if long_held_since is not None:
                days = (short_open_date - long_held_since).days
                return "long_term" if days >= self.lt_threshold_days else "short_term"
            return "short_term"
        # Long positions: standard holding period rule
        days_held = (sale_date - lot.acquisition_date).days
        return "long_term" if days_held >= self.lt_threshold_days else "short_term"

    def match_and_realize(
        self,
        asset: AssetId,
        lots: list[TaxLot],
        sell_quantity: float,
        price: float,
        as_of: date,
    ) -> tuple[list[RealizedGain], list[TaxLot], float]:
        return _consume_lots(
            asset, _sort_lots(lots, self.lot_method, price),
            sell_quantity, price, as_of, self.classify_gain,
        )

    def netting_rules(self, totals_by_type: dict[str, float]) -> dict[str, float]:
        st = totals_by_type.get("short_term", 0.0)
        lt = totals_by_type.get("long_term",  0.0)
        if st < 0 < lt:
            offset = min(-st, lt)
            st += offset; lt -= offset
        elif lt < 0 < st:
            offset = min(-lt, st)
            lt += offset; st -= offset
        return {"short_term": st, "long_term": lt}

    def tax_liability(self, report: "TaxReport") -> float:  # type: ignore[name-defined]
        from .tax_report import TaxReport
        t = self.netting_rules(dict(report.totals_by_type))
        return (
            max(t.get("short_term", 0.0), 0.0) * self.st_rate
            + max(t.get("long_term",  0.0), 0.0) * self.lt_rate
        )


@dataclass
class NoTaxPolicy:
    lot_method: LotMethod = LotMethod.FIFO

    def classify_gain(self, lot: TaxLot, sale_date: date, gain: float) -> str:
        return "none"

    def match_and_realize(
        self,
        asset: AssetId,
        lots: list[TaxLot],
        sell_quantity: float,
        price: float,
        as_of: date,
    ) -> tuple[list[RealizedGain], list[TaxLot], float]:
        return _consume_lots(
            asset, _sort_lots(lots, self.lot_method, price),
            sell_quantity, price, as_of, self.classify_gain,
        )

    def netting_rules(self, totals_by_type: dict[str, float]) -> dict[str, float]:
        return totals_by_type
