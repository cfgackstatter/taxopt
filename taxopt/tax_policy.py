# tax_policy.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Protocol, cast, runtime_checkable

from .data_types import AssetId, TaxLot
from .tax_report import RealizedGain


@runtime_checkable
class TaxPolicy(Protocol):
    def classify_gain(self, lot: TaxLot, sale_date: date, gain: float) -> str: ...
    def netting_rules(self, totals_by_type: dict[str, float]) -> dict[str, float]: ...


class LotMethod(Enum):
    FIFO     = "fifo"
    LIFO     = "lifo"
    MIN_GAIN = "min_gain"
    MAX_GAIN = "max_gain"
    MAX_LOSS = "max_loss"


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

    def netting_rules(self, totals_by_type: dict[str, float]) -> dict[str, float]:
        return totals_by_type
