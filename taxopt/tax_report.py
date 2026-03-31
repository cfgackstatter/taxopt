# tax_report.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable, Mapping, Sequence

from .data_types import AssetId, TaxLot


@dataclass(frozen=True)
class RealizedGain:
    asset: AssetId
    lot: TaxLot
    quantity: float          # absolute quantity sold (>0)
    proceeds: float
    cost: float
    gain: float              # proceeds - cost
    gain_type: str           # e.g. "short_term", "long_term"


@dataclass(frozen=True)
class TaxReport:
    as_of: date
    events: Sequence[RealizedGain]
    totals_by_type: Mapping[str, float]
    total_gain: float

    @classmethod
    def from_events(
        cls,
        events: Iterable[RealizedGain],
        tax_policy: Any,
        as_of: date,
    ) -> "TaxReport":
        ev = list(events)
        totals: dict[str, float] = {}
        for e in ev:
            totals[e.gain_type] = totals.get(e.gain_type, 0.0) + e.gain
        totals = tax_policy.netting_rules(totals)
        return cls(
            as_of=as_of,
            events=ev,
            totals_by_type=totals,
            total_gain=sum(totals.values()),
        )
