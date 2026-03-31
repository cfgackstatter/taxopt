from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .portfolio import Portfolio
from .tax_report import TaxReport


class _RatePolicy(Protocol):
    st_rate: float
    lt_rate: float
    lt_threshold_days: int


@dataclass
class TaxLedger:
    """
    Tracks cumulative realized gains/losses over the backtest.
    Assumes the investor has external gains to absorb harvested losses
    immediately — standard assumption in the TLH literature.
    No year-end reset, no carryforward complexity.
    """
    st_realized: float = 0.0
    lt_realized: float = 0.0

    def record(self, report: TaxReport) -> None:
        self.st_realized += report.totals_by_type.get("short_term", 0.0)
        self.lt_realized += report.totals_by_type.get("long_term",  0.0)

    def cumulative_tax_value(self, policy: _RatePolicy) -> float:
        """Net tax impact of all realized activity. Negative = net tax saving."""
        return self.st_realized * policy.st_rate + self.lt_realized * policy.lt_rate

    def after_tax_nav(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        as_of: date,
        policy: _RatePolicy,
    ) -> float:
        """
        Pre-tax NAV
        − DTL on unrealized gains (hypothetical liquidation tax today)
        − cumulative tax on realized gains (positive = owed, negative = saved)
        """
        unreal_st, unreal_lt = 0.0, 0.0
        for asset, lots in portfolio.lots.items():
            px = prices[asset]
            for lot in lots:
                if lot.quantity <= 0:
                    continue
                gain = (px - lot.cost_basis) * lot.quantity
                days = (as_of - lot.acquisition_date).days
                if days >= policy.lt_threshold_days:
                    unreal_lt += gain
                else:
                    unreal_st += gain

        dtl = unreal_st * policy.st_rate + unreal_lt * policy.lt_rate
        return portfolio.total_value(prices) - dtl - self.cumulative_tax_value(policy)