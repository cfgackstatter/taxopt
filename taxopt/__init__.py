# taxopt/__init__.py
from .data_types import (
    AssetId,
    TaxLot,
    LotClose,
    LongOpen,
    ShortOpen,
    PortfolioAction,
    OptimizationInputs,
    OptimizationResult,
    Optimizer,          # ← now lives in data_types
)
from .optimizer import CvxpyOptimizer
from .portfolio import Portfolio
from .tax_policy import USCapitalGainsPolicy, NoTaxPolicy, LotMethod
from .tax_report import TaxReport, RealizedGain


__all__ = [
    "AssetId",
    "TaxLot",
    "Portfolio",
    "PortfolioAction",
    "LotClose",
    "LongOpen",
    "ShortOpen",
    "RealizedGain",
    "TaxReport",
    "USCapitalGainsPolicy",
    "NoTaxPolicy",
    "LotMethod",
    "OptimizationInputs",
    "OptimizationResult",
    "CvxpyOptimizer",
    "Optimizer",
]
