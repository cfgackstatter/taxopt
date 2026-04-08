# optimizer.py
from __future__ import annotations
from dataclasses import dataclass, replace
from datetime import date
from typing import cast

import cvxpy as cp
from cvxpy.constraints.constraint import Constraint as CpConstraint
import numpy as np

from .data_types import (
     _MIN_ACTION_QTY, AssetId, TaxLot, LotClose, LongOpen, ShortOpen,
    OptimizationInputs, OptimizationResult, PortfolioAction,
)
from .portfolio import Portfolio
from .tax_policy import TaxPolicy


@dataclass
class PortfolioPolicy:
    risk_aversion:  float       = 2.0
    tax_aversion:   float       = 1.0
    gross_leverage: float       = 1.0
    net_exposure:   float       = 1.0
    max_weight:     float       = 0.25
    max_turnover:   float | None = 0.25


@dataclass
class CvxpyOptimizer:
    solver: str = "SCIP"
    verbose: bool = False
    tax_aware: bool = True
    relax_turnover: bool = False
    turnover_relax_step: float = 0.05
    turnover_relax_max_attempts: int = 5
    mip_gap: float | None = 0.0
    periods_per_year: float = 12.0

    def solve(
        self,
        portfolio: Portfolio,
        inputs: OptimizationInputs,
        tax_policy: TaxPolicy,
        total_value: float,
    ) -> OptimizationResult:
        if total_value <= 0:
            raise ValueError(f"total_value must be positive, got {total_value}")

        effective_inputs = inputs
        prob: cp.Problem | None = None
        attempts = (
            self.turnover_relax_max_attempts
            if (self.relax_turnover and inputs.max_turnover is not None)
            else 1
        )

        for attempt in range(attempts):
            ctx  = _LotContext.build(portfolio, effective_inputs, tax_policy)
            v    = _build_variables(ctx)
            lcv, scv = _close_vecs(ctx, v)
            cons = _build_constraints(ctx, v, effective_inputs, portfolio, total_value, lcv, scv)
            obj  = _build_objective(ctx, v, effective_inputs, total_value, self.tax_aware,
                            self.periods_per_year, lcv, scv)

            prob = cp.Problem(obj, cons)

            # Analytic warm start for first rebalance (no existing lots)
            if ctx.m == 0:
                _set_first_rebal_hint(v, ctx, effective_inputs, total_value)

            solver_kwargs: dict = {}
            if self.mip_gap is not None and self.solver == "SCIP":
                solver_kwargs["limits/gap"] = self.mip_gap

            prob.solve(solver=self.solver, verbose=self.verbose, **solver_kwargs)

            if prob.status in ("optimal", "optimal_inaccurate"):
                return _extract_result(ctx, v, effective_inputs, total_value, prob.status)

            if (
                self.relax_turnover
                and effective_inputs.max_turnover is not None
                and attempt < attempts - 1
            ):
                relaxed = effective_inputs.max_turnover + self.turnover_relax_step
                if self.verbose:
                    print(
                        f"  [relax] attempt {attempt+1} failed ({prob.status}), "
                        f"relaxing turnover {effective_inputs.max_turnover:.2%} → {relaxed:.2%}"
                    )
                effective_inputs = replace(effective_inputs, max_turnover=relaxed)

        return OptimizationResult(
            weights={}, weights_long={}, weights_short={},
            actions=[], realized_gain=0.0, tax_cost=0.0,
            status=prob.status if prob is not None else "failed",
            _prices=inputs.prices,
            effective_turnover=0.0,
        )
        

# ---------------------------------------------------------------------------
# Lot context
# ---------------------------------------------------------------------------

@dataclass
class _LotContext:
    assets: list[AssetId]
    n: int
    prices: dict[AssetId, float]
    all_lots: list[TaxLot]
    lot_asset_idx: list[int]
    lot_is_long: list[bool]
    lot_qty: np.ndarray
    lot_prices: np.ndarray
    lot_gain_per_unit: np.ndarray
    lot_tax_cost_per_unit: np.ndarray
    cur_long_dollars: np.ndarray
    cur_shrt_dollars: np.ndarray
    P_long: np.ndarray           # shape (n, m): price-weighted long indicator
    P_shrt: np.ndarray           # shape (n, m): price-weighted short indicator
    lots_by_asset_long: list[list[int]]
    lots_by_asset_shrt: list[list[int]]

    @property
    def m(self) -> int:
        return len(self.all_lots)

    @classmethod
    def build(
        cls,
        portfolio: Portfolio,
        inputs: OptimizationInputs,
        tax_policy: TaxPolicy,
    ) -> _LotContext:
        assets = inputs.assets
        n      = len(assets)
        prices = inputs.prices
        idx    = {a: i for i, a in enumerate(assets)}
        as_of  = inputs.as_of

        all_lots:      list[TaxLot] = []
        lot_asset_idx: list[int]    = []
        lot_is_long:   list[bool]   = []

        for asset, lots in portfolio.lots.items():
            if asset not in idx:
                continue
            for lot in lots:
                all_lots.append(lot)
                lot_asset_idx.append(idx[asset])
                lot_is_long.append(lot.quantity > 0)

        m = len(all_lots)

        if m > 0:
            lot_qty = np.array([abs(l.quantity) for l in all_lots], dtype=float)
            lot_px  = np.array([prices[assets[lot_asset_idx[j]]] for j in range(m)], dtype=float)
            gain_pu = np.array(
                [(lot_px[j] - all_lots[j].cost_basis) * (1 if lot_is_long[j] else -1)
                for j in range(m)],
                dtype=float,
            )
            tax_rate  = np.array([_marginal_tax_rate(all_lots[j], gain_pu[j], as_of, tax_policy)
                                   for j in range(m)], dtype=float)
        else:
            lot_qty = lot_px = gain_pu = tax_rate = np.zeros(0)

        if m > 0:
            P_long = np.zeros((n, m))
            P_shrt = np.zeros((n, m))
            for j in range(m):
                if lot_is_long[j]:
                    P_long[lot_asset_idx[j], j] = lot_px[j]
                else:
                    P_shrt[lot_asset_idx[j], j] = lot_px[j]
        else:
            P_long = np.zeros((n, 0))
            P_shrt = np.zeros((n, 0))

        if m > 0:
            lot_asset_arr = np.array(lot_asset_idx)
            is_long_arr   = np.array(lot_is_long)
            dollar_arr    = lot_qty * lot_px
            cur_long = np.bincount(lot_asset_arr[ is_long_arr], weights=dollar_arr[ is_long_arr], minlength=n)
            cur_shrt = np.bincount(lot_asset_arr[~is_long_arr], weights=dollar_arr[~is_long_arr], minlength=n)
        else:
            cur_long = np.zeros(n)
            cur_shrt = np.zeros(n)

        lots_by_asset_long: list[list[int]] = [[] for _ in range(n)]
        lots_by_asset_shrt: list[list[int]] = [[] for _ in range(n)]
        for j in range(m):
            if lot_is_long[j]:
                lots_by_asset_long[lot_asset_idx[j]].append(j)
            else:
                lots_by_asset_shrt[lot_asset_idx[j]].append(j)

        return cls(
            assets=assets, n=n, prices=prices,
            all_lots=all_lots, lot_asset_idx=lot_asset_idx, lot_is_long=lot_is_long,
            lot_qty=lot_qty, lot_prices=lot_px,
            lot_gain_per_unit=gain_pu,
            lot_tax_cost_per_unit=gain_pu * tax_rate,
            cur_long_dollars=cur_long,
            cur_shrt_dollars=cur_shrt,
            P_long=P_long,
            P_shrt=P_shrt,
            lots_by_asset_long=lots_by_asset_long,
            lots_by_asset_shrt=lots_by_asset_shrt,
        )


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

@dataclass
class _Vars:
    s:      cp.Variable | None
    b_long: cp.Variable
    b_shrt: cp.Variable


def _build_variables(ctx: _LotContext) -> _Vars:
    return _Vars(
        s      = cp.Variable(ctx.m, nonneg=True, name="lot_closes") if ctx.m > 0 else None,
        b_long = cp.Variable(ctx.n, nonneg=True, name="buys_long"),
        b_shrt = cp.Variable(ctx.n, nonneg=True, name="buys_short"),
    )


def _close_vecs(
    ctx: _LotContext,
    v: _Vars,
) -> tuple[cp.Expression, cp.Expression]:
    if v.s is not None:
        return ctx.P_long @ v.s, ctx.P_shrt @ v.s
    z = cp.Constant(np.zeros(ctx.n))
    return z, z


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

def _build_constraints(
    ctx: _LotContext,
    v: _Vars,
    inputs: OptimizationInputs,
    portfolio: Portfolio,
    nav: float,
    long_close_vec: cp.Expression,
    shrt_close_vec: cp.Expression,
) -> list[CpConstraint]:
    C: list[CpConstraint] = []

    if v.s is not None:
        s_m = v.s

        # Per-lot cap
        C.append(cast(CpConstraint, s_m <= ctx.lot_qty))

        # Wash-sale: scalar binary per asset, created only when that side has lots
        max_dollars = float(inputs.max_weight * nav)
        for i in range(ctx.n):
            lfa = ctx.lots_by_asset_long[i]
            sfa = ctx.lots_by_asset_shrt[i]
            if lfa:
                tq = float(sum(ctx.lot_qty[j] for j in lfa))
                cl = cp.sum([s_m[j] for j in lfa])
                z  = cp.Variable(boolean=True, name=f"z_long_{i}")
                C += [cast(CpConstraint, cl <= tq * z),
                      cast(CpConstraint, v.b_long[i] <= max_dollars * (1 - z))]
            if sfa:
                tq = float(sum(ctx.lot_qty[j] for j in sfa))
                cs = cp.sum([s_m[j] for j in sfa])
                z  = cp.Variable(boolean=True, name=f"z_shrt_{i}")
                C += [cast(CpConstraint, cs <= tq * z),
                      cast(CpConstraint, v.b_shrt[i] <= max_dollars * (1 - z))]

    final_long = ctx.cur_long_dollars - long_close_vec + v.b_long
    final_shrt = ctx.cur_shrt_dollars - shrt_close_vec + v.b_shrt

    max_dollars = float(inputs.max_weight * nav)
    gross       = float(inputs.gross_leverage * nav)
    net         = float(inputs.net_exposure   * nav)

    C += cast(list[CpConstraint], [
        # Weight caps
        final_long <= max_dollars, final_shrt <= max_dollars,
        final_long >= 0,           final_shrt >= 0,
        # Leverage / exposure bands
        cp.sum(final_long) + cp.sum(final_shrt) == gross,
        cp.sum(final_long) - cp.sum(final_shrt) == net,
        # Net settlement
        cp.sum(v.b_long) + cp.sum(shrt_close_vec) <= portfolio.cash + cp.sum(long_close_vec) + cp.sum(v.b_shrt),
    ])

    if inputs.max_turnover is not None:
        cur_net = ctx.cur_long_dollars - ctx.cur_shrt_dollars
        C.append(cast(CpConstraint,
            cp.norm1(final_long - final_shrt - cur_net) / nav <= 2.0 * inputs.max_turnover
        ))

    return C


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def _build_objective(
    ctx: _LotContext,
    v: _Vars,
    inputs: OptimizationInputs,
    nav: float,
    tax_aware: bool,
    periods_per_year: float,
    long_close_vec: cp.Expression,
    shrt_close_vec: cp.Expression,
) -> cp.Maximize:
    final_long = ctx.cur_long_dollars - long_close_vec + v.b_long
    final_shrt = ctx.cur_shrt_dollars - shrt_close_vec + v.b_shrt
    w = (final_long - final_shrt) / nav

    if v.s is not None and tax_aware:
        tax_term = cast(cp.Expression, (ctx.lot_tax_cost_per_unit @ v.s) / nav) * periods_per_year
    else:
        tax_term = cp.Constant(0.0)

    alpha_vec = np.array([inputs.alpha.get(a, 0.0) for a in ctx.assets])
    return cp.Maximize(
        alpha_vec @ w
        - inputs.risk_aversion * cp.quad_form(w, cp.psd_wrap(inputs.covariance))
        - inputs.tax_aversion * tax_term
    )


# ---------------------------------------------------------------------------
# Initial solution
# ---------------------------------------------------------------------------

def _set_first_rebal_hint(
    v: _Vars,
    ctx: _LotContext,
    inputs: OptimizationInputs,
    nav: float,
) -> None:
    """Construct an analytic equal-weight warm-start for the first rebalance."""
    alpha_arr = np.array([inputs.alpha.get(a, 0.0) for a in ctx.assets])
    order     = np.argsort(alpha_arr)[::-1]   # descending alpha rank

    long_pool  = (inputs.gross_leverage + inputs.net_exposure) / 2.0 * nav
    short_pool = (inputs.gross_leverage - inputs.net_exposure) / 2.0 * nav
    max_dollars = inputs.max_weight * nav

    def _equal_weight(pool: float, indices: np.ndarray) -> np.ndarray:
        out = np.zeros(ctx.n)
        if pool < 1e-9:
            return out
        per_name = min(pool / len(indices), max_dollars)
        for idx in indices:
            out[idx] = per_name
        return out

    # How many names fit given max_weight cap?
    k_long  = max(1, int(np.ceil(long_pool  / max_dollars)))
    k_short = max(1, int(np.ceil(short_pool / max_dollars))) if short_pool > 0 else 0

    v.b_long.value = _equal_weight(long_pool,  order[:k_long])
    v.b_shrt.value = _equal_weight(short_pool, order[-k_short:]) if k_short > 0 else np.zeros(ctx.n)


# ---------------------------------------------------------------------------
# Extract result
# ---------------------------------------------------------------------------

def _extract_result(
    ctx: _LotContext,
    v: _Vars,
    inputs: OptimizationInputs,
    nav: float,
    status: str,
) -> OptimizationResult:
    def _val(var: cp.Variable) -> np.ndarray:
        if var.value is None:
            raise RuntimeError(f"Variable '{var.name()}' has no value after solve — status may be infeasible")
        return np.asarray(var.value).ravel()

    s_arr      = np.maximum(_val(v.s), 0.0) if v.s is not None else np.zeros(0)
    b_long_arr = np.maximum(_val(v.b_long), 0.0)
    b_shrt_arr = np.maximum(_val(v.b_shrt), 0.0)

    # Recompute final dollar positions numerically
    close_long = ctx.P_long @ s_arr if ctx.m > 0 else 0.0
    close_shrt = ctx.P_shrt @ s_arr if ctx.m > 0 else 0.0
    final_long = ctx.cur_long_dollars + b_long_arr - close_long
    final_shrt = ctx.cur_shrt_dollars + b_shrt_arr - close_shrt
    w_long = final_long / nav
    w_shrt = final_shrt / nav

    w_long_init = ctx.cur_long_dollars / nav
    w_shrt_init = ctx.cur_shrt_dollars / nav
    actual_turnover = float(
        np.sum(np.abs((w_long - w_shrt) - (w_long_init - w_shrt_init))) / 2
    )

    actions: list[PortfolioAction] = []
    for j, lot in enumerate(ctx.all_lots):
        units = float(s_arr[j])
        if units > _MIN_ACTION_QTY:
            actions.append(LotClose(
                asset=lot.asset,
                quantity=min(units, float(ctx.lot_qty[j])),
                lot_ref=lot,
            ))
    for i, asset in enumerate(ctx.assets):
        if b_long_arr[i] > _MIN_ACTION_QTY:
            actions.append(LongOpen(asset=asset, quantity=b_long_arr[i] / ctx.prices[asset]))
        if b_shrt_arr[i] > _MIN_ACTION_QTY:
            actions.append(ShortOpen(asset=asset, quantity=b_shrt_arr[i] / ctx.prices[asset]))

    return OptimizationResult(
        weights      = {a: float(w_long[i] - w_shrt[i]) for i, a in enumerate(ctx.assets)},
        weights_long = {a: float(w_long[i])              for i, a in enumerate(ctx.assets)},
        weights_short= {a: float(w_shrt[i])              for i, a in enumerate(ctx.assets)},
        actions=actions,
        realized_gain=float(ctx.lot_gain_per_unit      @ s_arr) if ctx.m > 0 else 0.0,
        tax_cost     =float(ctx.lot_tax_cost_per_unit  @ s_arr) if ctx.m > 0 else 0.0,
        status=status,
        _prices=dict(inputs.prices),
        effective_turnover=actual_turnover,
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _marginal_tax_rate(
    lot: TaxLot, gain_per_unit: float, as_of: date, tax_policy: TaxPolicy,
) -> float:
    gain_type = tax_policy.classify_gain(lot, as_of, gain_per_unit)
    if gain_type == "long_term":  return getattr(tax_policy, "lt_rate", 0.0)
    if gain_type == "short_term": return getattr(tax_policy, "st_rate", 0.0)
    return 0.0
