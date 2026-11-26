"""Deterministic Risk Engine (Overseer) with absolute veto authority over all orders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Any

from bot.config import BotConfig
from bot.math_utils import fee_drag_per_share, first_present, to_decimal


@dataclass(frozen=True)
class TradeProposal:
    market_id: str
    question: str
    category: str
    outcome: str  # "Yes" or "No"
    side: str     # "BUY" or "SELL"
    order_mode: str  # "maker_limit" or "marketable_limit"
    limit_price: Decimal
    tick_size: Decimal
    min_order_size: Decimal
    p_true_outcome: Decimal  # probability that THIS chosen outcome resolves True
    edge_net: Decimal
    strategy_type: str
    rationale: str
    est_fee_bps: int = 0


@dataclass(frozen=True)
class ApprovedOrder:
    market_id: str
    question: str
    category: str
    outcome: str
    side: str
    order_mode: str
    price: Decimal
    quantity: Decimal
    notional_usd: Decimal
    edge_net: Decimal
    strategy_type: str
    rationale: str
    est_fee_bps: int
    kelly_fraction_used: Decimal


@dataclass(frozen=True)
class VetoDecision:
    approved: bool
    reason: str
    order: ApprovedOrder | None = None


def choose_entry_execution(
    *,
    strategy_type: str,
    catalyst_freshness: str,
    p_true_outcome: Decimal,
    maker_price: Decimal,
    best_ask: Decimal,
    fee_rate_bps: int,
    breaking_taker_min_edge_net: Decimal,
) -> tuple[str, Decimal, Decimal, int]:
    """Return (order_mode, entry_price, edge_net, est_fee_bps).

    Defaults to 0%-fee post-only maker_limit. Switches to marketable_limit (FAK taker)
    only for news_lag + breaking_lt_6h when edge after taker fees still meets
    breaking_taker_min_edge_net.
    """
    maker_edge = p_true_outcome - maker_price
    if (
        strategy_type == "news_lag"
        and catalyst_freshness == "breaking_lt_6h"
        and Decimal("0") < best_ask < Decimal("1")
    ):
        taker_fee = fee_drag_per_share(best_ask, fee_rate_bps)
        taker_edge = p_true_outcome - best_ask - taker_fee
        if taker_edge >= breaking_taker_min_edge_net:
            return "marketable_limit", best_ask, taker_edge, fee_rate_bps
    return "maker_limit", maker_price, maker_edge, 0


class Overseer:
    """Pure mathematical risk & capital allocation governor (0 LLM tokens)."""

    def __init__(self, config: BotConfig) -> None:
        self.cfg = config

    def compute_portfolio_exposure(
        self,
        positions: list[dict[str, Any]],
        open_orders: list[dict[str, Any]],
        market_categories: dict[str, str],
    ) -> tuple[Decimal, dict[str, Decimal], dict[str, Decimal]]:
        """Return (total_exposure_usd, by_market_usd, by_category_usd)."""
        by_market: dict[str, Decimal] = {}
        by_category: dict[str, Decimal] = {}
        total = Decimal("0")

        for pos in positions:
            mid = str(first_present(pos, "market_id", "condition_id") or "")
            qty = to_decimal(first_present(pos, "quantity", "size"))
            cur_p = to_decimal(first_present(pos, "current_price", "avg_entry_price", "avg_price"))
            val_raw = first_present(pos, "current_value", "market_value")
            val = to_decimal(val_raw) if val_raw is not None else (qty * cur_p)
            if val <= Decimal("0") or not mid:
                continue
            cat = market_categories.get(mid, str(first_present(pos, "category") or "other")).lower()
            by_market[mid] = by_market.get(mid, Decimal("0")) + val
            by_category[cat] = by_category.get(cat, Decimal("0")) + val
            total += val

        # Also count resting BUY orders toward committed exposure so concurrent orders cannot breach caps
        for ord_row in open_orders:
            if str(ord_row.get("side", "")).upper() != "BUY":
                continue
            mid = str(first_present(ord_row, "market_id", "market") or "")
            rem_raw = first_present(ord_row, "remaining_quantity")
            if rem_raw is not None:
                unfilled = max(Decimal("0"), to_decimal(rem_raw))
            else:
                orig = to_decimal(first_present(ord_row, "quantity", "original_size"))
                filled = to_decimal(first_present(ord_row, "size_matched"), "0")
                unfilled = max(Decimal("0"), orig - filled)
            price = to_decimal(first_present(ord_row, "price"), "0")
            committed = unfilled * price
            if committed <= Decimal("0") or not mid:
                continue
            cat = market_categories.get(mid, "other").lower()
            by_market[mid] = by_market.get(mid, Decimal("0")) + committed
            by_category[cat] = by_category.get(cat, Decimal("0")) + committed
            total += committed

        return total, by_market, by_category

    def evaluate_buy(
        self,
        proposal: TradeProposal,
        *,
        equity_usd: Decimal,
        cash_usd: Decimal,
        total_exposure_usd: Decimal,
        market_exposure_usd: Decimal,
        category_exposure_usd: Decimal,
        has_open_order_on_market: bool,
        circuit_breaker_active: bool,
    ) -> VetoDecision:
        if circuit_breaker_active:
            dd_pct = self.cfg.circuit_breaker_24h_dd * Decimal("100")
            return VetoDecision(False, f"VETO: 24h drawdown circuit breaker ({dd_pct:.0f}%) is active.")

        if equity_usd <= Decimal("0"):
            return VetoDecision(False, "VETO: Equity is non-positive.")

        if has_open_order_on_market:
            return VetoDecision(False, "VETO: Resting order already exists on this market.")

        min_edge = self.cfg.min_edge_net
        if proposal.strategy_type == "structural_arb":
            min_edge = Decimal("0.015")
        elif proposal.strategy_type == "carry":
            if proposal.limit_price < self.cfg.carry_min_prob:
                return VetoDecision(
                    False,
                    f"VETO: Carry trade requires limit_price >= {self.cfg.carry_min_prob}, got {proposal.limit_price}.",
                )
            min_edge = Decimal("0.015")

        if proposal.edge_net < min_edge:
            return VetoDecision(
                False,
                f"VETO: Net edge {proposal.edge_net:.4f} < minimum {min_edge:.4f}.",
            )

        p_entry = proposal.limit_price
        p_true = proposal.p_true_outcome
        if p_entry <= Decimal("0") or p_entry >= Decimal("1"):
            return VetoDecision(False, f"VETO: Invalid limit price {p_entry}.")
        if p_true <= p_entry:
            return VetoDecision(False, f"VETO: p_true ({p_true}) <= entry price ({p_entry}).")

        # Exact binary Quarter-Kelly: f* = (p_true - p_entry) / (1 - p_entry)
        full_kelly = (p_true - p_entry) / (Decimal("1") - p_entry)
        raw_fraction = self.cfg.kelly_fraction * full_kelly
        if raw_fraction <= Decimal("0"):
            return VetoDecision(False, "VETO: Kelly stake <= 0.")

        kelly_stake_usd = equity_usd * raw_fraction

        single_cap_rem = (equity_usd * self.cfg.max_single_market_pct) - market_exposure_usd
        cat_cap_rem = (equity_usd * self.cfg.max_category_exposure_pct) - category_exposure_usd
        total_cap_rem = (equity_usd * self.cfg.max_total_exposure_pct) - total_exposure_usd
        cash_rem = cash_usd - self.cfg.cash_buffer_usd

        hard_cap_usd = min(single_cap_rem, cat_cap_rem, total_cap_rem, cash_rem)
        if hard_cap_usd <= Decimal("0"):
            return VetoDecision(
                False,
                f"VETO: Exposure/cash cap reached (single_rem={single_cap_rem:.2f}, "
                f"cat_rem={cat_cap_rem:.2f}, total_rem={total_cap_rem:.2f}, cash_rem={cash_rem:.2f}).",
            )

        target_stake_usd = min(kelly_stake_usd, hard_cap_usd)
        min_shares = max(Decimal("1"), proposal.min_order_size)
        min_notional = min_shares * p_entry

        if target_stake_usd < min_notional:
            # Allow bumping to min_order_size ONLY if high conviction edge >= 10% AND strictly within all hard caps
            if proposal.edge_net >= self.cfg.min_high_conv_bump_edge and min_notional <= hard_cap_usd:
                target_stake_usd = min_notional
            else:
                return VetoDecision(
                    False,
                    f"VETO: Sized stake ${target_stake_usd:.2f} below min order notional ${min_notional:.2f} "
                    f"(min_shares={min_shares}).",
                )

        shares = (target_stake_usd / p_entry).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        if shares < min_shares:
            return VetoDecision(False, f"VETO: Rounded shares {shares} < min_order_size {min_shares}.")

        notional = (shares * p_entry).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        return VetoDecision(
            True,
            "APPROVED",
            ApprovedOrder(
                market_id=proposal.market_id,
                question=proposal.question,
                category=proposal.category,
                outcome=proposal.outcome,
                side="BUY",
                order_mode=proposal.order_mode,
                price=p_entry,
                quantity=shares,
                notional_usd=notional,
                edge_net=proposal.edge_net,
                strategy_type=proposal.strategy_type,
                rationale=proposal.rationale,
                est_fee_bps=proposal.est_fee_bps,
                kelly_fraction_used=raw_fraction.quantize(Decimal("0.0001")),
            ),
        )

    def evaluate_position_exit(
        self,
        *,
        market_id: str,
        question: str,
        category: str,
        outcome: str,
        shares_held: Decimal,
        avg_entry_price: Decimal,
        best_bid: Decimal,
        best_ask: Decimal,
        tick_size: Decimal,
        min_order_size: Decimal,
        p_true_outcome: Decimal,
        fee_rate_bps: int,
    ) -> TradeProposal | None:
        """Return a SELL proposal for take-profit (edge collapsed) or thesis-break."""
        if shares_held < min_order_size or best_bid <= Decimal("0"):
            return None

        # 1. Thesis break: true probability dropped clearly below exit bid
        remaining_edge_at_bid = p_true_outcome - best_bid
        if remaining_edge_at_bid <= self.cfg.thesis_break_edge:
            return TradeProposal(
                market_id=market_id,
                question=question,
                category=category,
                outcome=outcome,
                side="SELL",
                order_mode="marketable_limit",
                limit_price=best_bid,
                tick_size=tick_size,
                min_order_size=min_order_size,
                p_true_outcome=p_true_outcome,
                edge_net=remaining_edge_at_bid,
                strategy_type="exit_thesis_break",
                rationale=f"Thesis-Break: Fair Value {p_true_outcome:.2f} < Bid {best_bid:.2f}; Kapital schützen.",
                est_fee_bps=fee_rate_bps,
            )

        # 2. Take-profit / Edge Collapse: in profit and remaining upside < 2%
        spread = best_ask - best_bid
        maker_sell_price = (best_ask - tick_size) if spread >= Decimal("2") * tick_size else best_ask
        if maker_sell_price <= best_bid:
            maker_sell_price = best_ask
        remaining_edge = p_true_outcome - maker_sell_price
        in_profit = maker_sell_price > (avg_entry_price + Decimal("0.02"))
        if in_profit and remaining_edge < self.cfg.take_profit_edge_remaining:
            return TradeProposal(
                market_id=market_id,
                question=question,
                category=category,
                outcome=outcome,
                side="SELL",
                order_mode="maker_limit",
                limit_price=maker_sell_price,
                tick_size=tick_size,
                min_order_size=min_order_size,
                p_true_outcome=p_true_outcome,
                edge_net=remaining_edge,
                strategy_type="exit_take_profit",
                rationale=(
                    f"Take-Profit: Kurs {maker_sell_price:.2f} hat Fair Value {p_true_outcome:.2f} erreicht "
                    f"(Entry {avg_entry_price:.2f}); Kapital wird für neue Chancen frei."
                ),
                est_fee_bps=0,
            )
        return None

    def find_stale_orders(self, open_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Identify resting orders older than stale_order_minutes."""
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.cfg.stale_order_minutes)
        stale: list[dict[str, Any]] = []
        for o in open_orders:
            created = o.get("created_at") or o.get("timestamp")
            if not created:
                continue
            try:
                dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt < cutoff:
                    stale.append(o)
            except ValueError:
                continue
        return stale
