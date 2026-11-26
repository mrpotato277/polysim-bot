"""Order execution module: constructs post-only maker orders first and logs every attempt."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from polysim_sdk import ApiError, PolySimError

from bot.config import BotConfig
from bot.math_utils import calc_taker_fee, maker_buy_price, maker_sell_price
from bot.overseer import ApprovedOrder, TradeProposal
from bot.polysim_client import PolySimWrapper
from bot.state import StateStore

log = logging.getLogger(__name__)


def build_client_order_id(market_id: str, side: str, outcome: str) -> str:
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S")
    short_mkt = market_id.replace("0x", "")[:10]
    suffix = uuid.uuid4().hex[:6]
    return f"psb-{stamp}-{short_mkt}-{side.lower()}-{outcome.lower()[:1]}-{suffix}"


class ExecutionEngine:
    """Executes Overseer-approved orders against PolySimulator or records dry-run intents."""

    def __init__(self, client: PolySimWrapper, state: StateStore, config: BotConfig) -> None:
        self.client = client
        self.state = state
        self.cfg = config

    @staticmethod
    def price_for_entry(
        *,
        best_bid: Decimal,
        best_ask: Decimal,
        tick_size: Decimal,
        side: str,
        order_mode: str,
    ) -> Decimal:
        if order_mode == "maker_limit":
            if side.upper() == "BUY":
                return maker_buy_price(best_bid, best_ask, tick_size)
            return maker_sell_price(best_bid, best_ask, tick_size)
        return best_bid if side.upper() == "SELL" else best_ask

    def execute_buy(self, order: ApprovedOrder, *, tick_size: Decimal) -> tuple[bool, str]:
        coid = build_client_order_id(order.market_id, "BUY", order.outcome)
        est_fee = calc_taker_fee(order.quantity, order.price, order.est_fee_bps)

        if self.cfg.dry_run:
            self.state.record_trade(
                {
                    "client_order_id": coid,
                    "order_id": "DRY_RUN",
                    "market_id": order.market_id,
                    "question": order.question,
                    "outcome": order.outcome,
                    "side": "BUY",
                    "order_mode": order.order_mode,
                    "price": f"{order.price}",
                    "quantity": f"{order.quantity}",
                    "notional_usd": f"{order.notional_usd}",
                    "est_fee_usd": f"{est_fee}",
                    "edge_net": f"{order.edge_net:.4f}",
                    "strategy_type": order.strategy_type,
                    "status": "DRY_RUN",
                    "dry_run": True,
                    "rationale": order.rationale,
                }
            )
            return True, f"DRY_RUN BUY {order.quantity} {order.outcome} @ {order.price}"

        try:
            if order.order_mode == "maker_limit":
                resp = self.client.place_maker_limit_order(
                    market_id=order.market_id,
                    side="BUY",
                    outcome=order.outcome,
                    price=order.price,
                    quantity=order.quantity,
                    tick_size=tick_size,
                    client_order_id=coid,
                )
            else:
                resp = self.client.place_marketable_limit_order(
                    market_id=order.market_id,
                    side="BUY",
                    outcome=order.outcome,
                    limit_price=order.price,
                    quantity=order.quantity,
                    tick_size=tick_size,
                    client_order_id=coid,
                )
            status = str(resp.get("status") or "SUBMITTED")
            oid = str(resp.get("order_id") or resp.get("id") or "")
            actual_fee = str(resp.get("fee") or est_fee)
            self.state.record_trade(
                {
                    "client_order_id": coid,
                    "order_id": oid,
                    "market_id": order.market_id,
                    "question": order.question,
                    "outcome": order.outcome,
                    "side": "BUY",
                    "order_mode": order.order_mode,
                    "price": f"{order.price}",
                    "quantity": f"{order.quantity}",
                    "notional_usd": f"{order.notional_usd}",
                    "est_fee_usd": actual_fee,
                    "edge_net": f"{order.edge_net:.4f}",
                    "strategy_type": order.strategy_type,
                    "status": status,
                    "dry_run": False,
                    "rationale": order.rationale,
                }
            )
            return True, f"LIVE BUY {status} oid={oid}"
        except (ApiError, PolySimError, Exception) as exc:
            err_msg = str(exc)[:240]
            log.warning("BUY failed for %s: %s", order.market_id, err_msg)
            self.state.record_trade(
                {
                    "client_order_id": coid,
                    "order_id": "",
                    "market_id": order.market_id,
                    "question": order.question,
                    "outcome": order.outcome,
                    "side": "BUY",
                    "order_mode": order.order_mode,
                    "price": f"{order.price}",
                    "quantity": f"{order.quantity}",
                    "notional_usd": f"{order.notional_usd}",
                    "est_fee_usd": f"{est_fee}",
                    "edge_net": f"{order.edge_net:.4f}",
                    "strategy_type": order.strategy_type,
                    "status": "ERROR",
                    "dry_run": False,
                    "rationale": f"{order.rationale} | ERR: {err_msg}",
                }
            )
            return False, err_msg

    def execute_sell_exit(self, proposal: TradeProposal, shares_held: Decimal) -> tuple[bool, str]:
        coid = build_client_order_id(proposal.market_id, "SELL", proposal.outcome)
        qty = shares_held.quantize(Decimal("0.01"))
        notional = (qty * proposal.limit_price).quantize(Decimal("0.01"))
        est_fee = calc_taker_fee(qty, proposal.limit_price, proposal.est_fee_bps)

        if self.cfg.dry_run:
            self.state.record_trade(
                {
                    "client_order_id": coid,
                    "order_id": "DRY_RUN",
                    "market_id": proposal.market_id,
                    "question": proposal.question,
                    "outcome": proposal.outcome,
                    "side": "SELL",
                    "order_mode": proposal.order_mode,
                    "price": f"{proposal.limit_price}",
                    "quantity": f"{qty}",
                    "notional_usd": f"{notional}",
                    "est_fee_usd": f"{est_fee}",
                    "edge_net": f"{proposal.edge_net:.4f}",
                    "strategy_type": proposal.strategy_type,
                    "status": "DRY_RUN",
                    "dry_run": True,
                    "rationale": proposal.rationale,
                }
            )
            return True, f"DRY_RUN SELL {qty} {proposal.outcome} @ {proposal.limit_price}"

        try:
            if proposal.order_mode == "maker_limit":
                resp = self.client.place_maker_limit_order(
                    market_id=proposal.market_id,
                    side="SELL",
                    outcome=proposal.outcome,
                    price=proposal.limit_price,
                    quantity=qty,
                    tick_size=proposal.tick_size,
                    client_order_id=coid,
                )
            else:
                resp = self.client.place_marketable_limit_order(
                    market_id=proposal.market_id,
                    side="SELL",
                    outcome=proposal.outcome,
                    limit_price=proposal.limit_price,
                    quantity=qty,
                    tick_size=proposal.tick_size,
                    client_order_id=coid,
                )
            status = str(resp.get("status") or "SUBMITTED")
            oid = str(resp.get("order_id") or resp.get("id") or "")
            self.state.record_trade(
                {
                    "client_order_id": coid,
                    "order_id": oid,
                    "market_id": proposal.market_id,
                    "question": proposal.question,
                    "outcome": proposal.outcome,
                    "side": "SELL",
                    "order_mode": proposal.order_mode,
                    "price": f"{proposal.limit_price}",
                    "quantity": f"{qty}",
                    "notional_usd": f"{notional}",
                    "est_fee_usd": str(resp.get("fee") or est_fee),
                    "edge_net": f"{proposal.edge_net:.4f}",
                    "strategy_type": proposal.strategy_type,
                    "status": status,
                    "dry_run": False,
                    "rationale": proposal.rationale,
                }
            )
            return True, f"LIVE SELL {status} oid={oid}"
        except Exception as exc:
            err_msg = str(exc)[:240]
            log.warning("SELL failed for %s: %s", proposal.market_id, err_msg)
            self.state.record_trade(
                {
                    "client_order_id": coid,
                    "order_id": "",
                    "market_id": proposal.market_id,
                    "question": proposal.question,
                    "outcome": proposal.outcome,
                    "side": "SELL",
                    "order_mode": proposal.order_mode,
                    "price": f"{proposal.limit_price}",
                    "quantity": f"{qty}",
                    "notional_usd": f"{notional}",
                    "est_fee_usd": f"{est_fee}",
                    "edge_net": f"{proposal.edge_net:.4f}",
                    "strategy_type": proposal.strategy_type,
                    "status": "ERROR",
                    "dry_run": False,
                    "rationale": f"{proposal.rationale} | ERR: {err_msg}",
                }
            )
            return False, err_msg

    def cancel_stale_order(self, order_row: dict[str, Any]) -> bool:
        oid = order_row.get("order_id") or order_row.get("id")
        if not oid:
            return False
        if self.cfg.dry_run:
            log.info("[DRY_RUN] Would cancel stale order %s", oid)
            return True
        try:
            self.client.cancel_order(oid)
            self.state.update_trade_status_by_order_id(str(oid), "CANCELLED")
            return True
        except Exception as exc:
            log.warning("Failed to cancel stale order %s: %s", oid, exc)
            return False
