"""Thin, rate-limited wrapper around the official polysim_sdk.PolySimClient."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

from polysim_sdk import ApiError, PolySimClient, PolySimError

from bot.config import BotConfig
from bot.math_utils import best_bid_ask, round_to_tick, to_decimal

log = logging.getLogger(__name__)


class PolySimWrapper:
    """Wraps official PolySimClient with Decimal-safe book parsing and batch helpers."""

    def __init__(self, config: BotConfig) -> None:
        self.cfg = config
        base_url = config.polysim_base_url.removesuffix("/v1").rstrip("/")
        self.sdk = PolySimClient(
            api_key=config.polysim_api_key,
            base_url=base_url,
            floor_interval=config.min_request_spacing_sec,
            pace_from_headers=True,
            timeout=15.0,
            max_retries=3,
        )

    def close(self) -> None:
        self.sdk.close()

    def __enter__(self) -> "PolySimWrapper":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── Account & Portfolio ────────────────────────────────────────────

    def get_balance(self) -> dict[str, Any]:
        return self.sdk.balance()

    def get_portfolio(self) -> dict[str, Any]:
        return self.sdk.portfolio(wallet_id="api")

    def get_positions(self, status: str = "OPEN") -> list[dict[str, Any]]:
        return self.sdk.positions(status=status, wallet_id="api")

    def get_open_orders(self, max_orders: int = 500) -> list[dict[str, Any]]:
        orders: list[dict[str, Any]] = []
        offset = 0
        page_size = min(100, max_orders)
        while len(orders) < max_orders:
            batch = self.sdk.list_orders(
                status="PENDING",
                limit=page_size,
                offset=offset,
                wallet_id="api",
            )
            if not batch:
                break
            orders.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)
        if len(orders) >= max_orders:
            log.warning("Open orders reached pagination cap (%d); list may be truncated.", max_orders)
        return orders[:max_orders]

    def get_equity_history(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.sdk.equity(limit=limit, wallet_id="api")

    def get_trade_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.sdk.history(limit=limit, wallet_id="api")

    def get_settlements(self, limit: int = 100) -> list[dict[str, Any]]:
        resp = self.sdk.settlements(limit=limit, wallet_id="api")
        if isinstance(resp, dict):
            return list(resp.get("data") or [])
        return []

    def get_profile_analysis(self, recent_trades: int = 30, equity_days: int = 30) -> dict[str, Any]:
        return self.sdk._transport.request(
            "GET",
            "/v1/account/profile-analysis",
            params={"recent_trades": recent_trades, "equity_days": equity_days},
        )

    # ── Market Data ────────────────────────────────────────────────────

    def list_active_markets(self, limit: int = 250, sort: str | None = "volume_24h") -> list[dict[str, Any]]:
        markets: list[dict[str, Any]] = []
        offset = 0
        page_size = min(100, limit)
        while len(markets) < limit:
            batch = self.sdk.list_markets(limit=page_size, offset=offset, active_only=True, sort=sort)
            if not batch:
                break
            markets.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)
        return markets[:limit]

    def list_events(self, limit: int = 100) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        offset = 0
        page_size = min(50, limit)
        while len(events) < limit:
            batch = self.sdk.list_events(limit=page_size, offset=offset)
            if not batch:
                break
            events.extend(batch)
            if len(batch) < page_size:
                break
            offset += len(batch)
        return events[:limit]

    def get_market(self, condition_id: str) -> dict[str, Any]:
        return self.sdk.get_market(condition_id)

    def get_orderbook(self, condition_id: str, *, outcome: str | None = None, depth: int = 20) -> dict[str, Any]:
        return self.sdk.get_book(condition_id, outcome=outcome, depth=depth)

    def get_orderbook_by_token(self, token_id: str, *, depth: int = 20) -> dict[str, Any]:
        return self.sdk.get_book_by_token(token_id, depth=depth)

    def get_books_batch(self, token_ids: list[str]) -> list[dict[str, Any]]:
        """Batch fetch CLOB order books (up to 50 tokens per call)."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(token_ids), 50):
            chunk = [{"token_id": tid} for tid in token_ids[i : i + 50]]
            resp = self.sdk._transport.request("POST", "/v1/books", json_body=chunk)
            if isinstance(resp, dict) and "_list" in resp:
                results.extend(resp["_list"])
            elif isinstance(resp, list):
                results.extend(resp)
            elif isinstance(resp, dict):
                results.extend(resp.get("books") or resp.get("items") or resp.get("data") or [])
        return results

    def get_prices_batch(self, market_ids: list[str]) -> list[dict[str, Any]]:
        """Batch fetch prices (up to 50 market condition_ids per call)."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(market_ids), 50):
            chunk = market_ids[i : i + 50]
            resp = self.sdk._transport.request("POST", "/v1/prices/batch", json_body={"market_ids": chunk})
            if isinstance(resp, dict) and "_list" in resp:
                results.extend(resp["_list"])
            elif isinstance(resp, dict):
                results.extend(resp.get("prices") or resp.get("items") or resp.get("data") or [])
        return results

    def get_fee_rate_bps(self, token_id: str) -> int:
        """Query effective per-market taker fee rate in bps (defaults safely to 500 bps)."""
        try:
            resp = self.sdk._transport.request("GET", "/v1/fee-rate", params={"token_id": token_id})
            if isinstance(resp, dict) and resp.get("fee_rate_bps") is not None:
                return int(resp["fee_rate_bps"])
        except (ApiError, PolySimError, ValueError, TypeError) as exc:
            log.debug("Fee rate fallback 500 bps for token %s: %s", token_id, exc)
        return 500

    # ── Book Analytics ─────────────────────────────────────────────────

    @staticmethod
    def parse_book_metrics(book: dict[str, Any]) -> dict[str, Any] | None:
        bids = book.get("bids") or []
        asks = book.get("asks") or []
        best_bid, best_ask = best_bid_ask(bids, asks)
        if best_bid is None or best_ask is None:
            return None
        if best_bid <= Decimal("0") or best_ask <= Decimal("0") or best_ask < best_bid:
            log.warning("Rejected invalid/crossed orderbook: best_bid=%s best_ask=%s", best_bid, best_ask)
            return None
        tick_size = to_decimal(book.get("tick_size"), "0.01")
        if tick_size <= Decimal("0"):
            tick_size = Decimal("0.01")
        min_order_size = to_decimal(book.get("min_order_size"), "5")
        if min_order_size <= Decimal("0"):
            min_order_size = Decimal("5")
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / Decimal("2")
        top_bid_sz = max(
            (to_decimal(b.get("size")) for b in bids if to_decimal(b.get("price")) == best_bid),
            default=Decimal("0"),
        )
        top_ask_sz = max(
            (to_decimal(a.get("size")) for a in asks if to_decimal(a.get("price")) == best_ask),
            default=Decimal("0"),
        )
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": spread,
            "mid": mid,
            "tick_size": tick_size,
            "min_order_size": min_order_size,
            "top_bid_size": top_bid_sz,
            "top_ask_size": top_ask_sz,
        }

    # ── Order Placement & Cancellation ─────────────────────────────────

    def place_maker_limit_order(
        self,
        *,
        market_id: str,
        side: str,
        outcome: str,
        price: Decimal,
        quantity: Decimal,
        tick_size: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        aligned_price = round_to_tick(price, tick_size, side=side)
        return self.sdk.place_order(
            market_id=market_id,
            side=side.upper(),
            outcome=outcome,
            quantity=str(quantity),
            order_type="limit",
            price=str(aligned_price),
            time_in_force="GTC",
            post_only=True,
            client_order_id=client_order_id,
        )

    def place_marketable_limit_order(
        self,
        *,
        market_id: str,
        side: str,
        outcome: str,
        limit_price: Decimal,
        quantity: Decimal,
        tick_size: Decimal,
        client_order_id: str,
    ) -> dict[str, Any]:
        aligned_price = round_to_tick(limit_price, tick_size, side=side)
        return self.sdk.place_order(
            market_id=market_id,
            side=side.upper(),
            outcome=outcome,
            quantity=str(quantity),
            order_type="limit",
            price=str(aligned_price),
            time_in_force="FAK",
            post_only=False,
            client_order_id=client_order_id,
        )

    def cancel_order(self, order_id: str | int) -> dict[str, Any]:
        return self.sdk.cancel_order(order_id)
