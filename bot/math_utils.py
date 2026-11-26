"""Pure quantitative math utilities using Decimal arithmetic (zero network or LLM calls)."""

from __future__ import annotations

import math
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Iterable, Mapping, Any


def to_decimal(value: Any, default: str = "0") -> Decimal:
    if value is None or value == "":
        return Decimal(default)
    return Decimal(str(value))


def best_bid_ask(
    bids: Iterable[Mapping[str, Any]],
    asks: Iterable[Mapping[str, Any]],
) -> tuple[Decimal | None, Decimal | None]:
    """Return (best_bid, best_ask) order-independently using Decimal comparisons."""
    bid_prices = [to_decimal(b["price"]) for b in bids if b.get("price") is not None]
    ask_prices = [to_decimal(a["price"]) for a in asks if a.get("price") is not None]
    best_bid = max(bid_prices) if bid_prices else None
    best_ask = min(ask_prices) if ask_prices else None
    return best_bid, best_ask


def calc_taker_fee(
    shares: Decimal,
    price: Decimal,
    fee_rate_bps: int | Decimal = 500,
    *,
    quantize_cents: bool = True,
) -> Decimal:
    """Compute Polymarket V2 taker fee: shares * feeRate * p * (1 - p)."""
    if shares <= Decimal("0") or price <= Decimal("0") or price >= Decimal("1") or int(fee_rate_bps) <= 0:
        return Decimal("0.00")
    fee_rate = Decimal(str(fee_rate_bps)) / Decimal("10000")
    raw_fee = shares * fee_rate * price * (Decimal("1") - price)
    if quantize_cents:
        return raw_fee.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    return raw_fee.quantize(Decimal("0.00001"), rounding=ROUND_DOWN)


def fee_drag_per_share(price: Decimal, fee_rate_bps: int | Decimal = 500) -> Decimal:
    """Fee per share in probability/price units for net-edge checks."""
    if price <= Decimal("0") or price >= Decimal("1") or int(fee_rate_bps) <= 0:
        return Decimal("0")
    fee_rate = Decimal(str(fee_rate_bps)) / Decimal("10000")
    return fee_rate * price * (Decimal("1") - price)


def round_to_tick(price: Decimal, tick_size: Decimal, *, side: str) -> Decimal:
    """Align price to market tick_size: ROUND_DOWN for BUY, ROUND_UP for SELL."""
    if tick_size <= Decimal("0"):
        tick_size = Decimal("0.01")
    steps = price / tick_size
    rounding = ROUND_DOWN if side.upper() == "BUY" else ROUND_UP
    aligned = steps.to_integral_value(rounding=rounding) * tick_size
    min_p = tick_size
    max_p = Decimal("1") - tick_size
    if aligned < min_p:
        aligned = min_p
    if aligned > max_p:
        aligned = max_p
    return aligned.normalize() if aligned == aligned.to_integral() else aligned.quantize(tick_size)


def maker_buy_price(best_bid: Decimal, best_ask: Decimal, tick_size: Decimal) -> Decimal:
    """Choose an aggressive non-crossing maker bid strictly below best_ask (0% fee)."""
    spread = best_ask - best_bid
    raw = (best_bid + tick_size) if spread >= Decimal("2") * tick_size else best_bid
    aligned = round_to_tick(raw, tick_size, side="BUY")
    if aligned >= best_ask:
        aligned = round_to_tick(best_bid, tick_size, side="BUY")
    return aligned


def maker_sell_price(best_bid: Decimal, best_ask: Decimal, tick_size: Decimal) -> Decimal:
    """Choose an aggressive non-crossing maker ask strictly above best_bid (0% fee)."""
    spread = best_ask - best_bid
    raw = (best_ask - tick_size) if spread >= Decimal("2") * tick_size else best_ask
    aligned = round_to_tick(raw, tick_size, side="SELL")
    if aligned <= best_bid:
        aligned = round_to_tick(best_ask, tick_size, side="SELL")
    return aligned


def clamp_prob(p: float, eps: float = 0.01) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def logit(p: float) -> float:
    pc = clamp_prob(p)
    return math.log(pc / (1.0 - pc))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def bayesian_blend(
    p_blind: Decimal,
    p_mkt: Decimal,
    confidence: Decimal,
    catalyst_freshness: str,
) -> Decimal:
    """Combine blind LLM estimate and market price in log-odds space to avoid anchoring."""
    freshness_mult = {
        "breaking_lt_6h": 1.0,
        "recent_24h": 0.72,
        "stale_or_none": 0.40,
    }.get(catalyst_freshness, 0.45)

    conf = max(0.0, min(1.0, float(confidence)))
    weight_blind = max(0.15, min(0.65, 0.15 + 0.45 * conf * freshness_mult))
    blended_logit = weight_blind * logit(float(p_blind)) + (1.0 - weight_blind) * logit(float(p_mkt))
    blended = clamp_prob(sigmoid(blended_logit), eps=0.01)
    return Decimal(f"{blended:.4f}")
