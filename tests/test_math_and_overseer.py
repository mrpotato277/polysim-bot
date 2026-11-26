"""Unit tests for quantitative math utilities and Overseer risk governance."""

from __future__ import annotations

from decimal import Decimal

from bot.config import BotConfig
from bot.math_utils import bayesian_blend, best_bid_ask, calc_taker_fee, round_to_tick
from bot.overseer import Overseer, TradeProposal


def test_order_independent_best_bid_ask() -> None:
    # PolySimulator wire order: bids ascending (best at [-1]), asks descending (best at [-1])
    bids = [{"price": "0.10", "size": "50"}, {"price": "0.42", "size": "100"}]
    asks = [{"price": "0.90", "size": "50"}, {"price": "0.45", "size": "100"}]
    bb, ba = best_bid_ask(bids, asks)
    assert bb == Decimal("0.42")
    assert ba == Decimal("0.45")

    # Reversed order (highest bid first, lowest ask first) must give identical result
    bb2, ba2 = best_bid_ask(list(reversed(bids)), list(reversed(asks)))
    assert bb2 == Decimal("0.42")
    assert ba2 == Decimal("0.45")

    # String-sort pitfall check: "0.9" > "0.10" numerically
    bb3, ba3 = best_bid_ask(
        [{"price": "0.10"}, {"price": "0.9"}],
        [{"price": "0.95"}, {"price": "0.91"}],
    )
    assert bb3 == Decimal("0.9")
    assert ba3 == Decimal("0.91")


def test_polymarket_v2_taker_vs_maker_fees() -> None:
    # At p=0.50, 100 shares, 500 bps (5%): 100 * 0.05 * 0.5 * 0.5 = $1.25
    fee = calc_taker_fee(Decimal("100"), Decimal("0.50"), 500)
    assert fee == Decimal("1.25")
    # Maker pays 0%
    assert calc_taker_fee(Decimal("100"), Decimal("0.50"), 0) == Decimal("0.00")
    # Cent quantization rounds down at 2 dp
    small_fee = calc_taker_fee(Decimal("5"), Decimal("0.90"), 500, quantize_cents=True)
    # 5 * 0.05 * 0.9 * 0.1 = 0.0225 -> $0.02
    assert small_fee == Decimal("0.02")


def test_tick_size_rounding() -> None:
    assert round_to_tick(Decimal("0.456"), Decimal("0.01"), side="BUY") == Decimal("0.45")
    assert round_to_tick(Decimal("0.451"), Decimal("0.01"), side="SELL") == Decimal("0.46")
    assert round_to_tick(Decimal("0.4567"), Decimal("0.001"), side="BUY") == Decimal("0.456")
    assert round_to_tick(Decimal("0.4561"), Decimal("0.001"), side="SELL") == Decimal("0.457")


def test_bayesian_blend_bounds_and_direction() -> None:
    # Blending 0.80 blind with 0.50 market should stay strictly between 0.50 and 0.80
    b_fresh = bayesian_blend(Decimal("0.80"), Decimal("0.50"), Decimal("0.9"), "breaking_lt_6h")
    b_stale = bayesian_blend(Decimal("0.80"), Decimal("0.50"), Decimal("0.9"), "stale_or_none")
    assert Decimal("0.50") < b_stale < b_fresh < Decimal("0.80")


def _make_proposal(
    *,
    market_id: str = "mkt-1",
    category: str = "politics",
    price: str = "0.50",
    p_true: str = "0.70",
    edge: str = "0.20",
    min_order_size: str = "5",
) -> TradeProposal:
    return TradeProposal(
        market_id=market_id,
        question="Test question?",
        category=category,
        outcome="Yes",
        side="BUY",
        order_mode="maker_limit",
        limit_price=Decimal(price),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal(min_order_size),
        p_true_outcome=Decimal(p_true),
        edge_net=Decimal(edge),
        strategy_type="news_lag",
        rationale="Unit test",
    )


def test_overseer_quarter_kelly_and_caps() -> None:
    cfg = BotConfig(polysim_api_key="x", gemini_api_key="y")
    ov = Overseer(cfg)
    prop = _make_proposal(price="0.50", p_true="0.70", edge="0.20")
    # full kelly = (0.70 - 0.50) / 0.50 = 0.40; quarter kelly = 0.10 -> 10% of $10,000 = $1,000
    dec = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("10000"),
        total_exposure_usd=Decimal("0"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("0"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert dec.approved and dec.order is not None
    assert dec.order.notional_usd <= Decimal("1000.00")
    assert dec.order.quantity >= Decimal("5")

    # Single-market cap already used -> veto
    dec_single = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("9000"),
        total_exposure_usd=Decimal("1000"),
        market_exposure_usd=Decimal("1000"),
        category_exposure_usd=Decimal("1000"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert not dec_single.approved

    # Category cap (25% = 2500) already used -> veto
    dec_cat = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("7500"),
        total_exposure_usd=Decimal("2500"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("2500"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert not dec_cat.approved

    # Total exposure cap (65% = 6500) already used -> veto
    dec_tot = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("3500"),
        total_exposure_usd=Decimal("6500"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("0"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert not dec_tot.approved

    # Circuit breaker active -> veto
    dec_cb = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("10000"),
        total_exposure_usd=Decimal("0"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("0"),
        has_open_order_on_market=False,
        circuit_breaker_active=True,
    )
    assert not dec_cb.approved

    # Carry strategy MUST be vetoed if limit_price < 0.88
    bad_carry = TradeProposal(
        market_id="mkt-carry",
        question="Cheap longshot?",
        category="politics",
        outcome="Yes",
        side="BUY",
        order_mode="maker_limit",
        limit_price=Decimal("0.10"),
        tick_size=Decimal("0.01"),
        min_order_size=Decimal("5"),
        p_true_outcome=Decimal("0.20"),
        edge_net=Decimal("0.10"),
        strategy_type="carry",
        rationale="Should fail carry min price",
    )
    dec_bad_carry = ov.evaluate_buy(
        bad_carry,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("10000"),
        total_exposure_usd=Decimal("0"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("0"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert not dec_bad_carry.approved
