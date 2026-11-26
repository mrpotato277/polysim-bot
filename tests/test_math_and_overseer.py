"""Unit tests for quantitative math utilities and Overseer risk governance."""

from __future__ import annotations

from decimal import Decimal

import pytest

import bot.news as news_mod
from bot.config import BotConfig
from bot.math_utils import bayesian_blend, best_bid_ask, calc_taker_fee, round_to_tick
from bot.overseer import Overseer, TradeProposal, choose_entry_execution
from bot.scanner import is_cumulative_deadline_question


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
    assert cfg.max_new_buys_per_run == 1
    assert cfg.kelly_fraction == Decimal("0.15")
    assert cfg.circuit_breaker_24h_dd == Decimal("-0.05")
    ov = Overseer(cfg)
    prop = _make_proposal(price="0.50", p_true="0.70", edge="0.20")
    # full kelly = (0.70 - 0.50) / 0.50 = 0.40; 0.15 * kelly = 0.06 -> capped by 4% single market ($400)
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
    assert dec.order.notional_usd <= Decimal("400.00")
    assert dec.order.quantity >= Decimal("5")

    # Single-market cap (4% = 400) already used -> veto
    dec_single = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("9600"),
        total_exposure_usd=Decimal("400"),
        market_exposure_usd=Decimal("400"),
        category_exposure_usd=Decimal("400"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert not dec_single.approved

    # Category cap (12% = 1200) already used -> veto
    dec_cat = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("8800"),
        total_exposure_usd=Decimal("1200"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("1200"),
        has_open_order_on_market=False,
        circuit_breaker_active=False,
    )
    assert not dec_cat.approved

    # Total exposure cap (35% = 3500) already used -> veto
    dec_tot = ov.evaluate_buy(
        prop,
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("6500"),
        total_exposure_usd=Decimal("3500"),
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


def test_is_cumulative_deadline_question() -> None:
    assert is_cumulative_deadline_question("Will the US invade Iran before December 31?") is True
    assert is_cumulative_deadline_question("Will a deal be signed by November 30?") is True
    assert is_cumulative_deadline_question("Israel x Iran ceasefire continues through December 31?") is False
    assert is_cumulative_deadline_question("Will Bitcoin stay above $80,000 through December?") is False


def test_specialist_news_matching_and_dedup(monkeypatch: pytest.MonkeyPatch) -> None:
    crypto_urls = [url for _, url in news_mod.SPECIALIST_FEEDS["crypto"]]
    fake_cache = {url: [] for url in crypto_urls}
    fake_cache[crypto_urls[0]] = [
        {
            "title": "Bitcoin rallies toward $55,000 after Fed liquidity note",
            "source": "CoinDesk",
            "published_at": "2026-11-26T14:00:00+00:00",
            "age_hours": 1.0,
            "link": "https://example.com/btc",
            "snippet": "Bitcoin approaches 55000 level.",
        },
        {
            "title": "European sailing regatta concludes in calm winds",
            "source": "CoinDesk",
            "published_at": "2026-11-26T12:00:00+00:00",
            "age_hours": 3.0,
            "link": "https://example.com/unrelated",
            "snippet": "Unrelated sailing article.",
        },
    ]
    monkeypatch.setattr(news_mod, "_SPECIALIST_FEED_CACHE", fake_cache)

    spec = news_mod.fetch_specialist_news("Bitcoin 55000", "crypto")
    assert len(spec) == 1
    assert "Bitcoin" in spec[0]["title"]

    monkeypatch.setattr(
        news_mod,
        "fetch_google_news",
        lambda *_args, **_kwargs: [
            {
                "title": "Bitcoin rallies toward $55,000 after Fed liquidity note!",
                "source": "Reuters",
                "published_at": "2026-11-26T13:00:00+00:00",
                "age_hours": 2.0,
                "link": "https://example.com/dup",
                "snippet": "Duplicate story.",
            },
            {
                "title": "Spot Bitcoin ETF volume hits weekly high",
                "source": "Bloomberg",
                "published_at": "2026-11-26T14:30:00+00:00",
                "age_hours": 0.5,
                "link": "https://example.com/etf",
                "snippet": "Fresh ETF flow.",
            },
        ],
    )

    merged = news_mod.fetch_market_news("Bitcoin 55000", category="crypto", max_items=10)
    assert len(merged) == 2
    assert [a["age_hours"] for a in merged] == [0.5, 1.0]
    assert merged[0]["source"] == "Bloomberg"
    assert merged[1]["source"] == "CoinDesk"


def test_hybrid_breaking_taker_selection() -> None:
    # 1. Breaking news + strong edge after 500 bps taker fee (>= 0.12) -> marketable_limit
    mode, price, edge, fee = choose_entry_execution(
        strategy_type="news_lag",
        catalyst_freshness="breaking_lt_6h",
        p_true_outcome=Decimal("0.75"),
        maker_price=Decimal("0.54"),
        best_ask=Decimal("0.55"),
        fee_rate_bps=500,
        breaking_taker_min_edge_net=Decimal("0.12"),
    )
    assert mode == "marketable_limit"
    assert price == Decimal("0.55")
    assert fee == 500
    assert edge >= Decimal("0.12")

    # 2. Breaking news, but net edge after taker fee falls below 0.12 -> stay maker_limit
    mode2, price2, edge2, fee2 = choose_entry_execution(
        strategy_type="news_lag",
        catalyst_freshness="breaking_lt_6h",
        p_true_outcome=Decimal("0.65"),
        maker_price=Decimal("0.54"),
        best_ask=Decimal("0.55"),
        fee_rate_bps=500,
        breaking_taker_min_edge_net=Decimal("0.12"),
    )
    assert mode2 == "maker_limit"
    assert price2 == Decimal("0.54")
    assert fee2 == 0
    assert edge2 == Decimal("0.11")

    # 3. Stale news or carry strategy -> always stay maker_limit regardless of edge
    for strat, fresh in [("news_lag", "stale_or_none"), ("carry", "breaking_lt_6h")]:
        mode3, price3, _edge3, fee3 = choose_entry_execution(
            strategy_type=strat,
            catalyst_freshness=fresh,
            p_true_outcome=Decimal("0.98"),
            maker_price=Decimal("0.54"),
            best_ask=Decimal("0.55"),
            fee_rate_bps=500,
            breaking_taker_min_edge_net=Decimal("0.12"),
        )
        assert mode3 == "maker_limit"
        assert price3 == Decimal("0.54")
        assert fee3 == 0
