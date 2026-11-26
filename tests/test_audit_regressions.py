"""Deterministic offline regression tests for audit fixes (zero network calls)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from bot.config import BotConfig, _parse_bool
from bot.dashboard import export_dashboard_data
from bot.forecaster import Superforecaster
from bot.news import _parse_feed_entries, build_news_query
from bot.orchestrator import _update_brier_scores
from bot.overseer import Overseer
from bot.polysim_client import PolySimWrapper
from bot.scanner import MarketScanner, normalize_category
from bot.state import StateStore


def test_parse_bool_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _parse_bool(None, True) is True
    assert _parse_bool(None, False) is False
    assert _parse_bool("", True) is True
    assert _parse_bool("   ", False) is False
    assert _parse_bool("true", False) is True
    assert _parse_bool("false", True) is False
    assert _parse_bool(" TRUE ", False) is True

    with pytest.raises(ValueError):
        _parse_bool("treu", True)

    monkeypatch.setenv("DRY_RUN", "treu")
    monkeypatch.setenv("POLYSIM_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    with pytest.raises(ValueError):
        BotConfig.from_env()


def test_overseer_remaining_quantity_and_zero_price() -> None:
    cfg = BotConfig(polysim_api_key="x", gemini_api_key="y")
    ov = Overseer(cfg)

    # Case A: remaining_quantity present alongside size_matched -> must NOT subtract twice
    exp_a, _, _ = ov.compute_portfolio_exposure(
        positions=[],
        open_orders=[
            {
                "side": "BUY",
                "market_id": "m1",
                "quantity": "100",
                "remaining_quantity": "60",
                "size_matched": "40",
                "price": "0.50",
            }
        ],
        market_categories={"m1": "politics"},
    )
    assert exp_a == Decimal("30.00")

    # Case B: only original quantity + size_matched
    exp_b, _, _ = ov.compute_portfolio_exposure(
        positions=[],
        open_orders=[
            {
                "side": "BUY",
                "market_id": "m1",
                "quantity": "100",
                "size_matched": "40",
                "price": "0.50",
            }
        ],
        market_categories={"m1": "politics"},
    )
    assert exp_b == Decimal("30.00")

    # Case C: explicit remaining_quantity=0 stays 0
    exp_c, _, _ = ov.compute_portfolio_exposure(
        positions=[],
        open_orders=[
            {
                "side": "BUY",
                "market_id": "m1",
                "quantity": "100",
                "remaining_quantity": 0,
                "size_matched": 100,
                "price": "0.50",
            }
        ],
        market_categories={"m1": "politics"},
    )
    assert exp_c == Decimal("0")

    # Case D: position with numeric 0 current_price and current_value stays 0 (never falls back to avg_entry_price)
    exp_zero_pos, _, _ = ov.compute_portfolio_exposure(
        positions=[
            {
                "market_id": "m1",
                "quantity": 100,
                "avg_entry_price": 0.50,
                "current_price": 0,
                "current_value": 0,
            }
        ],
        open_orders=[],
        market_categories={"m1": "politics"},
    )
    assert exp_zero_pos == Decimal("0")

    # Circuit breaker veto text reflects actual configured threshold (-5%)
    from tests.test_math_and_overseer import _make_proposal

    dec = ov.evaluate_buy(
        _make_proposal(),
        equity_usd=Decimal("10000"),
        cash_usd=Decimal("10000"),
        total_exposure_usd=Decimal("0"),
        market_exposure_usd=Decimal("0"),
        category_exposure_usd=Decimal("0"),
        has_open_order_on_market=False,
        circuit_breaker_active=True,
    )
    assert not dec.approved
    assert "-5%" in dec.reason


def test_dashboard_zero_value_and_no_position_fair_value(tmp_path: Any) -> None:
    db_path = tmp_path / "dash.db"
    dash_path = tmp_path / "dashboard_data.json"
    cfg = BotConfig(
        polysim_api_key="x",
        gemini_api_key="y",
        db_path=db_path,
        dashboard_json_path=dash_path,
    )
    fake_client = SimpleNamespace(
        sdk=SimpleNamespace(list_wallets=lambda: [{"kind": "API", "baseline": "50"}]),
        get_equity_history=lambda limit=200: [],
    )
    with StateStore(db_path) as state:
        state.record_forecast(
            {
                "market_id": "mkt-no",
                "question": "Will X happen?",
                "category": "politics",
                "end_date": "2026-12-10T00:00:00Z",
                "market_price": "0.5000",
                "p_blind": "0.2000",
                "p_blended": "0.2000",
                "confidence": "0.8000",
                "catalyst_freshness": "recent_24h",
                "edge_net": "0.3000",
                "strategy_type": "news_lag",
                "resolution_risk": "low",
                "reasoning": {"summary_de": "Test"},
            }
        )
        payload = export_dashboard_data(
            config=cfg,
            client=fake_client,  # type: ignore[arg-type]
            state=state,
            balance={"balance": 50, "total_value": 50},
            portfolio={"total_trades": 1},
            positions=[
                {
                    "market_id": "mkt-no",
                    "outcome": "No",
                    "quantity": 100,
                    "avg_entry_price": 0.50,
                    "current_price": 0,
                    "current_value": 0,
                }
            ],
            market_categories={"mkt-no": "politics"},
            circuit_breaker_active=False,
        )

    assert payload["kpis"]["equity_usd"] == 50.0
    assert payload["kpis"]["positions_value_usd"] == 0.0
    pos = payload["open_positions"][0]
    assert pos["p_yes"] == 0.20
    assert pos["p_blended"] == 0.80


def test_equity_snapshots_beyond_500_points(tmp_path: Any) -> None:
    base_dt = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    with StateStore(tmp_path / "eq.db") as state:
        for i in range(1, 502):
            ts = (base_dt + timedelta(minutes=5 * i)).isoformat()
            state.record_equity_snapshot(
                {
                    "ts": ts,
                    "cash_usd": str(10000 + i),
                    "positions_value_usd": "0",
                    "equity_usd": str(10000 + i),
                    "deployed_pct": "0",
                    "unrealized_pnl_usd": "0",
                    "realized_pnl_usd": str(i),
                    "open_positions_count": 0,
                    "circuit_breaker_active": False,
                }
            )
        first = state.get_first_equity_snapshot()
        latest = state.get_latest_equity_snapshot()
        recent = state.get_recent_equity_snapshots(limit=500)

    assert first is not None and first["equity_usd"] == "10001"
    assert latest is not None and latest["equity_usd"] == "10501"
    assert len(recent) == 500
    assert recent[0]["equity_usd"] == "10002"
    assert recent[-1]["equity_usd"] == "10501"


def test_unresolved_market_rotation_and_brier_comparison(tmp_path: Any) -> None:
    with StateStore(tmp_path / "brier.db") as state:
        # 12 unresolved markets; m01 gets 3 duplicate forecasts, m02..m12 get 1 each
        for _ in range(3):
            state.record_forecast(
                {
                    "market_id": "m01",
                    "question": "Market 1?",
                    "category": "politics",
                    "market_price": "0.50",
                    "p_blind": "0.20",
                    "p_blended": "0.20",
                    "confidence": "0.70",
                    "catalyst_freshness": "recent_24h",
                    "edge_net": "0.30",
                    "strategy_type": "news_lag",
                    "resolution_risk": "low",
                    "reasoning": {},
                }
            )
        for idx in range(2, 13):
            state.record_forecast(
                {
                    "market_id": f"m{idx:02d}",
                    "question": f"Market {idx}?",
                    "category": "politics",
                    "market_price": "0.40",
                    "p_blind": "0.80",
                    "p_blended": "0.80",
                    "confidence": "0.70",
                    "catalyst_freshness": "recent_24h",
                    "edge_net": "0.40",
                    "strategy_type": "news_lag",
                    "resolution_risk": "low",
                    "reasoning": {},
                }
            )

        # Round-robin limit=10 reaches m11 & m12 on second batch
        batch1 = state.list_unresolved_forecast_market_ids(limit=10)
        assert len(batch1) == 10
        assert "m12" not in batch1
        for mid in batch1:
            state.mark_forecast_market_checked(mid)
        batch2 = state.list_unresolved_forecast_market_ids(limit=10)
        assert "m12" in batch2

        fake_client = SimpleNamespace(
            get_settlements=lambda limit=100: [],
            get_market=lambda mid: {"resolved_outcome": "Yes"} if mid == "m12" else {},
        )
        updated = _update_brier_scores(fake_client, state)  # type: ignore[arg-type]
        assert updated == 1

        # Resolve m01 as No (3 rows) and verify event-weighted comparison ignores duplicate row weight
        state.update_resolved_market_brier("m01", False)
        comp = state.get_brier_comparison()
        assert comp["unique_markets"] == 2
        assert comp["resolved_rows"] == 4
        assert comp["brier_blended"] == 0.04
        assert comp["brier_blind"] == 0.04
        assert comp["brier_market"] == 0.305


def test_journal_preserves_order_id_on_timeout_and_cancel_status(tmp_path: Any) -> None:
    with StateStore(tmp_path / "journal.db") as state:
        state.record_trade(
            {
                "client_order_id": "psb-retry-1",
                "order_id": "ex-1",
                "market_id": "m1",
                "question": "Q?",
                "outcome": "Yes",
                "side": "BUY",
                "order_mode": "maker_limit",
                "price": "0.50",
                "quantity": "10",
                "notional_usd": "5.00",
                "est_fee_usd": "0.00",
                "edge_net": "0.10",
                "strategy_type": "news_lag",
                "status": "PENDING",
                "dry_run": False,
                "rationale": "Initial submit",
            }
        )
        # Simulated timeout retry with empty order_id and ERROR status must NOT erase ex-1 or PENDING
        state.record_trade(
            {
                "client_order_id": "psb-retry-1",
                "order_id": "",
                "market_id": "m1",
                "question": "Q?",
                "outcome": "Yes",
                "side": "BUY",
                "order_mode": "maker_limit",
                "price": "0.50",
                "quantity": "10",
                "notional_usd": "5.00",
                "est_fee_usd": "0.00",
                "edge_net": "0.10",
                "strategy_type": "news_lag",
                "status": "ERROR",
                "dry_run": False,
                "rationale": "ReadTimeout",
            }
        )
        rows = state.get_trade_journal(limit=10)
        assert rows[0]["order_id"] == "ex-1"
        assert rows[0]["status"] == "PENDING"
        assert "RETRY_ERR" in rows[0]["rationale"]

        changed = state.update_trade_status_by_order_id("ex-1", "CANCELLED")
        assert changed == 1
        assert state.get_trade_journal(limit=10)[0]["status"] == "CANCELLED"


def test_forecaster_rejects_empty_json_and_list(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = BotConfig(polysim_api_key="x", gemini_api_key="y")
    fc = Superforecaster(cfg)

    for bad_text in ("{}", "[]"):
        monkeypatch.setattr(
            fc.client.models,
            "generate_content",
            lambda **_kw: SimpleNamespace(text=bad_text),
        )
        with pytest.raises(RuntimeError):
            fc.forecast(
                question="Will X happen?",
                description="Official rules",
                end_date_iso="2026-12-10T00:00:00Z",
                market_price_yes=Decimal("0.50"),
                news_articles=[],
            )

    calls = {"n": 0}
    good_json = (
        '{"resolution_trap_risk":"low","resolution_note":"Clear",'
        '"base_rate":0.5,"bull_case":"A","bear_case":"B",'
        '"catalyst_freshness":"recent_24h","p_blind":0.65,'
        '"confidence":0.7,"summary_de":"Valid."}'
    )

    def _flaky_generate(**_kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] < 3:
            return SimpleNamespace(text="{}")
        return SimpleNamespace(text=good_json)

    monkeypatch.setattr(fc.client.models, "generate_content", _flaky_generate)
    res = fc.forecast(
        question="Will X happen?",
        description="Official rules",
        end_date_iso="2026-12-10T00:00:00Z",
        market_price_yes=Decimal("0.50"),
        news_articles=[],
    )
    assert res.api_attempts == 3
    assert res.p_blind == Decimal("0.6500")


def test_news_unicode_unknown_date_and_category_word_boundary() -> None:
    query = build_news_query("Will Pedro Sánchez open the library?")
    assert "Sánchez" in query

    future_rfc = (datetime.now(timezone.utc) + timedelta(hours=2)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    parsed = _parse_feed_entries(
        SimpleNamespace(
            entries=[
                {"title": "Undated report", "summary": "No timestamp"},
                {"title": "Future stamped report", "published": future_rfc, "summary": "Clock skew"},
            ]
        )
    )
    by_title = {a["title"]: a for a in parsed}
    assert by_title["Undated report"]["age_hours"] is None
    assert by_title["Future stamped report"]["age_hours"] == 0.0
    assert parsed[0]["title"] == "Future stamped report"

    assert normalize_category("politics", "Whether the city builds a bridge") == "politics"
    assert normalize_category("", "Whether the city builds a bridge") != "crypto"


def test_monotonicity_keeps_quantities_and_enforces_horizon() -> None:
    now = datetime.now(timezone.utc)
    fake_client = SimpleNamespace(
        get_orderbook=lambda *_args, **_kw: pytest.fail("Should not fetch orderbook for invalid pairs")
    )
    scanner = MarketScanner(fake_client, BotConfig(polysim_api_key="x", gemini_api_key="y"))  # type: ignore[arg-type]

    merged = {
        "m_early": {
            "condition_id": "m_early",
            "event_group_id": "ev1",
            "question": "Will 2 parks open before November 30?",
            "end_date": (now + timedelta(days=5)).isoformat(),
            "synced_price": "0.75",
            "clob_token_ids": ["t1", "t2"],
            "active": True,
        },
        "m_late_diff_qty": {
            "condition_id": "m_late_diff_qty",
            "event_group_id": "ev1",
            "question": "Will 20 parks open before January 15?",
            "end_date": (now + timedelta(days=15)).isoformat(),
            "synced_price": "0.35",
            "clob_token_ids": ["t3", "t4"],
            "active": True,
        },
        "m_same_early": {
            "condition_id": "m_same_early",
            "event_group_id": "ev2",
            "question": "Will the bridge open before November 30?",
            "end_date": (now + timedelta(days=5)).isoformat(),
            "synced_price": "0.75",
            "clob_token_ids": ["t5", "t6"],
            "active": True,
        },
        "m_same_too_far": {
            "condition_id": "m_same_too_far",
            "event_group_id": "ev2",
            "question": "Will the bridge open before January 15?",
            "end_date": (now + timedelta(days=45)).isoformat(),
            "synced_price": "0.35",
            "clob_token_ids": ["t7", "t8"],
            "active": True,
        },
    }
    hits = scanner._scan_structural_anomalies(
        merged,
        {"ev1": {"market_count": 2}, "ev2": {"market_count": 2}},
        {"m_early": "politics", "m_late_diff_qty": "politics", "m_same_early": "politics", "m_same_too_far": "politics"},
    )
    assert hits == []


def test_crossed_orderbook_rejected() -> None:
    met = PolySimWrapper.parse_book_metrics(
        {
            "bids": [{"price": "0.70", "size": "10"}],
            "asks": [{"price": "0.50", "size": "10"}],
        }
    )
    assert met is None
