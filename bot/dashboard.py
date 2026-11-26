"""Exports portfolio, forecasts, Brier calibration, and trade journal to docs/dashboard_data.json."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

from bot.config import BotConfig
from bot.math_utils import to_decimal
from bot.polysim_client import PolySimWrapper
from bot.state import StateStore, utc_now_iso

log = logging.getLogger(__name__)


def _json_safe(val: Any) -> Any:
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, dict):
        return {str(k): _json_safe(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_json_safe(v) for v in val]
    return val


def resolve_true_starting_balance(
    client: PolySimWrapper,
    state: StateStore,
    balance: dict[str, Any],
) -> Decimal:
    """Resolve true API wallet baseline (guards against /account/balance returning 1000 on 10k wallets)."""
    try:
        wallets = client.sdk.list_wallets()
        for w in wallets:
            if str(w.get("kind") or "").upper() == "API":
                base = to_decimal(w.get("baseline"))
                if base > Decimal("0"):
                    return base
    except Exception as exc:
        log.debug("list_wallets baseline lookup failed: %s", exc)

    snaps = state.get_equity_snapshots(limit=1)
    if snaps:
        first_eq = to_decimal(snaps[0].get("equity_usd"))
        if first_eq > Decimal("0"):
            return first_eq

    return to_decimal(balance.get("starting_balance"), "10000")


def export_dashboard_data(
    *,
    config: BotConfig,
    client: PolySimWrapper,
    state: StateStore,
    balance: dict[str, Any],
    portfolio: dict[str, Any],
    positions: list[dict[str, Any]],
    market_categories: dict[str, str],
    circuit_breaker_active: bool,
) -> dict[str, Any]:
    """Assemble clean JSON payload for docs/index.html (strictly zero secrets)."""
    cash = to_decimal(balance.get("balance"))
    starting_balance = resolve_true_starting_balance(client, state, balance)
    total_value = to_decimal(balance.get("total_value"), str(cash))

    positions_val = Decimal("0")
    unrealized_pnl = Decimal("0")
    category_exposure: dict[str, Decimal] = {}
    enriched_positions: list[dict[str, Any]] = []

    for pos in positions:
        mid = str(pos.get("market_id") or pos.get("condition_id") or "")
        qty = to_decimal(pos.get("quantity") or pos.get("size"))
        avg_entry = to_decimal(pos.get("avg_entry_price") or pos.get("avg_price"))
        cur_price = to_decimal(pos.get("current_price") or avg_entry)
        val = to_decimal(pos.get("current_value") or pos.get("market_value") or (qty * cur_price))
        upnl = to_decimal(pos.get("unrealized_pnl"), str((cur_price - avg_entry) * qty))
        cat = market_categories.get(mid, str(pos.get("category") or "other")).lower()

        positions_val += val
        unrealized_pnl += upnl
        category_exposure[cat] = category_exposure.get(cat, Decimal("0")) + val

        latest_fc = state.get_latest_forecast(mid) if mid else None
        enriched_positions.append(
            {
                "market_id": mid,
                "question": str(pos.get("question") or (latest_fc or {}).get("question") or mid),
                "category": cat,
                "outcome": str(pos.get("outcome") or "Yes"),
                "quantity": float(qty),
                "avg_entry_price": float(avg_entry),
                "current_price": float(cur_price),
                "current_value": float(val),
                "unrealized_pnl": float(upnl),
                "p_blended": float(to_decimal((latest_fc or {}).get("p_blended"), "0")) if latest_fc else None,
                "edge_net": float(to_decimal((latest_fc or {}).get("edge_net"), "0")) if latest_fc else None,
                "rationale": ((latest_fc or {}).get("reasoning") or {}).get("summary_de", ""),
            }
        )

    equity = max(total_value, cash + positions_val)
    total_pnl_usd = equity - starting_balance
    total_pnl_pct = (total_pnl_usd / starting_balance * Decimal("100")) if starting_balance > 0 else Decimal("0")
    deployed_pct = (positions_val / equity * Decimal("100")) if equity > 0 else Decimal("0")

    brier_avg, brier_count = state.get_brier_stats()
    max_dd = state.get_max_drawdown_pct() * Decimal("100")
    win_rate = portfolio.get("win_rate")
    total_trades = int(portfolio.get("total_trades") or 0)

    # Merge PolySimulator equity history + local equity snapshots
    curve_by_ts: dict[str, dict[str, Any]] = {}
    try:
        api_equity = client.get_equity_history(limit=200)
        for pt in api_equity:
            ts = str(pt.get("timestamp") or pt.get("ts") or pt.get("created_at") or "")
            eq = pt.get("total_value") or pt.get("equity") or pt.get("balance")
            if ts and eq is not None:
                curve_by_ts[ts[:16]] = {
                    "ts": ts,
                    "equity": float(to_decimal(eq)),
                    "cash": float(to_decimal(pt.get("cash") or pt.get("balance"), str(eq))),
                    "source": "polysim",
                }
    except Exception as exc:
        log.debug("Could not fetch API equity history: %s", exc)

    for snap in state.get_equity_snapshots(limit=500):
        ts = str(snap["ts"])
        curve_by_ts[ts[:16]] = {
            "ts": ts,
            "equity": float(to_decimal(snap["equity_usd"])),
            "cash": float(to_decimal(snap["cash_usd"])),
            "deployed_pct": float(to_decimal(snap["deployed_pct"])),
            "source": "bot",
        }

    equity_curve = sorted(curve_by_ts.values(), key=lambda x: x["ts"])

    payload = {
        "generated_at": utc_now_iso(),
        "dry_run": config.dry_run,
        "model": config.gemini_model,
        "risk_limits": {
            "max_total_exposure_pct": float(config.max_total_exposure_pct * Decimal("100")),
            "max_single_market_pct": float(config.max_single_market_pct * Decimal("100")),
            "max_category_exposure_pct": float(config.max_category_exposure_pct * Decimal("100")),
            "kelly_fraction": float(config.kelly_fraction),
            "circuit_breaker_24h_dd_pct": float(config.circuit_breaker_24h_dd * Decimal("100")),
        },
        "kpis": {
            "starting_balance_usd": float(starting_balance),
            "equity_usd": float(equity),
            "cash_usd": float(cash),
            "positions_value_usd": float(positions_val),
            "deployed_pct": round(float(deployed_pct), 2),
            "total_pnl_usd": round(float(total_pnl_usd), 2),
            "total_pnl_pct": round(float(total_pnl_pct), 2),
            "unrealized_pnl_usd": round(float(unrealized_pnl), 2),
            "total_trades": total_trades,
            "win_rate_pct": round(float(win_rate) * 100, 1) if win_rate is not None else None,
            "max_drawdown_pct": round(float(max_dd), 2),
            "brier_score": round(float(brier_avg), 4) if brier_avg is not None else None,
            "brier_resolved_count": brier_count,
            "total_forecasts": state.count_forecasts_total(),
            "circuit_breaker_active": circuit_breaker_active,
        },
        "category_exposure": {k: round(float(v), 2) for k, v in category_exposure.items()},
        "equity_curve": equity_curve,
        "open_positions": enriched_positions,
        "recent_forecasts": state.get_recent_forecasts(limit=25),
        "trade_journal": state.get_trade_journal(limit=50),
        "recent_runs": state.get_recent_runs(limit=20),
    }

    config.dashboard_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.dashboard_json_path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload
