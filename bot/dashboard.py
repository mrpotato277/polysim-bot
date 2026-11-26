"""Exports portfolio, forecasts, Brier calibration, and trade journal to docs/dashboard_data.json."""

from __future__ import annotations

import json
import logging
import math
from decimal import Decimal
from typing import Any

from bot.config import BotConfig
from bot.math_utils import first_present, to_decimal
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


def _parse_win_rate_pct(raw: Any) -> float | None:
    """Parse PolySimulator win_rate whether formatted as '0.0%' string or 0.0..1.0 ratio."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if "%" in s:
            try:
                val = float(s.replace("%", "").strip())
            except (ValueError, TypeError):
                return None
            return round(val, 1) if math.isfinite(val) else None
        raw = s
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(val):
        return None
    return round(val * 100.0, 1) if 0.0 <= val <= 1.0 else round(val, 1)


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

    first_snap = state.get_first_equity_snapshot()
    if first_snap:
        first_eq = to_decimal(first_snap.get("equity_usd"))
        if first_eq > Decimal("0"):
            return first_eq

    return to_decimal(first_present(balance, "starting_balance"), "10000")


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
    cash = to_decimal(first_present(balance, "balance"))
    starting_balance = resolve_true_starting_balance(client, state, balance)
    total_value_raw = first_present(balance, "total_value")

    positions_val = Decimal("0")
    unrealized_pnl = Decimal("0")
    category_exposure: dict[str, Decimal] = {}
    enriched_positions: list[dict[str, Any]] = []

    for pos in positions:
        mid = str(first_present(pos, "market_id", "condition_id") or "")
        qty = to_decimal(first_present(pos, "quantity", "size"))
        avg_entry = to_decimal(first_present(pos, "avg_entry_price", "avg_price"))
        cur_price_raw = first_present(pos, "current_price")
        cur_price = to_decimal(cur_price_raw) if cur_price_raw is not None else avg_entry
        val_raw = first_present(pos, "current_value", "market_value")
        val = to_decimal(val_raw) if val_raw is not None else (qty * cur_price)
        upnl_raw = first_present(pos, "unrealized_pnl")
        upnl = to_decimal(upnl_raw) if upnl_raw is not None else ((cur_price - avg_entry) * qty)
        cat = market_categories.get(mid, str(first_present(pos, "category") or "other")).lower()
        outcome_str = str(first_present(pos, "outcome") or "Yes")

        positions_val += val
        unrealized_pnl += upnl
        category_exposure[cat] = category_exposure.get(cat, Decimal("0")) + val

        latest_fc = state.get_latest_forecast(mid) if mid else None
        p_yes = to_decimal(latest_fc.get("p_blended"), "0") if latest_fc else None
        if p_yes is not None:
            p_outcome = (Decimal("1") - p_yes) if outcome_str.lower() == "no" else p_yes
        else:
            p_outcome = None
        enriched_positions.append(
            {
                "market_id": mid,
                "question": str(first_present(pos, "question") or (latest_fc or {}).get("question") or mid),
                "category": cat,
                "outcome": outcome_str,
                "quantity": float(qty),
                "avg_entry_price": float(avg_entry),
                "current_price": float(cur_price),
                "current_value": float(val),
                "unrealized_pnl": float(upnl),
                "p_yes": float(p_yes) if p_yes is not None else None,
                "p_blended": float(p_outcome) if p_outcome is not None else None,
                "edge_net": float(to_decimal((latest_fc or {}).get("edge_net"), "0")) if latest_fc else None,
                "rationale": ((latest_fc or {}).get("reasoning") or {}).get("summary_de", ""),
            }
        )

    reconciled_equity = cash + positions_val
    equity = to_decimal(total_value_raw) if total_value_raw is not None else reconciled_equity
    total_pnl_usd = equity - starting_balance
    total_pnl_pct = (total_pnl_usd / starting_balance * Decimal("100")) if starting_balance > 0 else Decimal("0")
    deployed_pct = (positions_val / equity * Decimal("100")) if equity > 0 else Decimal("0")

    brier_avg, brier_count = state.get_brier_stats()
    brier_comp = state.get_brier_comparison()
    max_dd = state.get_max_drawdown_pct() * Decimal("100")
    win_rate = portfolio.get("win_rate")
    total_trades = int(portfolio.get("total_trades") or 0)

    # Merge PolySimulator equity history + local equity snapshots
    curve_by_ts: dict[str, dict[str, Any]] = {}
    try:
        api_equity = client.get_equity_history(limit=200)
        for pt in api_equity:
            ts = str(pt.get("timestamp") or pt.get("ts") or pt.get("created_at") or "")
            eq = first_present(pt, "total_value", "equity", "balance")
            if ts and eq is not None:
                cash_pt = first_present(pt, "cash", "balance")
                curve_by_ts[ts[:16]] = {
                    "ts": ts,
                    "equity": float(to_decimal(eq)),
                    "cash": float(to_decimal(cash_pt)) if cash_pt is not None else float(to_decimal(eq)),
                    "source": "polysim",
                }
    except Exception as exc:
        log.debug("Could not fetch API equity history: %s", exc)

    for snap in state.get_recent_equity_snapshots(limit=500):
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
            "reconciled_equity_usd": float(reconciled_equity),
            "equity_discrepancy_usd": round(float(equity - reconciled_equity), 2),
            "cash_usd": float(cash),
            "positions_value_usd": float(positions_val),
            "deployed_pct": round(float(deployed_pct), 2),
            "total_pnl_usd": round(float(total_pnl_usd), 2),
            "total_pnl_pct": round(float(total_pnl_pct), 2),
            "unrealized_pnl_usd": round(float(unrealized_pnl), 2),
            "total_trades": total_trades,
            "win_rate_pct": _parse_win_rate_pct(win_rate),
            "max_drawdown_pct": round(float(max_dd), 2),
            "brier_score": round(float(brier_avg), 4) if brier_avg is not None else None,
            "brier_resolved_count": brier_count,
            "brier_blind_score": brier_comp["brier_blind"],
            "brier_market_score": brier_comp["brier_market"],
            "brier_unique_markets": brier_comp["unique_markets"],
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
