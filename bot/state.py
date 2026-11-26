"""SQLite state persistence for news deduplication, forecasts, trade journal, and equity snapshots."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.strip().encode("utf-8")).hexdigest()[:32]


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS seen_news (
                url_hash TEXT PRIMARY KEY,
                market_id TEXT NOT NULL,
                title TEXT NOT NULL,
                published_at TEXT,
                first_seen_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                market_id TEXT NOT NULL,
                question TEXT NOT NULL,
                category TEXT NOT NULL,
                end_date TEXT,
                market_price TEXT NOT NULL,
                p_blind TEXT NOT NULL,
                p_blended TEXT NOT NULL,
                confidence TEXT NOT NULL,
                catalyst_freshness TEXT NOT NULL,
                edge_net TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                resolution_risk TEXT NOT NULL,
                reasoning_json TEXT NOT NULL,
                resolved_outcome TEXT,
                brier_score TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_forecasts_market_ts ON forecasts(market_id, ts);

            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                client_order_id TEXT UNIQUE NOT NULL,
                order_id TEXT,
                market_id TEXT NOT NULL,
                question TEXT NOT NULL,
                outcome TEXT NOT NULL,
                side TEXT NOT NULL,
                order_mode TEXT NOT NULL,
                price TEXT NOT NULL,
                quantity TEXT NOT NULL,
                notional_usd TEXT NOT NULL,
                est_fee_usd TEXT NOT NULL,
                edge_net TEXT NOT NULL,
                strategy_type TEXT NOT NULL,
                status TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                rationale TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trade_journal_ts ON trade_journal(ts);

            CREATE TABLE IF NOT EXISTS equity_snapshots (
                ts TEXT PRIMARY KEY,
                cash_usd TEXT NOT NULL,
                positions_value_usd TEXT NOT NULL,
                equity_usd TEXT NOT NULL,
                deployed_pct TEXT NOT NULL,
                unrealized_pnl_usd TEXT NOT NULL,
                realized_pnl_usd TEXT NOT NULL,
                open_positions_count INTEGER NOT NULL,
                circuit_breaker_active INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS run_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                dry_run INTEGER NOT NULL,
                markets_scanned INTEGER NOT NULL,
                candidates_found INTEGER NOT NULL,
                llm_calls INTEGER NOT NULL,
                trades_attempted INTEGER NOT NULL,
                trades_executed INTEGER NOT NULL,
                errors_json TEXT NOT NULL,
                duration_sec REAL NOT NULL
            );
            """
        )
        self.conn.commit()

    # ── News deduplication ─────────────────────────────────────────────

    def filter_and_record_news(
        self, market_id: str, articles: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        """Return all articles annotated with is_new, recording unseen ones. Returns (articles, new_count)."""
        cur = self.conn.cursor()
        now_iso = utc_now_iso()
        new_count = 0
        annotated: list[dict[str, Any]] = []
        for art in articles:
            key_basis = art.get("link") or art.get("title") or ""
            h = hash_text(f"{market_id}::{key_basis}")
            row = cur.execute("SELECT 1 FROM seen_news WHERE url_hash = ?", (h,)).fetchone()
            is_new = row is None
            if is_new:
                new_count += 1
                cur.execute(
                    "INSERT OR IGNORE INTO seen_news (url_hash, market_id, title, published_at, first_seen_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (h, market_id, str(art.get("title", ""))[:300], art.get("published_at"), now_iso),
                )
            item = dict(art)
            item["is_new"] = is_new
            annotated.append(item)
        self.conn.commit()
        return annotated, new_count

    # ── Forecasts & Brier calibration ──────────────────────────────────

    def record_forecast(self, data: dict[str, Any]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO forecasts (
                ts, market_id, question, category, end_date,
                market_price, p_blind, p_blended, confidence,
                catalyst_freshness, edge_net, strategy_type,
                resolution_risk, reasoning_json, resolved_outcome, brier_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                data.get("ts") or utc_now_iso(),
                data["market_id"],
                data["question"],
                data["category"],
                data.get("end_date"),
                str(data["market_price"]),
                str(data["p_blind"]),
                str(data["p_blended"]),
                str(data["confidence"]),
                data["catalyst_freshness"],
                str(data["edge_net"]),
                data["strategy_type"],
                data["resolution_risk"],
                json.dumps(data["reasoning"], ensure_ascii=False),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_latest_forecast(self, market_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM forecasts WHERE market_id = ? ORDER BY id DESC LIMIT 1",
            (market_id,),
        ).fetchone()
        return self._forecast_row_to_dict(row) if row else None

    def get_recent_forecast_time(self, market_id: str) -> datetime | None:
        row = self.conn.execute(
            "SELECT ts FROM forecasts WHERE market_id = ? ORDER BY id DESC LIMIT 1",
            (market_id,),
        ).fetchone()
        if not row:
            return None
        try:
            return datetime.fromisoformat(row["ts"])
        except ValueError:
            return None

    def list_unresolved_forecast_market_ids(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT market_id FROM forecasts WHERE brier_score IS NULL"
        ).fetchall()
        return [str(r["market_id"]) for r in rows]

    def update_resolved_market_brier(self, market_id: str, resolved_yes: bool) -> int:
        outcome_label = "Yes" if resolved_yes else "No"
        y = Decimal("1") if resolved_yes else Decimal("0")
        rows = self.conn.execute(
            "SELECT id, p_blended FROM forecasts WHERE market_id = ? AND brier_score IS NULL",
            (market_id,),
        ).fetchall()
        updated = 0
        for r in rows:
            p = Decimal(str(r["p_blended"]))
            brier = (p - y) ** 2
            self.conn.execute(
                "UPDATE forecasts SET resolved_outcome = ?, brier_score = ? WHERE id = ?",
                (outcome_label, f"{brier:.6f}", r["id"]),
            )
            updated += 1
        if updated:
            self.conn.commit()
        return updated

    def get_brier_stats(self) -> tuple[Decimal | None, int]:
        rows = self.conn.execute(
            "SELECT brier_score FROM forecasts WHERE brier_score IS NOT NULL"
        ).fetchall()
        if not rows:
            return None, 0
        scores = [Decimal(str(r["brier_score"])) for r in rows]
        avg = sum(scores) / Decimal(len(scores))
        return avg, len(scores)

    def get_recent_forecasts(self, limit: int = 25) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM forecasts ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._forecast_row_to_dict(r) for r in rows]

    @staticmethod
    def _forecast_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        try:
            d["reasoning"] = json.loads(d.pop("reasoning_json"))
        except Exception:
            d["reasoning"] = {}
        return d

    # ── Trade journal ──────────────────────────────────────────────────

    def record_trade(self, trade: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO trade_journal (
                ts, client_order_id, order_id, market_id, question, outcome,
                side, order_mode, price, quantity, notional_usd, est_fee_usd,
                edge_net, strategy_type, status, dry_run, rationale
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade.get("ts") or utc_now_iso(),
                trade["client_order_id"],
                str(trade.get("order_id") or ""),
                trade["market_id"],
                trade["question"],
                trade["outcome"],
                trade["side"],
                trade["order_mode"],
                str(trade["price"]),
                str(trade["quantity"]),
                str(trade["notional_usd"]),
                str(trade["est_fee_usd"]),
                str(trade["edge_net"]),
                trade["strategy_type"],
                trade["status"],
                1 if trade.get("dry_run") else 0,
                trade["rationale"],
            ),
        )
        self.conn.commit()

    def get_trade_journal(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM trade_journal ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Equity & drawdown ──────────────────────────────────────────────

    def record_equity_snapshot(self, snap: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO equity_snapshots (
                ts, cash_usd, positions_value_usd, equity_usd, deployed_pct,
                unrealized_pnl_usd, realized_pnl_usd, open_positions_count,
                circuit_breaker_active
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snap.get("ts") or utc_now_iso(),
                str(snap["cash_usd"]),
                str(snap["positions_value_usd"]),
                str(snap["equity_usd"]),
                str(snap["deployed_pct"]),
                str(snap["unrealized_pnl_usd"]),
                str(snap["realized_pnl_usd"]),
                int(snap["open_positions_count"]),
                1 if snap.get("circuit_breaker_active") else 0,
            ),
        )
        self.conn.commit()

    def get_24h_drawdown(self, current_equity: Decimal) -> Decimal:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        rows = self.conn.execute(
            "SELECT equity_usd FROM equity_snapshots WHERE ts >= ?", (cutoff,)
        ).fetchall()
        if not rows:
            return Decimal("0")
        peak = max([Decimal(str(r["equity_usd"])) for r in rows] + [current_equity])
        if peak <= Decimal("0"):
            return Decimal("0")
        return (current_equity - peak) / peak

    def get_max_drawdown_pct(self) -> Decimal:
        rows = self.conn.execute(
            "SELECT equity_usd FROM equity_snapshots ORDER BY ts ASC"
        ).fetchall()
        if not rows:
            return Decimal("0")
        peak = Decimal("0")
        max_dd = Decimal("0")
        for r in rows:
            eq = Decimal(str(r["equity_usd"]))
            if eq > peak:
                peak = eq
            if peak > Decimal("0"):
                dd = (eq - peak) / peak
                if dd < max_dd:
                    max_dd = dd
        return max_dd

    def get_equity_snapshots(self, limit: int = 500) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM equity_snapshots ORDER BY ts ASC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Run logs ───────────────────────────────────────────────────────

    def record_run(self, run: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO run_logs (
                ts, dry_run, markets_scanned, candidates_found, llm_calls,
                trades_attempted, trades_executed, errors_json, duration_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.get("ts") or utc_now_iso(),
                1 if run.get("dry_run") else 0,
                int(run["markets_scanned"]),
                int(run["candidates_found"]),
                int(run["llm_calls"]),
                int(run["trades_attempted"]),
                int(run["trades_executed"]),
                json.dumps(run.get("errors", []), ensure_ascii=False),
                float(run["duration_sec"]),
            ),
        )
        self.conn.commit()

    def get_recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM run_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            try:
                d["errors"] = json.loads(d.pop("errors_json"))
            except Exception:
                d["errors"] = []
            out.append(d)
        return out

    def count_forecasts_total(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS c FROM forecasts").fetchone()
        return int(row["c"]) if row else 0
