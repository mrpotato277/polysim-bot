"""Market scanner: filters multi-day liquid markets, batch-reads CLOB books, and detects Alpha A/B/C."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from bot.config import BotConfig
from bot.math_utils import to_decimal
from bot.polysim_client import PolySimWrapper

log = logging.getLogger(__name__)


@dataclass
class MarketCandidate:
    condition_id: str
    question: str
    slug: str
    category: str
    raw_category: str
    end_date: datetime | None
    days_to_resolution: float
    event_group_id: str
    event_title: str
    outcomes: list[str]
    yes_token_id: str
    no_token_id: str
    volume_24h: Decimal
    volume_total: Decimal
    best_bid: Decimal
    best_ask: Decimal
    spread: Decimal
    mid: Decimal
    tick_size: Decimal
    min_order_size: Decimal
    top_bid_size: Decimal
    top_ask_size: Decimal
    strategy_type: str  # "structural_arb" | "news_lag" | "carry"
    suggested_outcome: str  # "Yes" | "No"
    annualized_carry_yield: Decimal = Decimal("0")
    structural_note: str = ""
    score: float = 0.0


def normalize_category(raw_cat: str | None, question: str) -> str:
    c = (raw_cat or "").strip().lower()
    q = question.lower()
    if c in {"crypto", "cryptocurrency", "defi", "bitcoin"} or any(
        w in q for w in ("bitcoin", "btc", "ethereum", "eth", "solana", "crypto")
    ):
        return "crypto"
    if c in {"politics", "elections", "us-politics"} or any(
        w in q for w in ("election", "president", "senate", "congress", "trump", "vote", "minister")
    ):
        return "politics"
    if c in {"world", "geopolitics", "ukraine", "middle-east"} or any(
        w in q for w in ("invade", "war", "ceasefire", "nato", "treaty", "sanction", "israel", "iran", "russia", "china")
    ):
        return "geopolitics"
    if c in {"finance", "economics", "business", "macro"} or any(
        w in q for w in ("fed", "fomc", "cpi", "inflation", "gdp", "rate cut", "tariff", "stock", "s&p")
    ):
        return "economics"
    if c in {"sports", "nba", "nfl", "soccer", "ufc", "tennis"}:
        return "sports"
    if c in {"tech", "ai", "science"} or any(w in q for w in ("openai", "gpt", "spacex", "launch", "fda", "ai ")):
        return "tech"
    if c in {"culture", "pop-culture", "entertainment", "mentions"}:
        return "culture"
    return c or "other"


def _parse_iso_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _extract_token_ids(m: dict[str, Any]) -> tuple[str, str]:
    raw = m.get("clob_token_ids") or m.get("clobTokenIds") or m.get("token_ids") or []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            raw = []
    if isinstance(raw, list) and len(raw) >= 2:
        return str(raw[0]), str(raw[1])
    if isinstance(raw, list) and len(raw) == 1:
        return str(raw[0]), ""
    live = m.get("live_price") or {}
    outs = live.get("outcomes") if isinstance(live, dict) else None
    if isinstance(outs, list) and len(outs) >= 2:
        return str(outs[0].get("token_id") or ""), str(outs[1].get("token_id") or "")
    tokens = m.get("tokens")
    if isinstance(tokens, list) and len(tokens) >= 2:
        return str(tokens[0].get("token_id") or ""), str(tokens[1].get("token_id") or "")
    return "", ""


def _extract_outcomes(m: dict[str, Any]) -> list[str]:
    outs = m.get("outcomes")
    if isinstance(outs, str):
        try:
            outs = json.loads(outs)
        except Exception:
            outs = None
    if isinstance(outs, list) and len(outs) >= 2:
        return [str(o) for o in outs]
    return ["Yes", "No"]


class MarketScanner:
    """Scans PolySimulator for multi-day liquid opportunities across Alpha A, B, and C."""

    def __init__(self, client: PolySimWrapper, config: BotConfig) -> None:
        self.client = client
        self.cfg = config

    def scan(self) -> tuple[list[MarketCandidate], dict[str, str], int]:
        """Return (ranked_candidates, market_categories_map, total_markets_scanned)."""
        now = datetime.now(timezone.utc)
        merged: dict[str, dict[str, Any]] = {}
        event_meta: dict[str, dict[str, Any]] = {}

        # 1. Fetch events first (includes end_date, event_group_id, and sibling structure)
        events = self.client.list_events(limit=80)
        for ev in events:
            eg_id = str(ev.get("event_group_id") or "")
            if eg_id:
                event_meta[eg_id] = ev
            for m in ev.get("markets") or []:
                cid = str(m.get("condition_id") or "")
                if not cid:
                    continue
                row = dict(m)
                row.setdefault("category", ev.get("category"))
                row.setdefault("event_title", ev.get("event_title"))
                row.setdefault("event_group_id", eg_id)
                merged[cid] = row

        # 2. Fetch active markets (includes live_price, volume_24h, clob_token_ids)
        active_markets = self.client.list_active_markets(limit=self.cfg.max_markets_scan)
        for m in active_markets:
            cid = str(m.get("condition_id") or "")
            if not cid:
                continue
            if cid in merged:
                merged[cid].update({k: v for k, v in m.items() if v is not None})
            else:
                merged[cid] = dict(m)

        market_categories: dict[str, str] = {}
        for cid, m in merged.items():
            market_categories[cid] = normalize_category(m.get("category"), str(m.get("question") or ""))

        # 3. Enrich top active markets missing end_date or clob_token_ids sparingly
        # Pre-filter by tradeable price range and skip single-game same-day sports spreads/O/U
        need_detail: list[str] = []
        for cid, m in merged.items():
            if not m.get("active", True) or m.get("closed") or m.get("resolved_outcome"):
                continue
            q_low = str(m.get("question") or "").lower()
            if " vs. " in q_low or " vs " in q_low or ": o/u " in q_low or q_low.startswith("spread:"):
                continue
            p_approx = to_decimal((m.get("live_price") or {}).get("buy") or m.get("synced_price") or m.get("last_price"), "0")
            if not (Decimal("0.04") <= p_approx <= Decimal("0.96")):
                continue
            vol24 = to_decimal(m.get("volume_24h"))
            vol_tot = to_decimal(m.get("volume_total") or m.get("volume") or m.get("synced_volume"))
            if vol_tot < self.cfg.min_volume_usd and vol24 < Decimal("2500"):
                continue
            yes_tid, _ = _extract_token_ids(m)
            if m.get("end_date") is None:
                need_detail.append(cid)
            elif not yes_tid:
                end_dt = _parse_iso_dt(m.get("end_date"))
                if end_dt is not None:
                    days = (end_dt - now).total_seconds() / 86400.0
                    if self.cfg.min_horizon_days <= days <= self.cfg.max_horizon_days:
                        need_detail.append(cid)

        need_detail.sort(
            key=lambda c: float(
                to_decimal(merged[c].get("volume_24h"))
                + to_decimal(merged[c].get("volume_total") or merged[c].get("volume")) / 100
            ),
            reverse=True,
        )
        for cid in need_detail[:25]:
            try:
                detail = self.client.get_market(cid)
                merged[cid].update({k: v for k, v in detail.items() if v is not None})
            except Exception as exc:
                log.debug("get_market failed for %s: %s", cid, exc)

        # 4. Pre-filter for multi-day horizon & minimum liquidity before fetching books
        pre_filtered: list[dict[str, Any]] = []
        for cid, m in merged.items():
            if not m.get("active", True) or m.get("closed") or m.get("resolved_outcome"):
                continue
            outs = _extract_outcomes(m)
            if len(outs) != 2 or outs[0].lower() != "yes" or outs[1].lower() != "no":
                continue
            end_dt = _parse_iso_dt(m.get("end_date"))
            if end_dt is None:
                continue
            days = (end_dt - now).total_seconds() / 86400.0
            if not (self.cfg.min_horizon_days <= days <= self.cfg.max_horizon_days):
                continue
            vol_tot = to_decimal(m.get("volume_total") or m.get("volume") or m.get("synced_volume"))
            vol_24h = to_decimal(m.get("volume_24h"))
            if vol_tot < self.cfg.min_volume_usd and vol_24h < Decimal("3000"):
                continue
            yes_tid, no_tid = _extract_token_ids(m)
            m["_end_dt"] = end_dt
            m["_days"] = days
            m["_yes_tid"] = yes_tid
            m["_no_tid"] = no_tid
            m["_vol24"] = vol_24h
            m["_vol_tot"] = vol_tot
            pre_filtered.append(m)

        # Rank pre-filtered markets by activity & take top max_candidates_for_books
        pre_filtered.sort(
            key=lambda m: float(m["_vol24"]) * 5.0 + float(m["_vol_tot"]) * 0.05,
            reverse=True,
        )
        shortlist = pre_filtered[: self.cfg.max_candidates_for_books]

        # 5. Batch-fetch CLOB order books in 1-2 requests!
        token_ids = [m["_yes_tid"] for m in shortlist if m.get("_yes_tid")]
        books_by_token: dict[str, dict[str, Any]] = {}
        if token_ids:
            try:
                books = self.client.get_books_batch(token_ids)
                for b in books:
                    tid = str(b.get("token_id") or b.get("asset_id") or "")
                    if tid:
                        books_by_token[tid] = b
            except Exception as exc:
                log.warning("Batch book fetch failed, falling back to single books: %s", exc)

        candidates: list[MarketCandidate] = []
        for m in shortlist:
            cid = str(m["condition_id"])
            yes_tid = m["_yes_tid"]
            book = books_by_token.get(yes_tid)
            if not book:
                try:
                    book = self.client.get_orderbook(cid, outcome="Yes", depth=10)
                except Exception:
                    continue
            metrics = self.client.parse_book_metrics(book)
            if not metrics:
                continue

            spread = metrics["spread"]
            mid = metrics["mid"]
            best_bid = metrics["best_bid"]
            best_ask = metrics["best_ask"]
            days = float(m["_days"])

            if spread > self.cfg.max_spread:
                continue

            cat = market_categories.get(cid, "other")
            raw_cat = str(m.get("category") or "other")
            q = str(m.get("question") or "")

            # Classify Alpha-B (News-Lag) vs Alpha-C (High-Probability Carry)
            strategy = ""
            suggested_outcome = "Yes"
            ann_yield = Decimal("0")
            score = float(m["_vol24"]) * 2.0 + float(m["_vol_tot"]) * 0.01

            if Decimal("0.12") <= mid <= Decimal("0.88"):
                strategy = "news_lag"
                suggested_outcome = "Yes"
            elif (
                days <= self.cfg.carry_max_horizon_days
                and spread <= Decimal("0.03")
                and (
                    (self.cfg.carry_min_prob <= mid <= self.cfg.carry_max_prob)
                    or (Decimal("1") - self.cfg.carry_max_prob <= mid <= Decimal("1") - self.cfg.carry_min_prob)
                )
            ):
                # High-prob side (either Yes at 0.88-0.96, or No at 0.88-0.96 i.e. Yes at 0.04-0.12)
                if mid >= self.cfg.carry_min_prob:
                    suggested_outcome = "Yes"
                    eff_price = best_bid + metrics["tick_size"] if spread >= Decimal("2") * metrics["tick_size"] else best_bid
                else:
                    suggested_outcome = "No"
                    no_bid = Decimal("1") - best_ask
                    no_ask = Decimal("1") - best_bid
                    eff_price = no_bid + metrics["tick_size"] if (no_ask - no_bid) >= Decimal("2") * metrics["tick_size"] else no_bid
                if Decimal("0") < eff_price < Decimal("1") and days > 0:
                    raw_ret = (Decimal("1") - eff_price) / eff_price
                    ann_yield = raw_ret * (Decimal("365") / Decimal(f"{days:.2f}"))
                    if ann_yield >= self.cfg.carry_min_annualized:
                        strategy = "carry"
                        score += float(ann_yield) * 5000.0

            if not strategy:
                continue

            candidates.append(
                MarketCandidate(
                    condition_id=cid,
                    question=q,
                    slug=str(m.get("slug") or ""),
                    category=cat,
                    raw_category=raw_cat,
                    end_date=m["_end_dt"],
                    days_to_resolution=days,
                    event_group_id=str(m.get("event_group_id") or ""),
                    event_title=str(m.get("event_title") or ""),
                    outcomes=_extract_outcomes(m),
                    yes_token_id=yes_tid,
                    no_token_id=m["_no_tid"],
                    volume_24h=m["_vol24"],
                    volume_total=m["_vol_tot"],
                    best_bid=best_bid,
                    best_ask=best_ask,
                    spread=spread,
                    mid=mid,
                    tick_size=metrics["tick_size"],
                    min_order_size=metrics["min_order_size"],
                    top_bid_size=metrics["top_bid_size"],
                    top_ask_size=metrics["top_ask_size"],
                    strategy_type=strategy,
                    suggested_outcome=suggested_outcome,
                    annualized_carry_yield=ann_yield,
                    score=score,
                )
            )

        # 6. Alpha-A: Structural Math & Monotonicity Scanner across complete event groups (0 LLM tokens)
        structural_hits = self._scan_structural_anomalies(merged, event_meta, market_categories)
        candidates.extend(structural_hits)

        candidates.sort(key=lambda c: (1 if c.strategy_type == "structural_arb" else 0, c.score), reverse=True)
        return candidates, market_categories, len(merged)

    def _scan_structural_anomalies(
        self,
        merged: dict[str, dict[str, Any]],
        event_meta: dict[str, dict[str, Any]],
        market_categories: dict[str, str],
    ) -> list[MarketCandidate]:
        """Detect mathematical anomalies: complete exclusive-outcome sums < 0.985 or deadline monotonicity breaks."""
        hits: list[MarketCandidate] = []
        by_event: dict[str, list[dict[str, Any]]] = {}
        for m in merged.values():
            eg = str(m.get("event_group_id") or "")
            if eg and m.get("active", True) and not m.get("closed") and not m.get("resolved_outcome"):
                by_event.setdefault(eg, []).append(m)

        for eg_id, group in by_event.items():
            ev = event_meta.get(eg_id, {})
            # 1) Complete multi-outcome bundle check (sum of asks < 0.985)
            omitted = int(ev.get("omitted_live_count") or 0)
            resolved_cnt = int(ev.get("resolved_market_count") or 0)
            total_cnt = int(ev.get("market_count") or 0)
            if omitted == 0 and resolved_cnt == 0 and 3 <= len(group) == total_cnt and len(group) <= 15:
                approx_sum = sum(
                    to_decimal(m.get("synced_price") or (m.get("live_price") or {}).get("buy") or m.get("last_price"), "1")
                    for m in group
                )
                if approx_sum < Decimal("0.96"):
                    leg_books: list[tuple[dict[str, Any], dict[str, Any]]] = []
                    ask_sum = Decimal("0")
                    for m in group:
                        cid = str(m.get("condition_id") or "")
                        try:
                            b = self.client.get_orderbook(cid, outcome="Yes", depth=5)
                            met = self.client.parse_book_metrics(b)
                            if met:
                                ask_sum += met["best_ask"]
                                leg_books.append((m, met))
                        except Exception:
                            break
                    if len(leg_books) == len(group) and ask_sum < self.cfg.structural_arb_max_sum:
                        # Flag the best-liquidity leg with the edge information
                        leg_books.sort(key=lambda pair: float(pair[1]["top_ask_size"]), reverse=True)
                        m0, met0 = leg_books[0]
                        cid0 = str(m0["condition_id"])
                        dt0 = _parse_iso_dt(m0.get("end_date"))
                        if dt0:
                            days0 = (dt0 - datetime.now(timezone.utc)).total_seconds() / 86400.0
                            if self.cfg.min_horizon_days <= days0 <= self.cfg.max_horizon_days:
                                yes_t, no_t = _extract_token_ids(m0)
                                hits.append(
                                    MarketCandidate(
                                        condition_id=cid0,
                                        question=str(m0.get("question") or ""),
                                        slug=str(m0.get("slug") or ""),
                                        category=market_categories.get(cid0, "other"),
                                        raw_category=str(ev.get("category") or "other"),
                                        end_date=dt0,
                                        days_to_resolution=days0,
                                        event_group_id=eg_id,
                                        event_title=str(ev.get("event_title") or ""),
                                        outcomes=_extract_outcomes(m0),
                                        yes_token_id=yes_t,
                                        no_token_id=no_t,
                                        volume_24h=to_decimal(m0.get("volume_24h")),
                                        volume_total=to_decimal(m0.get("volume_total") or m0.get("volume")),
                                        best_bid=met0["best_bid"],
                                        best_ask=met0["best_ask"],
                                        spread=met0["spread"],
                                        mid=met0["mid"],
                                        tick_size=met0["tick_size"],
                                        min_order_size=met0["min_order_size"],
                                        top_bid_size=met0["top_bid_size"],
                                        top_ask_size=met0["top_ask_size"],
                                        strategy_type="structural_arb",
                                        suggested_outcome="Yes",
                                        structural_note=(
                                            f"Multi-Outcome-Summe aller {len(group)} Ausgänge = {ask_sum:.3f} < {self.cfg.structural_arb_max_sum}."
                                        ),
                                        score=1_000_000.0,
                                    )
                                )

            # 2) Temporal monotonicity check: "before/by Month Day" series
            dated: list[tuple[datetime, Decimal, dict[str, Any]]] = []
            for m in group:
                end_dt = _parse_iso_dt(m.get("end_date"))
                p_raw = m.get("synced_price") or (m.get("live_price") or {}).get("buy") or m.get("last_price")
                p = to_decimal(p_raw, "0")
                if end_dt and Decimal("0.02") < p < Decimal("0.98"):
                    dated.append((end_dt, p, m))
            # Only compare markets that look like deadline variants of the exact same proposition
            if len(dated) >= 2:
                dated.sort(key=lambda x: x[0])
                for i in range(len(dated) - 1):
                    dt_early, p_early, m_early = dated[i]
                    dt_late, p_late, m_late = dated[i + 1]
                    if dt_early >= dt_late:
                        continue
                    q_early_stem = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,4})\b", "", str(m_early.get("question", "")).lower()).strip()
                    q_late_stem = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december|\d{1,4})\b", "", str(m_late.get("question", "")).lower()).strip()
                    if q_early_stem and q_early_stem == q_late_stem and (p_early - p_late) > self.cfg.monotonicity_violation_threshold:
                        # Later deadline is priced CHEAPER than earlier deadline -> buy Yes on later deadline
                        cid = str(m_late["condition_id"])
                        yes_tid, no_tid = _extract_token_ids(m_late)
                        if not yes_tid:
                            continue
                        try:
                            book = self.client.get_orderbook(cid, outcome="Yes", depth=10)
                            metrics = self.client.parse_book_metrics(book)
                        except Exception:
                            metrics = None
                        if not metrics or metrics["spread"] > Decimal("0.05"):
                            continue
                        if p_early - metrics["best_ask"] >= Decimal("0.02"):
                            now = datetime.now(timezone.utc)
                            days = max(0.5, (dt_late - now).total_seconds() / 86400.0)
                            hits.append(
                                MarketCandidate(
                                    condition_id=cid,
                                    question=str(m_late.get("question") or ""),
                                    slug=str(m_late.get("slug") or ""),
                                    category=market_categories.get(cid, "other"),
                                    raw_category=str(ev.get("category") or "other"),
                                    end_date=dt_late,
                                    days_to_resolution=days,
                                    event_group_id=eg_id,
                                    event_title=str(ev.get("event_title") or ""),
                                    outcomes=_extract_outcomes(m_late),
                                    yes_token_id=yes_tid,
                                    no_token_id=no_tid,
                                    volume_24h=to_decimal(m_late.get("volume_24h")),
                                    volume_total=to_decimal(m_late.get("volume_total") or m_late.get("volume")),
                                    best_bid=metrics["best_bid"],
                                    best_ask=metrics["best_ask"],
                                    spread=metrics["spread"],
                                    mid=metrics["mid"],
                                    tick_size=metrics["tick_size"],
                                    min_order_size=metrics["min_order_size"],
                                    top_bid_size=metrics["top_bid_size"],
                                    top_ask_size=metrics["top_ask_size"],
                                    strategy_type="structural_arb",
                                    suggested_outcome="Yes",
                                    structural_note=(
                                        f"Monotonie-Verletzung: Früherer Termin ({m_early.get('question')}) "
                                        f"steht bei {p_early:.3f}, späterer bei {metrics['mid']:.3f}."
                                    ),
                                    score=1_000_000.0,
                                )
                            )
        return hits
