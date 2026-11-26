"""Deterministic 9-step trading orchestrator (runs locally or via GitHub Actions)."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from bot.config import BotConfig
from bot.dashboard import export_dashboard_data, resolve_true_starting_balance
from bot.execution import ExecutionEngine
from bot.forecaster import Superforecaster
from bot.market_data import build_market_data_context
from bot.math_utils import maker_buy_price, to_decimal
from bot.news import build_news_query, fetch_market_news
from bot.overseer import Overseer, TradeProposal, choose_entry_execution
from bot.polysim_client import PolySimWrapper
from bot.scanner import MarketCandidate, MarketScanner
from bot.state import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("orchestrator")


def _update_brier_scores(client: PolySimWrapper, state: StateStore) -> int:
    """Check settlements and unresolved forecast markets to update Brier scores."""
    total_updated = 0
    try:
        settlements = client.get_settlements(limit=100)
        for s in settlements:
            mid = str(s.get("market_id") or "")
            res = str(s.get("resolved_outcome") or s.get("winning_outcome") or "").lower()
            if mid and res in {"yes", "no"}:
                total_updated += state.update_resolved_market_brier(mid, res == "yes")
    except Exception as exc:
        log.debug("Settlement check skipped: %s", exc)

    for mid in state.list_unresolved_forecast_market_ids(limit=15):
        try:
            m = client.get_market(mid)
            res = str(m.get("resolved_outcome") or "").lower()
            if res in {"yes", "no"}:
                total_updated += state.update_resolved_market_brier(mid, res == "yes")
            else:
                state.mark_forecast_market_checked(mid)
        except Exception:
            state.mark_forecast_market_checked(mid)
            continue
    return total_updated


def _re_evaluate_positions(
    *,
    cfg: BotConfig,
    client: PolySimWrapper,
    state: StateStore,
    forecaster: Superforecaster,
    overseer: Overseer,
    execution: ExecutionEngine,
    positions: list[dict[str, Any]],
    market_categories: dict[str, str],
) -> tuple[int, int, int]:
    """Check open positions for breaking news, take-profit, or thesis-break exits."""
    llm_calls = 0
    attempted = 0
    executed = 0
    now = datetime.now(timezone.utc)

    for pos in positions:
        mid = str(pos.get("market_id") or pos.get("condition_id") or "")
        qty = to_decimal(pos.get("quantity") or pos.get("size"))
        if not mid or qty <= Decimal("0"):
            continue
        outcome = str(pos.get("outcome") or "Yes")
        question = str(pos.get("question") or "")
        avg_entry = to_decimal(pos.get("avg_entry_price") or pos.get("avg_price"))
        cat = market_categories.get(mid, str(pos.get("category") or "other")).lower()

        try:
            book = client.get_orderbook(mid, outcome=outcome, depth=10)
            met = client.parse_book_metrics(book)
        except Exception:
            met = None
        if not met:
            continue

        latest = state.get_latest_forecast(mid)
        need_reforecast = latest is None
        news_articles: list[dict[str, Any]] = []

        if question:
            q = build_news_query(question)
            raw_news = fetch_market_news(q, category=cat, max_age_hours=48, max_items=10)
            news_articles, new_cnt = state.filter_and_record_news(mid, raw_news)
            if new_cnt > 0:
                need_reforecast = True
            elif latest is not None:
                last_dt = state.get_recent_forecast_time(mid)
                if last_dt and (now - last_dt) > timedelta(hours=12):
                    need_reforecast = True

        p_true_outcome: Decimal | None = None
        if need_reforecast and question and llm_calls < cfg.max_forecasts_per_run:
            try:
                mkt_yes_mid = met["mid"] if outcome.lower() == "yes" else (Decimal("1") - met["mid"])
                desc = ""
                end_date_iso = str(latest.get("end_date") or "") if latest else ""
                try:
                    detail = client.get_market(mid)
                    desc = str(detail.get("description") or detail.get("rules") or detail.get("event_title") or "")
                    if detail.get("end_date"):
                        end_date_iso = str(detail.get("end_date"))
                except Exception:
                    pass
                try:
                    mkt_ctx = build_market_data_context(question, cat)
                except Exception:
                    mkt_ctx = ""
                fc = forecaster.forecast(
                    question=question,
                    description=desc,
                    end_date_iso=end_date_iso,
                    market_price_yes=mkt_yes_mid,
                    news_articles=news_articles,
                    market_context=mkt_ctx,
                )
                llm_calls += fc.api_attempts
                p_true_outcome = fc.p_blended if outcome.lower() == "yes" else (Decimal("1") - fc.p_blended)
                edge_now = p_true_outcome - met["mid"]
                state.record_forecast(
                    {
                        "market_id": mid,
                        "question": question,
                        "category": cat,
                        "end_date": end_date_iso,
                        "market_price": f"{mkt_yes_mid:.4f}",
                        "p_blind": f"{fc.p_blind:.4f}",
                        "p_blended": f"{fc.p_blended:.4f}",
                        "confidence": f"{fc.confidence:.4f}",
                        "catalyst_freshness": fc.catalyst_freshness,
                        "edge_net": f"{edge_now:.4f}",
                        "strategy_type": "position_review",
                        "resolution_risk": fc.resolution_risk,
                        "reasoning": fc.raw_reasoning,
                    }
                )
            except Exception as exc:
                log.warning("Position re-forecast failed for %s: %s", mid, exc)

        if p_true_outcome is None and latest is not None:
            p_yes = to_decimal(latest.get("p_blended"), "0.5")
            p_true_outcome = p_yes if outcome.lower() == "yes" else (Decimal("1") - p_yes)

        if p_true_outcome is None:
            continue

        token_id = str(book.get("token_id") or pos.get("token_id") or "")
        fee_bps = client.get_fee_rate_bps(token_id) if token_id else 500

        exit_prop = overseer.evaluate_position_exit(
            market_id=mid,
            question=question or mid,
            category=cat,
            outcome=outcome,
            shares_held=qty,
            avg_entry_price=avg_entry,
            best_bid=met["best_bid"],
            best_ask=met["best_ask"],
            tick_size=met["tick_size"],
            min_order_size=met["min_order_size"],
            p_true_outcome=p_true_outcome,
            fee_rate_bps=fee_bps,
        )
        if exit_prop:
            attempted += 1
            ok, msg = execution.execute_sell_exit(exit_prop, qty)
            log.info("Exit order %s: %s", mid, msg)
            if ok:
                executed += 1

    return llm_calls, attempted, executed


def run_cycle(dry_run_override: bool | None = None) -> int:
    t0 = time.monotonic()
    cfg = BotConfig.from_env(dry_run_override=dry_run_override)
    errors: list[str] = []

    if not cfg.polysim_api_key or not cfg.gemini_api_key:
        log.error("POLYSIM_API_KEY and GEMINI_API_KEY must be set in .env or environment.")
        return 1

    log.info("Starting cycle (DRY_RUN=%s, model=%s)", cfg.dry_run, cfg.gemini_model)

    with PolySimWrapper(cfg) as client, StateStore(cfg.db_path) as state:
        overseer = Overseer(cfg)
        forecaster = Superforecaster(cfg)
        execution = ExecutionEngine(client, state, cfg)
        scanner = MarketScanner(client, cfg)

        # 1. Sync account state & update Brier scores
        balance = client.get_balance()
        portfolio = client.get_portfolio()
        positions = client.get_positions(status="OPEN")
        open_orders = client.get_open_orders()
        brier_updates = _update_brier_scores(client, state)

        cash_usd = to_decimal(balance.get("balance"))
        equity_usd = to_decimal(balance.get("total_value"), str(cash_usd))
        unrealized_pnl = sum((to_decimal(p.get("unrealized_pnl")) for p in positions), Decimal("0"))
        starting_bal = resolve_true_starting_balance(client, state, balance)
        realized_pnl = equity_usd - starting_bal - unrealized_pnl

        dd_24h = state.get_24h_drawdown(equity_usd)
        circuit_breaker = dd_24h <= cfg.circuit_breaker_24h_dd
        if circuit_breaker:
            log.warning("CIRCUIT BREAKER ACTIVE: 24h drawdown %s <= %s", dd_24h, cfg.circuit_breaker_24h_dd)

        # 2. Cancel stale resting orders
        stale_cancelled = 0
        for stale_ord in overseer.find_stale_orders(open_orders):
            if execution.cancel_stale_order(stale_ord):
                stale_cancelled += 1
        if stale_cancelled > 0:
            try:
                open_orders = client.get_open_orders()
            except Exception as exc:
                log.warning("Failed to refresh open orders after cancel: %s", exc)

        # 3. Scan markets for Alpha A/B/C
        candidates, market_categories, total_scanned = scanner.scan()
        log.info("Scanned %d markets -> %d filtered candidates", total_scanned, len(candidates))

        total_exp, by_market, by_category = overseer.compute_portfolio_exposure(
            positions, open_orders, market_categories
        )
        deployed_pct = (total_exp / equity_usd * Decimal("100")) if equity_usd > 0 else Decimal("0")
        synced_total_exp = total_exp
        synced_cash_usd = cash_usd

        # 4. Re-evaluate open positions
        pos_llm, pos_att, pos_exec = _re_evaluate_positions(
            cfg=cfg,
            client=client,
            state=state,
            forecaster=forecaster,
            overseer=overseer,
            execution=execution,
            positions=positions,
            market_categories=market_categories,
        )

        llm_calls = pos_llm
        trades_attempted = pos_att
        trades_executed = pos_exec

        open_order_markets = {
            str(o.get("market_id") or o.get("market") or "") for o in open_orders
        }

        # 5. Fetch & deduplicate news for top non-structural candidates (0 LLM tokens)
        enriched_news: dict[str, tuple[list[dict[str, Any]], int]] = {}
        for cand in candidates[:15]:
            if cand.strategy_type == "structural_arb":
                enriched_news[cand.condition_id] = ([], 0)
                continue
            q_str = build_news_query(cand.question, cand.event_title)
            raw_arts = fetch_market_news(q_str, category=cand.category, max_age_hours=48, max_items=10)
            arts, new_cnt = state.filter_and_record_news(cand.condition_id, raw_arts)
            enriched_news[cand.condition_id] = (arts, new_cnt)

        # Prioritize: structural_arb > candidates with new unseen news > high-score carry/news_lag
        now = datetime.now(timezone.utc)
        def _priority_key(c: MarketCandidate) -> tuple[int, int, float]:
            _, n_new = enriched_news.get(c.condition_id, ([], 0))
            is_struct = 1 if c.strategy_type == "structural_arb" else 0
            return (is_struct, 1 if n_new > 0 else 0, c.score)

        ranked = sorted(candidates[:15], key=_priority_key, reverse=True)

        # 6. Forecast & evaluate trades
        new_buys_executed = 0
        for cand in ranked:
            if new_buys_executed >= cfg.max_new_buys_per_run:
                log.info(
                    "Ramp-up pace cap reached (max %d new buy/run); deferring remaining entries.",
                    cfg.max_new_buys_per_run,
                )
                break
            if cand.condition_id in open_order_markets:
                continue

            arts, new_cnt = enriched_news.get(cand.condition_id, ([], 0))
            last_fc_dt = state.get_recent_forecast_time(cand.condition_id)
            recently_forecasted = last_fc_dt is not None and (now - last_fc_dt) < timedelta(hours=6)

            proposal: TradeProposal | None = None

            if cand.strategy_type == "structural_arb":
                # Pure mathematical anomaly -> construct maker buy without LLM tokens
                entry_price = maker_buy_price(cand.best_bid, cand.best_ask, cand.tick_size)
                # Conservative implied fair probability for structural monotonicity/bundle edge
                p_true = min(Decimal("0.98"), entry_price + Decimal("0.04"))
                edge_net = p_true - entry_price
                proposal = TradeProposal(
                    market_id=cand.condition_id,
                    question=cand.question,
                    category=cand.category,
                    outcome="Yes",
                    side="BUY",
                    order_mode="maker_limit",
                    limit_price=entry_price,
                    tick_size=cand.tick_size,
                    min_order_size=cand.min_order_size,
                    p_true_outcome=p_true,
                    edge_net=edge_net,
                    strategy_type="structural_arb",
                    rationale=cand.structural_note,
                    est_fee_bps=0,
                )
            else:
                if llm_calls >= cfg.max_forecasts_per_run:
                    continue
                if recently_forecasted and new_cnt == 0:
                    continue

                # Fetch full market description when needed for resolution trap verification
                desc = ""
                try:
                    detail = client.get_market(cand.condition_id)
                    desc = str(detail.get("description") or detail.get("rules") or detail.get("event_title") or "")
                except Exception:
                    desc = cand.event_title

                try:
                    try:
                        mkt_ctx = build_market_data_context(cand.question, cand.category)
                    except Exception:
                        mkt_ctx = ""
                    fc = forecaster.forecast(
                        question=cand.question,
                        description=desc,
                        end_date_iso=cand.end_date.isoformat() if cand.end_date else "",
                        market_price_yes=cand.mid,
                        news_articles=arts,
                        market_context=mkt_ctx,
                    )
                    llm_calls += fc.api_attempts
                except Exception as exc:
                    err = f"Forecaster error for {cand.condition_id}: {exc}"
                    log.warning(err)
                    errors.append(err)
                    continue

                if fc.resolution_risk == "high":
                    state.record_forecast(
                        {
                            "market_id": cand.condition_id,
                            "question": cand.question,
                            "category": cand.category,
                            "end_date": cand.end_date.isoformat() if cand.end_date else "",
                            "market_price": f"{cand.mid:.4f}",
                            "p_blind": f"{fc.p_blind:.4f}",
                            "p_blended": f"{fc.p_blended:.4f}",
                            "confidence": f"{fc.confidence:.4f}",
                            "catalyst_freshness": fc.catalyst_freshness,
                            "edge_net": "0.0000",
                            "strategy_type": cand.strategy_type,
                            "resolution_risk": fc.resolution_risk,
                            "reasoning": fc.raw_reasoning,
                        }
                    )
                    log.info("Skipped %s due to high resolution_trap_risk", cand.question[:50])
                    continue

                # Determine optimal side (Yes vs No) & exact order book for that side
                yes_entry = maker_buy_price(cand.best_bid, cand.best_ask, cand.tick_size)
                yes_edge = fc.p_blended - yes_entry

                no_bid_est = Decimal("1") - cand.best_ask
                no_ask_est = Decimal("1") - cand.best_bid
                no_entry_est = maker_buy_price(no_bid_est, no_ask_est, cand.tick_size)
                no_edge_est = (Decimal("1") - fc.p_blended) - no_entry_est

                chosen_outcome = "Yes"
                entry_price = yes_entry
                chosen_ask = cand.best_ask
                chosen_token_id = cand.yes_token_id
                p_true_outcome = fc.p_blended
                edge_net = yes_edge
                tick_sz = cand.tick_size
                min_sz = cand.min_order_size

                if no_edge_est > yes_edge and no_edge_est > Decimal("0"):
                    # Fetch exact NO order book to verify real tick size, spread, and min_order_size
                    try:
                        no_book = client.get_orderbook(cand.condition_id, outcome="No", depth=10)
                        no_met = client.parse_book_metrics(no_book)
                    except Exception:
                        no_met = None
                    if no_met:
                        chosen_outcome = "No"
                        tick_sz = no_met["tick_size"]
                        min_sz = no_met["min_order_size"]
                        entry_price = maker_buy_price(no_met["best_bid"], no_met["best_ask"], tick_sz)
                        chosen_ask = no_met["best_ask"]
                        chosen_token_id = cand.no_token_id
                        p_true_outcome = Decimal("1") - fc.p_blended
                        edge_net = p_true_outcome - entry_price

                # Carry candidates are strictly for high-probability bonds; do not flip them into longshot bets
                effective_strategy = cand.strategy_type
                if cand.strategy_type == "carry" and chosen_outcome != cand.suggested_outcome:
                    edge_net = p_true_outcome - entry_price
                    state.record_forecast(
                        {
                            "market_id": cand.condition_id,
                            "question": cand.question,
                            "category": cand.category,
                            "end_date": cand.end_date.isoformat() if cand.end_date else "",
                            "market_price": f"{cand.mid:.4f}",
                            "p_blind": f"{fc.p_blind:.4f}",
                            "p_blended": f"{fc.p_blended:.4f}",
                            "confidence": f"{fc.confidence:.4f}",
                            "catalyst_freshness": fc.catalyst_freshness,
                            "edge_net": "0.0000",
                            "strategy_type": "carry_rejected",
                            "resolution_risk": fc.resolution_risk,
                            "reasoning": fc.raw_reasoning,
                        }
                    )
                    log.info("Overseer %s -> SKIP: Carry candidate disagreed with market consensus.", cand.question[:45])
                    continue

                # Default to 0% maker_limit; switch to marketable_limit (FAK taker) on high-edge breaking news
                fee_bps = 0
                if effective_strategy == "news_lag" and fc.catalyst_freshness == "breaking_lt_6h":
                    fee_bps = client.get_fee_rate_bps(chosen_token_id) if chosen_token_id else 500
                order_mode, entry_price, edge_net, est_fee_bps = choose_entry_execution(
                    strategy_type=effective_strategy,
                    catalyst_freshness=fc.catalyst_freshness,
                    p_true_outcome=p_true_outcome,
                    maker_price=entry_price,
                    best_ask=chosen_ask,
                    fee_rate_bps=fee_bps,
                    breaking_taker_min_edge_net=cfg.breaking_taker_min_edge_net,
                )
                if order_mode == "marketable_limit":
                    log.info(
                        "Hybrid-taker triggered for %s (%s): ask=%s fee_bps=%d edge_net=%.4f",
                        cand.question[:45],
                        chosen_outcome,
                        entry_price,
                        est_fee_bps,
                        edge_net,
                    )

                state.record_forecast(
                    {
                        "market_id": cand.condition_id,
                        "question": cand.question,
                        "category": cand.category,
                        "end_date": cand.end_date.isoformat() if cand.end_date else "",
                        "market_price": f"{cand.mid:.4f}",
                        "p_blind": f"{fc.p_blind:.4f}",
                        "p_blended": f"{fc.p_blended:.4f}",
                        "confidence": f"{fc.confidence:.4f}",
                        "catalyst_freshness": fc.catalyst_freshness,
                        "edge_net": f"{edge_net:.4f}",
                        "strategy_type": effective_strategy,
                        "resolution_risk": fc.resolution_risk,
                        "reasoning": fc.raw_reasoning,
                    }
                )

                proposal = TradeProposal(
                    market_id=cand.condition_id,
                    question=cand.question,
                    category=cand.category,
                    outcome=chosen_outcome,
                    side="BUY",
                    order_mode=order_mode,
                    limit_price=entry_price,
                    tick_size=tick_sz,
                    min_order_size=min_sz,
                    p_true_outcome=p_true_outcome,
                    edge_net=edge_net,
                    strategy_type=effective_strategy,
                    rationale=f"[{effective_strategy}] {fc.summary_de}",
                    est_fee_bps=est_fee_bps,
                )

            if proposal is None:
                continue

            # 7. Overseer risk check & Quarter-Kelly sizing
            decision = overseer.evaluate_buy(
                proposal,
                equity_usd=equity_usd,
                cash_usd=cash_usd,
                total_exposure_usd=total_exp,
                market_exposure_usd=by_market.get(cand.condition_id, Decimal("0")),
                category_exposure_usd=by_category.get(cand.category, Decimal("0")),
                has_open_order_on_market=(cand.condition_id in open_order_markets),
                circuit_breaker_active=circuit_breaker,
            )
            if not decision.approved or decision.order is None:
                log.info("Overseer %s -> %s", cand.question[:45], decision.reason)
                continue

            # 8. Execute order
            trades_attempted += 1
            ok, msg = execution.execute_buy(decision.order, tick_size=proposal.tick_size)
            log.info("Execution %s -> %s", cand.question[:45], msg)
            if ok:
                trades_executed += 1
                new_buys_executed += 1
                notional = decision.order.notional_usd
                cash_usd -= notional
                total_exp += notional
                by_market[cand.condition_id] = by_market.get(cand.condition_id, Decimal("0")) + notional
                by_category[cand.category] = by_category.get(cand.category, Decimal("0")) + notional
                open_order_markets.add(cand.condition_id)

        # 9. Export dashboard JSON & record run log (skip rewriting on idle 5-min checks)
        duration = round(time.monotonic() - t0, 2)
        last_snap = state.get_latest_equity_snapshot()
        now_snap = datetime.now(timezone.utc)
        snap_stale = True
        equity_changed = True
        if last_snap is not None:
            try:
                last_dt = datetime.fromisoformat(str(last_snap["ts"]).replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                snap_stale = (now_snap - last_dt) >= timedelta(minutes=12)
            except ValueError:
                snap_stale = True
            last_eq = to_decimal(last_snap.get("equity_usd"))
            last_pos = int(last_snap.get("open_positions_count") or 0)
            equity_changed = (last_eq != equity_usd.quantize(Decimal("0.01"))) or (last_pos != len(positions))

        should_persist = (
            llm_calls > 0
            or trades_attempted > 0
            or brier_updates > 0
            or len(errors) > 0
            or equity_changed
            or snap_stale
        )

        if should_persist:
            state.record_equity_snapshot(
                {
                    "cash_usd": f"{synced_cash_usd:.2f}",
                    "positions_value_usd": f"{synced_total_exp:.2f}",
                    "equity_usd": f"{equity_usd:.2f}",
                    "deployed_pct": f"{deployed_pct:.2f}",
                    "unrealized_pnl_usd": f"{unrealized_pnl:.2f}",
                    "realized_pnl_usd": f"{realized_pnl:.2f}",
                    "open_positions_count": len(positions),
                    "circuit_breaker_active": circuit_breaker,
                }
            )
            state.record_run(
                {
                    "dry_run": cfg.dry_run,
                    "markets_scanned": total_scanned,
                    "candidates_found": len(candidates),
                    "llm_calls": llm_calls,
                    "trades_attempted": trades_attempted,
                    "trades_executed": trades_executed,
                    "errors": errors,
                    "duration_sec": duration,
                }
            )
            export_dashboard_data(
                config=cfg,
                client=client,
                state=state,
                balance=balance,
                portfolio=portfolio,
                positions=positions,
                market_categories=market_categories,
                circuit_breaker_active=circuit_breaker,
            )
        else:
            log.info("Idle 5-min check (no new signals/changes); skipping commit-triggering dashboard rewrite.")
        log.info(
            "Cycle finished in %.2fs | scanned=%d candidates=%d llm=%d attempted=%d executed=%d",
            duration,
            total_scanned,
            len(candidates),
            llm_calls,
            trades_attempted,
            trades_executed,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PolySimulator quantitative trading bot orchestrator.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="Force DRY_RUN=true (no live orders sent).")
    group.add_argument("--live", action="store_true", help="Force DRY_RUN=false (submit simulation orders).")
    args = parser.parse_args()

    override: bool | None = None
    if args.dry_run:
        override = True
    elif args.live:
        override = False

    return run_cycle(dry_run_override=override)


if __name__ == "__main__":
    raise SystemExit(main())
