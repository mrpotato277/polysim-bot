"""Blind Superforecaster using Gemini Flash + Bayesian log-odds blending in Python."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from google import genai
from google.genai import types

from bot.config import BotConfig
from bot.math_utils import bayesian_blend, clamp_prob

log = logging.getLogger(__name__)

FALLBACK_MODELS = (
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.8-flash",
)


@dataclass
class ForecastResult:
    p_blind: Decimal
    p_blended: Decimal
    confidence: Decimal
    catalyst_freshness: str
    resolution_risk: str
    resolution_note: str
    base_rate: Decimal
    bull_case: str
    bear_case: str
    summary_de: str
    model_used: str
    raw_reasoning: dict[str, Any]


class Superforecaster:
    """Estimates blind event probabilities without anchoring on market price."""

    def __init__(self, config: BotConfig) -> None:
        self.cfg = config
        self.client = genai.Client(
            vertexai=self.cfg.use_vertex_express,
            api_key=config.gemini_api_key,
            http_options=types.HttpOptions(timeout=25000),
        )

    def forecast(
        self,
        *,
        question: str,
        description: str,
        end_date_iso: str,
        market_price_yes: Decimal,
        news_articles: list[dict[str, Any]],
    ) -> ForecastResult:
        """Run blind forecasting prompt (NO market price shown to LLM), then blend in Python."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        news_lines = []
        for idx, art in enumerate(news_articles[:6], start=1):
            tag = "[NEW]" if art.get("is_new") else "[SEEN]"
            age = art.get("age_hours", "?")
            src = art.get("source") or "News"
            title = art.get("title", "")
            snippet = art.get("snippet", "")
            news_lines.append(f"{idx}. {tag} ({age}h ago, {src}) {title} -- {snippet[:140]}")
        news_block = "\n".join(news_lines) if news_lines else "No recent news articles found in the last 48 hours."

        prompt = f"""You are a disciplined quantitative superforecaster evaluating a binary Polymarket contract.
Current UTC time: {now_utc}
Contract resolution deadline: {end_date_iso or "Unknown"}

Question (YES outcome): {question}
Resolution context / rules: {description[:900] if description else "Standard Polymarket binary resolution based on official announcement / primary sources."}

Recent News Headlines (last 48h):
{news_block}

Instructions:
1. Check for Polymarket resolution traps (e.g., 'announced' vs 'enacted', exact deadline cutoff, ambiguous wording). Rate resolution_trap_risk as "low", "medium", or "high".
2. Anchor on an outside-view historical base_rate (0.01 to 0.99).
3. State the strongest bull_case (for YES) and bear_case (for NO) concisely.
4. Assess catalyst_freshness: "breaking_lt_6h" if material news arrived <6h ago, "recent_24h" if <24h ago, else "stale_or_none".
5. Output your calibrated blind probability p_blind (0.01 to 0.99) for YES resolving True, and your confidence (0.0 to 1.0, stay humble: >0.85 only when facts are near-certain).
6. Write summary_de in clear, concise German (1-2 sentences, no emojis) explaining the thesis.

Return ONLY valid JSON with these exact keys:
{{
  "resolution_trap_risk": "low|medium|high",
  "resolution_note": "short string",
  "base_rate": 0.50,
  "bull_case": "short string",
  "bear_case": "short string",
  "catalyst_freshness": "breaking_lt_6h|recent_24h|stale_or_none",
  "p_blind": 0.50,
  "confidence": 0.50,
  "summary_de": "Kurze deutsche Begründung."
}}"""

        ordered_models = [self.cfg.gemini_model] + [m for m in FALLBACK_MODELS if m != self.cfg.gemini_model]
        parsed: dict[str, Any] | None = None
        used_model = self.cfg.gemini_model
        last_exc: Exception | None = None

        for model_id in ordered_models:
            try:
                resp = self.client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.15,
                    ),
                )
                raw_text = (resp.text or "").strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.strip("`").removeprefix("json").strip()
                parsed = json.loads(raw_text)
                used_model = model_id
                break
            except Exception as exc:
                log.warning("Forecast call failed on %s: %s", model_id, exc)
                last_exc = exc

        if parsed is None:
            raise RuntimeError(f"All Gemini models failed for forecast: {last_exc}")

        p_blind_f = clamp_prob(float(parsed.get("p_blind", 0.5)), eps=0.01)
        base_rate_f = clamp_prob(float(parsed.get("base_rate", 0.5)), eps=0.01)
        conf_f = max(0.0, min(1.0, float(parsed.get("confidence", 0.4))))

        freshness = str(parsed.get("catalyst_freshness", "stale_or_none")).strip().lower()
        if freshness not in {"breaking_lt_6h", "recent_24h", "stale_or_none"}:
            freshness = "stale_or_none"

        trap_risk = str(parsed.get("resolution_trap_risk", "medium")).strip().lower()
        if trap_risk not in {"low", "medium", "high"}:
            trap_risk = "medium"

        p_blind = Decimal(f"{p_blind_f:.4f}")
        confidence = Decimal(f"{conf_f:.4f}")
        p_blended = bayesian_blend(p_blind, market_price_yes, confidence, freshness)

        return ForecastResult(
            p_blind=p_blind,
            p_blended=p_blended,
            confidence=confidence,
            catalyst_freshness=freshness,
            resolution_risk=trap_risk,
            resolution_note=str(parsed.get("resolution_note", ""))[:300],
            base_rate=Decimal(f"{base_rate_f:.4f}"),
            bull_case=str(parsed.get("bull_case", ""))[:350],
            bear_case=str(parsed.get("bear_case", ""))[:350],
            summary_de=str(parsed.get("summary_de", ""))[:400],
            model_used=used_model,
            raw_reasoning={
                "model": used_model,
                "resolution_note": str(parsed.get("resolution_note", ""))[:300],
                "base_rate": base_rate_f,
                "bull_case": str(parsed.get("bull_case", ""))[:350],
                "bear_case": str(parsed.get("bear_case", ""))[:350],
                "summary_de": str(parsed.get("summary_de", ""))[:400],
                "news_used": [a.get("title", "")[:140] for a in news_articles[:4]],
            },
        )
