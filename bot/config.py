"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _parse_bool(value: str | None, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


@dataclass(frozen=True)
class BotConfig:
    polysim_api_key: str
    gemini_api_key: str
    polysim_base_url: str = "https://api.polysimulator.com/v1"
    gemini_model: str = "gemini-3.5-flash-lite"
    dry_run: bool = True

    # Capital & Risk Limits (strictly enforced by Overseer — scaled balanced profile)
    max_total_exposure_pct: Decimal = Decimal("0.55")
    max_single_market_pct: Decimal = Decimal("0.07")
    max_category_exposure_pct: Decimal = Decimal("0.22")
    kelly_fraction: Decimal = Decimal("0.22")
    cash_buffer_usd: Decimal = Decimal("1.00")
    circuit_breaker_24h_dd: Decimal = Decimal("-0.05")

    # Alpha & Edge Thresholds
    min_edge_net: Decimal = Decimal("0.06")
    breaking_taker_min_edge_net: Decimal = Decimal("0.12")
    min_high_conv_bump_edge: Decimal = Decimal("0.10")
    carry_min_prob: Decimal = Decimal("0.88")
    carry_max_prob: Decimal = Decimal("0.96")
    carry_min_annualized: Decimal = Decimal("0.40")
    take_profit_edge_remaining: Decimal = Decimal("0.02")
    thesis_break_edge: Decimal = Decimal("-0.04")
    structural_arb_max_sum: Decimal = Decimal("0.985")
    monotonicity_violation_threshold: Decimal = Decimal("0.03")

    # Market Scanner Filters
    min_horizon_days: int = 2
    max_horizon_days: int = 21
    carry_max_horizon_days: int = 15
    min_volume_usd: Decimal = Decimal("15000")
    max_spread: Decimal = Decimal("0.04")
    max_markets_scan: int = 1000
    max_candidates_for_books: int = 120
    max_forecasts_per_run: int = 8
    max_new_buys_per_run: int = 2
    stale_order_minutes: int = 90

    # Rate Limiting
    min_request_spacing_sec: float = 0.05

    # Paths
    db_path: Path = PROJECT_ROOT / "data" / "state.db"
    dashboard_json_path: Path = PROJECT_ROOT / "docs" / "dashboard_data.json"

    @property
    def use_vertex_express(self) -> bool:
        return False

    @classmethod
    def from_env(cls, dry_run_override: bool | None = None) -> "BotConfig":
        polysim_key = os.getenv("POLYSIM_API_KEY", "").strip()
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
        base_url = os.getenv("POLYSIM_API_URL", "https://api.polysimulator.com/v1").strip()
        dry_run = _parse_bool(os.getenv("DRY_RUN"), True) if dry_run_override is None else dry_run_override
        return cls(
            polysim_api_key=polysim_key,
            gemini_api_key=gemini_key,
            polysim_base_url=base_url,
            gemini_model=gemini_model,
            dry_run=dry_run,
        )
