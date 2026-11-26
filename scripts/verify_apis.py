"""Verify PolySimClient (SDK) and Google Gemini API access in read-only mode."""

from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from polysim_sdk import PolySimClient

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def mask_key(key: str | None) -> str:
    if not key:
        return "<MISSING>"
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def verify_polysim_sdk() -> None:
    print("=== 1. Verifying PolySimClient (Official SDK) ===")
    api_key = os.getenv("POLYSIM_API_KEY", "").strip()
    print(f"Using POLYSIM_API_KEY: {mask_key(api_key)}")
    if not api_key:
        raise RuntimeError("POLYSIM_API_KEY is missing in .env")

    with PolySimClient(api_key=api_key, floor_interval=0.55) as client:
        me = client.me()
        print("me():", json.dumps(me, indent=2)[:800])

        bal = client.balance()
        print("balance():", json.dumps(bal))

        pos = client.positions(status="OPEN")
        print(f"positions(OPEN): {len(pos)} open")

        open_orders = client.list_orders(status="PENDING")
        print(f"list_orders(PENDING): {len(open_orders)} pending")

        markets = client.list_markets(limit=5)
        print(f"list_markets(limit=5): {len(markets)} markets")
        if markets:
            m0 = markets[0]
            cond_id = m0["condition_id"]
            detail = client.get_market(cond_id)
            print("get_market(cond_id) keys:", list(detail.keys()))
            print("get_market(cond_id) preview:", json.dumps(detail, indent=2)[:1500])

            book = client.get_book(cond_id)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            best_bid = max((Decimal(str(b["price"])) for b in bids), default=None)
            best_ask = min((Decimal(str(a["price"])) for a in asks), default=None)
            print(
                f"get_book(cond_id): tick_size={book.get('tick_size')} "
                f"min_order_size={book.get('min_order_size')} best_bid={best_bid} best_ask={best_ask}"
            )

        events = client.list_events(limit=2)
        print(f"list_events(limit=2): {len(events)} events")
        if events:
            print("event #0 keys:", list(events[0].keys()))
            print("event #0 preview:", json.dumps(events[0], indent=2)[:1000])


def verify_gemini_api() -> None:
    print("\n=== 2. Verifying Google Gemini API ===")
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    print(f"Using GEMINI_API_KEY: {mask_key(api_key)}")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in .env")

    client = genai.Client(
        vertexai=False,
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )
    candidates = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]
    working: list[str] = []
    for model_id in candidates:
        try:
            print(f"Testing vertexai=False with model={model_id} ...", flush=True)
            resp = client.models.generate_content(
                model=model_id,
                contents='Return strictly JSON: {"status": "ok", "model": "' + model_id + '"}',
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            print(f"[SUCCESS] {model_id}: {resp.text}", flush=True)
            working.append(model_id)
        except Exception as exc:
            print(f"[FAIL] {model_id}: {exc}", flush=True)
    if not working:
        raise RuntimeError("No candidate Gemini Flash models responded!")
    print(f"Working Flash models in priority order: {working}", flush=True)

    print("Verify done.")


if __name__ == "__main__":
    verify_polysim_sdk()
    verify_gemini_api()
