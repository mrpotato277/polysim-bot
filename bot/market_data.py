"""Defensive, zero-auth live market data provider (Kraken, Coinbase, FRED, Yahoo Finance).

Fetches objective spot quotes, 30-day realized crypto volatility, and US Fed funds rate
at most once per process via in-memory caches, with silent fallbacks on network errors.
Never overrides risk limits or probability formulas blindly; provides verified facts only.
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

CRYPTO_ASSETS: dict[str, dict[str, Any]] = {
    "BTC": {
        "keywords": ("bitcoin", "btc"),
        "kraken_pair": "XBTUSD",
        "coinbase_pair": "BTC-USD",
    },
    "ETH": {
        "keywords": ("ethereum", "eth"),
        "kraken_pair": "ETHUSD",
        "coinbase_pair": "ETH-USD",
    },
    "SOL": {
        "keywords": ("solana", "sol"),
        "kraken_pair": "SOLUSD",
        "coinbase_pair": "SOL-USD",
    },
}

YAHOO_ASSETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("^GSPC", "S&P 500", ("s&p 500", "s&p", "spx")),
    ("^IXIC", "Nasdaq Composite", ("nasdaq",)),
    ("GC=F", "Gold Futures", ("gold",)),
    ("CL=F", "WTI Crude Oil", ("crude oil", "oil", "wti")),
    ("^VIX", "CBOE VIX", ("vix",)),
)

FED_KEYWORDS: tuple[str, ...] = (
    "fed",
    "fomc",
    "federal reserve",
    "interest rate",
    "rate cut",
    "rate hike",
    "bps",
    "basis points",
)

_USD_THRESHOLD_RE = re.compile(
    r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)(?:\s*([kKmM])(?![a-zA-Z]))?"
)

# Per-run in-memory caches (keyed by symbol / series / ticker)
_CRYPTO_CACHE: dict[str, dict[str, Any] | None] = {}
_OHLC_VOL_CACHE: dict[str, float | None] = {}
_FRED_CACHE: dict[str, tuple[str, float] | None] = {}
_YAHOO_CACHE: dict[str, dict[str, Any] | None] = {}


def _contains_any_keyword(text_lower: str, keywords: tuple[str, ...]) -> bool:
    if not text_lower or not keywords:
        return False
    pattern = r"(?<!\w)(?:" + "|".join(re.escape(k) for k in keywords) + r")(?!\w)"
    return bool(re.search(pattern, text_lower))


def extract_single_usd_threshold(question: str) -> float | None:
    """Return USD threshold ONLY when exactly one valid amount exists in the question."""
    if not question:
        return None
    matches = _USD_THRESHOLD_RE.findall(question)
    if len(matches) != 1:
        return None
    num_raw, suffix = matches[0]
    try:
        val = float(num_raw.replace(",", ""))
    except ValueError:
        return None
    mult = {"k": 1_000.0, "m": 1_000_000.0}.get((suffix or "").lower(), 1.0)
    val *= mult
    return val if val > 0.0 else None


def fetch_crypto_snapshot(symbol: str, timeout: float = 5.0) -> dict[str, Any] | None:
    """Fetch spot price + 24h range for BTC/ETH/SOL via Kraken with Coinbase fallback."""
    sym = symbol.upper()
    if sym in _CRYPTO_CACHE:
        return _CRYPTO_CACHE[sym]

    meta = CRYPTO_ASSETS.get(sym)
    if not meta:
        _CRYPTO_CACHE[sym] = None
        return None

    snap: dict[str, Any] | None = None

    # 1. Primary: Kraken public Ticker
    try:
        url = f"https://api.kraken.com/0/public/Ticker?pair={meta['kraken_pair']}"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            payload = resp.json()
            result = payload.get("result") if not payload.get("error") else None
            if isinstance(result, dict) and result:
                entry = next(iter(result.values()))
                last = float(entry["c"][0])
                low_24h = float(entry["l"][1])
                high_24h = float(entry["h"][1])
                if last > 0:
                    snap = {
                        "symbol": sym,
                        "last": last,
                        "low_24h": low_24h,
                        "high_24h": high_24h,
                        "source": "Kraken",
                    }
    except Exception as exc:
        log.debug("Kraken ticker failed for %s: %s", sym, exc)

    # 2. Backup: Coinbase Exchange public ticker
    if snap is None:
        try:
            url = f"https://api.exchange.coinbase.com/products/{meta['coinbase_pair']}/ticker"
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                payload = resp.json()
                last = float(payload["price"])
                if last > 0:
                    snap = {
                        "symbol": sym,
                        "last": last,
                        "low_24h": None,
                        "high_24h": None,
                        "source": "Coinbase",
                    }
        except Exception as exc:
            log.debug("Coinbase ticker failed for %s: %s", sym, exc)

    _CRYPTO_CACHE[sym] = snap
    return snap


def fetch_crypto_30d_vol(symbol: str, timeout: float = 5.0) -> float | None:
    """Compute annualized 30-day realized log-return volatility from Kraken daily OHLC."""
    sym = symbol.upper()
    if sym in _OHLC_VOL_CACHE:
        return _OHLC_VOL_CACHE[sym]

    meta = CRYPTO_ASSETS.get(sym)
    if not meta:
        _OHLC_VOL_CACHE[sym] = None
        return None

    vol: float | None = None
    try:
        url = f"https://api.kraken.com/0/public/OHLC?pair={meta['kraken_pair']}&interval=1440"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            payload = resp.json()
            result = payload.get("result") if not payload.get("error") else None
            if isinstance(result, dict):
                candles: list[Any] = []
                for k, v in result.items():
                    if k != "last" and isinstance(v, list):
                        candles = v
                        break
                closes = [
                    float(row[4])
                    for row in candles[-31:]
                    if isinstance(row, (list, tuple)) and len(row) > 4 and float(row[4]) > 0
                ]
                if len(closes) >= 10:
                    returns = [
                        math.log(closes[i] / closes[i - 1])
                        for i in range(1, len(closes))
                    ]
                    if len(returns) >= 2:
                        vol = statistics.stdev(returns) * math.sqrt(365.0)
    except Exception as exc:
        log.debug("Kraken OHLC volatility failed for %s: %s", sym, exc)

    _OHLC_VOL_CACHE[sym] = vol
    return vol


def fetch_fed_funds_rate(timeout: float = 5.0) -> tuple[str, float] | None:
    """Fetch latest Effective Federal Funds Rate (DFF) from St. Louis Fed CSV."""
    cache_key = "DFF"
    if cache_key in _FRED_CACHE:
        return _FRED_CACHE[cache_key]

    result: tuple[str, float] | None = None
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            for line in reversed(resp.text.splitlines()):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                date_str, val_str = parts
                if not date_str or date_str.lower() == "observation_date" or val_str in {"", "."}:
                    continue
                try:
                    rate = float(val_str)
                    result = (date_str, rate)
                    break
                except ValueError:
                    continue
    except Exception as exc:
        log.debug("FRED DFF fetch failed: %s", exc)

    _FRED_CACHE[cache_key] = result
    return result


def fetch_yahoo_quote(ticker: str, timeout: float = 5.0) -> dict[str, Any] | None:
    """Fetch quote summary from Yahoo Finance chart API (silently returns None on 429/error)."""
    if ticker in _YAHOO_CACHE:
        return _YAHOO_CACHE[ticker]

    quote_data: dict[str, Any] | None = None
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}?range=5d&interval=1d"
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            payload = resp.json()
            results = (payload.get("chart") or {}).get("result") or []
            if results and isinstance(results[0], dict):
                meta = results[0].get("meta") or {}
                price = float(meta["regularMarketPrice"])
                prev = float(meta.get("chartPreviousClose") or price)
                pct_chg = ((price - prev) / prev * 100.0) if prev > 0 else 0.0
                if price > 0:
                    quote_data = {
                        "ticker": ticker,
                        "price": price,
                        "prev_close": prev,
                        "pct_change": pct_chg,
                    }
    except Exception as exc:
        log.debug("Yahoo quote failed for %s: %s", ticker, exc)

    _YAHOO_CACHE[ticker] = quote_data
    return quote_data


def build_market_data_context(question: str, category: str = "other") -> str:
    """Assemble factual live market reference lines relevant to the question (or '')."""
    q_low = (question or "").strip().lower()
    if not q_low:
        return ""

    lines: list[str] = []

    # 1. Crypto assets (strict word-boundary matching)
    matched_cryptos = [
        sym
        for sym, meta in CRYPTO_ASSETS.items()
        if _contains_any_keyword(q_low, meta["keywords"])
    ]
    single_target = extract_single_usd_threshold(question) if len(matched_cryptos) == 1 else None

    for sym in matched_cryptos:
        snap = fetch_crypto_snapshot(sym)
        if not snap:
            continue
        spot = float(snap["last"])
        vol = fetch_crypto_30d_vol(sym)

        details: list[str] = []
        if snap.get("low_24h") is not None and snap.get("high_24h") is not None:
            details.append(f"24h range: ${snap['low_24h']:,.2f} - ${snap['high_24h']:,.2f}")
        if vol is not None:
            details.append(f"30d realized annualized vol: ~{vol * 100:.0f}%")
        detail_str = f" ({', '.join(details)})" if details else ""
        lines.append(f"{sym}/USD live spot ({snap['source']}): ${spot:,.2f}{detail_str}.")

        if single_target is not None and (0.05 * spot <= single_target <= 20.0 * spot):
            pct_diff = ((single_target - spot) / spot) * 100.0
            lines.append(
                f"Distance from live spot to ${single_target:,.2f}: {pct_diff:+.1f}% "
                "(NOTE: crypto has heavy fat tails and Polymarket resolution source e.g. Binance/Chainlink may differ by ~0.1-0.2%)."
            )

    # 2. US Fed / Macro rates
    if _contains_any_keyword(q_low, FED_KEYWORDS):
        dff = fetch_fed_funds_rate()
        if dff is not None:
            date_str, rate = dff
            lines.append(f"Effective Federal Funds Rate (FRED DFF): {rate:.2f}% (as of {date_str}).")

    # 3. Equity indices / Commodities / VIX
    for ticker, label, kws in YAHOO_ASSETS:
        if _contains_any_keyword(q_low, kws):
            qdata = fetch_yahoo_quote(ticker)
            if qdata:
                lines.append(
                    f"{label} ({ticker}) live reference: {qdata['price']:,.2f} "
                    f"(prev close: {qdata['prev_close']:,.2f}, {qdata['pct_change']:+.2f}%)."
                )

    return "\n".join(lines)
