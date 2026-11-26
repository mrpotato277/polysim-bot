"""News radar fetching and deduplicating recent Google News RSS articles per market."""

from __future__ import annotations

import html
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

import feedparser
import httpx

STOPWORDS = {
    "will",
    "the",
    "be",
    "a",
    "an",
    "in",
    "on",
    "at",
    "by",
    "before",
    "after",
    "to",
    "of",
    "for",
    "from",
    "with",
    "or",
    "and",
    "is",
    "are",
    "was",
    "were",
    "this",
    "that",
    "does",
    "do",
    "did",
    "has",
    "have",
    "had",
    "any",
    "over",
    "under",
    "than",
    "more",
    "less",
    "between",
    "during",
    "next",
    "end",
    "win",
    "lose",
    "yes",
    "no",
    "2025",
    "2026",
    "2027",
    "2028",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def build_news_query(question: str, event_title: str | None = None) -> str:
    """Extract crisp keywords from a Polymarket question for Google News RSS search."""
    text = question.strip()
    tokens = re.findall(r"[\$€£%]?[^\W_]+(?:[,\.\-'\$/%][^\W_]+)*", text)
    keywords: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        clean = tok.strip(".,?!:;\"'()[]")
        low = clean.lower()
        if not clean or len(clean) < 2 or low in STOPWORDS:
            continue
        if low not in seen:
            seen.add(low)
            keywords.append(clean)
        if len(keywords) >= 7:
            break

    if len(keywords) < 2 and event_title:
        for tok in re.findall(r"[^\W_]+(?:[\.\-'][^\W_]+)*", event_title):
            clean = tok.strip(".,?!:;\"'()[]")
            low = clean.lower()
            if clean and low not in STOPWORDS and low not in seen:
                seen.add(low)
                keywords.append(clean)
                if len(keywords) >= 6:
                    break

    return " ".join(keywords) if keywords else question[:80]


_SPECIALIST_FEED_CACHE: dict[str, list[dict[str, Any]]] = {}

SPECIALIST_FEEDS: dict[str, list[tuple[str, str]]] = {
    "economics": [
        ("Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
        (
            "CNBC Economy",
            "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
        ),
        ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ],
    "crypto": [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Cointelegraph", "https://cointelegraph.com/rss"),
    ],
    "politics": [
        ("NPR Politics", "https://feeds.npr.org/1014/rss.xml"),
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ],
    "geopolitics": [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ],
    "tech": [
        ("TechCrunch", "https://techcrunch.com/feed/"),
        ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ],
    "sports": [
        ("ESPN", "https://www.espn.com/espn/rss/news"),
        ("ESPN NBA", "https://www.espn.com/espn/rss/nba/news"),
    ],
}


def _strip_html(raw: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", raw or "")
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def _parse_pub_dt(entry: Any) -> datetime | None:
    published = entry.get("published") or entry.get("updated")
    if not published:
        return None
    try:
        dt = parsedate_to_datetime(published)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _age_sort_key(art: dict[str, Any]) -> float:
    age = art.get("age_hours")
    return float("inf") if age is None else float(age)


def _parse_feed_entries(
    feed: Any,
    *,
    default_source: str = "",
    max_age_hours: int | None = 48,
) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours) if max_age_hours is not None else None
    articles: list[dict[str, Any]] = []

    for entry in getattr(feed, "entries", []) or []:
        pub_dt = _parse_pub_dt(entry)
        if cutoff is not None and pub_dt is not None and pub_dt < cutoff:
            continue
        if pub_dt is None:
            age_hours: float | None = None
        elif pub_dt > now:
            age_hours = 0.0
        else:
            age_hours = round((now - pub_dt).total_seconds() / 3600.0, 2)
        source_obj = entry.get("source")
        source_name = (
            source_obj.get("title") if isinstance(source_obj, dict) else ""
        ) or default_source
        title = _strip_html(entry.get("title", ""))
        snippet = _strip_html(entry.get("summary", ""))
        if not title:
            continue
        articles.append(
            {
                "title": title[:240],
                "source": str(source_name or "")[:80],
                "published_at": pub_dt.isoformat() if pub_dt else "",
                "age_hours": age_hours,
                "link": str(entry.get("link", ""))[:500],
                "snippet": snippet[:280],
            }
        )

    articles.sort(key=_age_sort_key)
    return articles


def fetch_google_news(
    query: str,
    *,
    max_age_hours: int = 48,
    max_items: int = 8,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Fetch recent news articles from Google News RSS without using LLM tokens."""
    if not query.strip():
        return []
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PolySimBot/1.0)"})
            if resp.status_code != 200:
                return []
            feed = feedparser.parse(resp.text)
    except Exception:
        return []

    return _parse_feed_entries(feed, max_age_hours=max_age_hours)[:max_items]


def _tokenize_for_match(text: str) -> set[str]:
    """Extract lowercase Unicode alphanumeric tokens (len >= 3, excluding stopwords)."""
    normalized = re.sub(r"(?<=\d),(?=\d)", "", (text or "").lower())
    return {
        tok
        for tok in re.findall(r"[^\W_]+", normalized)
        if len(tok) >= 3 and tok not in STOPWORDS
    }


def _fetch_single_rss_feed(
    source_label: str,
    url: str,
    *,
    max_age_hours: int = 48,
    timeout: float = 6.0,
) -> list[dict[str, Any]]:
    """Fetch and cache an RSS feed once per process, returning items within max_age_hours."""
    cached = _SPECIALIST_FEED_CACHE.get(url)
    if cached is None:
        cached = []
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PolySimBot/1.0)"})
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.text)
                    cached = _parse_feed_entries(feed, default_source=source_label, max_age_hours=None)
        except Exception:
            cached = []
        _SPECIALIST_FEED_CACHE[url] = cached

    return [
        art
        for art in cached
        if art.get("age_hours") is None or float(art["age_hours"]) <= float(max_age_hours)
    ]


def fetch_specialist_news(
    query: str,
    category: str,
    *,
    max_age_hours: int = 48,
    max_items: int = 6,
    timeout: float = 6.0,
) -> list[dict[str, Any]]:
    """Match category-specific specialist RSS articles against market query tokens."""
    feeds = SPECIALIST_FEEDS.get((category or "").strip().lower(), [])
    q_tokens = _tokenize_for_match(query)
    if not feeds or not q_tokens:
        return []

    min_overlap = 1 if len(q_tokens) <= 2 else 2
    scored: list[tuple[int, float, dict[str, Any]]] = []

    for source_label, url in feeds:
        items = _fetch_single_rss_feed(
            source_label,
            url,
            max_age_hours=max_age_hours,
            timeout=timeout,
        )
        for art in items:
            haystack = f"{art.get('title', '')} {art.get('snippet', '')}"
            overlap = len(q_tokens & _tokenize_for_match(haystack))
            if overlap >= min_overlap:
                scored.append((overlap, -_age_sort_key(art), art))

    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [row[2] for row in scored[:max_items]]


def fetch_market_news(
    query: str,
    *,
    category: str = "other",
    max_age_hours: int = 48,
    max_items: int = 10,
    timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Combine Google News RSS and category-specific specialist feeds, deduplicated by title."""
    combined = fetch_google_news(
        query,
        max_age_hours=max_age_hours,
        max_items=max_items,
        timeout=timeout,
    ) + fetch_specialist_news(
        query,
        category,
        max_age_hours=max_age_hours,
        max_items=min(6, max_items),
        timeout=min(6.0, timeout),
    )

    combined.sort(key=_age_sort_key)
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for art in combined:
        norm_title = re.sub(r"[\W_]+", "", str(art.get("title") or "").lower())[:90]
        if not norm_title or norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)
        deduped.append(art)
        if len(deduped) >= max_items:
            break
    return deduped
