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
    # Remove trailing date / threshold boilerplate if clean enough, keep proper nouns & core verbs
    tokens = re.findall(r"[A-Za-z0-9$%.\-']+", text)
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
        for tok in re.findall(r"[A-Za-z0-9.\-']+", event_title):
            clean = tok.strip(".,?!:;\"'()[]")
            low = clean.lower()
            if clean and low not in STOPWORDS and low not in seen:
                seen.add(low)
                keywords.append(clean)
                if len(keywords) >= 6:
                    break

    return " ".join(keywords) if keywords else question[:80]


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

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max_age_hours)
    articles: list[dict[str, Any]] = []

    for entry in feed.entries:
        pub_dt = _parse_pub_dt(entry)
        if pub_dt is not None and pub_dt < cutoff:
            continue
        age_hours = round((now - pub_dt).total_seconds() / 3600.0, 2) if pub_dt else 24.0
        source_obj = entry.get("source")
        source_name = source_obj.get("title") if isinstance(source_obj, dict) else ""
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

    articles.sort(key=lambda a: a["age_hours"])
    return articles[:max_items]
