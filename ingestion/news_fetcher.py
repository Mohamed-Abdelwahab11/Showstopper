"""
Showstopper — News Fetcher
Fetches articles from NewsAPI.org /v2/everything endpoint.
Falls back to bundled sample articles when no API key is available (demo mode).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml

from models.schemas import Article, ArticleSource

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
_SAMPLE_ARTICLES_PATH = Path(__file__).parent.parent / "data" / "sample_articles.json"

with open(_CONFIG_PATH) as f:
    _CFG = yaml.safe_load(f)["ingestion"]

NEWSAPI_BASE_URL = "https://newsapi.org/v2/everything"
NEWSAPI_HEADLINES_URL = "https://newsapi.org/v2/top-headlines"


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)["ingestion"]


def _load_demo_articles(topic: str) -> list[Article]:
    """Load bundled sample articles when no API key is available."""
    if not _SAMPLE_ARTICLES_PATH.exists():
        logger.warning("No sample_articles.json found. Returning empty list.")
        return []

    with open(_SAMPLE_ARTICLES_PATH) as f:
        raw = json.load(f)

    articles = []
    for item in raw:
        src = item.get("source", {})
        articles.append(Article(
            url=item.get("url", ""),
            title=item.get("title", ""),
            description=item.get("description"),
            content=item.get("content"),
            source=ArticleSource(id=src.get("id"), name=src.get("name", "Demo Source")),
            author=item.get("author"),
            published_at=item.get("publishedAt"),
            query_topic=topic,
        ))
    logger.info(f"[DEMO] Loaded {len(articles)} sample articles.")
    return articles


def fetch_articles(
    topic: str,
    api_key: Optional[str] = None,
    max_articles: Optional[int] = None,
    days_back: Optional[int] = None,
    language: Optional[str] = None,
    sort_by: Optional[str] = None,
) -> list[Article]:
    """
    Fetch news articles for a given topic from NewsAPI.org.

    Parameters
    ----------
    topic       : Search query / topic string
    api_key     : NewsAPI key. Reads NEWS_API_KEY env var if not provided.
    max_articles: Maximum articles to return (default from config.yaml)
    days_back   : How many days back to search (default from config.yaml)
    language    : 2-letter language code (default from config.yaml)
    sort_by     : relevancy | popularity | publishedAt

    Returns
    -------
    List of Article objects
    """
    cfg = _load_config()
    key = api_key or os.getenv("NEWS_API_KEY", "")
    max_results = max_articles or cfg.get("max_articles", 100)
    n_days = days_back or cfg.get("days_back", 7)
    lang = language or cfg.get("language", "en")
    sort = sort_by or cfg.get("sort_by", "relevancy")

    if not key:
        logger.warning("NEWS_API_KEY not set — switching to demo mode.")
        return _load_demo_articles(topic)

    from_date = (datetime.utcnow() - timedelta(days=n_days)).strftime("%Y-%m-%d")

    params = {
        "q": topic,
        "apiKey": key,
        "language": lang,
        "sortBy": sort,
        "pageSize": min(max_results, 100),   # NewsAPI max per page
        "from": from_date,
    }

    try:
        logger.info(f"Fetching articles for topic='{topic}' from NewsAPI…")
        response = requests.get(NEWSAPI_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message', 'unknown')}")
            return _load_demo_articles(topic)

        raw_articles = data.get("articles", [])
        logger.info(f"NewsAPI returned {len(raw_articles)} articles.")

        articles: list[Article] = []
        seen_urls: set[str] = set()

        for item in raw_articles:
            url = item.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # Skip removed articles
            if item.get("title") == "[Removed]":
                continue

            src = item.get("source", {})
            articles.append(Article(
                url=url,
                title=item.get("title", ""),
                description=item.get("description"),
                content=item.get("content"),
                source=ArticleSource(
                    id=src.get("id"),
                    name=src.get("name", "Unknown"),
                ),
                author=item.get("author"),
                published_at=item.get("publishedAt"),
                query_topic=topic,
            ))

        logger.info(f"Parsed {len(articles)} unique articles for '{topic}'.")
        return articles

    except requests.RequestException as e:
        logger.error(f"NewsAPI request failed: {e} — falling back to demo mode.")
        return _load_demo_articles(topic)


def fetch_top_headlines(
    topic: str,
    api_key: Optional[str] = None,
    max_articles: int = 20,
    country: str = "us",
) -> list[Article]:
    """
    Fetch top headlines for a topic (faster, more current than /everything).
    Used as a supplement to fetch_articles.
    """
    key = api_key or os.getenv("NEWS_API_KEY", "")
    if not key:
        return []

    params = {
        "q": topic,
        "apiKey": key,
        "country": country,
        "pageSize": min(max_articles, 100),
    }

    try:
        response = requests.get(NEWSAPI_HEADLINES_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            return []

        articles: list[Article] = []
        for item in data.get("articles", []):
            url = item.get("url", "")
            if not url or item.get("title") == "[Removed]":
                continue
            src = item.get("source", {})
            articles.append(Article(
                url=url,
                title=item.get("title", ""),
                description=item.get("description"),
                content=item.get("content"),
                source=ArticleSource(id=src.get("id"), name=src.get("name", "Unknown")),
                author=item.get("author"),
                published_at=item.get("publishedAt"),
                query_topic=topic,
            ))
        logger.info(f"Top headlines: {len(articles)} articles for '{topic}'.")
        return articles

    except requests.RequestException as e:
        logger.warning(f"Headlines fetch failed: {e}")
        return []
