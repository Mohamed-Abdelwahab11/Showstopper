"""
Showstopper — Article Cleaner
Strips HTML, normalizes whitespace, deduplicates articles by URL hash,
and removes newsapi boilerplate (truncation markers, cookie notices, etc.)
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from models.schemas import Article

logger = logging.getLogger(__name__)

# ── Boilerplate patterns to strip from article content ────────
_BOILERPLATE_PATTERNS = [
    r"\[\+\d+ chars\]",                   # NewsAPI content truncation marker
    r"Read (more|full article)\.?",
    r"Subscribe to (continue reading|read more)\.?",
    r"This article (originally )?appeared (in|on) .+",
    r"© \d{4} .+?\. All rights reserved\.",
    r"Sign in to (read|access) .+",
    r"Cookie (Policy|Notice|Settings)",
    r"We use cookies .+?(\.|$)",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE | re.MULTILINE)

# ── Minimum useful content length ─────────────────────────────
MIN_CONTENT_LENGTH = 80


def _strip_html(text: Optional[str]) -> str:
    """Remove all HTML tags and decode HTML entities."""
    if not text:
        return ""
    soup = BeautifulSoup(text, "lxml")
    return soup.get_text(separator=" ", strip=True)


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines to single space."""
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _remove_boilerplate(text: str) -> str:
    """Remove common news article boilerplate patterns."""
    return _BOILERPLATE_RE.sub("", text).strip()


def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def clean_article(article: Article) -> Optional[Article]:
    """
    Clean a single article in place.
    Returns None if article has insufficient content after cleaning.
    """
    title = _normalize_whitespace(_strip_html(article.title))
    description = _normalize_whitespace(_strip_html(article.description or ""))
    content = _normalize_whitespace(_remove_boilerplate(_strip_html(article.content or "")))

    # Merge best available text
    combined = " ".join(filter(None, [title, description, content]))
    if len(combined) < MIN_CONTENT_LENGTH:
        logger.debug(f"Skipping short article: '{title[:60]}…'")
        return None

    return Article(
        url=article.url,
        title=title,
        description=description or None,
        content=content or None,
        source=article.source,
        author=article.author,
        published_at=article.published_at,
        query_topic=article.query_topic,
    )


def clean_articles(articles: list[Article]) -> list[Article]:
    """
    Clean and deduplicate a list of articles.

    Steps:
    1. Strip HTML from all text fields
    2. Remove boilerplate phrases
    3. Normalize whitespace
    4. Deduplicate by URL hash
    5. Filter articles with too little content

    Returns cleaned, deduplicated Article list.
    """
    seen_hashes: set[str] = set()
    cleaned: list[Article] = []

    for article in articles:
        # Deduplicate by URL
        h = _url_hash(article.url)
        if h in seen_hashes:
            logger.debug(f"Duplicate URL skipped: {article.url}")
            continue
        seen_hashes.add(h)

        result = clean_article(article)
        if result is not None:
            cleaned.append(result)

    logger.info(f"Cleaned {len(cleaned)} / {len(articles)} articles (removed {len(articles) - len(cleaned)}).")
    return cleaned
