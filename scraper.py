"""
scraper.py — TruthBot Dataset Builder
======================================
Scrapes ALL articles from Sebenarnya.my and saves them as a JSON dataset.
This builds the local knowledge base used by LangChain's retriever.

Run ONCE to build the initial dataset.
Re-run weekly to pick up new articles — already-scraped URLs are skipped.

Usage:
    python scraper.py

Output:
    sebenarnya_articles.json
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
import logging
from typing import List, Optional, Dict

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BASE_URL    = "https://sebenarnya.my"
HEADERS     = {"User-Agent": "TruthBot-Scraper/2.0 (Academic Research, UNIMAS)"}
OUTPUT_FILE = "sebenarnya_articles.json"
MAX_PAGES   = 200

# Delay settings to be respectful to the server
DELAY_BETWEEN_PAGES = 1.0  # seconds
DELAY_BETWEEN_ARTICLES = 0.5  # seconds


def get_article_links_from_page(page_num: int) -> List[str]:
    """
    Fetch one listing page and return all article URLs on it.
    
    Args:
        page_num: Page number to fetch (1 = homepage, 2+ = /page/N/)
    
    Returns:
        List of unique article URLs found on the page
    """
    try:
        # Build URL for the page
        if page_num == 1:
            url = BASE_URL
        else:
            url = f"{BASE_URL}/page/{page_num}/"
        
        logger.debug(f"Fetching listing page: {url}")
        
        r = requests.get(url, headers=HEADERS, timeout=10)
        
        # 404 means we've reached the end
        if r.status_code == 404:
            logger.info(f"Page {page_num} returned 404 — no more pages")
            return []
        
        # Other error codes
        if r.status_code != 200:
            logger.warning(f"Page {page_num} returned status {r.status_code}")
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        
        # Try primary selectors
        for a in soup.select("h2.entry-title a, h1.entry-title a, .post-title a"):
            href = a.get("href", "")
            if href and BASE_URL in href:
                links.append(href)
        
        # Fallback selectors if primary ones don't work
        if not links:
            logger.debug(f"Primary selectors found no links. Trying fallback selectors...")
            for a in soup.select("a[href*='/']"):
                href = a.get("href", "")
                if href and BASE_URL in href and "sebenarnya.my" in href and not href.endswith(("/", "/page/")):
                    # Filter out non-article pages
                    if any(pattern not in href.lower() for pattern in ["category", "tag", "author", "search"]):
                        links.append(href)
        
        # Remove duplicates
        unique_links = list(set(links))
        logger.debug(f"Found {len(unique_links)} unique links on page {page_num}")
        
        return unique_links
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching page {page_num}")
        return []
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error on page {page_num}: {e}")
        return []
    except Exception as e:
        logger.error(f"Error fetching listing page {page_num}: {e}", exc_info=True)
        return []


def scrape_article(url: str) -> Optional[Dict]:
    """
    Fetch one article and extract its title, date, and full body text.
    
    Args:
        url: Full URL of the article to scrape
    
    Returns:
        Dictionary with url, title, date, content, or None if failed
    """
    try:
        logger.debug(f"Scraping article: {url}")
        
        r = requests.get(url, headers=HEADERS, timeout=10)
        
        if r.status_code != 200:
            logger.warning(f"Article returned status {r.status_code}: {url}")
            return None
        
        soup = BeautifulSoup(r.text, "html.parser")

        # ── Extract title ──────────────────────────────────────────
        title_tag = soup.select_one("h1.entry-title, h2.entry-title, .post-title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        if not title:
            logger.warning(f"No title found for: {url}")
            return None

        # ── Extract date ──────────────────────────────────────────
        date_tag = soup.select_one("time.entry-date, .post-date, time")
        date = date_tag.get("datetime", "") if date_tag else ""

        # ── Extract body ──────────────────────────────────────────
        # Try multiple body selectors
        body_tag = soup.select_one(".entry-content, .post-content, article, main, [role='main']")
        
        if not body_tag:
            logger.warning(f"No content body found for: {url}")
            return None

        # ── Remove unwanted elements ───────────────────────────────
        for tag in body_tag.select(
            "script, style, nav, footer, .sharedaddy, .jp-relatedposts, .related-posts, "
            ".advertisement, .ads, [class*='sidebar'], [class*='widget'], iframe"
        ):
            tag.decompose()

        # ── Get clean text ────────────────────────────────────────
        body = body_tag.get_text(separator=" ", strip=True)

        # ── Validate extracted content ────────────────────────────
        if not body or len(body) < 100:
            logger.warning(f"Content too short ({len(body)} chars): {url}")
            return None

        # ── Create article object ─────────────────────────────────
        article = {
            "url": url,
            "title": title,
            "date": date,
            "content": body[:6000]  # Cap at 6000 chars
        }

        logger.debug(f"✓ Successfully scraped: {title[:50]}...")
        return article

    except requests.exceptions.Timeout:
        logger.error(f"Timeout scraping: {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error scraping {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}", exc_info=True)
        return None


def load_existing_articles() -> tuple[set, list]:
    """
    Load previously scraped articles to avoid duplicates.
    
    Returns:
        Tuple of (set of existing URLs, list of existing articles)
    """
    existing_urls = set()
    existing_articles = []
    
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                existing_articles = json.load(f)
                existing_urls = {a.get("url", "") for a in existing_articles if a.get("url")}
            
            logger.info(f"✓ Loaded {len(existing_articles)} existing articles from '{OUTPUT_FILE}'")
            logger.info(f"  Will skip these {len(existing_urls)} URLs to avoid re-scraping")
        except json.JSONDecodeError:
            logger.error(f"'{OUTPUT_FILE}' is corrupted. Starting fresh.")
            existing_articles = []
            existing_urls = set()
        except Exception as e:
            logger.error(f"Error loading existing articles: {e}", exc_info=True)
            existing_articles = []
            existing_urls = set()
    else:
        logger.info(f"No existing dataset. Starting fresh.")
    
    return existing_urls, existing_articles


def save_articles(articles: list) -> bool:
    """
    Save articles to JSON file.
    
    Args:
        articles: List of article dictionaries
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved {len(articles)} articles to '{OUTPUT_FILE}'")
        return True
    except Exception as e:
        logger.error(f"Error saving articles: {e}", exc_info=True)
        return False


def run_scraper():
    """Main scraper routine."""
    
    logger.info("=" * 70)
    logger.info("TruthBot — Sebenarnya.my Scraper")
    logger.info("=" * 70)
    logger.info(f"Base URL: {BASE_URL}")
    logger.info(f"Max pages to scan: {MAX_PAGES}")
    logger.info(f"Output file: {OUTPUT_FILE}")
    logger.info("=" * 70)

    # ── Step 1: Load existing articles ─────────────────────────────
    existing_urls, existing_articles = load_existing_articles()

    # ── Step 2: Scan all listing pages for article links ───────────
    logger.info("\nStep 1: Scanning listing pages for article links...")
    logger.info("-" * 70)
    
    all_links = []
    empty_page_count = 0
    
    for page in range(1, MAX_PAGES + 1):
        logger.info(f"Scanning page {page}/{MAX_PAGES}...")
        links = get_article_links_from_page(page)
        
        if not links:
            empty_page_count += 1
            # Stop after 3 consecutive empty pages (graceful termination)
            if empty_page_count >= 3:
                logger.info(f"No articles found on last 3 pages. Stopping scan.")
                break
        else:
            empty_page_count = 0  # Reset counter on successful page
            all_links.extend(links)
        
        time.sleep(DELAY_BETWEEN_PAGES)

    all_links = list(set(all_links))  # Remove duplicates
    logger.info(f"✓ Scan complete. Found {len(all_links)} unique article links")

    # ── Step 3: Filter to only new articles ────────────────────────
    logger.info("\nStep 2: Filtering new articles...")
    logger.info("-" * 70)
    
    new_links = [url for url in all_links if url not in existing_urls]
    
    logger.info(f"Total links found: {len(all_links)}")
    logger.info(f"Already scraped: {len(existing_urls)}")
    logger.info(f"New to scrape: {len(new_links)}")

    # ── Step 4: Scrape new articles ────────────────────────────────
    logger.info(f"\nStep 3: Scraping {len(new_links)} new articles...")
    logger.info("-" * 70)
    
    new_articles = []
    failed_urls = []
    
    for i, url in enumerate(new_links, 1):
        logger.info(f"[{i}/{len(new_links)}] Scraping: {url[:60]}...")
        
        article = scrape_article(url)
        
        if article:
            new_articles.append(article)
            logger.debug(f"✓ Success: {article['title'][:50]}...")
        else:
            failed_urls.append(url)
            logger.warning(f"✗ Failed to scrape: {url}")
        
        time.sleep(DELAY_BETWEEN_ARTICLES)

    # ── Step 5: Merge and save ─────────────────────────────────────
    logger.info("\nStep 4: Merging and saving...")
    logger.info("-" * 70)
    
    all_articles = existing_articles + new_articles
    
    if save_articles(all_articles):
        logger.info("✓ Successfully saved dataset")
    else:
        logger.error("✗ Failed to save dataset")
        return

    # ── Final Summary ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"📊 Statistics:")
    logger.info(f"   New articles scraped: {len(new_articles)}")
    logger.info(f"   Failed to scrape: {len(failed_urls)}")
    logger.info(f"   Total in dataset: {len(all_articles)}")
    logger.info(f"   Saved to: {OUTPUT_FILE}")
    logger.info("=" * 70)
    
    if failed_urls:
        logger.warning(f"\nFailed URLs ({len(failed_urls)}):")
        for url in failed_urls[:10]:  # Show first 10
            logger.warning(f"  - {url}")
        if len(failed_urls) > 10:
            logger.warning(f"  ... and {len(failed_urls) - 10} more")


if __name__ == "__main__":
    try:
        run_scraper()
    except KeyboardInterrupt:
        logger.info("\nScraping interrupted by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)