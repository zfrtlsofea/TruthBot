"""
TruthBot Dataset Builder
Scrapes Sebenarnya.my using WordPress REST API.

Output:
    sebenarnya_articles.json
"""

import requests
import json
import os
import time
import logging
from bs4 import BeautifulSoup

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

API_URL = "https://sebenarnya.my/wp-json/wp/v2/posts"
OUTPUT_FILE = "sebenarnya_articles.json"

PER_PAGE = 100
MAX_PAGES = 100

HEADERS = {
    "User-Agent": "TruthBot Academic Research (UNIMAS)"
}

DELAY = 1

# --------------------------------------------------
# LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------
# LOAD EXISTING DATA
# --------------------------------------------------

def load_existing_articles():
    if not os.path.exists(OUTPUT_FILE):
        logger.info("No existing dataset found.")
        return [], set()

    try:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)

        urls = {a["url"] for a in articles if "url" in a}

        logger.info(
            f"Loaded {len(articles)} existing articles."
        )

        return articles, urls

    except Exception as e:
        logger.error(f"Failed loading dataset: {e}")
        return [], set()


# --------------------------------------------------
# CLEAN HTML
# --------------------------------------------------

def clean_html(html_text):

    soup = BeautifulSoup(html_text, "html.parser")

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    return text


# --------------------------------------------------
# GET POSTS
# --------------------------------------------------

def fetch_posts(page_number):

    params = {
        "page": page_number,
        "per_page": PER_PAGE
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code == 400:
            return []

        response.raise_for_status()

        return response.json()

    except Exception as e:
        logger.error(
            f"Error fetching page {page_number}: {e}"
        )
        return []


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def run_scraper():

    logger.info("=" * 60)
    logger.info("TruthBot Sebenarnya.my Scraper")
    logger.info("=" * 60)

    existing_articles, existing_urls = load_existing_articles()

    new_articles = []

    for page in range(1, MAX_PAGES + 1):

        logger.info(f"Fetching API page {page}")

        posts = fetch_posts(page)

        if not posts:
            logger.info("No more posts found.")
            break

        for post in posts:

            try:

                article_url = post.get("link", "")

                if article_url in existing_urls:
                    continue

                title_html = post.get(
                    "title",
                    {}
                ).get(
                    "rendered",
                    ""
                )

                content_html = post.get(
                    "content",
                    {}
                ).get(
                    "rendered",
                    ""
                )

                title = clean_html(title_html)

                content = clean_html(content_html)

                if len(content) < 100:
                    continue

                article = {
                    "url": article_url,
                    "title": title,
                    "date": post.get("date", ""),
                    "content": content[:6000]
                }

                new_articles.append(article)

                logger.info(
                    f"Added: {title[:80]}"
                )

            except Exception as e:
                logger.error(
                    f"Error processing post: {e}"
                )

        time.sleep(DELAY)

    all_articles = existing_articles + new_articles

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_articles,
            f,
            ensure_ascii=False,
            indent=2
        )

    logger.info("=" * 60)
    logger.info("SCRAPING COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"New Articles: {len(new_articles)}"
    )
    logger.info(
        f"Total Articles: {len(all_articles)}"
    )
    logger.info(
        f"Saved To: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_scraper()