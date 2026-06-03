"""
TruthBot Dataset Builder
Scrape ALL Sebenarnya.my articles using WordPress REST API

Output:
    sebenarnya_articles.json
"""

import requests
import json
import time
import logging
from bs4 import BeautifulSoup

# ==================================================
# CONFIG
# ==================================================

API_URL = "https://sebenarnya.my/wp-json/wp/v2/posts"

OUTPUT_FILE = "sebenarnya_articles.json"

PER_PAGE = 100

DELAY = 1

HEADERS = {
    "User-Agent": "TruthBot Academic Research (UNIMAS)"
}

# ==================================================
# LOGGING
# ==================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==================================================
# CLEAN HTML
# ==================================================

def clean_html(html):

    if not html:
        return ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = soup.get_text(
        separator=" ",
        strip=True
    )

    return text

# ==================================================
# FETCH POSTS
# ==================================================

def fetch_posts(page):

    params = {
        "page": page,
        "per_page": PER_PAGE
    }

    try:

        response = requests.get(
            API_URL,
            params=params,
            headers=HEADERS,
            timeout=60
        )

        if response.status_code == 400:
            return []

        response.raise_for_status()

        return response.json()

    except Exception as e:

        logger.error(
            f"Failed page {page}: {e}"
        )

        return []

# ==================================================
# MAIN
# ==================================================

def run_scraper():

    logger.info("=" * 60)
    logger.info("TRUTHBOT DATASET SCRAPER")
    logger.info("=" * 60)

    all_articles = []

    seen_urls = set()

    page = 1

    while True:

        logger.info(
            f"Fetching page {page}"
        )

        posts = fetch_posts(page)

        if not posts:

            logger.info(
                "Reached end of website."
            )

            break

        added = 0

        for post in posts:

            try:

                url = post.get(
                    "link",
                    ""
                )

                if not url:
                    continue

                if url in seen_urls:
                    continue

                seen_urls.add(url)

                title = clean_html(
                    post.get(
                        "title",
                        {}
                    ).get(
                        "rendered",
                        ""
                    )
                )

                content = clean_html(
                    post.get(
                        "content",
                        {}
                    ).get(
                        "rendered",
                        ""
                    )
                )

                if len(content) < 50:
                    continue

                article = {

                    "id": post.get("id"),

                    "title": title,

                    "date": post.get(
                        "date",
                        ""
                    ),

                    "url": url,

                    "content": content

                }

                all_articles.append(
                    article
                )

                added += 1

            except Exception as e:

                logger.error(
                    f"Error processing article: {e}"
                )

        logger.info(
            f"Added {added} articles from page {page}"
        )

        logger.info(
            f"Current total: {len(all_articles)}"
        )

        page += 1

        time.sleep(DELAY)

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
        f"Total Articles: {len(all_articles)}"
    )
    logger.info(
        f"Saved to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    run_scraper()