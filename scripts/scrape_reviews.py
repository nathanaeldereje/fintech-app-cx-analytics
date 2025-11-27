import pandas as pd
from google_play_scraper import reviews, Sort
import time

def scrape_reviews_for_app(bank_name, package_id, total=400, chunk_size=200):
    """
    Scrape Google Play Store reviews for a single app.

    bank_name: str  -> Bank label (CBE, BOA, Dashen)
    package_id: str -> Google Play app package name
    total: int      -> Total number of reviews to fetch
    chunk_size: int -> Number of reviews per request
    """
    print(f"🔹 Scraping {bank_name}...")

    collected = []
    continuation_token = None
    count = 0

    while count < total:
        try:
            result, continuation_token = reviews(
                package_id,
                lang="en",
                country="et", #country region ethiopia
                sort=Sort.NEWEST,
                count=chunk_size,
                filter_score_with=None,
                continuation_token=continuation_token
            )

            # Tag each review
            for r in result:
                r["bank"] = bank_name

            collected.extend(result)
            count += len(result)

            time.sleep(2)  # prevent rate limits
        except Exception as e:
            print(f"❌ Error scraping {bank_name}: {e}")
            break

    print(f"✅ Finished {bank_name} ({len(collected)} reviews)")
    return collected
