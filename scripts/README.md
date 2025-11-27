# Scripts – Production Pipeline

These are clean, modular, well-commented Python scripts designed to be run sequentially (or via a main orchestrator).

| Script                     | Description                                      |
|----------------------------|--------------------------------------------------|
| scrape_reviews.py          | Collects 400+ reviews per bank using google-play-scraper |
| preprocess_reviews.py      | Deduplication, date normalization, cleaning     |
| sentiment_analysis.py      | Applies DistilBERT (and optional VADER) + saves results |
| theme_extraction.py        | TF-IDF + spaCy keyword extraction + rule-based theme clustering |
| load_to_postgres.py        | Creates schema and bulk-inserts processed data   |

Run order:
```bash
python scripts/scrape_reviews.py
python scripts/preprocess_reviews.py
python scripts/sentiment_analysis.py
python scripts/theme_extraction.py
python scripts/load_to_postgres.py
```