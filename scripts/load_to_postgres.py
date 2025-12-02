# scripts/load_to_postgres.py  ← FINAL, GUARANTEED TO WORK
import psycopg2
import pandas as pd
import os
print(os.path.exists("../data/processed/cleaned_reviews_with_themes.csv"))

# Load data
df = pd.read_csv("../data/processed/cleaned_reviews_with_themes.csv")
print(f"Loaded {len(df)} rows from CSV")

# Fix types
df["review_date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df["rating"] = df["rating"].astype(int)



# Bank mapping
bank_map = {"CBE": 1, "BOA": 2, "DASHEN": 3}
df["bank_id"] = df["bank"].map(bank_map)


# Connection
conn = psycopg2.connect(
    host="localhost",
    database="bank_reviews",
    user="postgres",
    password="abcd1234",
    port=5432
)
cur = conn.cursor()

# TRUNCATE FIRST (so we know it's clean)
cur.execute("TRUNCATE TABLE reviews RESTART IDENTITY")
conn.commit()
print("Truncated reviews table")

# NORMAL INSERT — NOT execute_values (this is the fix!)
insert_sql = """
    INSERT INTO reviews (
        bank_id, review_text, rating, review_date,
        sentiment_label, sentiment_score, sentiment_confidence,
        processed_review, theme, source
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

inserted = 0
for _, row in df.iterrows():
    cur.execute(insert_sql, (
        int(row["bank_id"]),
        str(row["review"]),
        int(row["rating"]),
        row["review_date"],
        str(row["sentiment"]),
        float(row["sentiment_score"]) if pd.notna(row["sentiment_score"]) else None,
        float(row["confidence"]) if pd.notna(row["confidence"]) else None,
        str(row["processed_review"]) if pd.notna(row["processed_review"]) else "",
        str(row["theme"]),
        "Google Play Store"
    ))
    inserted += 1
    if inserted % 100 == 0:
        print(f"Inserted {inserted} rows...")

conn.commit()
print(f"\nSUCCESS: {inserted} rows inserted!")

# Verify
cur.execute("SELECT COUNT(*) FROM reviews")
count = cur.fetchone()[0]
print(f"Verification to validate: reviews table now has {count} rows")

cur.close()
conn.close()
