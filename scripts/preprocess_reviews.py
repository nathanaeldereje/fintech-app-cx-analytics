import pandas as pd
import re

def remove_duplicates(df):
    """
    Remove duplicate entries based on reviewId or review text.

    Parameters:
        df (DataFrame): Raw scraped reviews.

    Returns:
        DataFrame: Reviews with duplicates removed.
    """
    before = len(df)

    # Drop duplicates using reviewId and review content
    df = df.drop_duplicates(subset=["reviewId", "content"], keep="first")

    after = len(df)
    print(f"🔹 Removed {before - after} duplicate reviews.")
    return df


def remove_empty_reviews(df):
    """
    Remove reviews where the content is empty or missing.

    Parameters:
        df (DataFrame): Reviews.

    Returns:
        DataFrame: Reviews with empty text removed.
    """
    before = len(df)

    # Remove rows with NaN or whitespace-only text
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip() != ""]

    after = len(df)
    print(f"🔹 Removed {before - after} empty reviews.")
    return df


def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'at' column to proper YYYY-MM-DD datetime64[ns] (without time).
    Safe for PostgreSQL, plotting, and analysis.
    """
    df["date"] = pd.to_datetime(df["at"], errors="coerce").dt.normalize()
    print(f"Date normalized → {df['date'].dtype}")
    return df

def standardize_bank_names(df):
    """
    Ensure all bank names follow a consistent uppercase format.

    Parameters:
        df (DataFrame): Reviews.

    Returns:
        DataFrame: Reviews with standardized bank names.
    """
    df["bank"] = df["bank"].str.strip().str.upper()
    print("🔹 Standardized bank names.")
    return df



def remove_non_latin_reviews(df):
    """
    Remove reviews containing non-Latin scripts (Amharic, Arabic, etc.)
    Keeps only English + basic punctuation + emojis.
    """
    before = len(df)
    
    # Anything outside basic ASCII = non-Latin script
    non_latin_mask = df["content"].fillna("").astype(str).str.contains(r'[^\x00-\x7F]', regex=True)
    
    df = df[~non_latin_mask].copy()
    
    after = len(df)
    print(f"Removed {before - after} reviews with non-Latin characters (Amharic/Arabic/etc)")
    print(f"→ Kept {after:,} clean English reviews for accurate sentiment analysis")
    
    return df

def select_required_columns(df):
    """
    Keep only the required columns:
    review, rating, date, bank, source.

    Parameters:
        df (DataFrame): Fully preprocessed reviews.

    Returns:
        DataFrame: Final clean dataset.
    """
    # Map original columns to final standardized names
    df["review"] = df["content"]
    df["rating"] = df["score"]
    df["source"] = "Google Play Store"
    print("🔹 Selected required final columns.")
    return df[["review", "rating", "date", "bank", "source"]]


def preprocess_pipeline(df):
    """
    Full preprocessing pipeline:
    1. Load raw data
    2. Remove duplicates
    3. Remove empty reviews
    4. Normalize dates
    5. Standardize bank names
    6. Select and rename final columns
    7. Save clean dataset

    Usage:
        preprocess_pipeline("../data/raw_reviews_2025.csv")
    """
    print("Starting preprocessing pipeline...\n")

    print(f"Raw data loaded: {len(df):,} reviews")

    # 1–5. Apply all cleaning steps
    df = remove_duplicates(df)
    df = remove_empty_reviews(df)
    df = remove_non_latin_reviews(df)
    df = normalize_dates(df)
    df = standardize_bank_names(df)
    df = select_required_columns(df)
    
    return df 