# scripts/theme_extraction.py

from typing import Dict, List
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# LOAD NLP MODEL
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# CUSTOM STOPWORDS
# -----------------------------
SENTIMENT_STOPWORDS = [
    "good", "nice", "great", "excellent", "bad", "poor", "worst", "best",
    "love", "hate", "amazing", "awesome", "terrible", "perfect", "wonderful"
]

DOMAIN_STOPWORDS = [
    "bank", "banking", "app", "application", "mobile", "mobile banking",
    "cbe", "boa", "dashen", "birr", "amole", "ethiopia", "ethiopian"
]


# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
def preprocess_text(text):
    """
    Clean and preprocess review text using spaCy.
    - lowercase
    - tokenization
    - stopword removal
    - keep only N/V/ADJ
    - lemmatization
    """
    doc = nlp(text.lower())

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and token.pos_ in ("NOUN", "VERB", "ADJ")
        and token.lemma_ not in SENTIMENT_STOPWORDS
        and token.lemma_ not in DOMAIN_STOPWORDS
    ]

    return " ".join(tokens)


# -----------------------------
# TF-IDF KEYWORD EXTRACTION
# -----------------------------
def extract_keywords(corpus, top_k=20):
    """
    Extract top TF-IDF keywords from corpus.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,           # appear in at least 3 reviews
        max_df=0.85,        # ignore terms in >85% of docs
        stop_words="english",
        sublinear_tf=True   # better for long reviews
    )
    X = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()
    avg_tfidf = np.asarray(X.mean(axis=0)).flatten()

    top_indices = avg_tfidf.argsort()[::-1][:top_k]

    return [(feature_names[i], float(avg_tfidf[i])) for i in top_indices]


# -----------------------------
# THEME KEYWORDS
# -----------------------------
THEME_KEYWORDS: Dict[str, List[str]] = {
    "Account Access Issues": [
        "login", "log in", "signin", "sign in", "signup", "register",
        "password", "pin", "forgot", "reset", "verify", "verification",
        "otp", "code", "blocked", "locked", "session", "timeout"
    ],
    "Transaction Problems": [
        "transfer", "send", "receive", "payment", "pay", "failed",
        "declined", "pending", "stuck", "error", "transaction", "money",
        "balance", "deducted", "charged", "double", "refund"
    ],
    "App Crashes & Bugs": [
        "crash", "freeze", "hang", "bug", "error", "force close",
        "not working", "stopped", "black screen", "white screen"
    ],
    "Slow Performance": [
        "slow", "lag", "loading", "take long", "delay", "wait",
        "hanging", "buffering", "timeout"
    ],
    "User Interface Issues": [
        "confusing", "complicated", "hard", "difficult", "layout",
        "design", "button", "menu", "find", "navigate"
    ],
    "Missing Features": [
        "feature", "add", "need", "should have", "why no", "no option",
        "statement", "history", "airtime", "bill", "notification"
    ],
    "Customer Support": [
        "support", "help", "contact", "call", "respond", "reply",
        "agent", "branch", "office", "complaint"
    ],
    "Security Concerns": [
        "secure", "safe", "fraud", "scam", "hack", "privacy", "data"
    ],
    "Positive Experience": [
        "easy", "simple", "fast", "smooth", "convenient", "helpful",
        "user friendly", "clear", "better", "improved"
    ]
}


# -----------------------------
# THEME ASSIGNMENT
# -----------------------------
def assign_themes(processed_text):
    """
    Assign the first matching theme.
    """
    if not processed_text.strip():
        return "No Content"

    words = set(processed_text.split())  # use set for O(1) lookup

    # Count matches per theme
    theme_scores = {}
    for theme, keywords in THEME_KEYWORDS.items():
        matches = sum(1 for kw in keywords if kw in words or any(word.startswith(kw) for word in words))
        if matches > 0:
            theme_scores[theme] = matches

    if not theme_scores:
        return "Other"

    # Return highest-scoring theme
    return max(theme_scores, key=theme_scores.get)


# -----------------------------
# FULL PIPELINE (CONVENIENT)
# -----------------------------
def apply_full_thematic_analysis(df):
    """
    Adds 'processed_review' and 'theme' to dataframe.
    """
    """
    Complete thematic analysis pipeline.
    Adds: processed_review, theme, dominant_theme (if multiple)
    """
    df = df.copy()
    
    print("Preprocessing reviews...")
    df["processed_review"] = df["review"].astype(str).apply(preprocess_text)
    
    print("Assigning themes...")
    df["theme"] = df["processed_review"].apply(assign_themes)
    
    # Optional: show distribution
    print("\nTheme Distribution:")
    print(df["theme"].value_counts().head(10))
    
    return df