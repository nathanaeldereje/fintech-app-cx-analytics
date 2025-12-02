# scripts/utils/insights.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Ensure output directory exists
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 1. Rating & sentiment distributions
def plot_rating_distribution(df, out_dir="reports/figures/"):
    ensure_dir(out_dir)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="rating", hue="bank", order=[1,2,3,4,5])
    plt.title("Rating Distribution per Bank")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}rating_distribution_per_bank.png", bbox_inches='tight')
    plt.close()

def plot_sentiment_distribution(df, out_dir="reports/figures/"):
    ensure_dir(out_dir)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="sentiment", hue="bank")
    plt.title("Sentiment Distribution per Bank")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{out_dir}sentiment_distribution_per_bank.png", bbox_inches='tight')
    plt.close()

# 2. Top themes per bank
def plot_top_themes_per_bank(df, out_dir="reports/figures/", top_n=7):
    ensure_dir(out_dir)
    df_ex = df.explode("theme")
    counts = df_ex.groupby(["bank", "theme"]).size().reset_index(name="count")
    
    for bank in df["bank"].unique():
        top = counts[counts["bank"] == bank].nlargest(top_n, "count")
        if top.empty: continue
        plt.figure(figsize=(10, 5))
        sns.barplot(data=top, y="theme", x="count", palette="viridis")
        plt.title(f"Top {top_n} Themes — {bank}")
        plt.tight_layout()
        plt.savefig(f"{out_dir}top_themes_{bank}.png", bbox_inches='tight')
        plt.close()

# 3. Monthly sentiment trend
def plot_sentiment_trends(df, out_dir="reports/figures/"):
    ensure_dir(out_dir)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    trend = df.groupby(["bank", "month"])["sentiment_score"].mean().reset_index()
    
    for bank in df["bank"].unique():
        data = trend[trend["bank"] == bank]
        if data.empty: continue
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=data, x="month", y="sentiment_score", marker="o")
        plt.title(f"Monthly Average Sentiment — {bank}")
        plt.xticks(rotation=45)
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{out_dir}sentiment_trend_{bank}.png", bbox_inches='tight')
        plt.close()

# 4. Negative keywords via TF-IDF
def print_negative_keywords(df, top_n=15):
    vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words='english')
    for bank in df["bank"].unique():
        neg = df[(df["bank"] == bank) & (df["sentiment"] == "negative")]
        texts = neg["review"].fillna("").tolist()
        if len(texts) < 5:
            print(f"\n{bank}: Not enough negative reviews")
            continue
        X = vec.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        terms = np.array(vec.get_feature_names_out())
        top_idx = scores.argsort()[-top_n*2:][::-1]
        print(f"\nTop {top_n} negative keywords for {bank}:")
        print(", ".join(terms[top_idx][:top_n]))

# 5. Sentiment by theme → CSV
def save_sentiment_by_theme(df, path="reports/sentiment_by_theme.csv"):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    df_ex = df.explode("theme").copy()
    df_ex["sentiment_score"] = pd.to_numeric(df_ex["sentiment_score"], errors='coerce')
    result = df_ex.groupby(["bank", "theme"])["sentiment_score"].mean().round(4).reset_index()
    result.to_csv(path, index=False)
    print(f"Saved sentiment by theme → {path}")

    # --- START: additions to satisfy Task 4 -----------------------
from collections import defaultdict
import csv
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

# safe check for required columns
def validate_df(df):
    required = {"bank","rating","sentiment","sentiment_score","review","date","theme"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors='coerce')
    return df

# 6. Extract drivers and pain points with counts + example quotes
def extract_drivers_and_painpoints(df, out_path="reports/drivers_and_painpoints.csv", min_examples=3, top_k_themes=3):
    """
    For each bank: pick top positive themes (drivers) and top negative themes (pain points)
    based on mean sentiment_score and counts. Save CSV with: bank, category(driver|pain), theme, count, mean_sentiment, examples (joined).
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    df = validate_df(df)
    # explode themes
    df_ex = df.explode("theme").copy()
    df_ex["sentiment_score"] = pd.to_numeric(df_ex["sentiment_score"], errors='coerce')
    # aggregate
    agg = df_ex.groupby(["bank","theme"]).agg(
        count=("review", "size"),
        mean_sentiment=("sentiment_score", "mean")
    ).reset_index()
    rows = []
    for bank in df["bank"].unique():
        bank_agg = agg[agg["bank"]==bank].dropna(subset=["theme"])
        if bank_agg.empty:
            continue
        # drivers: high mean_sentiment, require some minimum count
        drivers = bank_agg[bank_agg["count"]>=1].sort_values(["mean_sentiment","count"], ascending=[False, False]).head(top_k_themes)
        pain = bank_agg[bank_agg["count"]>=1].sort_values(["mean_sentiment","count"], ascending=[True, False]).head(top_k_themes)
        # helper to get example quotes
        for category, df_sel in [("driver", drivers), ("pain", pain)]:
            for _, r in df_sel.iterrows():
                theme = r["theme"]
                cnt = int(r["count"])
                mean_s = float(r["mean_sentiment"])
                examples = df_ex[(df_ex["bank"]==bank) & (df_ex["theme"]==theme)]["review"].dropna().head(min_examples).tolist()
                # sanitize examples (shorten)
                examples = [ (ex[:300] + "..." ) if len(ex)>300 else ex for ex in examples ]
                rows.append({
                    "bank": bank,
                    "category": category,
                    "theme": theme,
                    "count": cnt,
                    "mean_sentiment": round(mean_s,4),
                    "examples": " ||| ".join(examples)
                })
    # save CSV
    keys = ["bank","category","theme","count","mean_sentiment","examples"]
    with open(out_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved drivers & pain points → {out_path}")
    return pd.DataFrame(rows)

# 7. Generate word clouds (or fallback TF-IDF bar if wordcloud not available)
def generate_keyword_clouds(df, out_dir="reports/figures/", top_n=100):
    ensure_dir(out_dir)
    df = validate_df(df)
    for bank in df["bank"].unique():
        texts = df[(df["bank"]==bank)]["review"].dropna().astype(str).tolist()
        if not texts:
            continue
        joined = " ".join(texts)
        fname = f"{out_dir}wordcloud_{bank}.png"
        if WORDCLOUD_AVAILABLE:
            wc = WordCloud(width=1200, height=600, background_color="white", max_words=top_n)
            wc.generate(joined)
            plt.figure(figsize=(12,6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
        else:
            # fallback: top TF-IDF terms bar plot
            vec = TfidfVectorizer(ngram_range=(1,2), max_features=2000, stop_words='english')
            X = vec.fit_transform(texts)
            scores = np.asarray(X.sum(axis=0)).ravel()
            terms = np.array(vec.get_feature_names_out())
            top_idx = scores.argsort()[-20:][::-1]
            top_terms = terms[top_idx]
            top_scores = scores[top_idx]
            plt.figure(figsize=(10,6))
            sns.barplot(x=top_scores, y=top_terms)
            plt.title(f"Top TF-IDF terms — {bank} (wordcloud package not installed)")
            plt.tight_layout()
            plt.savefig(fname, bbox_inches="tight")
            plt.close()
        print(f"Saved keyword cloud → {fname}")

# 8. Consolidated plots: overall sentiment trend across banks + rating heatmap
def plot_consolidated_figures(df, out_dir="reports/figures/"):
    ensure_dir(out_dir)
    df = validate_df(df)
    # overall monthly sentiment trend (multiple banks on same plot)
    df["month"] = df["date"].dt.to_period("M").astype(str)
    trend = df.groupby(["month","bank"])["sentiment_score"].mean().reset_index()
    plt.figure(figsize=(12,6))
    sns.lineplot(data=trend, x="month", y="sentiment_score", hue="bank", marker="o")
    plt.title("Monthly Average Sentiment — All Banks")
    plt.xticks(rotation=45)
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.savefig(f"{out_dir}sentiment_trend_all_banks.png", bbox_inches="tight")
    plt.close()
    # rating heatmap: banks vs rating counts
    rating_table = df.pivot_table(index="bank", columns="rating", values="review", aggfunc="count", fill_value=0)
    plt.figure(figsize=(8, max(4, 0.6*len(rating_table))))
    sns.heatmap(rating_table, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Rating counts per Bank (heatmap)")
    plt.tight_layout()
    plt.savefig(f"{out_dir}rating_heatmap_per_bank.png", bbox_inches="tight")
    plt.close()
    print(f"Saved consolidated figures → {out_dir}")

# 9. Ethics / bias note saved to file
def save_ethics_note(path="reports/ethics_note.txt"):
    ensure_dir(os.path.dirname(path) or ".")
    note = (
        "Potential biases and limitations:\n"
        "- Selection bias: reviews come from a subset of users who choose to leave reviews (may be more negative).\n"
        "- Platform bias: review distribution depends on the store/platform and its user base.\n"
        "- Temporal bias: older reviews may not reflect current app state; analyze by month/year.\n"
        "- Language and parsing errors: theme extraction depends on the upstream theme-mapping quality.\n"
        "- Sample size: small theme counts are noisy; require a minimum count when making recommendations.\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(note)
    print(f"Saved ethics note → {path}")
    return path

# 10. Convenience runner to produce everything needed for Task 4
def produce_task4_outputs(df, base_out="reports/"):
    """
    Run the recommended pipeline to produce:
      - figures in reports/figures/
      - sentiment_by_theme.csv (existing function)
      - drivers_and_painpoints.csv
      - word clouds
      - consolidated figures
      - ethics note
    """
    df = validate_df(df)
    # existing plots
    plot_rating_distribution(df, out_dir=os.path.join(base_out,"figures/"))
    plot_sentiment_distribution(df, out_dir=os.path.join(base_out,"figures/"))
    plot_top_themes_per_bank(df, out_dir=os.path.join(base_out,"figures/"))
    plot_sentiment_trends(df, out_dir=os.path.join(base_out,"figures/"))
    # new outputs
    save_sentiment_by_theme(df, path=os.path.join(base_out,"sentiment_by_theme.csv"))
    extract_drivers_and_painpoints(df, out_path=os.path.join(base_out,"drivers_and_painpoints.csv"))
    generate_keyword_clouds(df, out_dir=os.path.join(base_out,"figures/"))
    plot_consolidated_figures(df, out_dir=os.path.join(base_out,"figures/"))
    save_ethics_note(path=os.path.join(base_out,"ethics_note.txt"))
    print("Finished producing Task 4 outputs.")
# --- END additions ---------------------------------------------------------
