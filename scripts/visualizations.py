import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud

def plot_bank_cx_insights(df: pd.DataFrame, bank_sentiment: pd.Series):
    """
    Generates a 2x2 grid of visualizations for Bank Customer Experience (CX) insights.

    Args:
        df (pd.DataFrame): The main DataFrame containing review data 
                           (requires 'bank', 'sentiment', 'sentiment_score', 'rating', 'confidence' columns).
        bank_sentiment (pd.Series): A Series where the index is the bank name and 
                                    the values are the calculated average sentiment score.
    """
    # Create the figure and 2x2 grid of axes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Plot 1: Sentiment Distribution per Bank (Top Left) ---
    sns.countplot(data=df, x="bank", hue="sentiment", palette="RdYlGn", ax=axes[0, 0])
    axes[0, 0].set_title("1. Sentiment Distribution by Bank")
    axes[0, 0].set_ylabel("Number of Reviews")
    axes[0, 0].set_xlabel("Bank")
    axes[0, 0].legend(title="Sentiment")

    # --- Plot 2: Average Sentiment Score per Bank (Top Right) ---
    # Sort and plot the bank sentiment Series horizontally
    bank_sentiment.sort_values().plot(
        kind="barh", 
        ax=axes[0, 1], 
        color=["red", "orange", "green"][:len(bank_sentiment)] # Dynamic coloring
    )
    axes[0, 1].set_title("2. Average Sentiment Score per Bank")
    axes[0, 1].set_xlabel("Sentiment Score")
    axes[0, 1].set_ylabel("Bank")

    # --- Plot 3: Sentiment vs Rating Heatmap (Bottom Left) ---
    # Prepare data: Mean sentiment_score for each bank-rating combination
    heatmap_data = df.groupby(["bank", "rating"])["sentiment_score"].mean().unstack()
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", # Format annotation to 2 decimal places
        cmap="RdYlGn", 
        center=0, # Center the color scale at 0 (neutral)
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("3. Sentiment Score vs. Star Rating")
    axes[1, 0].set_ylabel("Bank")
    axes[1, 0].set_xlabel("Star Rating")

    # --- Plot 4: Confidence Distribution (Bottom Right) ---
    # Assuming 'confidence' column exists and is between 0 and 1
    df["confidence"].hist(
        bins=30, 
        ax=axes[1, 1], 
        color="skyblue", 
        edgecolor="black"
    )
    axes[1, 1].set_title("4. DistilBERT Confidence Scores")
    axes[1, 1].set_xlabel("Confidence")
    axes[1, 1].set_ylabel("Frequency")

    # Adjust layout to prevent overlap and display the figure
    plt.tight_layout()
    plt.show()



def plot_top_keywords_per_bank(df_keywords: pd.DataFrame, top_n: int = 10):
    """Horizontal bar charts of top keywords per bank (one subplot per bank)"""
    banks = ["CBE", "BOA", "DASHEN"]  # adjust if you have more/less
    bank_names = {"CBE": "Commercial Bank of Ethiopia", "BOA": "Bank of Abyssinia", "DASHEN": "Dashen Bank"}

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    colors = {"CBE": "#66b3ff", "BOA": "#ff9999", "DASHEN": "#99ff99"}

    for idx, bank in enumerate(banks):
        top = df_keywords[df_keywords["Bank"] == bank].head(top_n)
        axes[idx].barh(top["Keyword"], top["TF-IDF Score"], color=colors[bank])
        axes[idx].set_title(f"Top {top_n} Keywords – {bank_names.get(bank, bank)}", fontsize=14, pad=10)
        axes[idx].invert_yaxis()
        axes[idx].set_xlabel("TF-IDF Score")

    plt.tight_layout()
    plt.show()


def plot_theme_distribution(theme_by_bank: pd.DataFrame):
    """Stacked horizontal bar chart of themes per bank"""
    plt.figure(figsize=(12, 8))
    theme_by_bank.plot(kind="barh", stacked=True, cmap="Set3", figsize=(12, 8))
    plt.title("Theme Distribution by Bank", fontsize=18, fontweight="bold", pad=20)
    plt.xlabel("Number of Reviews", fontsize=12)
    plt.ylabel("Bank", fontsize=12)
    plt.legend(title="Theme", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()


    


def plot_wordclouds(df: pd.DataFrame, banks: list = None):
    """Generate one word cloud per bank"""
    if banks is None:
        banks = ["CBE", "BOA", "DASHEN"]

    bank_names = {"CBE": "Commercial Bank of Ethiopia", "BOA": "Bank of Abyssinia", "DASHEN": "Dashen Bank"}

    fig, axes = plt.subplots(1, len(banks), figsize=(18, 8))
    if len(banks) == 1:
        axes = [axes]

    for ax, bank in zip(axes, banks):
        text = " ".join(review for review in df[df["bank"] == bank]["processed_review"].dropna())
        if not text:
            text = "no reviews"

        wc = WordCloud(
            background_color="white",
            max_words=80,
            colormap="viridis",
            width=800,
            height=400
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"{bank_names.get(bank, bank)}\nWord Cloud", fontsize=16)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    