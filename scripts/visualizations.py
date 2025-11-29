import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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