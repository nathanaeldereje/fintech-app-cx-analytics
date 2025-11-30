"""
sentiment_analysis.py

This module handles sentiment analysis for customer reviews using the
DistilBERT SST-2 model. It includes batching for efficient inference
and safe handling of empty review text.

Functions:
- apply_sentiment_pipeline(): Applies transformer sentiment analysis to review texts.
"""

from typing import List
from transformers import pipeline


def apply_sentiment_pipeline(reviews: List[str], sentiment_pipeline) -> List[dict]:
    """
    Apply DistilBERT sentiment analysis to a list of reviews in batches.

    Parameters
    ----------
    reviews : list of str
        List of review texts from the dataset.
    sentiment_pipeline : transformers.Pipeline
        Loaded HuggingFace sentiment analysis pipeline.

    Returns
    -------
    List[dict]
        A list of dictionaries containing:
        - 'label': 'POSITIVE' or 'NEGATIVE'
        - 'score': confidence score
    """
    results = []
    batch_size = 32

    for i in range(0, len(reviews), batch_size):
        # Replace empty or whitespace-only strings
        batch = [r if isinstance(r, str) and r.strip() else "neutral" 
                 for r in reviews[i:i + batch_size]]
        
        batch_results = sentiment_pipeline(batch)
        results.extend(batch_results)

    return results
