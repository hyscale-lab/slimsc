# scorers/tfidf_scorer.py
import numpy as np
import logging
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import the utility function from the same package
from .utils import average_pairwise_similarity

def calculate_tfidf_cosine_similarity(texts: List[str]) -> float:
    """Calculates average pairwise cosine similarity using TF-IDF vectors."""
    if not isinstance(texts, list) or len(texts) < 2:
        # logging.debug("TF-IDF: Requires a list of at least 2 texts.")
        return np.nan
    try:
        vectorizer = TfidfVectorizer()
        # Handle potential empty strings or non-string elements gracefully
        valid_texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
        if len(valid_texts) < 2:
             logging.warning("TF-IDF: Less than 2 valid non-empty strings found after filtering.")
             return np.nan

        tfidf_matrix = vectorizer.fit_transform(valid_texts)
        # Calculate pairwise cosine similarity
        cosine_sim_matrix = cosine_similarity(tfidf_matrix)
        # Ensure scores are within [0, 1] range (cosine similarity can be negative, but TF-IDF vectors are non-negative)
        cosine_sim_matrix = np.clip(cosine_sim_matrix, 0, 1)
        return average_pairwise_similarity(cosine_sim_matrix)
    except ValueError as ve:
        # Specifically catch errors like "empty vocabulary" if all texts are stop words
        logging.error(f"Error during TF-IDF vectorization (potentially empty vocabulary): {ve}")
        return np.nan
    except Exception as e:
        logging.error(f"Unexpected error during TF-IDF calculation: {e}")
        return np.nan