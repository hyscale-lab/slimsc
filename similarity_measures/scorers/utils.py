# scorers/utils.py
import numpy as np
import logging

def average_pairwise_similarity(similarity_matrix: np.ndarray) -> float:
    """Calculates the average similarity from a pairwise similarity matrix, excluding self-similarity."""
    if not isinstance(similarity_matrix, np.ndarray) or similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        logging.error(f"Invalid input: similarity_matrix must be a square numpy array. Shape: {similarity_matrix.shape}")
        return np.nan
    if similarity_matrix.shape[0] < 2:
        # logging.debug("Cannot calculate pairwise similarity with less than 2 items.")
        return np.nan # Not enough items to compare

    # Get upper triangle indices, excluding the diagonal (k=1)
    upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)

    if len(upper_triangle_indices[0]) == 0:
        # This case should be covered by shape[0] < 2, but as a safeguard
        # logging.debug("No pairs found for averaging (likely only 1 item).")
        return np.nan

    pairwise_scores = similarity_matrix[upper_triangle_indices]

    if pairwise_scores.size == 0:
        # logging.debug("No pairwise scores extracted.")
        return np.nan # Should not happen if indices were found

    return np.mean(pairwise_scores)