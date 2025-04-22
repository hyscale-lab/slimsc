# scorers/__init__.py

# Import the main calculation functions from each module
from .tfidf_scorer import calculate_tfidf_cosine_similarity
from .sbert_scorer import calculate_sbert_cosine_similarity
from .llm_scorer import calculate_llm_judge_similarity, llm_client # Expose client to check availability
from .gemini_scorer import calculate_gemini_cosine_similarity, gemini_client
from .sbert_threshold_evaluator import run_sbert_evaluation

# You can also define an __all__ list if you want to control `from scorers import *`
__all__ = [
    "calculate_tfidf_cosine_similarity",
    "calculate_sbert_cosine_similarity",
    "calculate_llm_judge_similarity",
    "calculate_gemini_cosine_similarity",
    "llm_client", # Make client accessible for checks in main.py
    "gemini_client",
    "run_sbert_evaluation",
]