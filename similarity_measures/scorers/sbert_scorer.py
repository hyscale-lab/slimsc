# scorers/sbert_scorer.py
import numpy as np
import logging
from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

import config # To get DEVICE setting
from .utils import average_pairwise_similarity

# Cache models and tokenizers using the method_key from the config
sbert_models_cache: Dict[str, tuple[Any, Any]] = {}

# Mean Pooling function (remains the same)
def _mean_pooling(model_output, attention_mask):
    """Helper function for mean pooling."""
    token_embeddings = model_output[0] # First element contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def _get_sbert_model_and_tokenizer(model_config: Dict[str, Any], device: str):
    """Loads or retrieves SBERT model and tokenizer based on config dict."""
    method_key = model_config["method_key"]
    model_id = model_config["model_id"]
    tokenizer_id = model_config["tokenizer_id"]
    model_kwargs = model_config.get("model_kwargs", {}) # Use .get for safety
    tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})

    if method_key not in sbert_models_cache:
        logging.info(f"Loading SBERT setup '{method_key}':")
        logging.info(f"  Tokenizer: {tokenizer_id}")
        logging.info(f"  Model: {model_id}")
        logging.info(f"  Model Kwargs: {model_kwargs}")
        logging.info(f"  Tokenizer Kwargs: {tokenizer_kwargs}")
        logging.info(f"  Target Device: {device}")

        if model_kwargs.get("trust_remote_code"):
            logging.warning(f"  Loading model {model_id} with trust_remote_code=True.")
        if tokenizer_kwargs.get("trust_remote_code"):
             logging.warning(f"  Loading tokenizer {tokenizer_id} with trust_remote_code=True.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, **tokenizer_kwargs)
            model = AutoModel.from_pretrained(model_id, **model_kwargs).to(device)
            model.eval() # Set model to evaluation mode
            sbert_models_cache[method_key] = (tokenizer, model)
            logging.info(f"Successfully loaded SBERT setup: {method_key}")
        except ImportError as e:
             logging.error(f"Failed to load {method_key}. Possible missing dependencies for remote code. Error: {e}")
             logging.error("Try installing necessary packages mentioned in the model card or error message.")
             raise # Re-raise error
        except Exception as e:
            logging.error(f"Failed to load SBERT setup {method_key}: {e}")
            raise # Re-raise error to be handled by the caller
    else:
        # logging.debug(f"Using cached SBERT setup: {method_key}") # Optional debug log
        pass

    return sbert_models_cache[method_key]

def _calculate_sbert_embeddings(texts: List[str], model_config: Dict[str, Any], device: str) -> np.ndarray | None:
    """Generates Sentence-BERT embeddings for a list of texts using the specified config."""
    method_key = model_config["method_key"]
    try:
        tokenizer, model = _get_sbert_model_and_tokenizer(model_config, device)

        valid_texts = [str(t) for t in texts if isinstance(t, str)]
        if not valid_texts:
             logging.warning(f"SBERT ({method_key}): No valid strings found in input.")
             return None

        # Nomic specific recommendation: Use mean pooling
        # For nomic-embed-text-v1.5 and v2, mean pooling is often suggested.
        # The current _mean_pooling function should work.
        # For some models like E5, you might need CLS pooling instead.
        # This implementation uses mean pooling for all models currently.

        encoded_input = tokenizer(valid_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])

        # --- Nomic Specific Normalization & Dimensionality (Optional but Recommended) ---
        # Nomic models often benefit from normalization and sometimes need truncation
        # if you only want the first N dimensions (e.g., 256 for v1.5 search).
        # For similarity, usually use the full dimension. Normalization is good practice.
        if "nomic" in method_key:
             embeddings_normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
             logging.debug(f"SBERT ({method_key}): Applied L2 normalization to embeddings.")
             final_embeddings = embeddings_normalized
        else:
            # Standard approach (or apply normalization universally if desired)
            final_embeddings = sentence_embeddings
            # Optionally normalize all:
            # final_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return final_embeddings.cpu().numpy() # Move embeddings to CPU

    except Exception as e:
        logging.error(f"Error generating SBERT embeddings with {method_key}: {e}")
        return None


def calculate_sbert_cosine_similarity(texts: List[str], model_config: Dict[str, Any]) -> float:
    """Calculates average pairwise cosine similarity using the SBERT setup from model_config."""
    method_key = model_config["method_key"] # Get the unique key for logging/cache
    if not isinstance(texts, list) or len(texts) < 2:
        return np.nan

    # Use device from global config
    device = config.DEVICE
    embeddings = _calculate_sbert_embeddings(texts, model_config, device)

    if embeddings is None or embeddings.shape[0] < 2:
        logging.warning(f"SBERT ({method_key}): Could not generate sufficient embeddings (needed >= 2).")
        return np.nan

    try:
        # Cosine similarity calculation expects normalized vectors for best results,
        # which we did optionally above, especially for Nomic.
        cosine_sim_matrix = cosine_similarity(embeddings)
        # Clip to [0, 1] range as requested.
        cosine_sim_matrix = np.clip(cosine_sim_matrix, 0, 1)
        return average_pairwise_similarity(cosine_sim_matrix)
    except Exception as e:
        logging.error(f"Error during SBERT cosine similarity calculation ({method_key}): {e}")
        return np.nan