# scorers/gemini_scorer.py
import numpy as np
import logging
import time
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
import google.api_core.exceptions
from google.genai import types

import config
from .utils import average_pairwise_similarity

# --- Initialize Gemini Client ---
gemini_client = None
if config.GEMINI_API_KEY:
    try:
        # Use the client pattern as shown in the user example
        gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        # No standard 'list_models' or easy test method via Client instance easily found,
        # assume constructor success implies basic configuration is okay.
        logging.info(f"Google GenAI client initialized for model {config.GEMINI_EMBEDDING_MODEL_ID}.")
    except Exception as e:
        logging.error(f"Failed to initialize Google GenAI client: {e}. Gemini scoring will be skipped.")
        gemini_client = None # Ensure client is None if init fails
else:
    logging.warning("GEMINI_API_KEY not found in environment variables or config. Gemini scoring will be skipped.")


def _get_gemini_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    """Generates Gemini embeddings for a list of texts using batch processing."""
    if not gemini_client:
        # logging.debug("Gemini embedding skipped: client not initialized.")
        return None
    if not texts:
        # logging.debug("Gemini embedding skipped: no texts provided.")
        return None

    # Prepend 'models/' prefix required by the API call
    model_api_name = f"models/{config.GEMINI_EMBEDDING_MODEL_ID}"
    model_task_type = config.GEMINI_TASK_TYPE

    retries = 3
    delay = 5 # seconds

    logging.debug(f"Requesting Gemini embeddings for {len(texts)} texts using model {model_api_name} and task_type {model_task_type}.")

    # Filter out empty strings which might cause API errors
    valid_texts = [text for text in texts if text and text.strip()]
    if not valid_texts:
        logging.warning("Gemini embedding skipped: no non-empty texts found after filtering.")
        return None
    if len(valid_texts) < len(texts):
        logging.warning(f"Gemini embedding: Filtered out {len(texts) - len(valid_texts)} empty texts.")


    for attempt in range(retries):
        try:
            # The 'requests' parameter takes the list of content strings directly
            result = gemini_client.models.embed_content(
                model=model_api_name,
                contents=valid_texts, # Pass the list of strings
                config=types.EmbedContentConfig(task_type=model_task_type)
            )
            time.sleep(delay)

             # Check if the result object has an 'embeddings' attribute which is a list
            if hasattr(result, 'embeddings') and isinstance(result.embeddings, list):
                raw_embeddings = []
                valid_embedding_count = 0
                for emb_obj in result.embeddings:
                    # Check if each item has a 'values' attribute which is a list
                    if hasattr(emb_obj, 'values') and isinstance(emb_obj.values, list):
                        raw_embeddings.append(emb_obj.values)
                        valid_embedding_count += 1
                    else:
                        logging.error(f"Gemini embedding item has unexpected structure: {emb_obj}")
                        # Handle error? Skip this embedding? For now, let's skip it.
                        # You might want to add a placeholder like [np.nan] * dimension if needed later.

                # Check if we got embeddings for all valid texts sent
                if valid_embedding_count == len(valid_texts):
                    embeddings_array = np.array(raw_embeddings, dtype=np.float32)
                    logging.debug(f"Successfully obtained {embeddings_array.shape[0]} Gemini embeddings for {len(valid_texts)} valid texts.")
                    # Return the embeddings corresponding only to the valid texts
                    return embeddings_array
                else:
                    logging.error(f"Gemini embedding count mismatch or invalid structure within embeddings: Expected {len(valid_texts)}, Got {valid_embedding_count} valid embeddings. Full response object might have more details: {result}")
                    return None # Mismatch or invalid structure found

            else:
                logging.error(f"Unexpected Gemini API response structure. Expected object with 'embeddings' attribute (list). Got: {type(result)} - {result}")
                return None

        except google.api_core.exceptions.ResourceExhausted as e:
            logging.warning(f"Gemini API rate limit exceeded (Attempt {attempt + 1}/{retries}). Retrying in {delay}s... Error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(f"Gemini rate limit error after {retries} retries. Skipping embedding generation.")
                return None # Failed after retries
        except google.api_core.exceptions.InvalidArgument as e:
             logging.error(f"Gemini API Invalid Argument error: {e}. Check model name ('{model_api_name}'), task type ('{model_task_type}'), or input text validity. Skipping.")
             return None # Likely non-retryable config or input error
        except google.api_core.exceptions.GoogleAPIError as e:
            logging.error(f"Google API error during Gemini embedding: {e}. Skipping embedding generation.")
            return None # Don't retry on general API errors for now
        except Exception as e:
            logging.error(f"Unexpected error during Gemini embedding: {e}. Skipping embedding generation.")
            # Consider logging traceback for unexpected errors: import traceback; traceback.print_exc()
            return None

    return None # Should only be reached if all retries fail


def calculate_gemini_cosine_similarity(texts: List[str]) -> float:
    """Calculates average pairwise cosine similarity using Gemini embeddings."""
    # Check client initialization status again
    global gemini_client
    if not gemini_client:
        logging.warning("Gemini similarity calculation skipped: Gemini client not available.")
        return np.nan

    # Filter valid texts upfront for the calculation logic
    valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 2:
        # logging.debug(f"Gemini: Requires at least 2 non-empty texts (found {len(valid_texts)}).")
        return np.nan

    embeddings = _get_gemini_embeddings(valid_texts)

    if embeddings is None or embeddings.shape[0] < 2:
        logging.warning(f"Gemini: Could not generate sufficient embeddings for similarity calculation (needed >= 2).")
        return np.nan

    try:
        # Gemini embeddings are usually expected to be used with cosine similarity
        cosine_sim_matrix = cosine_similarity(embeddings)
        # Clip to [0, 1] range. Dot product of normalized vectors is [-1, 1].
        # Cosine similarity IS the dot product of normalized vectors.
        # Some embedding models might be designed such that similarity maps more directly to 0-1.
        # Clipping ensures the 0-1 range required by the prompt.
        cosine_sim_matrix = np.clip(cosine_sim_matrix, 0, 1)
        return average_pairwise_similarity(cosine_sim_matrix)
    except Exception as e:
        logging.error(f"Error during Gemini cosine similarity calculation: {e}")
        return np.nan