# scorers/llm_scorer.py
import numpy as np
import logging
import time
from itertools import combinations
from typing import List, Optional
from openai import OpenAI, RateLimitError, APIError, APITimeoutError

import config # To get API key, model, prompt

# --- Initialize OpenAI Client ---
llm_client: Optional[OpenAI] = None
if config.OPENAI_API_KEY:
    try:
        llm_client = OpenAI(api_key=config.OPENAI_API_KEY, timeout=30.0) # Added timeout
        # Test connection (optional, but good practice)
        # try:
        #     llm_client.models.list()
        #     logging.info("OpenAI client initialized and connection verified.")
        # except APIError as api_err:
        #      logging.error(f"OpenAI API error during initialization test: {api_err}")
        #      llm_client = None # Disable client if test fails
        # except Exception as e:
        #     logging.error(f"Unexpected error during OpenAI client test: {e}")
        #     llm_client = None
        logging.info("OpenAI client initialized.") # Simplified message
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}. LLM scoring will be skipped.")
else:
    logging.warning("OPENAI_API_KEY not found in environment variables or config. LLM scoring will be skipped.")


def _get_llm_similarity_score(thought1: str, thought2: str) -> float | None:
    """Gets semantic similarity score for a single pair of thoughts using LLM."""
    if not llm_client:
        # logging.debug("LLM scoring skipped: client not initialized.")
        return None # Skip if client not initialized

    prompt = config.LLM_SIMILARITY_PROMPT_TEMPLATE.format(thought1=thought1, thought2=thought2)
    retries = 3
    delay = 5 # seconds

    for attempt in range(retries):
        try:
            response = llm_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, # For deterministic scoring
                max_tokens=10,   # Expecting just a number
                n=1,
                # request_timeout=20 # Timeout per request
            )
            score_text = response.choices[0].message.content.strip()

            # Attempt to convert the response to a float
            try:
                score = float(score_text)
                if 0.0 <= score <= 1.0:
                    # logging.debug(f"LLM score obtained: {score}")
                    return score
                else:
                    logging.warning(f"LLM returned score out of range [0.0, 1.0]: {score}. Raw text: '{score_text}'. Treating as invalid.")
                    return None # Score out of expected range
            except ValueError:
                logging.warning(f"LLM response could not be parsed as float: '{score_text}'. Treating as invalid.")
                return None # Cannot convert to float

        except RateLimitError as e:
            logging.warning(f"LLM Rate limit exceeded (Attempt {attempt + 1}/{retries}). Retrying in {delay}s... Error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                logging.error(f"LLM Rate limit error after {retries} retries. Skipping pair.")
                return None
        except APITimeoutError as e:
             logging.warning(f"LLM API request timed out (Attempt {attempt + 1}/{retries}). Retrying in {delay}s... Error: {e}")
             if attempt < retries - 1:
                 time.sleep(delay)
                 delay *= 2
             else:
                logging.error(f"LLM timeout error after {retries} retries. Skipping pair.")
                return None
        except APIError as e:
            # Catch broader API errors (e.g., server errors, auth issues)
            logging.error(f"OpenAI API error during scoring: {e}. Status code: {e.status_code}. Skipping pair.")
            return None # Don't retry on general API errors for now unless specified
        except Exception as e:
            logging.error(f"Unexpected error during LLM API call: {e}. Skipping pair.")
            return None # Don't retry on unexpected errors
    # Should only be reached if all retries fail for retryable errors
    return None


def calculate_llm_judge_similarity(texts: List[str]) -> float:
    """Calculates average pairwise similarity using LLM-as-a-Judge."""
    global llm_client # Ensure we're using the globally initialized client
    if not llm_client:
        logging.warning("LLM similarity calculation skipped: OpenAI client not available.")
        return np.nan
    if not isinstance(texts, list) or len(texts) < 2:
        # logging.debug("LLM: Requires a list of at least 2 texts.")
        return np.nan

    pair_scores = []
    valid_texts = [str(t) for t in texts if isinstance(t, str) and t.strip()]
    if len(valid_texts) < 2:
        logging.warning("LLM: Less than 2 valid non-empty strings found after filtering.")
        return np.nan

    logging.info(f"LLM: Evaluating {len(list(combinations(valid_texts, 2)))} pairs with {config.OPENAI_MODEL}.")
    # Use itertools.combinations to get unique pairs
    pair_iterator = combinations(valid_texts, 2)
    # Optional: Add tqdm here if many pairs per section are expected
    # from tqdm import tqdm
    # pair_iterator = tqdm(combinations(valid_texts, 2), total=..., desc="LLM Pairs")

    for thought1, thought2 in pair_iterator:
        score = _get_llm_similarity_score(thought1, thought2)
        if score is not None:
            pair_scores.append(score)
        else:
            # Warning/error is logged within _get_llm_similarity_score
            pass # Score for this pair could not be obtained

    if not pair_scores:
        logging.warning("LLM: No valid similarity scores were obtained for any pair in this section.")
        return np.nan # No valid scores to average

    average_score = np.mean(pair_scores)
    logging.info(f"LLM: Average score for section: {average_score:.4f} (from {len(pair_scores)} pairs)")
    return average_score