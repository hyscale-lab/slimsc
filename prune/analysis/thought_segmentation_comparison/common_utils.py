# prune/analysis/thought_segmentation_comparison/common_utils.py

import numpy as np
import logging
from typing import List, Tuple
from transformers import AutoTokenizer

# Assuming these are in the project's python path
from prune.clients import stream_vllm_request, process_stream_chunks

logger = logging.getLogger(__name__)

_tokenizer_cache = {}

def get_tokenizer(tokenizer_path: str):
    """Loads and caches a tokenizer."""
    global _tokenizer_cache
    if tokenizer_path not in _tokenizer_cache:
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        try:
            _tokenizer_cache[tokenizer_path] = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
            return None
    return _tokenizer_cache[tokenizer_path]


def get_entropy(logprobs: List[float]) -> float:
    """Calculates entropy from a list of log probabilities."""
    if not logprobs:
        return 0.0
    probs = np.exp(logprobs)
    total_prob = np.sum(probs)
    if total_prob == 0:
        return 0.0
    normalized_probs = probs / total_prob
    entropy = -np.sum(normalized_probs * np.log2(normalized_probs, where=(normalized_probs > 0)))
    return entropy

async def get_generation_data(
    prompt: str,
    vllm_url: str,
    model_identifier: str,
    request_id: str
) -> Tuple[List[str], List[float], str]:
    """
    Calls the vLLM server for a prompt and returns its tokens, entropies, and full text.
    """
    stream_generator = stream_vllm_request(
        prompt=prompt,
        vllm_url=vllm_url,
        model_name=model_identifier,
        request_id=request_id,
        temperature=0.6,
        top_logprobs=10,
    )
    
    chunks = [chunk async for chunk in stream_generator]
    processed_result = process_stream_chunks(chunks, 1)

    if not processed_result or "error" in processed_result:
        logger.error(f"Failed to get generation for request {request_id}. Response: {processed_result}")
        return [], [], ""

    full_text = processed_result.get("full_content", "")
    logprobs_content = processed_result.get("logprobs", [])

    if not logprobs_content:
        logger.warning(f"No logprobs received for request {request_id}")
        return [], [], full_text
    
    tokens = [item['token'] for item in logprobs_content]
    logprob_values = [
        [d['logprob'] for d in item['top_logprobs']] if item.get('top_logprobs') else []
        for item in logprobs_content
    ]
    entropies = [get_entropy(lp_values) for lp_values in logprob_values]
    
    return tokens, entropies, full_text