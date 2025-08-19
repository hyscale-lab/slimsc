import os
import pandas as pd
import argparse
from tqdm import tqdm
import time
import json
import random
import asyncio
import glob
import numpy as np
import csv
import collections
from typing import List, Dict, Optional, Set, AsyncGenerator

from transformers import AutoTokenizer

# Add project root to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from prune.clients import stream_vllm_request, process_stream_chunks, close_aiohttp_session
from prune.utils import DatasetHandler
from prune.utils.similarity_utils import TARGET_PHRASES, find_thought_boundaries

from rich.logging import RichHandler
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
MIN_SEGMENT_TOKENS_ENTROPY = 25
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

async def process_question_for_segmentation(
    example: Dict,
    iteration: int,
    paths: Dict[str, str],
    vllm_url: str,
    model_name: str,
    model_identifier: str,
    dataset_name: str
) -> None:
    logger.info(f"--- Processing Question {iteration} for {model_name} on {dataset_name} ---")
    
    tokenizer = get_tokenizer(model_identifier)
    if not tokenizer:
        logger.error(f"Cannot proceed for question {iteration} without a tokenizer.")
        return

    handler = DatasetHandler(dataset_name=dataset_name)
    prompt, _ = handler.create_prompt(example)

    request_id = f"{dataset_name}_q{iteration}_c1_stream_segmentation"
    
    stream_generator = stream_vllm_request(
        prompt=prompt,
        vllm_url=vllm_url,
        model_name=model_identifier,
        request_id=request_id,
        temperature=0.6,
        top_logprobs=10,
    )

    try:
        chunks = [chunk async for chunk in stream_generator]
        processed_result = process_stream_chunks(chunks, 1)
    except Exception as e:
        logger.error(f"Error during streaming or processing for request {request_id}: {e}")
        return

    if not processed_result or "error" in processed_result:
        logger.error(f"Failed to get a valid generation for question {iteration}. API response: {processed_result}")
        return

    full_text = processed_result.get("full_content", "")
    logprobs_content = processed_result.get("logprobs", [])

    if not full_text:
        logger.warning(f"No content generated for question {iteration}. Aborting processing for this question.")
        return

    logger.info(f"Successfully generated {len(full_text)} characters for question {iteration}.")

    # Method 1: TARGET_PHRASES
    try:
        boundaries = find_thought_boundaries(full_text, TARGET_PHRASES)
        segmented_text_phrases = full_text
        for boundary in sorted(boundaries, reverse=True):
            if boundary > 0:
                segmented_text_phrases = segmented_text_phrases[:boundary] + "\n\n---------------\n\n" + segmented_text_phrases[boundary:]
        
        output_path_phrases = os.path.join(paths["phrases"], f"{model_name}_{dataset_name}_q{iteration}_phrases.txt")
        with open(output_path_phrases, "w", encoding="utf-8") as f:
            f.write(segmented_text_phrases)
        logger.info(f"Saved target phrase segmentation to {output_path_phrases}")
    except Exception as e:
        logger.error(f"Error during TARGET_PHRASES segmentation for question {iteration}: {e}")

    # Method 2: High Entropy Tokens
    if not logprobs_content:
        logger.warning(f"No logprobs data received for question {iteration}. Skipping entropy-based segmentation.")
        return
        
    try:
        tokens = [item['token'] for item in logprobs_content]
        
        logprob_values_per_token = [
            [d['logprob'] for d in item['top_logprobs']] if item.get('top_logprobs') else []
            for item in logprobs_content
        ]
        entropies = [get_entropy(lp_values) for lp_values in logprob_values_per_token]
        
        if not entropies:
             logger.warning(f"Could not calculate entropies for question {iteration}.")
             return

        for percentile in [1, 5, 10, 15, 20]:
            threshold = np.percentile(entropies, 100 - percentile)
            
            final_segments_as_tokens = []
            current_segment_tokens = []
            high_entropy_tokens_log = []
            
            for i, token in enumerate(tokens):
                current_segment_tokens.append(token)
                
                is_high_entropy = entropies[i] >= threshold
                is_long_enough = len(current_segment_tokens) >= MIN_SEGMENT_TOKENS_ENTROPY
                
                if is_high_entropy and is_long_enough:
                    final_segments_as_tokens.append(current_segment_tokens)
                    high_entropy_tokens_log.append(token)
                    current_segment_tokens = []
            
            if current_segment_tokens:
                final_segments_as_tokens.append(current_segment_tokens)
            
            decoded_segments = [
                tokenizer.convert_tokens_to_string(seg) for seg in final_segments_as_tokens
            ]
            
            segmented_text_entropy = "\n\n---------------\n\n".join(decoded_segments)

            output_path_entropy = os.path.join(paths[f"entropy_{percentile}"], f"{model_name}_{dataset_name}_q{iteration}_entropy_{percentile}.txt")
            with open(output_path_entropy, "w", encoding="utf-8") as f:
                f.write(segmented_text_entropy)
            logger.info(f"Saved {percentile}% entropy segmentation to {output_path_entropy}")
            
            if high_entropy_tokens_log:
                # 1. Count the frequency of each raw token using collections.Counter
                token_counts = collections.Counter(high_entropy_tokens_log)

                # 2. Define the new CSV output path
                output_path_csv = os.path.join(paths[f"entropy_{percentile}"], f"{model_name}_{dataset_name}_q{iteration}_he_token_freq_{percentile}.csv")
                
                # 3. Write the data to the CSV file
                with open(output_path_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    
                    # Write the header row
                    writer.writerow(["token", "frequency"])
                    
                    # Iterate through the tokens sorted by frequency (most common first)
                    for raw_token, count in token_counts.most_common():
                        # Decode the raw token into a readable string
                        decoded_token = tokenizer.decode(tokenizer.convert_tokens_to_ids(raw_token))
                        
                        # Make special characters explicit for clarity in the CSV
                        readable_token = decoded_token.replace('\n', '<newline>').replace('\r', '')
                        
                        # Write the token and its frequency to the CSV
                        writer.writerow([readable_token, count])
                
                logger.info(f"Saved high entropy token frequencies to {output_path_csv}")
            else:
                logger.info(f"No high entropy tokens found for percentile {percentile} to log.")

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Error processing logprobs data structure for question {iteration}: {e}")
        logger.error(f"Sample of logprobs_content causing error: {logprobs_content[:2] if logprobs_content else 'None'}")


def setup_output_directories(base_output_dir: str, model_name: str, dataset_name: str) -> Dict[str, str]:
    """Creates directories for storing analysis results."""
    paths = {}
    base_path = os.path.join(base_output_dir, model_name, dataset_name)
    
    paths["phrases"] = os.path.join(base_path, "target_phrases")
    os.makedirs(paths["phrases"], exist_ok=True)

    for percentile in [1, 5, 10, 15, 20]:
        dir_name = f"entropy_{percentile}p"
        paths[f"entropy_{percentile}"] = os.path.join(base_path, dir_name)
        os.makedirs(paths[f"entropy_{percentile}"], exist_ok=True)
        
    return paths


async def main_async(args):
    # Ensure the main output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # The `model_identifier` should be what the vLLM server expects in the 'model' field.
    # Usually, this is the path it was loaded with.
    models = {
        "r1-distill": args.tokenizer_path if args.model_name == "r1-distill" else "/path/to/r1-distill/on/server",
        "qwq": "/path/to/qwq/on/server" # You may need to adjust this path
    }
    
    # Update model identifier based on command-line args
    if args.model_name == "r1-distill":
        models["r1-distill"] = args.tokenizer_path # Assuming tokenizer_path is the model path
    elif args.model_name == "qwq":
         # Assuming you'd pass a path for qwq similarly
        models["qwq"] = args.tokenizer_path

    # datasets = ["gpqa_diamond"]
    datasets = ["aqua_rat", "gpqa_diamond", "aime"]
    
    tasks = []
    
    for model_short_name, model_identifier in models.items():
        if args.model_name and args.model_name != model_short_name:
            continue
            
        for dataset_name in datasets:
            logger.info(f"Setting up analysis for model: {model_identifier} on dataset: {dataset_name}")
            
            paths = setup_output_directories(args.output_dir, model_short_name, dataset_name)
            
            dataset_handler = DatasetHandler(dataset_name)
            examples = dataset_handler.load_dataset()
            
            random.seed(DEFAULT_SEED)
            # Use random.sample on the list of examples directly
            sampled_indices = random.sample(range(len(examples)), 3)
            
            for i, example_index in enumerate(sampled_indices):
                example = examples[example_index]
                # Pass a unique iteration number for logging/file naming
                iteration = example_index + 1 
                task = process_question_for_segmentation(
                    example=example,
                    iteration=iteration,
                    paths=paths,
                    vllm_url=args.vllm_url,
                    model_name=model_short_name,
                    model_identifier=model_identifier,
                    dataset_name=dataset_name
                )
                tasks.append(task)
                
    await asyncio.gather(*tasks)
    await close_aiohttp_session()

def main():
    parser = argparse.ArgumentParser(description="Compare thought segmentation methods.")
    parser.add_argument('--vllm_url', type=str, required=True, help='URL of the vLLM server OpenAI-compatible endpoint.')
    parser.add_argument('--model_name', type=str, required=True, choices=["r1-distill", "qwq"], help='Short name for the model to run.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the model directory, used as the model identifier for the API call.')
    parser.add_argument('--output_dir', type=str, default="prune/analysis/thought_segmentation_comparison", help='Base directory to save analysis results.')
    
    args = parser.parse_args()

    # The script should be run from the project root (e.g., 'slimsc')
    # Adjust output_dir path if necessary to ensure it's correct relative to execution dir
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()