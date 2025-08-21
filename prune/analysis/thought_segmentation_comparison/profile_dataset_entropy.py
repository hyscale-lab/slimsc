# prune/analysis/thought_segmentation_comparison/profile_dataset_entropy.py

import os
import argparse
import json
import random
import asyncio
import numpy as np
import pandas as pd
import csv
from collections import defaultdict
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging

# Add project root to Python path
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(project_root)

from prune.utils import DatasetHandler
from prune.clients import close_aiohttp_session
from common_utils import get_generation_data # Import from shared utility file

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- NEW: This function contains the core logic for profiling a single dataset ---
async def profile_single_dataset(dataset_name: str, args):
    """
    Performs entropy profiling for a single dataset and saves the results.
    """
    logger.info(f"--- Starting profiling for dataset: {dataset_name} ---")
    
    # 1. Initialize data structures for this dataset's aggregation
    all_entropies = []
    aggregated_entropies_by_token = defaultdict(list)

    # 2. Load dataset and get samples
    dataset_handler = DatasetHandler(dataset_name)
    examples = dataset_handler.load_dataset()
    random.seed(args.seed)
    sampled_indices = random.sample(range(len(examples)), k=min(args.num_samples, len(examples)))
    
    # 3. Process each sample question to gather data
    tasks = []
    for i, example_index in enumerate(sampled_indices):
        example = examples[example_index]
        prompt, _ = dataset_handler.create_prompt(example)
        request_id = f"profile_{dataset_name}_q{example_index}"
        tasks.append(get_generation_data(prompt, args.vllm_url, args.tokenizer_path, request_id))

    results = await asyncio.gather(*tasks)

    # 4. Aggregate results from all samples
    for tokens, entropies, _ in results:
        if not tokens:
            continue
        all_entropies.extend(entropies)
        for token, entropy in zip(tokens, entropies):
            aggregated_entropies_by_token[token].append(entropy)
    
    if not all_entropies:
        logger.error(f"Failed to gather any entropy data for {dataset_name}. Cannot create profile.")
        return

    dataset_output_dir = os.path.join(args.output_dir, args.model_name, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
        
    # 5. Post-processing: Calculate thresholds and generate profile file
    logger.info(f"Calculating dataset-level entropy thresholds for {dataset_name}...")
    percentiles_to_calc = [1, 5, 10, 15, 20]
    thresholds = {
        p: np.percentile(all_entropies, 100 - p) for p in percentiles_to_calc
    }
    
    profile_data = {
        "model_name": args.model_name,
        "dataset": dataset_name,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "total_tokens_analyzed": len(all_entropies),
        "thresholds": thresholds
    }
    
    profile_filename = f"{args.model_name}_{dataset_name}_profile.json"
    profile_path = os.path.join(dataset_output_dir, profile_filename)
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=4)
    logger.info(f"Entropy profile for {dataset_name} saved to: {profile_path}")

    # 6. Calculate full stats for all tokens and find the top 100
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    all_token_stats = []
    for raw_token, entropy_list in aggregated_entropies_by_token.items():
        decoded_token = tokenizer.decode(tokenizer.convert_tokens_to_ids(raw_token))
        all_token_stats.append({
            "token": decoded_token.replace('\n', '<newline>'),
            "frequency": len(entropy_list),
            "mean_entropy": np.mean(entropy_list),
            "std_dev_entropy": np.std(entropy_list)
        })
        
    top_100_stats = sorted(all_token_stats, key=lambda x: x['mean_entropy'], reverse=True)[:100]

    stats_filename = f"{args.model_name}_{dataset_name}_top_100_entropy_tokens.csv"
    stats_path = os.path.join(dataset_output_dir, stats_filename)
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["token", "frequency", "mean_entropy", "std_dev_entropy"])
        writer.writeheader()
        writer.writerows(top_100_stats)
    logger.info(f"Top 100 token stats for {dataset_name} saved to: {stats_path}")

    # 7. Generate and save the word cloud
    wordcloud_data = {item['token']: item['mean_entropy'] for item in top_100_stats}
    wordcloud = WordCloud(
        width=1600, height=800, background_color='white', colormap='viridis',
        collocations=False, random_state=args.seed
    ).generate_from_frequencies(wordcloud_data)
    
    wc_filename = f"{args.model_name}_{dataset_name}_wordcloud.png"
    wc_path = os.path.join(dataset_output_dir, wc_filename)
    wordcloud.to_file(wc_path)
    logger.info(f"Word cloud for {dataset_name} saved to: {wc_path}")


async def main_async(args):
    logger.info(f"Starting parallel entropy profiling for datasets: {args.datasets}")

    # Create a list of tasks, one for each dataset
    tasks = []
    for dataset_name in args.datasets:
        tasks.append(profile_single_dataset(dataset_name, args))

    # Run all profiling tasks concurrently
    await asyncio.gather(*tasks)

    # Close the shared session after all tasks are complete
    await close_aiohttp_session()
    logger.info("All profiling tasks completed.")

def main():
    parser = argparse.ArgumentParser(description="Profile dataset-level token entropy for multiple datasets in parallel.")
    parser.add_argument('--vllm_url', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument(
        '--datasets', 
        type=str, 
        required=True, 
        nargs='+',
        choices=["aqua_rat", "gpqa_diamond", "aime"],
        help="One or more datasets to profile."
    )
    parser.add_argument('--output_dir', type=str, default="prune/analysis/thought_segmentation_comparison/profiles")
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()