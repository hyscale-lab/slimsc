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
from common_utils import get_generation_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def profile_single_dataset(dataset_name: str, args):
    """
    Performs entropy profiling for a single dataset and saves the results.
    """
    logger.info(f"--- Starting profiling for dataset: {dataset_name} ---")
    
    all_entropies = []
    aggregated_entropies_by_token = defaultdict(list)

    dataset_handler = DatasetHandler(dataset_name)
    examples = dataset_handler.load_dataset()
    random.seed(args.seed)
    sampled_indices = random.sample(range(len(examples)), k=min(args.num_samples, len(examples)))
    
    tasks = []
    for i, example_index in enumerate(sampled_indices):
        example = examples[example_index]
        prompt, _ = dataset_handler.create_prompt(example)
        request_id = f"profile_{dataset_name}_q{example_index}"
        tasks.append(get_generation_data(prompt, args.vllm_url, args.tokenizer_path, request_id))

    results = await asyncio.gather(*tasks)

    for tokens, entropies, _ in results:
        if tokens and entropies:
            all_entropies.extend(entropies)
            for token, entropy in zip(tokens, entropies):
                aggregated_entropies_by_token[token].append(entropy)
    
    if not all_entropies:
        logger.error(f"Failed to gather any entropy data for {dataset_name}. Cannot create profile.")
        return
        
    dataset_output_dir = os.path.join(args.output_dir, args.model_name, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
        
    logger.info(f"Calculating dataset-level entropy thresholds for {dataset_name}...")
    percentiles_to_calc = [1, 5, 10, 15, 20]
    thresholds = {
        p: np.percentile(all_entropies, 100 - p) for p in percentiles_to_calc
    }
    thresholds["lowest_1"] = np.percentile(all_entropies, 1)
    
    profile_data = {
        "model_name": args.model_name, "dataset": dataset_name, "num_samples": args.num_samples,
        "seed": args.seed, "total_tokens_analyzed": len(all_entropies), "thresholds": thresholds
    }
    
    profile_filename = f"{args.model_name}_{dataset_name}_profile.json"
    profile_path = os.path.join(dataset_output_dir, profile_filename)
    with open(profile_path, 'w') as f:
        json.dump(profile_data, f, indent=4)
    logger.info(f"Entropy profile for {dataset_name} saved to: {profile_path}")

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
        
    # --- TOP 100 (HIGH ENTROPY) ---
    top_100_stats = sorted(all_token_stats, key=lambda x: x['mean_entropy'], reverse=True)[:100]
    stats_filename_top = f"{args.model_name}_{dataset_name}_top_100_entropy_tokens.csv"
    stats_path_top = os.path.join(dataset_output_dir, stats_filename_top)
    with open(stats_path_top, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["token", "frequency", "mean_entropy", "std_dev_entropy"])
        writer.writeheader()
        writer.writerows(top_100_stats)
    logger.info(f"Top 100 token stats saved to: {stats_path_top}")

    wordcloud_data_top = {item['token']: item['mean_entropy'] for item in top_100_stats}
    wordcloud_top = WordCloud(
        width=1600, height=800, background_color='white', colormap='viridis',
        collocations=False, random_state=args.seed, font_path=args.font_path
    ).generate_from_frequencies(wordcloud_data_top)
    
    wc_filename_top = f"{args.model_name}_{dataset_name}_top_100_wordcloud.png"
    wc_path_top = os.path.join(dataset_output_dir, wc_filename_top)
    wordcloud_top.to_file(wc_path_top)
    logger.info(f"Top 100 word cloud saved to: {wc_path_top}")

    # --- BOTTOM 100 (LOW ENTROPY) ---
    bottom_100_stats = sorted(all_token_stats, key=lambda x: x['mean_entropy'])[:100]
    stats_filename_bottom = f"{args.model_name}_{dataset_name}_bottom_100_entropy_tokens.csv"
    stats_path_bottom = os.path.join(dataset_output_dir, stats_filename_bottom)
    with open(stats_path_bottom, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["token", "frequency", "mean_entropy", "std_dev_entropy"])
        writer.writeheader()
        writer.writerows(bottom_100_stats)
    logger.info(f"Bottom 100 token stats saved to: {stats_path_bottom}")

    # For the low entropy word cloud, we want larger words for *lower* entropy.
    # We can invert the values for visualization: size = max_entropy - entropy
    if bottom_100_stats:
        max_entropy_in_bottom_100 = bottom_100_stats[-1]['mean_entropy']
        wordcloud_data_bottom = {
            item['token']: (max_entropy_in_bottom_100 - item['mean_entropy'] + 0.01) # Add epsilon to avoid zero
            for item in bottom_100_stats
        }
        wordcloud_bottom = WordCloud(
            width=1600, height=800, background_color='white', colormap='viridis',
            collocations=False, random_state=args.seed, font_path=args.font_path
        ).generate_from_frequencies(wordcloud_data_bottom)
        
        wc_filename_bottom = f"{args.model_name}_{dataset_name}_bottom_100_wordcloud.png"
        wc_path_bottom = os.path.join(dataset_output_dir, wc_filename_bottom)
        wordcloud_bottom.to_file(wc_path_bottom)
        logger.info(f"Bottom 100 word cloud saved to: {wc_path_bottom}")

async def main_async(args):
    logger.info(f"Starting parallel entropy profiling for datasets: {args.datasets}")
    tasks = [profile_single_dataset(dataset_name, args) for dataset_name in args.datasets]
    await asyncio.gather(*tasks)
    await close_aiohttp_session()
    logger.info("All profiling tasks completed.")

def main():
    parser = argparse.ArgumentParser(description="Profile dataset-level token entropy for multiple datasets in parallel.")
    parser.add_argument('--vllm_url', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True, nargs='+')
    parser.add_argument('--output_dir', type=str, default="prune/analysis/thought_segmentation_comparison/profiles")
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--font_path', type=str, default=None, help="Path to a .ttf or .otf font file for word cloud generation (e.g., for Mandarin support).")
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()