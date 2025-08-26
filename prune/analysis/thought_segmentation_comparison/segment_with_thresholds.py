# prune/analysis/thought_segmentation_comparison/segment_with_thresholds.py

import os
import argparse
import json
import random
import asyncio
import numpy as np
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
from prune.utils.similarity_utils import TARGET_PHRASES, find_thought_boundaries
from prune.clients import close_aiohttp_session
from common_utils import get_generation_data, get_tokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MIN_SEGMENT_TOKENS = 25

async def segment_single_question(
    example, dataset_handler, paths, thresholds, tokenizer, args, example_index, dataset_name
):
    """
    Processes and segments a single question and returns the raw token/entropy data for aggregation.
    """
    logger.info(f"Starting segmentation for {dataset_name} question index {example_index}...")
    prompt, _ = dataset_handler.create_prompt(example)
    request_id = f"segment_{dataset_name}_q{example_index}"

    tokens, entropies, full_text = await get_generation_data(
        prompt, args.vllm_url, args.tokenizer_path, request_id
    )

    if not full_text:
        logger.warning(f"No text generated for question index {example_index}. Skipping.")
        return None, None # Return None to indicate failure

    # --- METHOD 1: TARGET_PHRASES Segmentation with Minimum Token Length ---
    try:
        potential_boundaries = find_thought_boundaries(full_text, TARGET_PHRASES)
        
        # Filter boundaries to enforce minimum segment length
        final_boundaries = [0]
        for i in range(1, len(potential_boundaries)):
            start_pos = final_boundaries[-1]
            end_pos = potential_boundaries[i]
            segment_text = full_text[start_pos:end_pos]
            
            # Count tokens in the segment
            num_tokens = len(tokenizer.encode(segment_text))
            
            if num_tokens >= MIN_SEGMENT_TOKENS:
                final_boundaries.append(end_pos)
        
        # Reconstruct text with separators only at the valid, final boundaries
        segmented_text_phrases = full_text
        for boundary in sorted(final_boundaries, reverse=True):
            if boundary > 0:
                segmented_text_phrases = segmented_text_phrases[:boundary] + "\n\n---------------\n\n" + segmented_text_phrases[boundary:]

        path_phrases = os.path.join(paths["phrases"], f"{args.model_name}_{dataset_name}_q{example_index}_phrases.txt")
        with open(path_phrases, "w", encoding="utf-8") as f:
            f.write(segmented_text_phrases)

    except Exception as e:
        logger.error(f"Error during TARGET_PHRASES segmentation for q{example_index}: {e}")

    # --- METHOD 2: High Entropy Segmentation (using global thresholds) ---
    if tokens:
        for percentile_str, threshold_val in thresholds.items():
            percentile = int(percentile_str)
            
            final_segments_as_tokens = []
            current_segment_tokens = []
            
            for i, token in enumerate(tokens):
                current_segment_tokens.append(token)
                is_high_entropy = entropies[i] >= threshold_val
                is_long_enough = len(current_segment_tokens) >= MIN_SEGMENT_TOKENS
                
                if is_high_entropy and is_long_enough:
                    final_segments_as_tokens.append(current_segment_tokens)
                    current_segment_tokens = []
            
            if current_segment_tokens:
                final_segments_as_tokens.append(current_segment_tokens)
            
            decoded_segments = [tokenizer.convert_tokens_to_string(seg) for seg in final_segments_as_tokens]
            segmented_text_entropy = "\n\n---------------\n\n".join(decoded_segments)

            path_entropy = os.path.join(paths[f"entropy_{percentile}"], f"{args.model_name}_{dataset_name}_q{example_index}_entropy_{percentile}.txt")
            with open(path_entropy, "w", encoding="utf-8") as f:
                f.write(segmented_text_entropy)
    
    logger.info(f"Finished segmentation for {dataset_name} question index {example_index}.")
    return tokens, entropies


def setup_output_directories(base_dir, model_name, dataset_name):
    paths = {}
    base_path = os.path.join(base_dir, model_name, dataset_name)
    paths["phrases"] = os.path.join(base_path, "target_phrases")
    os.makedirs(paths["phrases"], exist_ok=True)
    for p in [1, 5, 10, 15, 20]:
        paths[f"entropy_{p}"] = os.path.join(base_path, f"entropy_{p}p")
        os.makedirs(paths[f"entropy_{p}"], exist_ok=True)
    return paths

async def main_async(args):
    tokenizer = get_tokenizer(args.tokenizer_path)
    if not tokenizer:
        return
        
    logger.info(f"Starting parallel segmentation for datasets: {args.datasets}")
    
    for dataset_name in args.datasets:
        logger.info(f"--- Processing dataset: {dataset_name} ---")
        
        # 1. Load the specific profile for this dataset
        profile_filename = f"{args.model_name}_{dataset_name}_profile.json"
        profile_path = os.path.join(args.profile_dir, args.model_name, dataset_name, profile_filename)
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
            thresholds = profile_data['thresholds']
            logger.info(f"Successfully loaded entropy profile for {dataset_name}.")
        except FileNotFoundError:
            logger.error(f"Profile file not found for {dataset_name} at {profile_path}. Skipping this dataset.")
            continue
            
        # 2. Setup directories, samples, and tasks
        paths = setup_output_directories(args.output_dir, args.model_name, dataset_name)
        dataset_handler = DatasetHandler(dataset_name)
        examples = dataset_handler.load_dataset()
        random.seed(args.seed)
        sampled_indices = random.sample(range(len(examples)), k=min(args.num_samples, len(examples)))
        
        tasks = []
        for example_index in sampled_indices:
            example = examples[example_index]
            tasks.append(segment_single_question(
                example, dataset_handler, paths, thresholds, tokenizer, args, example_index, dataset_name
            ))
        
        # 3. Run segmentation tasks for the current dataset
        logger.info(f"Executing {len(tasks)} segmentation tasks for {dataset_name} in parallel...")
        segmentation_results = await asyncio.gather(*tasks)

        # 4. Post-segmentation analysis for this dataset's run
        logger.info(f"--- Aggregating stats for the {dataset_name} segmentation run ---")
        
        all_entropies = []
        aggregated_entropies_by_token = defaultdict(list)
        
        for tokens, entropies in segmentation_results:
            if tokens and entropies:
                all_entropies.extend(entropies)
                for token, entropy in zip(tokens, entropies):
                    aggregated_entropies_by_token[token].append(entropy)

        if not all_entropies:
            logger.warning(f"No entropy data collected during segmentation for {dataset_name}. Cannot generate run stats.")
            continue

        # Create stats files in the SEGMENTED_OUTPUT directory
        dataset_output_dir = os.path.join(args.output_dir, args.model_name, dataset_name)

        # 5. Save comparison profile JSON
        run_thresholds = {p: np.percentile(all_entropies, 100 - p) for p in [1, 5, 10, 15, 20]}
        run_profile_data = {
            "source_profile": profile_path, "num_samples": len(segmentation_results),
            "total_tokens_analyzed": len(all_entropies), "thresholds": run_thresholds
        }
        run_profile_path = os.path.join(dataset_output_dir, f"segmentation_run_profile.json")
        with open(run_profile_path, 'w') as f: json.dump(run_profile_data, f, indent=4)
        logger.info(f"Segmentation run profile saved to: {run_profile_path}")

        # 6. Save top 100 stats CSV
        all_token_stats = [{
            "token": tokenizer.decode(tokenizer.convert_tokens_to_ids(raw_token)).replace('\n', '<newline>'),
            "frequency": len(entropy_list), "mean_entropy": np.mean(entropy_list), "std_dev_entropy": np.std(entropy_list)
        } for raw_token, entropy_list in aggregated_entropies_by_token.items()]
        
        top_100_stats = sorted(all_token_stats, key=lambda x: x['mean_entropy'], reverse=True)[:100]
        stats_path = os.path.join(dataset_output_dir, f"segmentation_run_top_100_tokens.csv")
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["token", "frequency", "mean_entropy", "std_dev_entropy"])
            writer.writeheader()
            writer.writerows(top_100_stats)
        logger.info(f"Segmentation run top 100 stats saved to: {stats_path}")

        # 7. Save Word Cloud
        wordcloud_data = {item['token']: item['mean_entropy'] for item in top_100_stats}
        wordcloud = WordCloud(
            width=1600, height=800, background_color='white', colormap='viridis',
            collocations=False, random_state=args.seed
        ).generate_from_frequencies(wordcloud_data)
        wc_path = os.path.join(dataset_output_dir, f"segmentation_run_wordcloud.png")
        wordcloud.to_file(wc_path)
        logger.info(f"Segmentation run word cloud saved to: {wc_path}")

    await close_aiohttp_session()
    logger.info("All segmentation tasks and analyses completed.")

def main():
    parser = argparse.ArgumentParser(description="Segment text in parallel and generate run-specific stats.")
    parser.add_argument('--vllm_url', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, required=True)
    parser.add_argument('--datasets', type=str, required=True, nargs='+')
    parser.add_argument('--profile_dir', type=str, default="prune/analysis/thought_segmentation_comparison/profiles")
    parser.add_argument('--output_dir', type=str, default="prune/analysis/thought_segmentation_comparison/segmented_output")
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()