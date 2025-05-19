import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import sys
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from slimsc.prune.utils import DatasetHandler

def read_summary_jsons(summaries_dir: str, dataset_name: str) -> Tuple[List[int], List[int]]:
    """
    Read all summary JSONs and extract completion tokens for correct and incorrect chains.
    Returns lists of completion tokens for correct and incorrect chains.
    """
    correct_tokens = []
    incorrect_tokens = []
    total_files = 0
    processed_files = 0
    dataset_handler = DatasetHandler(dataset_name=dataset_name)
    
    for filename in tqdm(os.listdir(summaries_dir)):
        if not filename.endswith('_summary.json'):
            continue
            
        total_files += 1
        with open(os.path.join(summaries_dir, filename), 'r') as f:
            summary = json.load(f)
            
        if "chains_for_voting_details" not in summary:
            continue
            
        processed_files += 1
        for chain in summary["chains_for_voting_details"]:
            if "completion_tokens" not in chain:
                continue
                
            tokens = chain["completion_tokens"]
            is_correct = dataset_handler.calculate_score(chain["extracted_answer"], summary["correct_answer_letter"]) == 1
            
            if is_correct:
                correct_tokens.append(tokens)
            else:
                incorrect_tokens.append(tokens)
    
    print(f"Found {total_files} summary files")
    print(f"Processed {processed_files} files with chain details")
    print(f"Found {len(correct_tokens)} correct chains and {len(incorrect_tokens)} incorrect chains")
                
    return correct_tokens, incorrect_tokens

def create_cdf_plots(correct_tokens: List[int], incorrect_tokens: List[int], 
                    output_dir: str, model_name: str, dataset_name: str, run_name: str):
    """
    Create two CDF plots:
    1. Proportion of chains vs completion tokens
    2. Number of chains vs completion tokens
    """
    # Check if we have data to plot
    if not correct_tokens and not incorrect_tokens:
        print("No token data found to plot!")
        return
        
    # Calculate CDFs
    all_tokens = correct_tokens + incorrect_tokens
    min_tokens = min(all_tokens)
    max_tokens = max(all_tokens)
    
    # Create x-axis points
    x = np.linspace(min_tokens, max_tokens, 1000)
    
    # Calculate CDFs for proportions
    correct_cdf_prop = []
    incorrect_cdf_prop = []
    
    if correct_tokens:
        correct_cdf_prop = [sum(1 for t in correct_tokens if t <= x_val) / len(correct_tokens) 
                           for x_val in x]
    if incorrect_tokens:
        incorrect_cdf_prop = [sum(1 for t in incorrect_tokens if t <= x_val) / len(incorrect_tokens) 
                             for x_val in x]
    
    # Calculate CDFs for counts
    correct_cdf_count = [sum(1 for t in correct_tokens if t <= x_val) for x_val in x]
    incorrect_cdf_count = [sum(1 for t in incorrect_tokens if t <= x_val) for x_val in x]
    
    # Create base filename with model, dataset, and run info
    base_filename = f"{model_name}_{dataset_name}_{run_name}"
    
    # Create proportion plot
    plt.figure(figsize=(10, 6))
    if correct_tokens:
        plt.plot(x, correct_cdf_prop, color='tab:blue', label='Correct Chains', linewidth=2)
    if incorrect_tokens:
        plt.plot(x, incorrect_cdf_prop, color='tab:orange', label='Incorrect Chains', linewidth=2)
    plt.xlabel('Completion Tokens', fontsize=24)
    plt.ylabel('Proportion of Chains', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_completion_token_cdf_proportions.png'), 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    # Create count plot
    plt.figure(figsize=(10, 6))
    if correct_tokens:
        plt.plot(x, correct_cdf_count, color='tab:blue', label='Correct Chains', linewidth=2)
    if incorrect_tokens:
        plt.plot(x, incorrect_cdf_count, color='tab:orange', label='Incorrect Chains', linewidth=2)
    plt.xlabel('Completion Tokens', fontsize=24)
    plt.ylabel('Number of Chains', fontsize=24)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{base_filename}_completion_token_cdf_counts.png'), 
                bbox_inches='tight', pad_inches=0.1)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate CDF plots for completion tokens')
    parser.add_argument('--base_slimsc_dir', type=str, required=True, help='Base directory of the slimsc project')
    parser.add_argument('--model_arch', type=str, required=True, help='Model architecture name')
    parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
    parser.add_argument('--control_run_name', type=str, required=True, help='Name of the control run directory')
    parser.add_argument('--output_dir', type=str, default='completion_token_plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Construct paths
    base_results_dir = os.path.join(args.base_slimsc_dir, "prune/results", args.model_arch, args.dataset_name)
    control_dir = os.path.join(base_results_dir, args.control_run_name)
    summaries_dir = os.path.join(control_dir, "summaries")
    
    print(f"Looking for summary files in: {summaries_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read summary JSONs and extract completion tokens
    correct_tokens, incorrect_tokens = read_summary_jsons(summaries_dir, args.dataset_name)
    
    # Create CDF plots
    create_cdf_plots(correct_tokens, incorrect_tokens, args.output_dir, 
                    args.model_arch, args.dataset_name, args.control_run_name)
    
    print(f"Plots saved in {args.output_dir}")

if __name__ == "__main__":
    main()

    """
    python completion_token_cdf.py --base_slimsc_dir /home/users/ntu/colinhon/slimsc --model_arch R1-Distill-Qwen-14B --dataset_name aime --control_run_name sc_16_control --output_dir plots/R1-Distill-Qwen-14B
    """
