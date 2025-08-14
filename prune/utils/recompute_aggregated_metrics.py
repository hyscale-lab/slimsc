import os
import pandas as pd
import numpy as np
import json
import argparse
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

def recompute_aggregated_metrics(run_dir: str):
    """
    Recomputes the aggregated_metrics.json file for a single run
    based on its evaluation_summary.csv file.
    
    This script is generic and works for both Control and Similarity Pruning runs.
    """
    logging.info(f"Starting recomputation for run directory: {run_dir}")

    # Define file paths
    csv_path = os.path.join(run_dir, "evaluation_summary.csv")
    json_path = os.path.join(run_dir, "aggregated_metrics.json")

    # --- Step 1: Validate inputs and load existing metadata ---
    if not os.path.exists(csv_path):
        logging.error(f"Required file not found: {csv_path}. Cannot proceed.")
        return

    # Load existing JSON to preserve config and metadata (like run_type)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        logging.info(f"Successfully loaded existing metadata from {json_path}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Could not read or parse existing {json_path}: {e}")
        logging.error("Metadata (config, run_type, etc.) is required. Cannot proceed.")
        return

    # Load the CSV data
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file {csv_path}: {e}")
        return

    if df.empty:
        logging.warning("The evaluation_summary.csv file is empty. Cannot compute metrics.")
        return

    # --- Step 2: Data Cleaning and Preparation ---
    # Ensure all potentially numeric columns are converted, coercing errors to NaN
    numeric_cols = [
        "final_score", "prompt_tokens", "total_completion_tokens", "total_tokens",
        "total_reasoning_tokens", "total_non_reasoning_tokens",
        "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec",
        "n_chains_requested", "n_chains_received", # For Control runs
        "n_chains_start", "n_chains_completed_stream_for_voting", "n_chains_error" # For Pruning runs
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Step 3: Calculate Aggregated Metrics ---
    logging.info("Calculating new aggregated metrics from CSV data...")
    num_processed_questions = len(df)
    num_qns_with_score = df['final_score'].dropna().shape[0]

    # Helper function to safely calculate and cast stats from a pandas Series
    def get_stat(series, func, cast_to=float):
        if not isinstance(series, pd.Series) or series.dropna().empty:
            return None
        val = func(series.dropna())
        return cast_to(val) if pd.notna(val) else None

    # Common metrics for both run types
    overall_accuracy = get_stat(df['final_score'], np.mean)
    mean_total_completion_tokens = get_stat(df.get('total_completion_tokens'), np.mean)
    max_total_completion_tokens = get_stat(df.get('total_completion_tokens'), np.max)
    mean_mean_kv_usage = get_stat(df.get('avg_kv_cache_usage'), np.mean)
    mean_max_kv_usage = get_stat(df.get('max_kv_cache_usage'), np.mean)
    max_max_kv_usage = get_stat(df.get('max_kv_cache_usage'), np.max)
    mean_processing_duration = get_stat(df.get('processing_duration_sec'), np.mean)
    max_processing_duration = get_stat(df.get('processing_duration_sec'), np.max)

    # Initialize the new metrics dictionary with common values
    new_metrics = {
        "overall_accuracy": overall_accuracy,
        "mean_total_completion_tokens_per_question": mean_total_completion_tokens,
        "max_total_completion_tokens_per_question": max_total_completion_tokens,
        "mean_mean_kv_cache_usage_per_question_perc": mean_mean_kv_usage,
        "mean_max_kv_cache_usage_per_question_perc": mean_max_kv_usage,
        "max_max_kv_cache_usage_across_all_questions_perc": max_max_kv_usage,
        "mean_processing_duration_sec_per_question": mean_processing_duration,
        "max_processing_duration_sec_per_question": max_processing_duration,
    }

    # === Logic branch based on run_type from the original JSON ===
    run_type = original_data.get("run_type", "")
    is_pruning_run = "Pruning" in run_type

    if is_pruning_run:
        logging.info("Detected Pruning run. Calculating pruning-specific metrics.")
        new_metrics.update({
            "num_qns_processed": num_processed_questions,
            "num_qns_with_score": num_qns_with_score,
            "mean_chains_started_per_question": get_stat(df.get('n_chains_start'), np.mean),
            "mean_chains_completed_stream_for_voting_per_question": get_stat(df.get('n_chains_completed_stream_for_voting'), np.mean),
            "mean_chains_error_per_question": get_stat(df.get('n_chains_error'), np.mean),
            "max_chains_error_per_question": get_stat(df.get('n_chains_error'), np.max),
        })
    else: # Assume control run
        logging.info("Detected Control run. Calculating control-specific metrics.")
        new_metrics.update({
            "num_questions_processed": num_processed_questions,
            "num_questions_with_score": num_qns_with_score,
            "mean_chains_requested_per_question": get_stat(df.get('n_chains_requested'), np.mean),
            "mean_chains_received_per_question": get_stat(df.get('n_chains_received'), np.mean),
        })
        
        tokenizer_provided = original_data.get("config", {}).get("tokenizer_path_provided", False)
        if tokenizer_provided and 'total_reasoning_tokens' in df.columns:
            logging.info("Tokenizer was provided. Calculating counted token metrics.")
            new_metrics["counted_tokens_aggregated"] = {
                "mean_total_reasoning_tokens_per_question": get_stat(df.get('total_reasoning_tokens'), np.mean),
                "max_total_reasoning_tokens_per_question": get_stat(df.get('total_reasoning_tokens'), np.max),
                "mean_total_non_reasoning_tokens_per_question": get_stat(df.get('total_non_reasoning_tokens'), np.mean),
                "max_total_non_reasoning_tokens_per_question": get_stat(df.get('total_non_reasoning_tokens'), np.max),
            }
        else:
             new_metrics["counted_tokens_aggregated"] = None

    # --- Step 4: Assemble and Save the Final JSON ---
    recomputed_data = {
        "dataset": original_data.get("dataset"),
        "model_name": original_data.get("model_name"),
        "run_type": original_data.get("run_type"),
        "config": original_data.get("config"),
        "metrics": new_metrics,
    }

    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(recomputed_data, f, indent=2)
        logging.info(f"Successfully recomputed and saved metrics to {json_path}")
    except (IOError, TypeError) as e:
        logging.error(f"Failed to save recomputed metrics to {json_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute aggregated_metrics.json for a specific run from its evaluation_summary.csv. Works for Control and Pruning runs."
    )
    parser.add_argument(
        '--run_dir',
        type=str,
        required=True,
        help='The path to the specific run directory (e.g., ".../config_name/run1").'
    )
    args = parser.parse_args()
    
    recompute_aggregated_metrics(args.run_dir)