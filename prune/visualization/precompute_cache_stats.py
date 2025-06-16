# precompute_cache_stats.py

import os
import re
import json # Import json library
import numpy as np
from pathlib import Path
import warnings

# --- Configuration ---
user = os.environ.get("USER", "default_user")
BASE_RESULTS_PATH = Path("/home/users/ntu/{user}/slimsc/prune/results".format(user=user))
MODELS_TO_PROCESS = ["R1-Distill-Qwen-14B"]
DATASETS_TO_PROCESS = ["aime", "gpqa_diamond"]
OUTPUT_FILENAME = "precomputed_mean_gpu_cache_perc.txt"
SUMMARIES_DIR = "summaries"
SUMMARY_FILE_PATTERN = "question_*_summary.json"

# --- Helper Function (Updated) ---
def calculate_overall_avg_kv_cache_for_run(run_path: Path):
    """
    Calculates the mean of the 'avg_kv_cache_usage' values
    across all question summary JSONs for a given run.
    A run_path points to a specific sc_i_control directory.
    """
    summaries_path = run_path / SUMMARIES_DIR
    json_files = list(summaries_path.glob(SUMMARY_FILE_PATTERN))

    if not json_files:
        warnings.warn(f"No summary JSON files found in {summaries_path} using pattern '{SUMMARY_FILE_PATTERN}'")
        return None

    per_question_avg_usages = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
                # Extract the avg_kv_cache_usage value
                # Use .get() for safer access in case the key is missing
                avg_usage = data.get("avg_kv_cache_usage") 
                
                if avg_usage is not None:
                    # Ensure it's a number, not None or other unexpected type
                    if isinstance(avg_usage, (int, float)):
                        per_question_avg_usages.append(avg_usage)
                    else:
                         warnings.warn(f"Unexpected type for 'avg_kv_cache_usage' in {json_file}: {type(avg_usage)}")
                else:
                    # warnings.warn(f"Key 'avg_kv_cache_usage' not found in {json_file}") # Can be noisy
                    pass # Silently skip files missing the key
                    
        except FileNotFoundError:
            warnings.warn(f"Summary file not found during processing: {json_file}")
        except json.JSONDecodeError:
            warnings.warn(f"Error decoding JSON from file: {json_file}")
        except Exception as e:
            warnings.warn(f"An unexpected error occurred processing {json_file}: {e}")
            
    if not per_question_avg_usages:
        warnings.warn(f"No valid 'avg_kv_cache_usage' data found across all summary JSONs in {run_path}")
        return None
        
    # Calculate the mean of the collected 'avg_kv_cache_usage' values
    overall_mean = np.mean(per_question_avg_usages)
    return overall_mean

# --- Main Script Logic ---
def main():
    print(f"Starting precomputation of '{OUTPUT_FILENAME}' based on summary JSONs...")
    processed_count = 0
    skipped_count = 0
    error_count = 0

    for model_name in MODELS_TO_PROCESS:
        for dataset_name in DATASETS_TO_PROCESS:
            dataset_path = BASE_RESULTS_PATH / model_name / dataset_name
            if not dataset_path.is_dir():
                print(f"INFO: Dataset path not found, skipping: {dataset_path}")
                continue

            sc_run_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and re.match(r"sc_\d+_control", d.name)]

            if not sc_run_dirs:
                # print(f"INFO: No sc_X_control directories found in {dataset_path}") # Can be noisy
                continue

            for run_dir in sc_run_dirs:
                run_name = run_dir.name
                print(f"Processing: {model_name} / {dataset_name} / {run_name}")

                output_file_path = run_dir / OUTPUT_FILENAME
                # Uncomment the block below if you want to skip recalculating if the file exists
                # if output_file_path.exists():
                #     print(f"  Skipping, {OUTPUT_FILENAME} already exists.")
                #     skipped_count +=1
                #     continue

                # Call the updated function
                overall_avg_perc = calculate_overall_avg_kv_cache_for_run(run_dir)

                if overall_avg_perc is not None:
                    try:
                        with open(output_file_path, 'w') as f:
                            f.write(f"{overall_avg_perc:.10f}") # Save with good precision
                        print(f"  Successfully wrote {overall_avg_perc:.6f} to {output_file_path}")
                        processed_count += 1
                    except IOError as e:
                        print(f"  ERROR: Could not write to {output_file_path}: {e}")
                        error_count += 1
                else:
                    print(f"  Skipping {run_name}, could not calculate overall average KV cache usage from summaries.")
                    skipped_count += 1
    
    print("\nPrecomputation complete.")
    print(f"Successfully processed and saved: {processed_count} directories.")
    print(f"Skipped (no data or other reasons): {skipped_count} directories.")
    print(f"Errors during writing: {error_count} directories.")

if __name__ == "__main__":
    main()