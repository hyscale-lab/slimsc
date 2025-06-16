# combine_sc_results.py
import os
import pandas as pd
import json
import shutil
import argparse
from typing import Dict, Optional, List
import random
import tarfile # For handling .tar.gz files
import tempfile # For temporary directories

# Configure logging
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

# --- Helper function (adapted from your original script) ---
CSV_COLS_EXPECTED = [
    "iteration", "question_id", "n_chains_requested", "n_chains_received",
    "correct_answer", "voted_answer", "final_score",
    "prompt_tokens", "total_completion_tokens",
    "total_reasoning_tokens", "total_non_reasoning_tokens",
    "total_tokens", "individual_answers_str",
    "avg_kv_cache_usage", "max_kv_cache_usage",
    "processing_duration_sec",
]
DEFAULT_SEED = 12

def calculate_and_save_aggregated_metrics(
    final_df: pd.DataFrame,
    aggregated_metrics_path: str,
    dataset_name: str,
    model_name: str,
    n_chains: int,
    tokenizer_path: Optional[str],
    specific_iterations: Optional[List[int]] = None
):
    if final_df.empty:
        logger.warning("[yellow]Combined DataFrame is empty. Cannot calculate aggregated metrics.[/yellow]")
        empty_metrics = {
            "dataset": dataset_name,
            "model_name": model_name,
            "run_type": "Self-Consistency (Streaming) - COMBINED",
            "config": {
                "n_chains": n_chains,
                "tokenizer_path_provided": tokenizer_path is not None,
                 "iterations_selected_by": "specific_list" if specific_iterations is not None else "range (combined)",
                 "random_seed": DEFAULT_SEED if specific_iterations is not None and any(specific_iterations) and hasattr(random, 'getstate') else None
            },
            "metrics": {"num_questions_processed": 0}
        }
        try:
            with open(aggregated_metrics_path, "w", encoding='utf-8') as f:
                json.dump(empty_metrics, f, indent=2)
            logger.info(f"Empty aggregated metrics saved to {aggregated_metrics_path}")
        except IOError as e:
            logger.exception(f"[red]Error writing empty aggregated metrics file {aggregated_metrics_path}[/red]")
        return

    logger.info("Recalculating overall aggregated metrics from combined data...")

    for col in ["final_score", "prompt_tokens", "total_completion_tokens", "total_tokens",
                "total_reasoning_tokens", "total_non_reasoning_tokens",
                "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec",
                "n_chains_requested", "n_chains_received"]:
        if col in final_df.columns:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        else:
            logger.warning(f"Column '{col}' not found in combined DataFrame. Adding it as NaN for aggregation.")
            final_df[col] = pd.NA

    num_processed_questions = len(final_df)
    num_questions_with_score = final_df['final_score'].dropna().shape[0]
    overall_accuracy = final_df['final_score'].dropna().mean() if num_questions_with_score > 0 else None
    mean_total_completion_tokens = final_df['total_completion_tokens'].dropna().mean() if num_processed_questions > 0 else None
    max_total_completion_tokens = final_df['total_completion_tokens'].dropna().max() if num_processed_questions > 0 else None
    mean_max_kv_usage = final_df['max_kv_cache_usage'].dropna().mean() if num_processed_questions > 0 else None
    max_max_kv_usage = final_df['max_kv_cache_usage'].dropna().max() if num_processed_questions > 0 else None
    mean_processing_duration = final_df['processing_duration_sec'].dropna().mean() if num_processed_questions > 0 else None
    max_processing_duration = final_df['processing_duration_sec'].dropna().max() if num_processed_questions > 0 else None
    mean_chains_requested = final_df['n_chains_requested'].dropna().mean() if num_processed_questions > 0 else None
    mean_chains_received = final_df['n_chains_received'].dropna().mean() if num_processed_questions > 0 else None
    mean_total_reasoning_tokens = final_df['total_reasoning_tokens'].dropna().mean() if tokenizer_path and 'total_reasoning_tokens' in final_df.columns and num_processed_questions > 0 else None
    max_total_reasoning_tokens = final_df['total_reasoning_tokens'].dropna().max() if tokenizer_path and 'total_reasoning_tokens' in final_df.columns and num_processed_questions > 0 else None
    mean_total_non_reasoning_tokens = final_df['total_non_reasoning_tokens'].dropna().mean() if tokenizer_path and 'total_non_reasoning_tokens' in final_df.columns and num_processed_questions > 0 else None
    max_total_non_reasoning_tokens = final_df['total_non_reasoning_tokens'].dropna().max() if tokenizer_path and 'total_non_reasoning_tokens' in final_df.columns and num_processed_questions > 0 else None

    aggregated_metrics = {
        "dataset": dataset_name, "model_name": model_name, "run_type": "Self-Consistency (Streaming) - COMBINED",
        "config": {
            "n_chains": n_chains, "tokenizer_path_provided": tokenizer_path is not None,
            "iterations_selected_by": "specific_list" if specific_iterations is not None else "range (combined)",
            "random_seed": DEFAULT_SEED if specific_iterations is not None and specific_iterations and hasattr(random, 'getstate') else None
        },
        "metrics": {
            "num_questions_processed": num_processed_questions, "num_questions_with_score": num_questions_with_score,
            "overall_accuracy": f'{overall_accuracy:.2f}' if overall_accuracy is not None else None,
            "mean_total_completion_tokens_per_question": f'{mean_total_completion_tokens:.1f}' if mean_total_completion_tokens is not None else None,
            "max_total_completion_tokens_per_question": f'{max_total_completion_tokens:.1f}' if max_total_completion_tokens is not None else None,
            "mean_max_kv_cache_usage_per_question_perc": f'{mean_max_kv_usage:.4f}' if mean_max_kv_usage is not None else None,
            "max_max_kv_cache_usage_across_all_questions_perc": f'{max_max_kv_usage:.4f}' if max_max_kv_usage is not None else None,
            "mean_processing_duration_sec_per_question": f'{mean_processing_duration:.2f}' if mean_processing_duration is not None else None,
            "max_processing_duration_sec_per_question": f'{max_processing_duration:.2f}' if max_processing_duration is not None else None,
            "mean_chains_requested_per_question": f'{mean_chains_requested:.2f}' if mean_chains_requested is not None else None,
            "mean_chains_received_per_question": f'{mean_chains_received:.2f}' if mean_chains_received is not None else None,
            "counted_tokens_aggregated": {
                "mean_total_reasoning_tokens_per_question": f'{mean_total_reasoning_tokens:.1f}' if mean_total_reasoning_tokens is not None else None,
                "max_total_reasoning_tokens_per_question": f'{max_total_reasoning_tokens:.1f}' if max_total_reasoning_tokens is not None else None,
                "mean_total_non_reasoning_tokens_per_question": f'{mean_total_non_reasoning_tokens:.1f}' if mean_total_non_reasoning_tokens is not None else None,
                "max_total_non_reasoning_tokens_per_question": f'{max_total_non_reasoning_tokens:.1f}' if max_total_non_reasoning_tokens is not None else None,
            } if tokenizer_path else None,
        }
    }
    try:
        with open(aggregated_metrics_path, "w", encoding='utf-8') as f:
            json.dump(aggregated_metrics, f, indent=2)
        logger.info(f"[bold green]Combined aggregated metrics saved to {aggregated_metrics_path}[/bold green]")
    except IOError as e: logger.exception(f"[red]Error writing combined aggregated metrics file {aggregated_metrics_path}[/red]")
    except TypeError as e: logger.exception(f"[red]Error serializing combined aggregated metrics data to JSON: {e}[/red]")
    logger.info("\n--- Combined Overall Aggregated Results ---")
    if overall_accuracy is not None: logger.info(f"Overall Accuracy: {overall_accuracy:.4f}")

def combine_directories(source_dirs_list: List[str], target_dir_path: str, subdir_name: str):
    target_subdir = os.path.join(target_dir_path, subdir_name)
    os.makedirs(target_subdir, exist_ok=True)
    logger.info(f"Combining subdirectory: {subdir_name} into {target_subdir}")
    copied_count = 0
    for source_base_dir in source_dirs_list:
        src_subdir = os.path.join(source_base_dir, subdir_name)
        if os.path.isdir(src_subdir):
            for item in os.listdir(src_subdir):
                s_item = os.path.join(src_subdir, item)
                d_item = os.path.join(target_subdir, item)
                if os.path.isfile(s_item):
                    if os.path.exists(d_item):
                        logger.warning(f"File {d_item} already exists. Overwriting with file from {source_base_dir}.")
                    shutil.copy2(s_item, d_item)
                    copied_count +=1
        else:
            logger.warning(f"Source subdirectory not found: {src_subdir} (from base: {source_base_dir})")
    logger.info(f"Copied {copied_count} files into {target_subdir} for subdirectory '{subdir_name}'")


def extract_and_find_data_dir(tar_path: str, temp_extraction_parent_dir: str) -> Optional[str]:
    """
    Extracts a tar.gz file and tries to find the relevant data directory within it.
    Returns the path to the data directory, or None if extraction fails or data dir not found.
    The data directory is expected to contain 'evaluation_summary.csv' or 'summaries/'.
    """
    try:
        # Create a unique subdirectory within the temp_extraction_parent_dir for this tarball
        # to avoid name clashes if tarballs have same top-level folder names.
        tar_basename = os.path.splitext(os.path.splitext(os.path.basename(tar_path))[0])[0]
        temp_extract_target = tempfile.mkdtemp(prefix=f"{tar_basename}_", dir=temp_extraction_parent_dir)

        logger.info(f"Extracting {tar_path} to {temp_extract_target}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=temp_extract_target)
        logger.info(f"Extraction of {tar_path} complete.")

        # Heuristic to find the actual data directory:
        # Option 1: The extracted content is directly the data (e.g., evaluation_summary.csv at the root)
        if os.path.exists(os.path.join(temp_extract_target, "evaluation_summary.csv")) or \
           os.path.isdir(os.path.join(temp_extract_target, "summaries")):
            return temp_extract_target

        # Option 2: The tarball contained a single top-level directory, and data is inside it.
        extracted_items = os.listdir(temp_extract_target)
        if len(extracted_items) == 1:
            potential_data_dir = os.path.join(temp_extract_target, extracted_items[0])
            if os.path.isdir(potential_data_dir) and \
               (os.path.exists(os.path.join(potential_data_dir, "evaluation_summary.csv")) or \
                os.path.isdir(os.path.join(potential_data_dir, "summaries"))):
                return potential_data_dir
        
        # If multiple items and none of the above, search for a directory that looks like the target
        for item in extracted_items:
            potential_data_dir = os.path.join(temp_extract_target, item)
            if os.path.isdir(potential_data_dir) and \
               (os.path.exists(os.path.join(potential_data_dir, "evaluation_summary.csv")) or \
                os.path.isdir(os.path.join(potential_data_dir, "summaries"))):
                logger.info(f"Found data in subdirectory: {potential_data_dir}")
                return potential_data_dir


        logger.warning(f"Could not identify a valid data directory structure in extracted contents of {tar_path} within {temp_extract_target}. "
                       f"Expected 'evaluation_summary.csv' or 'summaries/' directory at the root or within a single top-level extracted folder.")
        return None

    except tarfile.TarError as e:
        logger.error(f"Error extracting {tar_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during extraction or processing of {tar_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Combine results from multiple SC evaluation run archives (.tar.gz).")
    parser.add_argument("--source_tars", required=True, nargs='+',
                        help="List of paths to the source .tar.gz archives. Each archive should contain one run's output.")
    parser.add_argument("--combined_dir", required=True, help="Path to the target directory for combined results.")
    parser.add_argument("--model_name", required=True, help="Model name (used in aggregated_metrics.json).")
    parser.add_argument("--dataset_name", default="gpqa_diamond", help="Dataset name (used in aggregated_metrics.json).")
    parser.add_argument("--sc_value", type=int, default=64, help="The N_chains/SC value used (e.g., 64 for sc_64_control).")
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to tokenizer (if original runs used it).')

    args = parser.parse_args()

    os.makedirs(args.combined_dir, exist_ok=True)
    logger.info(f"Combined results will be saved in: {args.combined_dir}")

    # Create a single parent temporary directory for all extractions
    # This parent temp dir will be cleaned up at the end.
    parent_temp_dir = tempfile.mkdtemp(prefix="sc_combine_extract_")
    extracted_data_paths = [] # Will store paths to the actual data within each temp extraction

    try:
        for tar_path in args.source_tars:
            if not os.path.exists(tar_path):
                logger.error(f"Source archive not found: {tar_path}. Skipping.")
                continue
            if not tarfile.is_tarfile(tar_path):
                logger.error(f"File is not a valid tar archive: {tar_path}. Skipping.")
                continue

            data_dir = extract_and_find_data_dir(tar_path, parent_temp_dir)
            if data_dir:
                extracted_data_paths.append(data_dir)
            else:
                logger.warning(f"Failed to process or find data in {tar_path}. It will be excluded from combination.")

        if not extracted_data_paths:
            logger.error("No valid source data directories found after attempting extraction. Aborting.")
            return

        # --- The rest of the script uses extracted_data_paths instead of args.source_dirs ---
        combined_eval_summary_csv = os.path.join(args.combined_dir, "evaluation_summary.csv")
        combined_kv_summary_csv = os.path.join(args.combined_dir, "kvcache_usages.csv")
        combined_aggregated_metrics_json = os.path.join(args.combined_dir, "aggregated_metrics.json")

        # --- 1. Combine `summaries/` directory ---
        combine_directories(extracted_data_paths, args.combined_dir, "summaries")
        # --- 2. Combine `individual_chains/` directory ---
        combine_directories(extracted_data_paths, args.combined_dir, "individual_chains")
        # --- 3. (Optional) Combine `kvcache_usages/` directory ---
        # combine_directories(extracted_data_paths, args.combined_dir, "kvcache_usages")

        # --- 4. Combine `evaluation_summary.csv` ---
        all_dfs = []
        for source_dir in extracted_data_paths:
            eval_csv_path = os.path.join(source_dir, "evaluation_summary.csv")
            if os.path.exists(eval_csv_path):
                try:
                    df_current = pd.read_csv(eval_csv_path)
                    all_dfs.append(df_current)
                    logger.info(f"Loaded {len(df_current)} rows from {eval_csv_path} (extracted from a tar)")
                except Exception as e: logger.error(f"Could not load {eval_csv_path}: {e}")
            else: logger.warning(f"File not found: {eval_csv_path} in {source_dir}")
        
        combined_eval_df = pd.DataFrame()
        if all_dfs:
            combined_eval_df = pd.concat(all_dfs, ignore_index=True)
            combined_eval_df['iteration'] = pd.to_numeric(combined_eval_df['iteration'], errors='coerce').astype('Int64')
            combined_eval_df = combined_eval_df.dropna(subset=['iteration'])
            for col in CSV_COLS_EXPECTED:
                if col not in combined_eval_df.columns:
                    logger.warning(f"Column '{col}' missing. Adding with default.")
                    if col in ["n_chains_requested", "n_chains_received", "prompt_tokens", "total_completion_tokens", "total_tokens", "total_reasoning_tokens", "total_non_reasoning_tokens"]: combined_eval_df[col] = 0
                    elif col in ["final_score", "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec"]: combined_eval_df[col] = pd.NA
                    else: combined_eval_df[col] = None
            combined_eval_df = combined_eval_df[CSV_COLS_EXPECTED].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")
            try:
                combined_eval_df.to_csv(combined_eval_summary_csv, index=False)
                logger.info(f"Combined evaluation_summary.csv saved to {combined_eval_summary_csv} with {len(combined_eval_df)} rows.")
            except Exception as e: logger.error(f"Error saving combined evaluation_summary.csv: {e}")
        else: logger.warning("No dataframes to combine for evaluation_summary.csv.")

        # --- 5. Combine `kvcache_usages.csv` ---
        all_kv_dfs = []
        for source_dir in extracted_data_paths:
            kv_csv_path = os.path.join(source_dir, "kvcache_usages.csv")
            if os.path.exists(kv_csv_path):
                try:
                    kv_df_current = pd.read_csv(kv_csv_path)
                    all_kv_dfs.append(kv_df_current)
                    logger.info(f"Loaded {len(kv_df_current)} rows from {kv_csv_path} (extracted)")
                except Exception as e: logger.error(f"Could not load {kv_csv_path}: {e}")
            else: logger.warning(f"File not found: {kv_csv_path} (optional) in {source_dir}")
        if all_kv_dfs:
            combined_kv_df = pd.concat(all_kv_dfs, ignore_index=True)
            if 'iteration' in combined_kv_df.columns:
                combined_kv_df['iteration'] = pd.to_numeric(combined_kv_df['iteration'], errors='coerce').astype('Int64')
                combined_kv_df = combined_kv_df.dropna(subset=['iteration'])
                combined_kv_df = combined_kv_df.sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")
            else:
                logger.warning("'iteration' col not found in kvcache_usages.csv. Dropping all duplicate rows.")
                combined_kv_df = combined_kv_df.drop_duplicates()
            try:
                combined_kv_df.to_csv(combined_kv_summary_csv, index=False)
                logger.info(f"Combined kvcache_usages.csv saved to {combined_kv_summary_csv} with {len(combined_kv_df)} rows.")
            except Exception as e: logger.error(f"Error saving combined kvcache_usages.csv: {e}")
        else: logger.info("No data to combine for kvcache_usages.csv, or files not found.")

        # --- 6. Recalculate `aggregated_metrics.json` ---
        if not combined_eval_df.empty:
            calculate_and_save_aggregated_metrics(
                final_df=combined_eval_df, aggregated_metrics_path=combined_aggregated_metrics_json,
                dataset_name=args.dataset_name, model_name=args.model_name, n_chains=args.sc_value,
                tokenizer_path=args.tokenizer_path, specific_iterations=None
            )
        else: logger.error("Cannot generate aggregated_metrics.json because combined_eval_df is empty.")
        logger.info("[bold green]Result combination process finished.[/bold green]")

    finally:
        logger.info(f"Cleaning up temporary extraction directory: {parent_temp_dir}")
        shutil.rmtree(parent_temp_dir, ignore_errors=True)
        logger.info("Temporary directory cleanup complete.")

if __name__ == "__main__":
    main()