# slimsc/prune/evaluation/esc_eval.py

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
import collections.abc
from typing import List, Dict, Optional, Set

# Assuming the script is in slimsc/prune/evaluation/
# Adjust the path to add the project root
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from prune.clients import close_aiohttp_session
from prune.utils import DatasetHandler
from prune.evaluation.kv_cache_extraction import clear_source_kv_cache
# The `process_question_sc_stream` is not needed here, but the new esc one is.
# Make sure the new `process_question_esc_stream` is in your processing.py
from prune.evaluation.processing import process_question_esc_stream 

from rich.logging import RichHandler
import logging

logger = logging.getLogger(__name__)

DEFAULT_SEED = 0

def flatten_dict(d, parent_key='', sep='_'):
    """ Flattens a nested dictionary. """
    items = []
    for k, v in d.items():
        # Convert keys to strings to handle numeric keys
        k_str = str(k)
        parent_key_str = str(parent_key) if parent_key else ''
        new_key = parent_key_str + sep + k_str if parent_key_str else k_str
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_and_save_mean_stats(base_run_dir: str):
    """
    Finds all aggregated_metrics.json files in run* subdirectories,
    calculates the mean and std dev, and saves to mean_aggregated_metrics.json.
    """
    logger.info(f"Recalculating mean stats in parent directory: {base_run_dir}")
    run_dirs = glob.glob(os.path.join(base_run_dir, "run*"))
    
    all_metrics_data = []
    first_run_config = None
    for run_dir in sorted(run_dirs):
        metrics_file = os.path.join(run_dir, "aggregated_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    if 'metrics' in data and isinstance(data['metrics'], dict):
                        if first_run_config is None and 'config' in data:
                            first_run_config = data.get('config')
                        
                        flat_metrics = flatten_dict(data['metrics'])
                        all_metrics_data.append(flat_metrics)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not read or parse {metrics_file}: {e}")
    
    if not all_metrics_data:
        logger.warning("No valid aggregated_metrics.json files found to average. Skipping.")
        return

    df = pd.DataFrame(all_metrics_data)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        logger.warning("No numeric metrics found to average. Skipping.")
        return

    mean_stats = df[numeric_cols].mean().to_dict()
    std_stats = df[numeric_cols].std().to_dict()
    
    final_mean_metrics = {
        "num_runs_aggregated": len(df),
        "mean": {k: v for k, v in mean_stats.items() if pd.notna(v)},
        "std_dev": {f"{k}_std": v for k, v in std_stats.items() if pd.notna(v)},
        "config": first_run_config
    }

    output_path = os.path.join(base_run_dir, "mean_aggregated_metrics.json")
    try:
        with open(output_path, 'w') as f:
            json.dump(final_mean_metrics, f, indent=2)
        logger.info(f"[bold blue]Successfully saved mean stats for {len(df)} runs to {output_path}[/bold blue]")
    except (IOError, TypeError) as e:
        logger.error(f"Failed to save mean stats to {output_path}: {e}")


def setup_output_directories(base_output_dir: str, model_name: str, dataset_name: str, n_chains: int, window_size: int, run_index: int) -> Dict[str, str]:
    """Creates directories for storing ESC evaluation results."""
    run_name = f"esc_n{n_chains}_w{window_size}"
    
    base_run_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name)
    run_specific_dir = os.path.join(base_run_dir, f"run{run_index}")

    chains_output_dir = os.path.join(run_specific_dir, "individual_chains")
    summary_output_dir = os.path.join(run_specific_dir, "summaries")
    results_csv_path = os.path.join(run_specific_dir, "evaluation_summary.csv")
    kvcache_usages_dir = os.path.join(run_specific_dir, "kvcache_usages")
    aggregated_metrics_path = os.path.join(run_specific_dir, "aggregated_metrics.json")

    os.makedirs(run_specific_dir, exist_ok=True)
    os.makedirs(chains_output_dir, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(kvcache_usages_dir, exist_ok=True)

    source_kv_file = os.path.join(run_specific_dir, "kvcache_usages.csv")

    return {
        "base": run_specific_dir,
        "chains": chains_output_dir,
        "summaries": summary_output_dir,
        "csv": results_csv_path,
        "kvcache_usages_dir": kvcache_usages_dir,
        "source_usage_file": source_kv_file,
        "aggregated_metrics_json": aggregated_metrics_path,
        "base_run_dir": base_run_dir
    }


async def run_esc_evaluation_async(
    dataset_name: str,
    model_name: str,
    model_identifier: str,
    tokenizer_path: Optional[str],
    n_chains: int,
    window_size: int,
    vllm_url: str,
    base_output_dir: str,
    run_index: int,
    start_iteration: int = 1,
    end_iteration: Optional[int] = None,
    specific_iterations: Optional[List[int]] = None
):
    """Runs the Early-Stopping Self-Consistency evaluation loop."""
    logger.info(f"Starting Run {run_index} - ESC Eval: N_max={n_chains}, Window_size={window_size}, Model={model_name}")
    paths = setup_output_directories(base_output_dir, model_name, dataset_name, n_chains, window_size, run_index)

    clear_source_kv_cache(paths.get("source_usage_file"))

    results_list = []
    processed_iterations = set()
    csv_cols = [
        "iteration", "question_id", "n_chains_max", "window_size", "n_chains_generated",
        "stopped_early", "correct_answer", "final_answer", "final_score",
        "prompt_tokens", "total_completion_tokens", "total_tokens",
        "total_reasoning_tokens", "total_non_reasoning_tokens",
        "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec",
        "individual_answers_str",
    ]

    if os.path.exists(paths["csv"]):
        try:
            existing_df = pd.read_csv(paths["csv"])
            for col in csv_cols:
                 if col not in existing_df.columns:
                      existing_df[col] = pd.NA
            existing_df['iteration'] = pd.to_numeric(existing_df['iteration'], errors='coerce').astype('Int64')
            existing_df = existing_df.dropna(subset=['iteration']).drop_duplicates(subset=["iteration"], keep="last")
            results_list = existing_df.to_dict('records')
            processed_iterations = set(existing_df['iteration'].unique())
            logger.info(f"Resuming. Found {len(processed_iterations)} previously processed iterations.")
        except Exception as e:
            logger.exception(f"[red]Could not read existing results file {paths['csv']}. Starting fresh.[/red]")

    dataset_handler = DatasetHandler(dataset_name=dataset_name)
    examples = dataset_handler.load_dataset()
    total_examples = len(examples)
    target_iterations_set: Set[int] = set()

    if specific_iterations is not None:
        valid_specific_iterations = sorted([i for i in specific_iterations if 1 <= i <= total_examples])
        if len(valid_specific_iterations) != len(specific_iterations):
            logger.warning(f"[yellow]Some specified iterations were out of range (1 to {total_examples}). Skipping them.[/yellow]")
        target_iterations_set = set(valid_specific_iterations)
    else:
        start = max(1, start_iteration)
        end = min(total_examples, end_iteration) if end_iteration is not None else total_examples
        if start > end:
            logger.error(f"[red]Start iteration ({start}) > End iteration ({end}). No iterations to process.[/red]")
            iterations_to_process = []
        else:
            target_iterations_set = set(range(start, end + 1))

    iterations_to_process = sorted(list(target_iterations_set - processed_iterations))

    if not iterations_to_process:
        logger.info("No new iterations to process.")
    else:
        logger.info(f"Need to process {len(iterations_to_process)} iterations.")
        pbar = tqdm(total=len(iterations_to_process), desc=f"{dataset_name} N={n_chains} W={window_size} ESC")
        for i in iterations_to_process:
            example = examples[i-1]
            result = await process_question_esc_stream(
                example, i, n_chains, window_size, paths, vllm_url, model_identifier, tokenizer_path, dataset_name
            )
            if result:
                results_list.append(result)
                try:
                    df = pd.DataFrame(results_list)
                    for col in csv_cols:
                        if col not in df.columns:
                             df[col] = pd.NA
                    df = df.dropna(subset=['iteration'])[csv_cols].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")
                    df.to_csv(paths["csv"], index=False)
                except Exception:
                    logger.exception(f"[red]\nError saving intermediate CSV[/red]")
            else:
                logger.error(f"[red]Skipping results storage for question {i} due to processing error.[/red]")
            pbar.update(1)
        pbar.close()

    clear_source_kv_cache(paths.get("source_usage_file"))
    
    if results_list:
        final_df = pd.DataFrame(results_list)
        numeric_cols = [col for col in csv_cols if "id" not in col and "answer" not in col]
        for col in numeric_cols:
             if col in final_df.columns:
                  final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        final_df[csv_cols].sort_values(by="iteration").to_csv(paths["csv"], index=False)
        logger.info(f"[bold green]Evaluation complete. Final results saved to {paths['csv']}[/bold green]")

        logger.info("Calculating overall aggregated metrics for this run...")
        aggregated_metrics = {
            "dataset": dataset_name, "model_name": model_name, "run_type": "ESC",
            "config": {
                "n_chains_max": n_chains, "window_size": window_size,
                "tokenizer_path_provided": tokenizer_path is not None,
                "random_seed": DEFAULT_SEED if specific_iterations else None
            },
            "metrics": {
                "num_questions_processed": len(final_df),
                "overall_accuracy": final_df['final_score'].mean(),
                "early_stop_rate": final_df['stopped_early'].mean(),
                "mean_chains_generated": final_df['n_chains_generated'].mean(),
                "mean_total_completion_tokens_per_question": final_df['total_completion_tokens'].mean(),
                "mean_processing_duration_sec_per_question": final_df['processing_duration_sec'].mean(),
                "mean_max_kv_cache_usage_per_question_perc": final_df['max_kv_cache_usage'].mean(),
                "mean_mean_kv_cache_usage_per_question_perc": final_df['avg_kv_cache_usage'].mean()
            }
        }
        
        try:
            with open(paths["aggregated_metrics_json"], "w") as f:
                json.dump(aggregated_metrics, f, indent=4)
            logger.info(f"[bold green]Aggregated metrics for run {run_index} saved to {paths['aggregated_metrics_json']}[/bold green]")
            calculate_and_save_mean_stats(paths['base_run_dir'])
        except Exception:
            logger.exception(f"[red]Error writing metrics or mean stats file[/red]")
    else:
        logger.warning("[yellow]No results were processed or loaded for aggregation.[/yellow]")

    await close_aiohttp_session()

def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(markup=True, rich_tracebacks=True)])
    logging.getLogger("httpx").setLevel(logging.WARNING)

def main():
    configure_logging()
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description='Run Early-Stopping Self-Consistency (ESC) Evaluation.')
    parser.add_argument('--n_chains', type=int, required=True, help='Maximum number of self-consistency chains (N).')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000", help='URL of the vLLM server endpoint.')
    parser.add_argument('--model_name', type=str, required=True, help='Short name for the model for directory structures.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Full model identifier for vLLM API.')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to tokenizer for token counting.')
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help='Dataset name.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(home, "slimsc/prune/results"), help='Base directory for results.')
    parser.add_argument('--num_qns', type=int, default=None, help=f'Number of random questions to run (overrides --start/--end). Seed: {DEFAULT_SEED}.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed).')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive).')
    parser.add_argument('--run_index', type=int, default=1, help='Index of this run for output subdirectories.')
    args = parser.parse_args()

    window_size = max(2, int(args.n_chains / 8))
    logger.info(f"Using a calculated window size of {window_size} for n_chains={args.n_chains}.")

    if args.n_chains < window_size:
        logger.error(f"[red]Invalid arguments: n_chains ({args.n_chains}) must be greater than or equal to the calculated window_size ({window_size}).[/red]")
        return

    specific_iterations_list: Optional[List[int]] = None
    if args.num_qns is not None:
        try:
            total_examples = len(DatasetHandler(dataset_name=args.dataset_name).load_dataset())
            num_to_select = min(args.num_qns, total_examples)
            random.seed(DEFAULT_SEED)
            specific_iterations_list = sorted(random.sample(range(1, total_examples + 1), num_to_select))
            logger.info(f"Selected {num_to_select} random questions using seed {DEFAULT_SEED}.")
        except Exception:
            logger.exception(f"[red]Failed to load dataset '{args.dataset_name}'[/red]")
            return

    try:
        asyncio.run(run_esc_evaluation_async(
            dataset_name=args.dataset_name, model_name=args.model_name, model_identifier=args.model_identifier,
            tokenizer_path=args.tokenizer_path, n_chains=args.n_chains, 
            window_size=window_size,
            vllm_url=args.vllm_url, base_output_dir=args.output_dir, run_index=args.run_index,
            start_iteration=args.start, end_iteration=args.end, specific_iterations=specific_iterations_list
        ))
    except KeyboardInterrupt:
        logger.exception("[red]\nEvaluation interrupted.[/red]")

if __name__ == "__main__":
    main()