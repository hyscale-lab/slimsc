# slimsc/prune/evaluation/similarity_prune_eval.py
import os
import pandas as pd
import argparse
from tqdm import tqdm
import time
import asyncio
import json
from typing import List, Dict, Optional

# Ensure correct relative imports if running as part of the package
try:
    from ..clients import close_aiohttp_session
    from ..utils import load_data_gpqa
    from ..utils.similarity_utils import get_embedding_model # To preload model if desired
    from .processing_similarity import process_question_similarity_prune
    from .kv_cache_extraction import clear_source_kv_cache, extract_kv_cache_usage_for_question
except ImportError:
     # Fallback for running script directly
     import sys
     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
     from slimsc.prune.clients import close_aiohttp_session
     from slimsc.prune.utils import load_data_gpqa
     from slimsc.prune.utils.similarity_utils import get_embedding_model
     from slimsc.prune.evaluation.processing_similarity import process_question_similarity_prune
     from slimsc.prune.evaluation.kv_cache_extraction import clear_source_kv_cache, extract_kv_cache_usage_for_question


from rich.logging import RichHandler
import logging

logger = logging.getLogger(__name__)

def setup_output_directories_prune(
        base_output_dir: str,
        model_name: str,
        dataset_name: str,
        n_start: int,
        threshold: float,
        pruning_strategy: str) -> Dict[str, str]:
    """Creates directories for storing similarity pruning evaluation results."""
    run_name = f"sim_prune_{pruning_strategy}_n{n_start}_thresh{threshold:.2f}"
    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name)
    chains_output_dir = os.path.join(model_dataset_dir, "individual_chains")
    summary_output_dir = os.path.join(model_dataset_dir, "summaries")
    results_csv_path = os.path.join(model_dataset_dir, "evaluation_summary.csv")
    kvcache_usages_dir = os.path.join(model_dataset_dir, "kvcache_usages")
    aggregated_metrics_path = os.path.join(model_dataset_dir, "aggregated_metrics.json") 

    os.makedirs(chains_output_dir, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(kvcache_usages_dir, exist_ok=True)

    home_dir = os.path.expanduser('~')
    source_kv_file = os.path.join(home_dir, 'scratch', 'kvcache_usage.csv')

    return {
        "base": model_dataset_dir,
        "chains": chains_output_dir,
        "summaries": summary_output_dir,
        "csv": results_csv_path,
        "kvcache_usages_dir": kvcache_usages_dir,
        "source_usage_file": source_kv_file,
        "aggregated_metrics_json": aggregated_metrics_path
    }


async def run_similarity_pruning_evaluation_async(
    dataset_name: str,
    model_name: str,
    model_identifier: str,
    tokenizer_path: str,
    n_chains_start: int,
    similarity_threshold: float,
    pruning_strategy: str,
    vllm_url: str,
    base_output_dir: str,
    start_iteration: int = 1,
    end_iteration: Optional[int] = None,
    specific_iterations: Optional[List[int]] = None
):
    """Runs the Similarity Pruning evaluation loop (Continuous Stream Version)."""
    logger.info(f"Starting Similarity Pruning Eval ({pruning_strategy}): N_start={n_chains_start}, Threshold={similarity_threshold}, Model={model_name}")
    paths = setup_output_directories_prune(
        base_output_dir, model_name, dataset_name, n_chains_start, similarity_threshold, pruning_strategy
    )

    # Pre-load embedding model
    try:
        get_embedding_model()
    except Exception:
        logger.error("[red]Failed to load embedding model. Cannot proceed.[/red]")
        return

    clear_source_kv_cache(paths.get("source_usage_file"))

    results_list = []
    processed_iterations = set()
    # Define CSV columns - must match keys returned by process_question_similarity_prune
    csv_cols = [
        "iteration", "question_id", "n_chains_start", 
        "n_chains_completed_stream_for_voting", "n_chains_error",
        "similarity_threshold", "correct_answer", "voted_answer", "final_score",
        "prompt_tokens", "total_completion_tokens", "total_tokens", # total_completion_tokens is sum across ALL finished streams
        "individual_answers_str", "total_analysis_intervals",
        "avg_kv_cache_usage", "max_kv_cache_usage", # Per-question KV stats
        "processing_duration_sec", # Per-question duration
    ]

    # Load existing results if resuming
    if os.path.exists(paths["csv"]):
        try:
            existing_df = pd.read_csv(paths["csv"])
            # Ensure all expected columns exist in loaded data, add if missing
            for col in csv_cols:
                 if col not in existing_df.columns:
                    # Add missing column with a default (e.g., None or 0 for numeric)
                    # Use None/NaN for float/object types, 0 for counts if appropriate
                    if col in ["n_chains_start", "n_chains_completed_stream_for_voting", 
                               "n_chains_pruned", "n_chains_error",
                               "prompt_tokens", "total_completion_tokens", "total_tokens",
                               "total_analysis_intervals"]:
                         existing_df[col] = 0
                    elif col in ["final_score", "similarity_threshold",
                                 "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec"]:
                         existing_df[col] = pd.NA # Use pandas NA for missing numeric data
                    else:
                         existing_df[col] = None # Use None for object/string types

            # Ensure correct dtypes for merging/comparison, especially iteration
            existing_df['iteration'] = pd.to_numeric(existing_df['iteration'], errors='coerce').astype('Int64') # Use nullable Int64
            # Drop rows where iteration failed to convert
            existing_df = existing_df.dropna(subset=['iteration']).drop_duplicates(subset=["iteration"], keep="last")

            results_list = existing_df.to_dict('records') # Use defined columns
            processed_iterations = set(existing_df['iteration'].unique())
            logger.info(f"Resuming. Found {len(processed_iterations)} previously processed iterations.")
        except Exception as e:
            logger.exception(f"[red]Could not read existing results file {paths['csv']}. Starting fresh.[/red]")
            results_list = []
            processed_iterations = set()

    examples = load_data_gpqa(dataset_name=dataset_name)
    total_examples = len(examples)

    target_iterations = set()
    if specific_iterations:
        target_iterations = {i for i in specific_iterations if 1 <= i <= total_examples}
    else:
        start = max(1, start_iteration)
        end = min(total_examples, end_iteration) if end_iteration else total_examples
        if start > end:
            logger.error(f"[red]Start iteration ({start}) > End iteration ({end}).[/red]")
            return
        target_iterations = set(range(start, end + 1))

    iterations_to_process = sorted(list(target_iterations - processed_iterations))

    if not iterations_to_process:
        logger.info("No new iterations to process.")
    else:
        logger.info(f"Need to process {len(iterations_to_process)} iterations.")
        pbar = tqdm(total=len(iterations_to_process), 
                    desc=f"GPQA SimPrune-{pruning_strategy} N={n_chains_start} T={similarity_threshold}")
        for i in iterations_to_process:
            example = examples[i-1]
            result = await process_question_similarity_prune(
                example=example,
                iteration=i,
                n_chains_start=n_chains_start,
                paths=paths,
                vllm_url=vllm_url,
                model_name=model_identifier,
                tokenizer_path=tokenizer_path,
                similarity_threshold=similarity_threshold,
                pruning_strategy=pruning_strategy
            )
            if result:
                results_list.append(result)
                try:
                    df = pd.DataFrame(results_list)
                    # Ensure correct dtypes for merging/comparison before drop_duplicates
                    df['iteration'] = pd.to_numeric(df['iteration'], errors='coerce').astype('Int64')
                    # Reorder columns and add any missing ones from csv_cols before saving
                    for col in csv_cols:
                        if col not in df.columns:
                             if col in ["n_chains_start", "n_chains_completed_stream_for_voting", 
                                        "n_chains_pruned", "n_chains_error",
                                        "prompt_tokens", "total_completion_tokens", "total_tokens",
                                        "total_analysis_intervals"]:
                                  df[col] = 0
                             elif col in ["final_score", "similarity_threshold",
                                          "avg_kv_cache_usage", "max_kv_cache_usage",
                                          "processing_duration_sec"]:
                                  df[col] = pd.NA
                             else:
                                  df[col] = None
                    # Filter out rows where iteration is NA after conversion
                    df = df.dropna(subset=['iteration'])[csv_cols].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")
                    df.to_csv(paths["csv"], index=False)
                except Exception as e:
                    logger.exception(f"[red]\nError saving intermediate CSV[/red]")
            else:
                logger.error(f"[red]Skipping results storage for question {i} due to processing error.[/red]")
            pbar.update(1)
            # Optional: Small delay between questions if server needs cooldown
            # await asyncio.sleep(0.5)
        pbar.close()

    # --- Final processing and summary ---
    aggregated_metrics = {}
    if results_list:
        final_df = pd.DataFrame(results_list)
        # Ensure correct dtypes for aggregation, handling potential NA values
        # Define columns intended to be numeric
        numeric_cols = [
            "iteration", # Already handled during load/merge, but good to include
            "n_chains_start", "n_chains_completed_stream_for_voting", "n_chains_error",
            "similarity_threshold", "final_score",
            "prompt_tokens", "total_completion_tokens", "total_tokens",
            "total_analysis_intervals", "avg_kv_cache_usage", "max_kv_cache_usage",
            "processing_duration_sec",
        ]
        # Apply numeric conversion only to specified columns
        for col in numeric_cols:
             if col in final_df.columns:
                  # Attempt numeric conversion, coercing errors to NaN
                  final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        # Drop rows where iteration is NA if any slipped through
        final_df = final_df.dropna(subset=['iteration'])

        # Save the final CSV one last time
        try:
             # Ensure all expected columns exist before saving
             for col in csv_cols:
                 if col not in final_df.columns:
                     if col in ["n_chains_start", "n_chains_completed_stream_for_voting",
                                "n_chains_error", "prompt_tokens", 
                                "total_completion_tokens", "total_tokens",
                                "total_analysis_intervals"]:
                          final_df[col] = 0
                     elif col in ["final_score", "similarity_threshold",
                                  "avg_kv_cache_usage", "max_kv_cache_usage",
                                  "processing_duration_sec"]:
                          final_df[col] = pd.NA
                     else:
                          final_df[col] = None

             final_df[csv_cols].sort_values(by="iteration").to_csv(paths["csv"], index=False)
             logger.info(f"[bold green]Evaluation complete. Final results saved to {paths['csv']}[/bold green]")
        except Exception as e:
             logger.exception(f"[red]\nError performing final save to CSV[/red]")


        # --- Calculate and Save Aggregated Metrics ---
        logger.info("Calculating overall aggregated metrics...")
        num_processed_questions = len(final_df)
        # Calculate score-based metrics only for questions where a score was successfully recorded
        num_questions_with_score = final_df['final_score'].dropna().shape[0]

        # Calculate required aggregates, handling potential NaNs (dropna) and ensuring non-zero division (num_processed_questions > 0)
        overall_accuracy = final_df['final_score'].dropna().mean() if num_questions_with_score > 0 else None

        mean_total_completion_tokens = final_df['total_completion_tokens'].dropna().mean() if num_processed_questions > 0 else None
        max_total_completion_tokens = final_df['total_completion_tokens'].dropna().max() if num_processed_questions > 0 else None

        mean_max_kv_usage = final_df['max_kv_cache_usage'].dropna().mean() if num_processed_questions > 0 else None
        max_max_kv_usage = final_df['max_kv_cache_usage'].dropna().max() if num_processed_questions > 0 else None

        mean_processing_duration = final_df['processing_duration_sec'].dropna().mean() if num_processed_questions > 0 else None
        max_processing_duration = final_df['processing_duration_sec'].dropna().max() if num_processed_questions > 0 else None

        mean_chains_error = final_df['n_chains_error'].dropna().mean() if num_processed_questions > 0 else None
        max_chains_error = final_df['n_chains_error'].dropna().max() if num_processed_questions > 0 else None

        mean_chains_start = final_df['n_chains_start'].dropna().mean() if num_processed_questions > 0 else None
        mean_chains_completed_for_voting = final_df['n_chains_completed_stream_for_voting'].dropna().mean() if num_processed_questions > 0 else None


        aggregated_metrics = {
            "dataset": dataset_name,
            "model_name": model_name,
            "run_type": f"Similarity Pruning ({pruning_strategy.replace('_', ' ').title()})",
            "config": {
                "n_chains_start": n_chains_start,
                "similarity_threshold": similarity_threshold,
                "pruning_strategy": pruning_strategy,
            },
            "metrics": {
                "num_questions_processed": num_processed_questions,
                "num_questions_with_score": num_questions_with_score, # Number of questions where a score could be calculated
                "overall_accuracy": float(overall_accuracy) if overall_accuracy is not None else None,

                "mean_total_completion_tokens_per_question": float(mean_total_completion_tokens) if mean_total_completion_tokens is not None else None,
                "max_total_completion_tokens_per_question": float(max_total_completion_tokens) if max_total_completion_tokens is not None else None,

                "mean_max_kv_cache_usage_per_question_perc": float(mean_max_kv_usage) if mean_max_kv_usage is not None else None,
                "max_max_kv_cache_usage_across_all_questions_perc": float(max_max_kv_usage) if max_max_kv_usage is not None else None,

                "mean_processing_duration_sec_per_question": float(mean_processing_duration) if mean_processing_duration is not None else None,
                "max_processing_duration_sec_per_question": float(max_processing_duration) if max_processing_duration is not None else None,

                "mean_chains_started_per_question": float(mean_chains_start) if mean_chains_start is not None else None,
                "mean_chains_completed_stream_for_voting_per_question": float(mean_chains_completed_for_voting) if mean_chains_completed_for_voting is not None else None,
                "mean_chains_error_per_question": float(mean_chains_error) if mean_chains_error is not None else None,
                "max_chains_error_per_question": float(max_chains_error) if max_chains_error is not None else None,
            }
        }

        try:
            with open(paths["aggregated_metrics_json"], "w", encoding='utf-8') as f:
                 json.dump(aggregated_metrics, f, indent=2)
            logger.info(f"[bold green]Aggregated metrics saved to {paths['aggregated_metrics_json']}[/bold green]")
        except IOError as e:
            logger.exception(f"[red]Error writing aggregated metrics file {paths['aggregated_metrics_json']}[/red]")
        except TypeError as e:
             logger.exception(f"[red]Error serializing aggregated metrics data to JSON: {e}[/red]")


        # Print summary stats (optional, as they are in the JSON file now)
        logger.info("\n--- Overall Aggregated Results ---")
        if overall_accuracy is not None:
            logger.info(f"[green]Overall Accuracy: {overall_accuracy:.4f}[/green]")
        else:
            logger.error("[red]Overall Accuracy could not be calculated (no valid scores).[/red]")

        if mean_total_completion_tokens is not None:
             logger.info(f"[green]Avg Total Completion Tokens per Question (all chains): {mean_total_completion_tokens:.1f}[/green]")
             logger.info(f"[green]Max Total Completion Tokens per Question (all chains): {max_total_completion_tokens:.1f}[/green]")

        if mean_chains_completed_for_voting is not None:
             logger.info(f"[green]Avg Chains Completed (for voting): {mean_chains_completed_for_voting:.1f}[/green]")

        if mean_processing_duration is not None:
             logger.info(f"[green]Avg Processing Duration per Question (sec): {mean_processing_duration:.2f}[/green]")
             logger.info(f"[green]Max Processing Duration per Question (sec): {max_processing_duration:.2f}[/green]")

        if mean_max_kv_usage is not None:
             logger.info(f"[green]Avg Max KV Cache Usage per Question (%): {mean_max_kv_usage:.4f}[/green]")
             logger.info(f"[green]Overall Max KV Cache Usage Recorded (%): {max_max_kv_usage:.4f}[/green]")

    else:
        logger.warning("[yellow]No results were processed or loaded for aggregation.[/yellow]")

    await close_aiohttp_session() # Ensure session is closed

def configure_logging():
    logging.basicConfig(
        level=logging.INFO, # Set to DEBUG for more verbose output from utils/processing
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)] # Enable tracebacks
    )
    # Configure specific loggers
    # Set default level for potentially chatty libraries to WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING) # FAISS can be chatty


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description='Run Similarity Pruning Evaluation using vLLM Streaming.')
    parser.add_argument('--n_start', type=int, required=True, help='Initial number of chains (N_start).')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Cosine similarity threshold for pruning (e.g., 0.85).')
    parser.add_argument('--pruning_strategy', type=str, required=True, choices=['fewest_thoughts', 'diversity', 'most_thoughts'],
                        help='Strategy to use for pruning decision: "fewest_thoughts", "most_thoughts", or "diversity".')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000",
                        help='URL of the vLLM server OpenAI-compatible endpoint.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Short name for the model used for directory structures.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Full model identifier used by vLLM API.')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to HuggingFace tokenizer directory (REQUIRED for segment extraction AND tie-breaking fallback).')
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help='Name of the GPQA subset/dataset.')
    parser.add_argument('--output_dir', type=str, default="./prune/results",
                        help='Base directory to save evaluation results.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed).')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive).')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated list of specific iterations.')
    args = parser.parse_args()

    # Validate threshold
    if not (0.0 < args.threshold <= 1.0):
        logger.error("[red]Similarity threshold must be between 0.0 (exclusive) and 1.0 (inclusive).[/red]")
        return

    specific_iterations_list = None
    if args.iterations:
        try:
            specific_iterations_list = [int(x.strip()) for x in args.iterations.split(',') if x.strip()]
            if not specific_iterations_list: raise ValueError("Empty list.")
        except ValueError:
            logger.exception("[red]Invalid format for --iterations.[/red]")
            return

    try:
        asyncio.run(run_similarity_pruning_evaluation_async(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            model_identifier=args.model_identifier,
            tokenizer_path=args.tokenizer_path,
            n_chains_start=args.n_start,
            similarity_threshold=args.threshold,
            pruning_strategy=args.pruning_strategy,
            vllm_url=args.vllm_url,
            base_output_dir=args.output_dir,
            start_iteration=args.start,
            end_iteration=args.end,
            specific_iterations=specific_iterations_list
        ))
    except KeyboardInterrupt:
        logger.exception("[red]\nEvaluation interrupted by user.[/red]")
    finally:
        logger.info("Cleanup potentially needed if interrupted before session close.")


if __name__ == "__main__":
    main()