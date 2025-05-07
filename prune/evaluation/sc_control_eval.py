# slimsc/prune/evaluation/sc_control_eval.py
import os
import pandas as pd
import argparse
from tqdm import tqdm
import time
import json
import random
import asyncio
from typing import List, Dict, Optional, Set

from ..clients import close_aiohttp_session
from ..utils import load_data_gpqa
from .kv_cache_extraction import clear_source_kv_cache
from .processing import process_question_sc_stream

from rich.logging import RichHandler
import logging

logger = logging.getLogger(__name__)

DEFAULT_SEED = 12


def setup_output_directories(base_output_dir: str, model_name: str, dataset_name: str, sc_value: int) -> Dict[str, str]:
    """Creates directories for storing evaluation results."""
    run_name = f"sc_{sc_value}_control" # Added stream suffix
    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name)
    chains_output_dir = os.path.join(model_dataset_dir, "individual_chains")
    summary_output_dir = os.path.join(model_dataset_dir, "summaries")
    results_csv_path = os.path.join(model_dataset_dir, "evaluation_summary.csv")
    kvcache_usages_dir = os.path.join(model_dataset_dir, "kvcache_usages")
    aggregated_metrics_path = os.path.join(model_dataset_dir, "aggregated_metrics.json")

    os.makedirs(model_dataset_dir, exist_ok=True)
    os.makedirs(chains_output_dir, exist_ok=True)
    os.makedirs(summary_output_dir, exist_ok=True)
    os.makedirs(kvcache_usages_dir, exist_ok=True)

    source_kv_file = os.path.join(model_dataset_dir, "kvcache_usages.csv")

    return {
        "base": model_dataset_dir,
        "chains": chains_output_dir,
        "summaries": summary_output_dir,
        "csv": results_csv_path,
        "kvcache_usages_dir": kvcache_usages_dir,
        "source_usage_file": source_kv_file,
        "aggregated_metrics_json": aggregated_metrics_path
    }


# --- run_sc_evaluation_async needs to call process_question_sc_stream ---
async def run_sc_evaluation_async(
    dataset_name: str,
    model_name: str,
    model_identifier: str,
    tokenizer_path: Optional[str],
    n_chains: int,
    vllm_url: str,
    base_output_dir: str,
    start_iteration: int = 1,
    end_iteration: Optional[int] = None,
    specific_iterations: Optional[List[int]] = None
):
    """Runs the Self-Consistency evaluation loop using streaming."""
    logger.info(f"Starting Async SC Streaming Eval w/ Token Counting: N={n_chains}, Model={model_name}")
    if tokenizer_path is None:
        logger.warning("[yellow]--tokenizer_path not provided. Reasoning/Non-reasoning token counts will not be calculated.[/yellow]")
    paths = setup_output_directories(base_output_dir, model_name, dataset_name, n_chains)

    clear_source_kv_cache(paths.get("source_usage_file"))

    results_list = []
    processed_iterations = set()
     # Define CSV columns - must match keys returned by process_question_sc_stream
    csv_cols = [
        "iteration", "question_id", "n_chains_requested", "n_chains_received", # n_chains_received = chains whose stream finished without task error
        "correct_answer", "voted_answer", "final_score",
        "prompt_tokens", "total_completion_tokens", # From usage, total across ALL requested chains
        "total_reasoning_tokens",      # Counted, total for chains used for voting
        "total_non_reasoning_tokens",  # Counted, total for chains used for voting
        "total_tokens",                # Usage total across ALL requested chains + prompt
        "individual_answers_str",
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
                      if col in ["n_chains_requested", "n_chains_received",
                                 "prompt_tokens", "total_completion_tokens", "total_tokens",
                                 "total_reasoning_tokens", "total_non_reasoning_tokens"]:
                          existing_df[col] = 0
                      elif col in ["final_score", "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec"]:
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

    target_iterations_set: Set[int] = set() # Set of 1-indexed iterations the user *wants* to process

    if specific_iterations is not None:
        # User provided a list (--iterations or generated by --num_qns)
        # Filter out any iterations outside the dataset range (1 to total_examples)
        valid_specific_iterations = sorted([i for i in specific_iterations if 1 <= i <= total_examples])
        if len(valid_specific_iterations) != len(specific_iterations):
            invalid_count = len(specific_iterations) - len(valid_specific_iterations)
            logger.warning(f"[yellow]{invalid_count} specified iteration(s) were out of the dataset range (1 to {total_examples}). Skipping them.[/yellow]")
        target_iterations_set = set(valid_specific_iterations)
        logger.info(f"Targeting {len(target_iterations_set)} specific iteration(s) based on input list (e.g., --iterations or --num_qns).")
        if len(target_iterations_set) < 20: # Avoid printing huge lists
            logger.info(f"Target iterations: {sorted(list(target_iterations_set))}")

    else:
        # User provided start/end (or defaults)
        start = max(1, start_iteration)
        end = min(total_examples, end_iteration) if end_iteration is not None else total_examples
        if start > end:
            logger.error(f"[red]Start iteration ({start}) > End iteration ({end}). No iterations to process.[/red]")
            # Even if resuming, if the *new* range is invalid, don't try to process more
            iterations_to_process = [] # Ensure this is empty in this case
        else:
            target_iterations_set = set(range(start, end + 1))
            logger.info(f"Targeting iterations from {start} to {end} based on --start/--end.")

    # Determine which iterations actually need processing (target minus already processed)
    iterations_to_process = sorted(list(target_iterations_set - processed_iterations))

    if not iterations_to_process:
        logger.info("No new iterations to process.")
    else:
        logger.info(f"Need to process {len(iterations_to_process)} iterations.")
        pbar = tqdm(total=len(iterations_to_process), desc=f"GPQA N={n_chains} Stream")
        for i in iterations_to_process:
            example = examples[i-1]
            result = await process_question_sc_stream(
                example, i, n_chains, paths, vllm_url, model_identifier, tokenizer_path
            )
            if result:
                results_list.append(result)
                # Incremental save logic
                try:
                    df = pd.DataFrame(results_list)
                    # Ensure correct dtypes for merging/comparison before drop_duplicates
                    df['iteration'] = pd.to_numeric(df['iteration'], 
                                                    errors='coerce').astype('Int64')
                    # Reorder columns and add any missing ones from csv_cols before saving
                    for col in csv_cols:
                        if col not in df.columns:
                             if col in ["n_chains_requested", "n_chains_received",
                                        "prompt_tokens", "total_completion_tokens", 
                                        "total_tokens", "total_reasoning_tokens",
                                        "total_non_reasoning_tokens"]:
                                 df[col] = 0
                             elif col in ["final_score", "avg_kv_cache_usage", 
                                          "max_kv_cache_usage", "processing_duration_sec"]:
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

        clear_source_kv_cache(paths.get("source_usage_file"))
        pbar.close()

    # --- Final processing and summary ---
    aggregated_metrics = {}
    if results_list:
        final_df = pd.DataFrame(results_list)
        # Ensure correct dtypes for aggregation, handling potential NA values
        for col in ["final_score", "prompt_tokens", "total_completion_tokens", "total_tokens",
                    "total_reasoning_tokens", "total_non_reasoning_tokens",
                    "avg_kv_cache_usage", "max_kv_cache_usage", "processing_duration_sec",
                    "n_chains_requested", "n_chains_received"]:
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
                    if col in ["n_chains_requested", "n_chains_received",
                               "prompt_tokens", "total_completion_tokens", "total_tokens",
                               "total_reasoning_tokens", "total_non_reasoning_tokens"]:
                        final_df[col] = 0
                    elif col in ["final_score", "avg_kv_cache_usage", "max_kv_cache_usage", 
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

        # Include token counting stats if tokenizer was available
        mean_total_reasoning_tokens = final_df['total_reasoning_tokens'].dropna().mean() if tokenizer_path and num_processed_questions > 0 else None
        max_total_reasoning_tokens = final_df['total_reasoning_tokens'].dropna().max() if tokenizer_path and num_processed_questions > 0 else None
        mean_total_non_reasoning_tokens = final_df['total_non_reasoning_tokens'].dropna().mean() if tokenizer_path and num_processed_questions > 0 else None
        max_total_non_reasoning_tokens = final_df['total_non_reasoning_tokens'].dropna().max() if tokenizer_path and num_processed_questions > 0 else None

        # Also include average chains requested and received
        mean_chains_requested = final_df['n_chains_requested'].dropna().mean() if num_processed_questions > 0 else None
        mean_chains_received = final_df['n_chains_received'].dropna().mean() if num_processed_questions > 0 else None


        aggregated_metrics = {
            "dataset": dataset_name,
            "model_name": model_name,
            "run_type": "Self-Consistency (Streaming)",
            "config": {
                "n_chains": n_chains,
                "tokenizer_path_provided": tokenizer_path is not None,
                "iterations_selected_by": "specific_list" if specific_iterations is not None else "range",
                "random_seed": DEFAULT_SEED if specific_iterations is not None and any(specific_iterations) and hasattr(random, 'getstate') else None # Show seed if random was likely used
            },
            "metrics": {
                "num_questions_processed": num_processed_questions,
                "num_questions_with_score": num_questions_with_score, # Number of questions where a score could be calculated
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
            logger.info(f"[green]Overall Accuracy: {overall_accuracy:.4f}[/green]") # Increased precision
        else:
            logger.error("[red]Overall Accuracy could not be calculated (no valid scores).[/red]")

        if mean_total_completion_tokens is not None:
             logger.info(f"[green]Avg Total Completion Tokens per Question (all chains): {mean_total_completion_tokens:.1f}[/green]")
             logger.info(f"[green]Max Total Completion Tokens per Question (all chains): {max_total_completion_tokens:.1f}[/green]")

        if mean_processing_duration is not None:
             logger.info(f"[green]Avg Processing Duration per Question (sec): {mean_processing_duration:.2f}[/green]")
             logger.info(f"[green]Max Processing Duration per Question (sec): {max_processing_duration:.2f}[/green]")

        if mean_max_kv_usage is not None:
             logger.info(f"[green]Avg Max KV Cache Usage per Question (%): {mean_max_kv_usage:.4f}[/green]")
             logger.info(f"[green]Overall Max KV Cache Usage Recorded (%): {max_max_kv_usage:.4f}[/green]")

        if tokenizer_path:
             if mean_total_reasoning_tokens is not None:
                 logger.info(f"[green]Avg Reasoning Tokens per Question (counted): {mean_total_reasoning_tokens:.1f}[/green]")
             if mean_total_non_reasoning_tokens is not None:
                 logger.info(f"[green]Avg Non-Reasoning Tokens per Question (counted): {mean_total_non_reasoning_tokens:.1f}[/green]")

    else:
        logger.warning("[yellow]No results were processed or loaded for aggregation.[/yellow]")


    await close_aiohttp_session() # Ensure session is closed


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)] # Enable tracebacks
    )
    # Configure specific loggers
    # Set default level for potentially chatty libraries to WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def main():
    configure_logging()
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description='Run Async SC (Control) Evaluation using vLLM Streaming.')
    parser.add_argument('--n_start', type=int, required=True, help='Number of self-consistency chains (N).')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000", help='URL of the vLLM server OpenAI-compatible endpoint.')
    parser.add_argument('--model_name', type=str, required=True, help='Short name for the model used for directory structures.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Full model identifier used by vLLM API.')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to the HuggingFace tokenizer directory (REQUIRED for reasoning token counts AND tie-breaking fallback).')
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help='Name of the GPQA subset/dataset.')
    parser.add_argument('--output_dir', type=str, default=os.path.join(home, "slimsc/prune/results"), help='Base directory to save evaluation results.')
    parser.add_argument('--num_qns', type=int, default=None,
                        help=f'Number of random questions to run from the dataset. If specified, overrides --start, --end, and --iterations. Uses random seed {DEFAULT_SEED} for selection.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed). Used only if --num_qns and --iterations are not specified.')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive). Used only if --num_qns and --iterations are not specified.')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated list of specific iterations (e.g., "1,5,10-12"). Overridden by --num_qns.')
    args = parser.parse_args()

    if args.tokenizer_path is None:
         logger.warning("[yellow]--tokenizer_path not provided.[/yellow]")
         logger.warning("[yellow]Reasoning and Non-Reasoning token counts cannot be calculated and will be missing from results.[/yellow]")
         logger.warning("[yellow]Tokenizer-based tie-breaking for N=2 will also be unavailable.\n[/yellow]")
         time.sleep(2) # Pause briefly to ensure message is seen

    specific_iterations_list: Optional[List[int]] = None
    
    # Determine the list of target iterations based on argument priority
    if args.num_qns is not None:
        if args.num_qns <= 0:
            logger.error("[red]--num_qns must be a positive integer.[/red]")
            return
        # Load the dataset just to get the total number of questions
        try:
            examples = load_data_gpqa(dataset_name=args.dataset_name)
            total_examples = len(examples)
        except Exception as e:
            logger.exception(f"[red]Failed to load dataset '{args.dataset_name}' to determine total examples.[/red]")
            return

        if args.num_qns > total_examples:
            logger.warning(f"[yellow]Requested --num_qns ({args.num_qns}) is greater than the total number of examples ({total_examples}) in the dataset. Will process all {total_examples} examples.[/yellow]")
            num_to_select = total_examples
        else:
            num_to_select = args.num_qns

        # Generate random selection
        random.seed(DEFAULT_SEED) # Set the seed
        all_possible_iterations = list(range(1, total_examples + 1))
        random.shuffle(all_possible_iterations)
        specific_iterations_list = sorted(all_possible_iterations[:num_to_select])

        logger.info(f"Selected {num_to_select} random questions using seed {DEFAULT_SEED}.")
        if num_to_select < 20: # Avoid printing huge lists
            logger.info(f"Selected iterations: {specific_iterations_list}")

    elif args.iterations:
        # Parse comma-separated list and handle ranges (e.g., 1,5,10-12)
        specific_iterations_list = []
        parts = args.iterations.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            if '-' in part:
                try:
                    start_range, end_range = map(int, part.split('-'))
                    if start_range > end_range:
                        logger.warning(f"[yellow]Invalid range '{part}': start > end. Skipping.[/yellow]")
                        continue
                    specific_iterations_list.extend(range(start_range, end_range + 1))
                except ValueError:
                    logger.warning(f"[yellow]Invalid range format '{part}'. Skipping.[/yellow]")
            else:
                try:
                    specific_iterations_list.append(int(part))
                except ValueError:
                    logger.warning(f"[yellow]Invalid iteration number format '{part}'. Skipping.[/yellow]")

        specific_iterations_list = sorted(list(set(specific_iterations_list))) # Remove duplicates and sort

        if not specific_iterations_list:
            logger.error("[red]--iterations argument provided, but no valid iterations were parsed.[/red]")
            return
        logger.info(f"Targeting {len(specific_iterations_list)} specific iterations based on --iterations argument.")
        if len(specific_iterations_list) < 20: # Avoid printing huge lists
            logger.info(f"Target iterations: {specific_iterations_list}")

    # If neither --num_qns nor --iterations is provided, specific_iterations_list remains None,
    # and the async function will use the default start/end logic passed below.

    try:
        asyncio.run(run_sc_evaluation_async(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            model_identifier=args.model_identifier,
            tokenizer_path=args.tokenizer_path,
            n_chains=args.n_start,
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