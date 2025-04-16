# slimsc/prune/evaluation/sc_control_eval.py
import os
import pandas as pd
import argparse
from tqdm import tqdm
import time
import asyncio
from typing import List, Dict, Optional

from ..clients import close_aiohttp_session
from ..utils import load_data_gpqa
from .kv_cache_extraction import clear_source_kv_cache
from .processing import process_question_sc_stream

from rich.logging import RichHandler
import logging

logger = logging.getLogger(__name__)


def setup_output_directories(base_output_dir: str, model_name: str, dataset_name: str, sc_value: int) -> Dict[str, str]:
    """Creates directories for storing evaluation results."""
    run_name = f"sc_{sc_value}_control_stream" # Added stream suffix
    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name)
    chains_output_dir = os.path.join(model_dataset_dir, "individual_chains")
    summary_output_dir = os.path.join(model_dataset_dir, "summaries")
    results_csv_path = os.path.join(model_dataset_dir, "evaluation_summary.csv")
    kvcache_usages_dir = os.path.join(model_dataset_dir, "kvcache_usages")

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
        "source_usage_file": source_kv_file
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

    clear_source_kv_cache(paths["source_usage_file"])

    results_list = []
    processed_iterations = set()
    # **** Define CSV columns ****
    csv_cols = ["iteration", "question_id", "n_chains_requested", "n_chains_received",
                "correct_answer", "voted_answer", "final_score",
                "prompt_tokens", "total_completion_tokens", # From usage
                "total_reasoning_tokens",      # Counted
                "total_non_reasoning_tokens",  # Counted
                "total_tokens",                # From usage
                "individual_answers_str"]

    if os.path.exists(paths["csv"]):
        try:
            existing_df = pd.read_csv(paths["csv"])
            # Ensure all expected columns exist in loaded data, add if missing
            for col in csv_cols:
                 if col not in existing_df.columns:
                     existing_df[col] = None # Add missing column with None
            results_list = existing_df[csv_cols].to_dict('records') # Use defined columns
            processed_iterations = set(existing_df['iteration'].dropna().astype(int).unique())
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
                    for col in csv_cols:
                         if col not in df.columns: df[col] = None
                    df = df[csv_cols].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")
                    df.to_csv(paths["csv"], index=False)
                except Exception as e:
                    logger.exception(f"[red]\nError saving intermediate CSV[/red]")
            else:
                logger.error(f"[red]Skipping results storage for question {i} due to processing error.[/red]")
            pbar.update(1)
        pbar.close()

    # Final processing and summary
    if results_list:
        final_df = pd.DataFrame(results_list)
        for col in csv_cols:
            if col not in final_df.columns: final_df[col] = None
        final_df = final_df[csv_cols].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")

        try:
             final_df.to_csv(paths["csv"], index=False)
             logger.info(f"[bold green]Evaluation complete. Final results saved to {paths['csv']}[/bold green]")
        except Exception as e:
             logger.exception(f"[red]\nError performing final save to CSV[/red]")

        # Print summary stats
        if 'final_score' in final_df.columns and not final_df['final_score'].isnull().all():
            accuracy = final_df['final_score'].astype(float).mean()
            logger.info(f"[green]Overall Accuracy: {accuracy:.2f}[/green]")
        else:
            logger.error("[red]Accuracy could not be calculated (no valid scores).[/red]")

        if 'prompt_tokens' in final_df.columns and not final_df['prompt_tokens'].isnull().all():
             avg_prompt_tokens = final_df['prompt_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Prompt Tokens: {avg_prompt_tokens:.1f}[/green]")
        if 'total_completion_tokens' in final_df.columns and not final_df['total_completion_tokens'].isnull().all():
             avg_completion_tokens = final_df['total_completion_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Completion Tokens (Total across successful chains): {avg_completion_tokens:.1f}[/green]")
        if 'total_reasoning_tokens' in final_df.columns and not final_df['total_reasoning_tokens'].isnull().all():
             avg_reasoning = final_df['total_reasoning_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Reasoning Tokens (Counted): {avg_reasoning:.1f}[/green]")
        if 'total_non_reasoning_tokens' in final_df.columns and not final_df['total_non_reasoning_tokens'].isnull().all():
             avg_non_reasoning = final_df['total_non_reasoning_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Non-Reasoning Tokens (Counted): {avg_non_reasoning:.1f}[/green]")
        if 'total_tokens' in final_df.columns and not final_df['total_tokens'].isnull().all():
             avg_total_tokens = final_df['total_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Total Tokens (Aggregated): {avg_total_tokens:.1f}[/green]")
        if 'avg_kv_cache_usage' in final_df.columns and final_df['avg_kv_cache_usage'].notna().any():
             avg_kv = final_df['avg_kv_cache_usage'].dropna().astype(float).mean()
             logger.info(f"[green]Avg KV Cache Usage (Question Avg): {avg_kv:.4f}[/green]")
        if 'max_kv_cache_usage' in final_df.columns and final_df['max_kv_cache_usage'].notna().any():
             avg_max_kv = final_df['max_kv_cache_usage'].dropna().astype(float).mean()
             overall_max_kv = final_df['max_kv_cache_usage'].dropna().astype(float).max()
             logger.info(f"[green]Avg Max KV Cache Usage (Question Max): {avg_max_kv:.4f}[/green]")
             logger.info(f"[green]Overall Max KV Cache Usage Recorded: {overall_max_kv:.4f}[/green]")

    else:
        logger.warning("[yellow]No results were processed or loaded.[/yellow]")

    await close_aiohttp_session() # Ensure session is closed


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True)]
    )


# --- main function remains the same, it just calls run_sc_evaluation_async ---
def main():
    configure_logging()
    parser = argparse.ArgumentParser(description='Run Async SC (Control) Evaluation using vLLM Streaming.')
    # Arguments are the same
    parser.add_argument('--sc_chains', type=int, required=True, choices=[2, 3, 5], help='Number of self-consistency chains (N).')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000", help='URL of the vLLM server OpenAI-compatible endpoint.')
    parser.add_argument('--model_name', type=str, required=True, help='Short name for the model used for directory structures.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Full model identifier used by vLLM API.')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to the HuggingFace tokenizer directory (REQUIRED for reasoning token counts).')
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help='Name of the GPQA subset/dataset.')
    parser.add_argument('--output_dir', type=str, default="./prune/results", help='Base directory to save evaluation results.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed).')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive).')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated list of specific iterations.')
    args = parser.parse_args()

    if args.tokenizer_path is None:
         logger.warning("[yellow]--tokenizer_path not provided.[/yellow]")
         logger.warning("[yellow]Reasoning and Non-Reasoning token counts cannot be calculated and will be missing from results.\n[/yellow]")
         time.sleep(2)

    specific_iterations_list = None
    if args.iterations:
        try:
            specific_iterations_list = [int(x.strip()) for x in args.iterations.split(',') if x.strip()]
            if not specific_iterations_list: raise ValueError("Empty list.")
        except ValueError:
            logger.exception("[red]Invalid format for --iterations.[/red]")
            return

    try:
        asyncio.run(run_sc_evaluation_async(
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            model_identifier=args.model_identifier,
            tokenizer_path=args.tokenizer_path,
            n_chains=args.sc_chains,
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