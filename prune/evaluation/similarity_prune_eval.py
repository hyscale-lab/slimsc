# slimsc/prune/evaluation/similarity_prune_eval.py
import os
import pandas as pd
import argparse
from tqdm import tqdm
import time
import asyncio
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

def setup_output_directories_prune(base_output_dir: str, model_name: str, dataset_name: str, n_start: int, threshold: float) -> Dict[str, str]:
    """Creates directories for storing similarity pruning evaluation results."""
    run_name = f"sim_prune_n{n_start}_thresh{threshold:.2f}"
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


async def run_similarity_pruning_evaluation_async(
    dataset_name: str,
    model_name: str,
    model_identifier: str,
    tokenizer_path: str,
    n_chains_start: int,
    similarity_threshold: float,
    vllm_url: str,
    base_output_dir: str,
    start_iteration: int = 1,
    end_iteration: Optional[int] = None,
    specific_iterations: Optional[List[int]] = None
):
    """Runs the Similarity Pruning evaluation loop (Continuous Stream Version)."""
    logger.info(f"Starting Similarity Pruning Eval (Continuous Stream): N_start={n_chains_start}, Threshold={similarity_threshold}, Model={model_name}")
    paths = setup_output_directories_prune(base_output_dir, model_name, dataset_name, n_chains_start, similarity_threshold)

    # Pre-load embedding model
    try:
        get_embedding_model()
    except Exception:
        logger.error("[red]Failed to load embedding model. Cannot proceed.[/red]")
        return

    clear_source_kv_cache(paths.get("source_usage_file"))

    results_list = []
    processed_iterations = set()
    csv_cols = ["iteration", "question_id", "n_chains_start", "n_chains_final", "n_chains_pruned",
                "similarity_threshold", "correct_answer", "voted_answer", "final_score",
                "prompt_tokens", "total_completion_tokens", "total_tokens",
                "individual_answers_str", "total_steps",
                "avg_kv_cache_usage", "max_kv_cache_usage"
                ]

    if os.path.exists(paths["csv"]):
        try:
            existing_df = pd.read_csv(paths["csv"])
            for col in csv_cols:
                 if col not in existing_df.columns:
                     existing_df[col] = None
            results_list = existing_df[csv_cols].to_dict('records')
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
        pbar = tqdm(total=len(iterations_to_process), desc=f"GPQA SimPrune N={n_chains_start} T={similarity_threshold}")
        for i in iterations_to_process:
            example = examples[i-1]
            result = await process_question_similarity_prune(
                example=example,
                iteration=i,
                n_chains_start=n_chains_start,
                paths=paths, # Pass paths dict containing KV paths
                vllm_url=vllm_url,
                model_name=model_identifier,
                tokenizer_path=tokenizer_path,
                similarity_threshold=similarity_threshold
            )
            if result:
                results_list.append(result)
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
            # Optional: Small delay between questions if server needs cooldown
            # await asyncio.sleep(0.5)
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

        # Add stats about pruning
        if 'n_chains_start' in final_df.columns:
            avg_start = final_df['n_chains_start'].astype(float).mean()
            logger.info(f"[green]Avg Chains Started: {avg_start:.1f}[/green]")
        if 'n_chains_final' in final_df.columns:
             avg_final = final_df['n_chains_final'].astype(float).mean()
             logger.info(f"[green]Avg Chains Final (used for vote): {avg_final:.1f}[/green]")
        if 'n_chains_pruned' in final_df.columns:
             avg_pruned = final_df['n_chains_pruned'].astype(float).mean()
             logger.info(f"[green]Avg Chains Pruned: {avg_pruned:.1f}[/green]")

        if 'prompt_tokens' in final_df.columns and not final_df['prompt_tokens'].isnull().all():
             avg_prompt_tokens = final_df['prompt_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Prompt Tokens: {avg_prompt_tokens:.1f}[/green]")
        if 'total_completion_tokens' in final_df.columns and not final_df['total_completion_tokens'].isnull().all():
             avg_completion_tokens = final_df['total_completion_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Completion Tokens (Total across final chains): {avg_completion_tokens:.1f}[/green]")
        if 'total_tokens' in final_df.columns and not final_df['total_tokens'].isnull().all():
             avg_total_tokens = final_df['total_tokens'].astype(float).mean()
             logger.info(f"[green]Avg Total Tokens (Aggregated): {avg_total_tokens:.1f}[/green]")
        if 'total_steps' in final_df.columns and not final_df['total_steps'].isnull().all():
             avg_steps = final_df['total_steps'].astype(float).mean()
             logger.info(f"[green]Avg Processing Steps per Question: {avg_steps:.1f}[/green]")

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
        level=logging.INFO, # Set to DEBUG for more verbose output from utils/processing
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True)] # Enable tracebacks
    )
    # Optionally configure specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING) # Quieter http logs
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description='Run Similarity Pruning Evaluation using vLLM Streaming.')
    parser.add_argument('--n_start', type=int, required=True, choices=[3, 5, 8, 10], help='Initial number of chains (N_start).') # Adjusted choices maybe
    parser.add_argument('--threshold', type=float, required=True, help='Cosine similarity threshold for pruning (e.g., 0.85).')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000", help='URL of the vLLM server OpenAI-compatible endpoint.')
    parser.add_argument('--model_name', type=str, required=True, help='Short name for the model used for directory structures.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Full model identifier used by vLLM API.')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to HuggingFace tokenizer directory (REQUIRED for segment extraction).')
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help='Name of the GPQA subset/dataset.')
    parser.add_argument('--output_dir', type=str, default="./prune/results", help='Base directory to save evaluation results.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed).')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive).')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated list of specific iterations.')
    args = parser.parse_args()

    # Validate threshold
    if not (0.0 < args.threshold < 1.0):
        logger.error("[red]Similarity threshold must be between 0.0 and 1.0 (exclusive).[/red]")
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