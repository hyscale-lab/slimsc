# slimsc/prune/sc_control_eval.py
import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
from collections import Counter
import time
import asyncio
from typing import List, Dict, Tuple, Optional, AsyncGenerator

from vllm_client import stream_vllm_request, process_stream_chunks, close_aiohttp_session
from gpqa_utils import load_data_gpqa, create_prompt_gpqa, extract_answer_gpqa, calculate_score_gpqa, count_tokens

def clear_source_kv_cache(source_kv_file: str):
    """Clear the source KV cache file"""
    if source_kv_file:
        try:
            if os.path.exists(source_kv_file):
                print(f"Removing existing KV cache usage file: {source_kv_file}")
                os.remove(source_kv_file)
            else:
                print(f"KV cache usage file {source_kv_file} does not exist, proceeding.")
        except OSError as e:
            print(f"Warning: Could not remove KV cache file {source_kv_file}. Permissions issue? Error: {e}")
    else:
        print("Warning: Source KV cache file path not found in paths configuration.")


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


def analyze_kv_cache_usage_for_question(
    start_time: float,
    end_time: float,
    iteration: int,
    paths: Dict[str, str]
) -> Dict[str, Optional[float]]:
    """
    Reads the server's KV cache log file, filters by time window,
    saves the filtered log for the question, and calculates aggregate usage.

    Args:
        start_time (float): Unix timestamp when processing for the question started.
        end_time (float): Unix timestamp when processing for the question ended.
        iteration (int): The question iteration number (for logging and saving).
        paths (dict): Dictionary containing 'source_usage_file' and 'kvcache_usages_dir'.

    Returns:
        Dict[str, Optional[float]]: Dictionary with 'avg_kv_cache_usage'
                                     and 'max_kv_cache_usage', or None if unavailable.
    """
    source_file = paths.get('source_usage_file')
    save_dir = paths.get('kvcache_usages_dir')
    results = {'avg_kv_cache_usage': None, 'max_kv_cache_usage': None}
    saved_log_path = None # Keep track of where we saved it

    if not source_file:
        print(f"Warning [Q{iteration}]: Source KV cache usage file path not configured.")
        return results
    if not save_dir:
        print(f"Warning [Q{iteration}]: KV cache usage save directory path not configured.")
        # Continue to calculate aggregates if possible, but cannot save per-question log
    if not os.path.exists(source_file):
        print(f"Warning [Q{iteration}]: KV cache usage file not found at {source_file}. Cannot analyze or save.")
        return results

    try:
        # Define expected header names from server log
        header_names = ['timestamp', 'gpu_cache_usage_perc']
        # Try reading, explicitly providing names and setting header=0 if file has header
        # Adjust header=0/None based on whether your server log *actually* writes a header row
        try:
            df = pd.read_csv(source_file) # Try reading normally first
            if list(df.columns) != header_names:
                 print(f"Warning [Q{iteration}]: Header mismatch in {source_file}. Expected {header_names}, got {list(df.columns)}. Attempting recovery.")
                 # Reread forcing names, skipping the actual header row if present
                 df = pd.read_csv(source_file, names=header_names, header=0, skiprows=[0] if os.path.getsize(source_file)>0 else None)
        except pd.errors.ParserError:
             # Fallback if parsing fails, maybe due to bad lines or no header
             print(f"Warning [Q{iteration}]: ParserError reading {source_file}. Attempting read with explicit names and no header.")
             df = pd.read_csv(source_file, names=header_names, header=None)


        if df.empty:
             print(f"Warning [Q{iteration}]: KV cache usage file {source_file} is empty after reading.")
             # Optionally save an empty file with header
             if save_dir:
                  empty_save_path = os.path.join(save_dir, f"question_{iteration}_kvcache_usage.csv")
                  try:
                       pd.DataFrame(columns=header_names).to_csv(empty_save_path, index=False)
                       print(f"Saved empty KV cache log for Q{iteration} to {empty_save_path}")
                  except IOError as e_save:
                       print(f"Error saving empty KV log for Q{iteration} to {empty_save_path}: {e_save}")
             return results

        # Ensure columns are numeric
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df['gpu_cache_usage_perc'] = pd.to_numeric(df['gpu_cache_usage_perc'], errors='coerce')
        df = df.dropna(subset=['timestamp', 'gpu_cache_usage_perc'])

        # Filter based on the processing time window for this question
        df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)].copy()

        # Save the filtered df ****
        if save_dir:
            saved_log_path = os.path.join(save_dir, f"question_{iteration}_kvcache_usage.csv")
            try:
                # Save the filtered data, ensuring header is written
                df_filtered.to_csv(saved_log_path, index=False, header=True)
                status_msg = f"({len(df_filtered)} rows)" if not df_filtered.empty else "(empty)"
                print(f"Saved filtered KV cache log for Q{iteration} {status_msg} to {saved_log_path}")
            except IOError as e_save:
                print(f"Error saving filtered KV log for Q{iteration} to {saved_log_path}: {e_save}")
                saved_log_path = None # Indicate saving failed
        else:
            print(f"Info [Q{iteration}]: Save directory not configured. Skipping save of filtered KV log.")


        # Calculate aggregates from filtered data
        if df_filtered.empty:
            print(f"Warning [Q{iteration}]: No KV cache usage data found within the time window "
                  f"({start_time:.2f} - {end_time:.2f}) in {source_file}.")
            # Results remain None
        else:
            avg_usage = df_filtered['gpu_cache_usage_perc'].mean()
            max_usage = df_filtered['gpu_cache_usage_perc'].max()

            results['avg_kv_cache_usage'] = float(avg_usage) if pd.notna(avg_usage) else None
            results['max_kv_cache_usage'] = float(max_usage) if pd.notna(max_usage) else None

            print(f"KV Cache Usage % Aggregates [Q{iteration}]: "
                  f"Avg={results['avg_kv_cache_usage']:.4f}, Max={results['max_kv_cache_usage']:.4f}")

    except pd.errors.EmptyDataError:
         print(f"Warning [Q{iteration}]: KV cache usage file {source_file} is empty or header-only.")
    except FileNotFoundError:
         print(f"Warning [Q{iteration}]: KV cache usage file not found at {source_file}.")
    except Exception as e:
        print(f"Error processing KV cache usage for Question {iteration} from {source_file}: {e}")
        import traceback
        traceback.print_exc() # More detailed error

    # Add saved path to results for potential logging elsewhere if needed (optional)
    # results['saved_kv_log_path'] = saved_log_path
    return results


def majority_vote(
    chain_results: List[Dict],
    correct_answer_letter: str,
    tokenizer_path: Optional[str] = None # Pass tokenizer path if needed for tie-break
) -> Tuple[Optional[str], int, List[str]]:
    """
    Performs majority voting on the extracted answers from SC chains.
    Uses completion_tokens from chain_results for N=2 tie-breaking if available.
    """
    # Filter out chains that resulted in an error
    valid_chain_results = [cr for cr in chain_results if "error" not in cr]

    if not valid_chain_results:
        print("Warning: No valid chains found for majority vote.")
        return None, 0, []


    extracted_answers = [chain.get("extracted_answer") for chain in valid_chain_results]
    valid_answers = [ans for ans in extracted_answers if ans is not None]

    if not valid_answers:
        return None, 0, extracted_answers

    answer_counts = Counter(valid_answers)
    most_common = answer_counts.most_common()

    if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
        voted_answer = most_common[0][0]
        score = calculate_score_gpqa(voted_answer, correct_answer_letter)
        return voted_answer, score, extracted_answers

    # Handle Ties
    max_count = most_common[0][1]
    tied_answers = [ans for ans, count in most_common if count == max_count]
    final_voted_answer = None

    if len(valid_chain_results) == 2 and len(tied_answers) == 2: # Use count of valid chains
        print(f"N=2 tie detected ({tied_answers}). Applying tie-breaker (fewest completion tokens).")
        min_tokens = float('inf')
        best_chain_idx_in_valid = -1 # Index within the valid_chain_results list
        token_counts_available = True

        for i, chain in enumerate(valid_chain_results):
            if chain.get("extracted_answer") in tied_answers:
                tokens = chain.get("completion_tokens")
                if tokens is not None:
                    if tokens < min_tokens:
                        min_tokens = tokens
                        best_chain_idx_in_valid = i
                else:
                    token_counts_available = False
                    break

        if token_counts_available and best_chain_idx_in_valid != -1:
            best_chain_data = valid_chain_results[best_chain_idx_in_valid]
            final_voted_answer = best_chain_data.get("extracted_answer")
            print(f"Tie broken via usage stats: Chose chain {best_chain_data['chain_index']} with {min_tokens} tokens (Answer: {final_voted_answer})")
        else:
            print("Usage stats missing for tie-break, attempting fallback to tokenizer counting...")
            if tokenizer_path:
                 min_tokens_fallback = float('inf')
                 best_chain_idx_fallback_in_valid = -1
                 for i, chain in enumerate(valid_chain_results):
                     if chain.get("extracted_answer") in tied_answers:
                         # Use full content for counting, assuming answer format is robust
                         content_to_count = chain.get("full_content", "")
                         tokens_fallback = count_tokens(content_to_count, tokenizer_path)
                         if tokens_fallback != -1 and tokens_fallback < min_tokens_fallback:
                              min_tokens_fallback = tokens_fallback
                              best_chain_idx_fallback_in_valid = i

                 if best_chain_idx_fallback_in_valid != -1:
                     best_chain_data_fallback = valid_chain_results[best_chain_idx_fallback_in_valid]
                     final_voted_answer = best_chain_data_fallback.get("extracted_answer")
                     print(f"Tie broken via tokenizer count: Chose chain {best_chain_data_fallback['chain_index']} with {min_tokens_fallback} tokens (Answer: {final_voted_answer})")
                 else:
                     print("Tokenizer counting failed. Arbitrarily choosing first tied answer.")
                     final_voted_answer = tied_answers[0]
            else:
                print("Tokenizer path not provided for N=2 tie-breaking fallback. Arbitrarily choosing first tied answer.")
                final_voted_answer = tied_answers[0]

    elif len(tied_answers) > 1:
         print(f"Tie detected among: {tied_answers}. Arbitrarily choosing first tied answer.")
         final_voted_answer = tied_answers[0]
    else:
        print(f"Unexpected voting scenario. Tied answers: {tied_answers}. Choosing first.")
        final_voted_answer = tied_answers[0]

    score = calculate_score_gpqa(final_voted_answer, correct_answer_letter)
    return final_voted_answer, score, extracted_answers


async def process_single_stream(stream_generator: AsyncGenerator[Dict, None], chain_index: int) -> Dict:
    """Helper coroutine to consume a stream and process its chunks."""
    chunks = [chunk async for chunk in stream_generator]
    return process_stream_chunks(chunks, chain_index)


async def process_question_sc_stream(
    example: Dict,
    iteration: int,
    n_chains: int,
    paths: Dict[str, str],
    vllm_url: str,
    model_name: str,
    tokenizer_path: Optional[str] # Needed for tie-break fallback
) -> Optional[Dict]:
    """
    Processes a single question using Self-Consistency by consuming parallel streams.

    Args:
        example (Dict): The dataset example.
        iteration (int): The 1-based index of the question.
        n_chains (int): Number of SC chains to run.
        paths (Dict[str, str]): Dictionary of output paths.
        vllm_url (str): URL of the vLLM server.
        model_name (str): Name of the model in vLLM.
        tokenizer_path (Optional[str]): Path to tokenizer model/files.

    Returns:
        Optional[Dict]: Aggregated results for this question, or None on failure.
    """
    print(f"\n--- Processing Question {iteration} (N={n_chains}, Streaming) ---")
    question_id = example.get("id", f"index_{iteration-1}")

    # Create Prompt
    try:
        prompt, choices, correct_answer_letter = create_prompt_gpqa(example)
    except Exception as e:
        print(f"Error creating prompt for question {iteration}: {e}")
        return None

    # Create and Run Async Tasks to consume N streams
    stream_consumers = []
    for i in range(n_chains):
        request_id = f"q{iteration}_c{i+1}_stream"
        stream_generator = stream_vllm_request( # Get the async generator
            prompt=prompt,
            vllm_url=vllm_url,
            model_name=model_name,
            request_id=request_id,
            temperature=0.6,
            max_tokens=32768,
            # stop_sequences=["\nAnswer:", "Final Answer:"], # Optional
            logprobs=None # Logprobs in stream not standard/reliable yet
        )
        # Create a task that consumes the generator and processes chunks
        stream_consumers.append(process_single_stream(stream_generator, i + 1))

    start_process_time = time.time()
    processed_chain_results = await asyncio.gather(*stream_consumers, return_exceptions=True)

    end_process_time = time.time()
    gather_duration = end_process_time - start_process_time
    print(f"Gathered & Processed {n_chains} streams for Q{iteration} in {gather_duration:.2f}s")

    # Analyze KV Cache Usage
    kv_cache_stats = analyze_kv_cache_usage_for_question(
        start_process_time, end_process_time, iteration, paths
    )
    avg_kv_usage = kv_cache_stats.get('avg_kv_cache_usage')
    max_kv_usage = kv_cache_stats.get('max_kv_cache_usage')

    # Process Gathered Results (which are already processed chains)
    successful_chains_data = []
    failed_chains_count = 0
    total_reasoning_tokens_agg = 0
    total_non_reasoning_tokens_agg = 0
    total_completion_tokens_agg = 0 # Aggregate from usage stats
    first_prompt_tokens = None
    token_counting_possible = (tokenizer_path is not None) # Flag if counting is attempted

    for result in processed_chain_results:
        if isinstance(result, Exception) or "error" in result:
            error_info = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            chain_idx_err = result.get("chain_index", "N/A") if isinstance(result, dict) else "N/A"
            print(f"Stream consumer task/chain {chain_idx_err} failed: {error_info}")
            failed_chains_count += 1
        else:
            # Successfully processed chain data
            chain_data = result
            successful_chains_data.append(chain_data)

            # Extract answer (assuming it's in final_answer_text)
            chain_data["extracted_answer"] = extract_answer_gpqa(chain_data.get("final_answer_text"))

            # **** Count Tokens using Tokenizer ****
            chain_reasoning_tokens = None
            chain_non_reasoning_tokens = None
            if token_counting_possible:
                reasoning_text = chain_data.get("reasoning_text", "")
                final_answer_text = chain_data.get("final_answer_text", "")

                chain_reasoning_tokens = count_tokens(reasoning_text, tokenizer_path)
                chain_non_reasoning_tokens = count_tokens(final_answer_text, tokenizer_path)

                chain_data["reasoning_tokens"] = chain_reasoning_tokens # Store per-chain count (can be None)
                chain_data["non_reasoning_tokens"] = chain_non_reasoning_tokens # Store per-chain count (can be None)

                # Add to aggregate if counting was successful
                if chain_reasoning_tokens is not None:
                    total_reasoning_tokens_agg += chain_reasoning_tokens
                if chain_non_reasoning_tokens is not None:
                    total_non_reasoning_tokens_agg += chain_non_reasoning_tokens
            else:
                 chain_data["reasoning_tokens"] = None # Mark as not counted
                 chain_data["non_reasoning_tokens"] = None

            # Aggregate usage stats from server report
            if chain_data.get("completion_tokens") is not None:
                total_completion_tokens_agg += chain_data["completion_tokens"]
            if first_prompt_tokens is None and chain_data.get("prompt_tokens") is not None:
                 first_prompt_tokens = chain_data["prompt_tokens"]
            elif chain_data["prompt_tokens"] is not None and first_prompt_tokens is not None and chain_data["prompt_tokens"] != first_prompt_tokens:
                  print(f"Warning: Prompt token mismatch in Q{iteration}. Chain {chain_data['chain_index']}: {chain_data['prompt_tokens']}, First: {first_prompt_tokens}")

            # Save individual chain output
            chain_filename = os.path.join(paths["chains"], f"question_{iteration}_chain_{chain_data['chain_index']}.txt")
            try:
                with open(chain_filename, "w") as f:
                    f.write(f"--- Chain {chain_data['chain_index']} for Question {iteration} ---\n")
                    f.write(f"Finish Reason: {chain_data.get('finish_reason', 'N/A')}\n")
                    f.write(f"Extracted Answer: {chain_data.get('extracted_answer', 'N/A')}\n")
                    f.write(f"Prompt Tokens (Usage): {chain_data.get('prompt_tokens', 'N/A')}\n")
                    f.write(f"Completion Tokens (Usage): {chain_data.get('completion_tokens', 'N/A')}\n")
                    f.write(f"Reasoning Tokens (Counted): {chain_reasoning_tokens if chain_reasoning_tokens is not None else 'N/A'}\n") # Display counted tokens
                    f.write(f"Non-Reasoning Tokens (Counted): {chain_non_reasoning_tokens if chain_non_reasoning_tokens is not None else 'N/A'}\n") # Display counted tokens
                    f.write("\n--- Reasoning Content ---\n")
                    f.write(chain_data.get('reasoning_text', 'N/A'))
                    f.write("\n\n--- Final Answer Content ---\n")
                    f.write(chain_data.get('final_answer_text', 'N/A'))
            except IOError as e:
                print(f"Error writing chain output file {chain_filename}: {e}")


    n_chains_received = len(successful_chains_data)
    if n_chains_received == 0:
         print(f"All chains failed for question {iteration}. Skipping.")
         # Save failure summary
         summary_data = {
            "iteration": iteration, "question_id": question_id, "status": "ALL_CHAINS_FAILED",
            "prompt": prompt, "choices": choices, "correct_answer": correct_answer_letter,
            "n_chains_requested": n_chains, "n_chains_received": 0,
            "avg_kv_cache_usage": avg_kv_usage, # Record KV if analysis ran
            "max_kv_cache_usage": max_kv_usage,
         }
         summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
         try:
            with open(summary_filename, "w") as f:
                json.dump(summary_data, f, indent=2)
         except IOError as e:
            print(f"Error writing failure summary file: {e}")
         return None

    # Perform Majority Vote on successful chains data
    voted_answer, final_score, all_extracted_answers = majority_vote(
        successful_chains_data, correct_answer_letter, tokenizer_path
    )

    # Aggregate Results and Save Summary
    prompt_tokens = first_prompt_tokens if first_prompt_tokens is not None else 0
    total_tokens = prompt_tokens + total_completion_tokens_agg

    summary_data = {
        "iteration": iteration,
        "question_id": question_id,
        "status": "SUCCESS" if failed_chains_count == 0 else f"PARTIAL_SUCCESS ({failed_chains_count}_failed)",
        "n_chains_requested": n_chains,
        "n_chains_received": n_chains_received,
        "prompt_len": len(prompt),
        "correct_answer_letter": correct_answer_letter,
        "individual_answers": all_extracted_answers,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "avg_kv_cache_usage": f"{avg_kv_usage:.4f}",
        "max_kv_cache_usage": f"{max_kv_usage:.4f}",
        "processing_duration_sec": f"{gather_duration:.1f}",
        "usage_aggregated": {
             "prompt_tokens": prompt_tokens,
             "total_completion_tokens_usage": total_completion_tokens_agg,
             "total_reasoning_tokens_counted": total_reasoning_tokens_agg if token_counting_possible else None,
             "total_non_reasoning_tokens_counted": total_non_reasoning_tokens_agg if token_counting_possible else None,
             "total_tokens_usage": total_tokens
        },
        "chain_details": [ # Save key details per successful chain
             {
                "chain_index": cr.get("chain_index"),
                "finish_reason": cr.get("finish_reason"),
                "extracted_answer": cr.get("extracted_answer"),
                "prompt_tokens": cr.get("prompt_tokens"),
                "completion_tokens": cr.get("completion_tokens"),
                "reasoning_tokens": cr.get("reasoning_tokens"),
                "non_reasoning_tokens": cr.get("non_reasoning_tokens"),
             } for cr in successful_chains_data
         ]
    }

    summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
    try:
        with open(summary_filename, "w") as f:
            json.dump(summary_data, f, indent=2)
    except IOError as e:
        print(f"Error writing summary file {summary_filename}: {e}")

    # Return data for final CSV
    return {
        "iteration": iteration,
        "question_id": question_id,
        "n_chains_requested": n_chains,
        "n_chains_received": n_chains_received,
        "correct_answer": correct_answer_letter,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "prompt_tokens": prompt_tokens,
        "total_completion_tokens": total_completion_tokens_agg, # Report usage total completion
        "total_reasoning_tokens": total_reasoning_tokens_agg if token_counting_possible else None, # Report counted reasoning
        "total_non_reasoning_tokens": total_non_reasoning_tokens_agg if token_counting_possible else None, # Report counted non-reasoning
        "total_tokens": total_tokens, # Report usage total
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
        "individual_answers_str": json.dumps(all_extracted_answers)
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
    print(f"Starting Async SC Streaming Eval w/ Token Counting: N={n_chains}, Model={model_name}")
    if tokenizer_path is None:
        print("WARNING: --tokenizer_path not provided. Reasoning/Non-reasoning token counts will not be calculated.")
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
            print(f"Resuming. Found {len(processed_iterations)} previously processed iterations.")
        except Exception as e:
            print(f"Warning: Could not read existing results file {paths['csv']}. Starting fresh. Error: {e}")
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
            print(f"Error: Start iteration ({start}) > End iteration ({end}).")
            return
        target_iterations = set(range(start, end + 1))

    iterations_to_process = sorted(list(target_iterations - processed_iterations))

    if not iterations_to_process:
        print("No new iterations to process.")
    else:
        print(f"Need to process {len(iterations_to_process)} iterations.")
        pbar = tqdm(total=len(iterations_to_process), desc=f"GPQA N={n_chains} Stream")
        for i in iterations_to_process:
            example = examples[i-1]
            # **** CALL THE STREAMING VERSION ****
            result = await process_question_sc_stream(
                example, i, n_chains, paths, vllm_url, model_identifier, tokenizer_path
            )
            if result:
                results_list.append(result)
                # Incremental save logic (same as before)
                try:
                    df = pd.DataFrame(results_list)
                    for col in csv_cols:
                         if col not in df.columns: df[col] = None
                    df = df[csv_cols].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")
                    df.to_csv(paths["csv"], index=False)
                except Exception as e:
                    print(f"\nError saving intermediate CSV: {e}")
            else:
                print(f"Skipping results storage for question {i} due to processing error.")
            pbar.update(1)
        pbar.close()

    # Final processing and summary (same as before)
    if results_list:
        final_df = pd.DataFrame(results_list)
        for col in csv_cols:
            if col not in final_df.columns: final_df[col] = None
        final_df = final_df[csv_cols].sort_values(by="iteration").drop_duplicates(subset=["iteration"], keep="last")

        try:
             final_df.to_csv(paths["csv"], index=False)
             print(f"\nEvaluation complete. Final results saved to {paths['csv']}")
        except Exception as e:
             print(f"\nError performing final save to CSV: {e}")

        # Print summary stats (same as before)
        if 'final_score' in final_df.columns and not final_df['final_score'].isnull().all():
            accuracy = final_df['final_score'].astype(float).mean()
            print(f"Overall Accuracy: {accuracy:.2f}")
        else:
            print("Accuracy could not be calculated (no valid scores).")

        if 'prompt_tokens' in final_df.columns and not final_df['prompt_tokens'].isnull().all():
             avg_prompt_tokens = final_df['prompt_tokens'].astype(float).mean()
             print(f"Avg Prompt Tokens: {avg_prompt_tokens:.1f}")
        if 'total_completion_tokens' in final_df.columns and not final_df['total_completion_tokens'].isnull().all():
             avg_completion_tokens = final_df['total_completion_tokens'].astype(float).mean()
             print(f"Avg Completion Tokens (Total across successful chains): {avg_completion_tokens:.1f}")
        if 'total_reasoning_tokens' in final_df.columns and not final_df['total_reasoning_tokens'].isnull().all():
             avg_reasoning = final_df['total_reasoning_tokens'].astype(float).mean()
             print(f"Avg Reasoning Tokens (Counted): {avg_reasoning:.1f}")
        if 'total_non_reasoning_tokens' in final_df.columns and not final_df['total_non_reasoning_tokens'].isnull().all():
             avg_non_reasoning = final_df['total_non_reasoning_tokens'].astype(float).mean()
             print(f"Avg Non-Reasoning Tokens (Counted): {avg_non_reasoning:.1f}")
        if 'total_tokens' in final_df.columns and not final_df['total_tokens'].isnull().all():
             avg_total_tokens = final_df['total_tokens'].astype(float).mean()
             print(f"Avg Total Tokens (Aggregated): {avg_total_tokens:.1f}")
        if 'avg_kv_cache_usage' in final_df.columns and final_df['avg_kv_cache_usage'].notna().any():
             avg_kv = final_df['avg_kv_cache_usage'].dropna().astype(float).mean()
             print(f"Avg KV Cache Usage (Question Avg): {avg_kv:.4f}")
        if 'max_kv_cache_usage' in final_df.columns and final_df['max_kv_cache_usage'].notna().any():
             avg_max_kv = final_df['max_kv_cache_usage'].dropna().astype(float).mean()
             overall_max_kv = final_df['max_kv_cache_usage'].dropna().astype(float).max()
             print(f"Avg Max KV Cache Usage (Question Max): {avg_max_kv:.4f}")
             print(f"Overall Max KV Cache Usage Recorded: {overall_max_kv:.4f}")
        print("------------------------\n")

    else:
        print("No results were processed or loaded.")

    await close_aiohttp_session() # Ensure session is closed


# --- main function remains the same, it just calls run_sc_evaluation_async ---
def main():
    parser = argparse.ArgumentParser(description='Run Async SC (Control) Evaluation using vLLM Streaming.')
    # Arguments are the same
    parser.add_argument('--sc_chains', type=int, required=True, choices=[2, 3, 5], help='Number of self-consistency chains (N).')
    parser.add_argument('--vllm_url', type=str, default="http://localhost:8000", help='URL of the vLLM server OpenAI-compatible endpoint.')
    parser.add_argument('--model_name', type=str, required=True, help='Short name for the model used for directory structures.')
    parser.add_argument('--model_identifier', type=str, required=True, help='Full model identifier used by vLLM API.')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Path to the HuggingFace tokenizer directory (REQUIRED for reasoning token counts).')
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help='Name of the GPQA subset/dataset.')
    parser.add_argument('--output_dir', type=str, default="./slimsc/prune/results", help='Base directory to save evaluation results.')
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed).')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive).')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated list of specific iterations.')
    args = parser.parse_args()

    if args.tokenizer_path is None:
         print("\n*** WARNING: --tokenizer_path not provided. ***")
         print("Reasoning and Non-Reasoning token counts cannot be calculated and will be missing from results.\n")
         time.sleep(2)

    specific_iterations_list = None
    if args.iterations:
        try:
            specific_iterations_list = [int(x.strip()) for x in args.iterations.split(',') if x.strip()]
            if not specific_iterations_list: raise ValueError("Empty list.")
        except ValueError:
            print("Error: Invalid format for --iterations.")
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
        print("\nEvaluation interrupted by user.")
    finally:
        print("Cleanup potentially needed if interrupted before session close.")


if __name__ == "__main__":
    main()