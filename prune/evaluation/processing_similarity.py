# slimsc/prune/evaluation/processing_similarity.py
import os
import json
import time
import asyncio
import numpy as np
from typing import Dict, Optional, List, Tuple, Set, AsyncGenerator

# Keep necessary imports
from ..clients import stream_vllm_request, close_aiohttp_session
from ..utils import create_prompt_gpqa, extract_answer_gpqa
from ..utils.similarity_utils import (
    FaissIndexManager, embed_segments, find_newly_completed_thoughts,
    extract_final_thought, get_embedding_model, MIN_SEGMENT_TOKENS
)
from .voting import majority_vote
from .kv_cache_extraction import extract_kv_cache_usage_for_question

import logging
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_TOKENS_PER_STREAM = 32768 # Max completion tokens per chain
ANALYSIS_INTERVAL_SECONDS = 3 # How often to check for new thoughts/prune
# Target phrases for identifying new thoughts (move to config?)
TARGET_PHRASES = ["alternative", "Alternative", "Another", "But another", "perhaps another", "Wait", "Oh wait", "But wait"]


async def stream_processing_worker(
    chain_id: str,
    stream_generator: AsyncGenerator[Dict, None],
    chain_state: Dict, # Reference to the shared state dict for this chain
    all_tasks_done_event: asyncio.Event # Event to signal when this worker finishes
):
    """
    Consumes a stream for a single chain, updates shared state, and handles termination.
    """
    logger.debug(f"Worker started for chain {chain_id}")
    try:
        async for chunk in stream_generator:
            if not chain_state["is_active"]: # Check if pruned by main loop
                logger.warning(f"Worker {chain_id}: Chain marked inactive, stopping consumption.")
                # Force finish reason if not already set by pruning logic
                if not chain_state.get("finish_reason"):
                    chain_state["finish_reason"] = "cancelled_inactive"
                break

            # Process chunk data
            if "error" in chunk:
                err_msg = chunk['error'].get('message', 'Unknown stream error')
                logger.error(f"Worker {chain_id}: Error chunk received: {err_msg}")
                chain_state["error"] = err_msg
                chain_state["finish_reason"] = f"error: {err_msg}"
                break # Stop processing on error

            if not chunk or "choices" not in chunk or not chunk["choices"]:
                continue # Skip empty/invalid chunks

            delta = chunk["choices"][0].get("delta", {})
            content = None
            if "content" in delta and delta["content"] is not None:
                content = delta["content"]
            elif "reasoning_content" in delta and delta["reasoning_content"] is not None:
                 content = delta["reasoning_content"]

            if content:
                 chain_state["full_text"] += content

            # Update token counts from usage if present in chunk
            if "usage" in chunk and chunk["usage"] is not None:
                 usage = chunk["usage"]
                 if chain_state["prompt_tokens"] is None: # Capture initial prompt tokens
                      chain_state["prompt_tokens"] = usage.get("prompt_tokens", 0)
                 # Completion tokens might be cumulative or per-chunk depending on server
                 # Assuming cumulative or final chunk has total:
                 chain_state["completion_tokens"] = usage.get("completion_tokens", chain_state["completion_tokens"])
                 # Alternative if only per-chunk completion:
                 # chain_state["completion_tokens"] += usage.get("completion_tokens", 0)

            # Check for natural stop
            chunk_finish_reason = chunk["choices"][0].get("finish_reason")
            if chunk_finish_reason and chunk_finish_reason == 'stop':
                 logger.info(f"Worker {chain_id}: Stream finished naturally (stop sequence).")
                 chain_state["finish_reason"] = chunk_finish_reason
                 break # Stop processing

        # End of stream loop (natural finish, error, or break due to inactivity)
        logger.debug(f"Worker {chain_id}: Stream loop finished. Reason: {chain_state.get('finish_reason', 'N/A')}")

    except asyncio.CancelledError:
        logger.warning(f"Worker {chain_id}: Task cancelled (likely due to pruning).")
        chain_state["finish_reason"] = chain_state.get("finish_reason", "cancelled_pruned") # Keep prune reason if set
    except Exception as e:
        logger.exception(f"Worker {chain_id}: Unhandled exception during stream consumption.")
        chain_state["error"] = f"Worker Exception: {e}"
        chain_state["finish_reason"] = chain_state.get("finish_reason", "error_worker_exception")
    finally:
        # Mark as finished regardless of how it stopped
        chain_state["finished"] = True
        if not chain_state.get("finish_reason"): # Ensure some reason is set
             chain_state["finish_reason"] = "worker_terminated"
        logger.debug(f"Worker {chain_id} concluding. Final state: finished={chain_state['finished']}, reason={chain_state['finish_reason']}")
        # Signal that this worker is done
        all_tasks_done_event.set() # Set event to potentially wake up main loop


async def process_question_similarity_prune(
    example: Dict,
    iteration: int,
    n_chains_start: int,
    paths: Dict[str, str], # Will contain KV cache paths
    vllm_url: str,
    model_name: str,
    tokenizer_path: str,
    similarity_threshold: float,
    max_analysis_steps: int = 100 # Limit analysis intervals
) -> Optional[Dict]:
    """
    Processes a question using Similarity Pruning with continuous streams.
    """
    logger.info(f"--- Processing Question {iteration} (N={n_chains_start}, SimPrune-Continuous Thresh={similarity_threshold}) ---")
    question_id = example.get("id", f"index_{iteration-1}")
    start_process_time = time.time()

    # --- Setup ---
    try:
        prompt, choices, correct_answer_letter = create_prompt_gpqa(example)
    except Exception as e: logger.exception(f"[red]Error creating prompt[/red]"); return None
    try:
        embedding_model = get_embedding_model()
        index_manager = FaissIndexManager(dimension=embedding_model.get_sentence_embedding_dimension())
    except Exception as e: logger.exception(f"[red]Failed init embedding/FAISS[/red]"); return None

    # --- Shared State and Task Management ---
    active_chains: Dict[str, Dict] = {}
    consumer_tasks: Dict[str, asyncio.Task] = {}
    all_tasks_done_event = asyncio.Event() # Used to signal worker completion

    for i in range(n_chains_start):
        chain_id = f"q{iteration}_c{i+1}"
        # Initialize state
        active_chains[chain_id] = {
            "id": chain_id, "full_text": "", "processed_boundaries": [],
            "completed_thought_count": 0, "is_active": True, "finished": False,
            "finish_reason": None, "error": None, "prompt_tokens": None, "completion_tokens": 0
        }
        # Create the long-running stream request
        stream_generator = stream_vllm_request(
            prompt=prompt, vllm_url=vllm_url, model_name=model_name,
            request_id=chain_id, # Use chain_id as the persistent request_id
            temperature=0.6,
            max_tokens=MAX_TOKENS_PER_STREAM, # Request a large number of tokens
            # stop_sequences=["\nAnswer:", "Final Answer:", "\n---"],
            logprobs=None
        )
        # Create and store the consumer task
        task = asyncio.create_task(
            stream_processing_worker(chain_id, stream_generator, active_chains[chain_id], all_tasks_done_event),
            name=f"worker_{chain_id}" # Name task for easier debugging
        )
        consumer_tasks[chain_id] = task

    # --- Periodic Analysis Loop ---
    analysis_step = 0
    while analysis_step < max_analysis_steps:
        analysis_step += 1
        logger.info(f"[bold cyan][Q{iteration} Analysis Interval {analysis_step}] Checking chains...[/bold cyan]")

        # --- Check for newly completed thoughts across all *active* chains ---
        active_chain_ids_now = [cid for cid, state in active_chains.items() if state["is_active"] and not state["finished"]]
        if not active_chain_ids_now:
             logger.info(f"[Q{iteration} Analysis Interval {analysis_step}] No active chains remain. Stopping analysis.")
             break # Exit analysis loop if all chains are done/pruned/errored

        all_new_thoughts_texts: List[Tuple[str, int, str]] = [] # (chain_id, thought_idx, text)

        for chain_id in active_chain_ids_now:
             chain_state = active_chains[chain_id]
             # Check based on current full_text vs processed_boundaries
             new_segments, updated_boundaries = find_newly_completed_thoughts(
                 chain_state["full_text"],
                 chain_state["processed_boundaries"],
                 tokenizer_path,
                 target_phrases=TARGET_PHRASES,
                 min_segment_tokens=MIN_SEGMENT_TOKENS
             )
             if new_segments:
                 logger.debug(f"Chain {chain_id}: Found {len(new_segments)} new completed thoughts in interval {analysis_step}.")
                 for start_idx, end_idx, text in new_segments:
                     thought_idx = chain_state["completed_thought_count"]
                     all_new_thoughts_texts.append((chain_id, thought_idx, text))
                     chain_state["completed_thought_count"] += 1
                 chain_state["processed_boundaries"] = updated_boundaries # Update state

        # --- Embed new thoughts ---
        newly_completed_thoughts_for_embedding: List[Tuple[str, int, str, np.ndarray]] = []
        if all_new_thoughts_texts:
            texts_only = [t[2] for t in all_new_thoughts_texts]
            embeddings = embed_segments(texts_only)
            if embeddings is None or len(embeddings) != len(all_new_thoughts_texts):
                logger.error("[red]Embedding failed or returned incorrect number. Skipping similarity check this interval.[/red]")
            else:
                for i, (chain_id, thought_idx, text) in enumerate(all_new_thoughts_texts):
                    newly_completed_thoughts_for_embedding.append((chain_id, thought_idx, text, embeddings[i]))

        # --- Check Similarity and Prune ---
        chains_to_prune_this_interval: Set[str] = set()
        embeddings_to_add_to_index: List[Tuple[np.ndarray, str, int, str]] = []

        if newly_completed_thoughts_for_embedding:
            logger.info(f"[Q{iteration} Int {analysis_step}] Checking similarity for {len(newly_completed_thoughts_for_embedding)} new thoughts.")
            for chain_id, thought_idx, text, embedding in newly_completed_thoughts_for_embedding:
                 # Skip if chain already pruned in this interval or inactive
                 if chain_id in chains_to_prune_this_interval or not active_chains[chain_id]["is_active"]: continue

                 chain_state = active_chains[chain_id]
                 # Pruning check (3rd thought onwards)
                 if thought_idx >= 2 and index_manager.get_num_active_chains() >= 1:
                     neighbor_result = index_manager.search_nearest_neighbor(embedding, chain_id)
                     if neighbor_result:
                         sim_score, neighbor_chain_id, _, _ = neighbor_result
                         if sim_score > similarity_threshold:
                             logger.warning(f"[bold yellow]PRUNING condition![/bold yellow] Chain {chain_id} (T{thought_idx}) vs {neighbor_chain_id}, score={sim_score:.4f}")
                             if len(chain_state['full_text']) <= len(active_chains[neighbor_chain_id]['full_text']):
                                 logger.warning(f"--> Pruning Chain {chain_id} (Shorter/Equal).")
                                 chains_to_prune_this_interval.add(chain_id)

                 # Prepare to add embedding if not pruned
                 if chain_id not in chains_to_prune_this_interval:
                     embeddings_to_add_to_index.append((embedding, chain_id, thought_idx, text))

        # --- Apply Pruning and Update FAISS ---
        if chains_to_prune_this_interval:
            for prune_id in chains_to_prune_this_interval:
                if active_chains[prune_id]["is_active"]: # Check if not already inactive
                    logger.info(f"Marking chain {prune_id} as inactive/pruned.")
                    active_chains[prune_id]["is_active"] = False
                    active_chains[prune_id]["finished"] = True
                    active_chains[prune_id]["finish_reason"] = "pruned_similarity"
                    # Cancel the corresponding worker task
                    if prune_id in consumer_tasks:
                        logger.info(f"Cancelling worker task for pruned chain {prune_id}")
                        consumer_tasks[prune_id].cancel()
                        # Optionally remove task from dict? Or let it be handled by wait
                    index_manager.remove_chain_embeddings(prune_id)

        # Add embeddings for non-pruned thoughts
        for emb, cid, tidx, txt in embeddings_to_add_to_index:
             if active_chains[cid]["is_active"] and not active_chains[cid]["finished"]:
                  index_manager.add_embedding(emb, cid, tidx, txt)

        # --- Wait for next interval or task completion ---
        # Use asyncio.wait to sleep but wake up if any task finishes
        logger.debug(f"Analysis interval {analysis_step} done. Waiting...")
        # Get tasks that are still running (not cancelled or done)
        running_consumer_tasks = [t for t in consumer_tasks.values() if not t.done()]
        if not running_consumer_tasks:
            logger.info("No running consumer tasks left. Exiting analysis loop.")
            break # All tasks are finished/cancelled

        # Reset the event before waiting
        all_tasks_done_event.clear()
        # Wait for the specified interval OR for the event to be set by a finishing worker
        try:
            # Wait for timeout OR the event being set
            await asyncio.wait_for(all_tasks_done_event.wait(), timeout=ANALYSIS_INTERVAL_SECONDS)
            logger.debug(f"Woke up from wait: Event was set (a task finished).")
        except asyncio.TimeoutError:
            logger.debug(f"Woke up from wait: Timeout reached.")
            pass # Timeout is normal, continue to next analysis interval

    # --- Analysis loop finished ---
    logger.info(f"Analysis loop completed after {analysis_step} intervals.")

    # --- Ensure all tasks are truly finished ---
    # Wait for any potentially cancelled tasks to finish cleanup
    remaining_tasks = [t for t in consumer_tasks.values() if not t.done()]
    if remaining_tasks:
        logger.info(f"Waiting for {len(remaining_tasks)} remaining worker tasks to finalize...")
        await asyncio.wait(remaining_tasks, timeout=10) # Short timeout for cleanup

    end_process_time = time.time() # Capture time after all workers should be done
    total_duration = end_process_time - start_process_time
    logger.info(f"[bold green]Finished processing Q{iteration} in {total_duration:.2f}s.[/bold green]")

    # --- KV Cache Extraction ---
    # Done *after* all generation and processing for the question finishes.
    kv_cache_stats = extract_kv_cache_usage_for_question(
        start_process_time, end_process_time, iteration, paths
    )
    avg_kv_usage = kv_cache_stats.get('avg_kv_cache_usage')
    max_kv_usage = kv_cache_stats.get('max_kv_cache_usage')


    # --- Post-Processing, Voting, Saving ---
    # Collect final states (check 'finished' flag set by workers or pruning)
    final_active_chains_data = []
    pruned_chains_data = []
    error_chains_data = []
    for chain_id, state in active_chains.items():
        if state.get("finish_reason") == "pruned_similarity":
            pruned_chains_data.append(state)
        elif state.get("error"):
            error_chains_data.append(state)
        elif state.get("finished"): # Finished naturally, by token limit, or cancelled
             # Only count as 'active final' if not pruned or errored
             if state.get("finish_reason") != "cancelled_pruned" and \
                state.get("finish_reason") != "cancelled_inactive":
                 final_active_chains_data.append(state)
        # else: chain didn't finish within max_analysis_steps? Treat as incomplete/error?

    logger.info(f"Q{iteration} Final Status: {len(final_active_chains_data)} chains completed, {len(pruned_chains_data)} pruned, {len(error_chains_data)} errors.")

    # Prepare results for voting (using completed chains)
    successful_chain_results = []
    total_completion_tokens_agg = 0
    total_prompt_tokens_agg = 0

    for chain_state in final_active_chains_data:
        extracted_answer = extract_answer_gpqa(chain_state["full_text"])
        result_data = {
            "chain_index": int(chain_state['id'].split('_c')[-1]),
            "full_content": chain_state["full_text"],
            "reasoning_text": chain_state["full_text"],
            "final_answer_text": "",
            "extracted_answer": extracted_answer,
            "finish_reason": chain_state["finish_reason"],
            "prompt_tokens": chain_state.get("prompt_tokens", 0),
            "completion_tokens": chain_state.get("completion_tokens", 0),
            "completed_thought_count": chain_state.get("completed_thought_count", 0)
        }
        successful_chain_results.append(result_data)
        total_completion_tokens_agg += result_data["completion_tokens"]
        if total_prompt_tokens_agg == 0 and result_data["prompt_tokens"] > 0:
             total_prompt_tokens_agg = result_data["prompt_tokens"]

    # Majority Vote
    if not successful_chain_results:
        logger.warning(f"[Q{iteration}] No successful chains remaining for majority vote.")
        voted_answer, final_score, all_extracted_answers = None, 0, []
    else:
        voted_answer, final_score, all_extracted_answers = majority_vote(
            successful_chain_results, correct_answer_letter, tokenizer_path
        )

    # --- Save Outputs ---
    # (Save individual chain files - logic mostly same, use final status determined above)
    for chain_id, chain_state in active_chains.items():
         chain_idx = int(chain_id.split('_c')[-1])
         final_status = "unknown"
         if chain_state.get("error"): final_status = "error"
         elif chain_state.get("finish_reason") == "pruned_similarity": final_status = "pruned"
         elif chain_state.get("finish_reason") == "cancelled_pruned": final_status = "pruned" # Treat cancelled also as pruned status
         elif chain_state.get("finish_reason") == "cancelled_inactive": final_status = "pruned" # Treat cancelled also as pruned status
         elif chain_state.get("finished"): final_status = "finished" # Naturally or max tokens
         else: final_status = "incomplete" # If loop finished before chain did

         chain_filename = os.path.join(paths["chains"], f"question_{iteration}_chain_{chain_idx}_{final_status}.txt")
         try:
             with open(chain_filename, "w", encoding='utf-8') as f:
                 f.write(f"--- Chain {chain_idx} for Question {iteration} ---\n")
                 f.write(f"Status: {final_status.upper()}\n")
                 f.write(f"Finish Reason: {chain_state.get('finish_reason', 'N/A')}\n")
                 f.write(f"Error: {chain_state.get('error', 'N/A')}\n")
                 f.write(f"Prompt Tokens (Usage): {chain_state.get('prompt_tokens', 'N/A')}\n")
                 f.write(f"Completion Tokens (Usage): {chain_state.get('completion_tokens', 'N/A')}\n")
                 f.write(f"Completed Thoughts Processed: {chain_state.get('completed_thought_count', 0)}\n")
                 f.write(f"Final Processed Boundaries: {chain_state.get('processed_boundaries', [])}\n")
                 f.write("\n--- Full Content ---\n")
                 f.write(chain_state.get('full_text', 'N/A'))
         except IOError as e:
             logger.exception(f"[red]Error writing chain output file {chain_filename}[/red]")

    # Save summary JSON
    summary_data = {
        "iteration": iteration, "question_id": question_id,
        "status": "SUCCESS" if successful_chain_results else "ALL_PRUNED_OR_ERROR",
        "n_chains_start": n_chains_start,
        "n_chains_final_completed": len(successful_chain_results), # Chains that finished ok
        "n_chains_pruned": len(pruned_chains_data),
        "n_chains_error": len(error_chains_data),
        "similarity_threshold": similarity_threshold,
        "correct_answer_letter": correct_answer_letter,
        "individual_answers_final": all_extracted_answers,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "processing_duration_sec": f"{total_duration:.1f}",
        "total_analysis_intervals": analysis_step,
        "avg_kv_cache_usage": f"{avg_kv_usage:.4f}" if avg_kv_usage is not None else None,
        "max_kv_cache_usage": f"{max_kv_usage:.4f}" if max_kv_usage is not None else None,
        "usage_aggregated": { # Aggregated from *successfully completed* chains
             "prompt_tokens": total_prompt_tokens_agg,
             "total_completion_tokens_usage": total_completion_tokens_agg,
             "total_tokens_usage": total_prompt_tokens_agg + total_completion_tokens_agg
        },
         "final_chain_details": [
             {
                "chain_index": cr.get("chain_index"),
                "finish_reason": cr.get("finish_reason"),
                "extracted_answer": cr.get("extracted_answer"),
                "completion_tokens": cr.get("completion_tokens"),
                "completed_thought_count": cr.get("completed_thought_count")
             } for cr in successful_chain_results
         ]
    }
    summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
    try:
        with open(summary_filename, "w", encoding='utf-8') as f:
            # Custom serializer for numpy floats potentially in KV stats
            def default_serializer(obj):
                if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
            json.dump(summary_data, f, indent=2, default=default_serializer)
    except IOError as e:
        logger.exception(f"[red]Error writing summary file {summary_filename}[/red]")
    except TypeError as e:
         logger.exception(f"[red]Error serializing summary data to JSON: {e}[/red]")

    # --- Return data for final CSV ---
    return {
        "iteration": iteration,
        "question_id": question_id,
        "n_chains_start": n_chains_start,
        "n_chains_final": len(successful_chain_results), # Use count of successfully completed chains
        "n_chains_pruned": len(pruned_chains_data),
        "similarity_threshold": similarity_threshold,
        "correct_answer": correct_answer_letter,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "prompt_tokens": total_prompt_tokens_agg,
        "total_completion_tokens": total_completion_tokens_agg,
        "total_tokens": total_prompt_tokens_agg + total_completion_tokens_agg,
        "individual_answers_str": json.dumps(all_extracted_answers),
        "total_steps": analysis_step, # Renamed from total_steps to total_analysis_intervals
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
    }