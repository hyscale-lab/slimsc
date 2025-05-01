# slimsc/prune/evaluation/processing_similarity.py
import os
import json
import time
import asyncio
import numpy as np
from typing import Dict, Optional, List, Tuple, Set, AsyncGenerator, Any

# Keep necessary imports
from ..clients import stream_vllm_request, close_aiohttp_session
from ..utils import create_prompt_gpqa, extract_answer_gpqa
from ..utils.similarity_utils import (
    FaissIndexManager, embed_segments, find_newly_completed_thoughts,
    extract_final_thought, get_embedding_model, MIN_SEGMENT_TOKENS, TARGET_PHRASES
)
from .voting import majority_vote_for_sim_prune
from .kv_cache_extraction import extract_kv_cache_usage_for_question

import logging
logger = logging.getLogger(__name__)

# --- Constants ---
MAX_TOKENS_PER_STREAM = 32768 # Max completion tokens per chain
ANALYSIS_INTERVAL_SECONDS = 3 # How often to check for new thoughts/prune


async def stream_processing_worker(
    chain_id: str,
    stream_generator: AsyncGenerator[Dict, None],
    chain_state: Dict, # Reference to the shared state dict for this chain
    all_tasks_done_event: asyncio.Event # Event to signal when this worker finishes
):
    """
    Consumes a stream for a single chain, updates shared state, and handles termination.
    Tracks reasoning completion based on server sending 'content' vs 'reasoning_content'.
    """
    logger.debug(f"Worker started for chain {chain_id}")
    try:
        async for chunk in stream_generator:
            if not chain_state["is_active"]: # Check if pruned by main loop
                logger.warning(f"Worker {chain_id}: Chain marked inactive, stopping consumption.")
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
            reasoning_delta = None
            content_delta = None

            # Check for specific reasoning content
            if "reasoning_content" in delta and delta["reasoning_content"] is not None:
                 reasoning_delta = delta["reasoning_content"]
                 chain_state["full_text"] += reasoning_delta
                 # logger.debug(f"Chain {chain_id}: Received reasoning_content") # Very verbose

            # Check for standard/final content - MUST BE NON-EMPTY
            # This marks the end of the reasoning phase
            # This must be processed after reasoning_content to ensure correct text accumulation
            if "content" in delta and delta["content"]: # Check for non-empty string truthiness
                 content_delta = delta["content"]
                 chain_state["full_text"] += content_delta
                 # logger.debug(f"Chain {chain_id}: Received non-empty content: '{content_delta[:20]}...'") # Verbose
                 if not chain_state["reasoning_complete"]:
                      chain_state["reasoning_complete"] = True
                      # Log the specific content that triggered the switch for debugging
                      logger.info(f"Chain {chain_id}: First non-empty 'content' chunk received. Pruning checks will stop.'")

            # Update token counts from usage if present in chunk
            if "usage" in chunk and chunk["usage"] is not None:
                 usage = chunk["usage"]
                 if chain_state["prompt_tokens"] is None: # Capture initial prompt tokens
                      chain_state["prompt_tokens"] = usage.get("prompt_tokens", 0)
                 chain_state["completion_tokens"] = usage.get("completion_tokens", chain_state["completion_tokens"])

            # Check for natural stop
            chunk_finish_reason = chunk["choices"][0].get("finish_reason")
            if chunk_finish_reason and chunk_finish_reason == 'stop':
                 logger.info(f"Worker {chain_id}: Stream finished naturally (stop sequence).")
                 chain_state["finish_reason"] = chunk_finish_reason
                 # If the stream stops, the reasoning phase is effectively over for pruning
                 # even if no 'content' was received.
                 if not chain_state["reasoning_complete"]:
                    logger.warning(f"Chain {chain_id}: Stopped naturally without receiving non-empty 'content' delta. Marking reasoning_complete anyway.")
                    chain_state["reasoning_complete"] = True
                 break # Stop processing

        # End of stream loop
        logger.debug(f"Worker {chain_id}: Stream loop finished. Reason: {chain_state.get('finish_reason', 'N/A')}")

    except asyncio.CancelledError:
        logger.debug(f"Worker {chain_id}: Task cancelled (likely due to pruning).")
        chain_state["finish_reason"] = chain_state.get("finish_reason", "cancelled_pruned")
        # If cancelled, reasoning is definitely not going to complete naturally via 'content' signal
        if not chain_state["reasoning_complete"]:
            chain_state["reasoning_complete"] = True # Mark as complete for pruning loop check consistency
    except Exception as e:
        logger.exception(f"Worker {chain_id}: Unhandled exception during stream consumption.")
        chain_state["error"] = f"Worker Exception: {e}"
        chain_state["finish_reason"] = chain_state.get("finish_reason", "error_worker_exception")
        if not chain_state["reasoning_complete"]:
            chain_state["reasoning_complete"] = True # Reasoning effectively complete on error
    finally:
        # Mark as finished regardless of how it stopped
        chain_state["finished"] = True
        if not chain_state.get("finish_reason"): # Ensure some reason is set if none captured
             chain_state["finish_reason"] = "worker_terminated_unexpectedly"
        logger.debug(f"Worker {chain_id} concluding. Final state: finished={chain_state['finished']}, reason={chain_state['finish_reason']}, reasoning_complete={chain_state['reasoning_complete']}, completion_tokens={chain_state['completion_tokens']}")
        # Signal that this worker is done
        all_tasks_done_event.set() # Set event to potentially wake up main loop


def calculate_mean_pairwise_similarity(embeddings_list: List[np.ndarray]) -> float:
    """
    Calculates the mean cosine similarity between all distinct pairs of embeddings in a list.
    Assumes embeddings are already normalized.
    Returns 0.0 if fewer than 2 embeddings are present.
    """
    num_embeddings = len(embeddings_list)
    if num_embeddings < 2:
        return 0.0 # No pairs to compare

    try:
        # Stack embeddings into a single NumPy array (ensure float32 for consistency)
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        # Calculate the dot product matrix (cosine similarity for normalized vectors)
        similarity_matrix = np.dot(embeddings_array, embeddings_array.T)

        # Extract the upper triangle, excluding the diagonal (k=1)
        # This gives a 1D array of all pairwise similarities
        upper_triangle_indices = np.triu_indices_from(similarity_matrix, k=1)
        pairwise_similarities = similarity_matrix[upper_triangle_indices]

        # Calculate the mean
        # Handle case where pairwise_similarities might be empty (shouldn't happen if num_embeddings >= 2)
        if pairwise_similarities.size == 0:
             return 0.0

        mean_sim = np.mean(pairwise_similarities)

        # Clamp the result between -1.0 and 1.0 in case of floating point inaccuracies
        mean_sim = np.clip(mean_sim, -1.0, 1.0)

        return float(mean_sim)

    except Exception as e:
        logger.exception(f"Error calculating mean pairwise similarity: {e}")
        return 0.0 # Return default value on error


async def process_question_similarity_prune(
    example: Dict,
    iteration: int,
    n_chains_start: int,
    paths: Dict[str, str], # Will contain KV cache paths
    vllm_url: str,
    model_name: str,
    tokenizer_path: str,
    similarity_threshold: float,
    pruning_strategy: str,
    max_analysis_steps: int = 100 # Limit analysis intervals
) -> Optional[Dict]:
    """
    Processes a question using Similarity Pruning with continuous streams.
    Prunes based on the specified strategy ('fewest_thoughts', 'most_thoughts' or 'diversity').
    Pruning only occurs during the reasoning phase (before server sends non-empty 'content').
    The first two thoughts (idx 0, 1) are never pruned but their embeddings are added.
    Stops analysis loop if only one chain remains active.
    """
    # Validate strategy
    if pruning_strategy not in ["fewest_thoughts", "most_thoughts", "diversity"]:
        logger.error(f"[red]Invalid pruning strategy: {pruning_strategy}. Must be 'fewest_thoughts', 'most_thoughts' or 'diversity'.[/red]")
        # You might want to raise an error or return None here depending on desired behavior
        raise ValueError(f"Invalid pruning strategy: {pruning_strategy}")

    logger.info(f"--- Processing Question {iteration} (N={n_chains_start}, SimPrune-{pruning_strategy} Thresh={similarity_threshold}) ---")
    question_id = example.get("id", f"index_{iteration-1}")
    start_process_time = time.time()

    # --- Setup ---
    try:
        prompt, choices, correct_answer_letter = create_prompt_gpqa(example)
    except Exception as e:
        logger.exception(f"[red]Error creating prompt[/red]")
        # Save failure summary for this question
        summary_data: Dict[str, Any] = { # Add type hint for clarity
           "iteration": iteration, "question_id": question_id, "status": "PROMPT_ERROR",
           "n_chains_start": n_chains_start, "error_message": str(e),
           "processing_duration_sec": 0.0, # Mark duration as 0 or N/A
        }
        summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
        try:
           with open(summary_filename, "w", encoding='utf-8') as f:
               json.dump(summary_data, f, indent=2)
        except IOError as e_save:
           logger.exception(f"[red]Error writing prompt error summary file[/red]")
        return None # Return None to indicate failure
    try:
        embedding_model = get_embedding_model()
        index_manager = FaissIndexManager(dimension=embedding_model.get_sentence_embedding_dimension())
    except Exception as e:
        logger.exception(f"[red]Failed init embedding/FAISS[/red]");
        # Save failure summary for this question
        summary_data_init_fail: Dict[str, Any] = { # Add type hint for clarity
            "iteration": iteration, "question_id": question_id, "status": "INIT_ERROR",
            "n_chains_start": n_chains_start, "error_message": str(e),
            "processing_duration_sec": 0.0, # Mark duration as 0 or N/A
         }
        summary_filename_init_fail = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
        try:
           with open(summary_filename_init_fail, "w", encoding='utf-8') as f:
               json.dump(summary_data_init_fail, f, indent=2)
        except IOError as e_save:
           logger.exception(f"[red]Error writing init error summary file[/red]")
        return None # Return None to indicate failure

    # --- Shared State and Task Management ---
    active_chains: Dict[str, Dict] = {} # State for ALL chains that started
    consumer_tasks: Dict[str, asyncio.Task] = {}
    # The event is set by any worker finishing, and cleared by the main loop before waiting
    all_tasks_done_event = asyncio.Event() # Used to signal worker completion

    for i in range(n_chains_start):
        chain_id = f"q{iteration}_c{i+1}"
        # Initialize state
        active_chains[chain_id] = {
            "id": chain_id, "full_text": "", "processed_boundaries": [],
            "completed_thought_count": 0, # Continuous thought_idx
            "embeddings": [],
            "reasoning_complete": False, # Track if non-reasoning 'content' has started
            "is_active": True, "finished": False,
            "finish_reason": None, "error": None, 
            "prompt_tokens": None, "completion_tokens": 0
        }
        # Create the long-running stream request
        stream_generator = stream_vllm_request(
            prompt=prompt, vllm_url=vllm_url, model_name=model_name,
            request_id=chain_id,
            temperature=0.6,
            max_tokens=MAX_TOKENS_PER_STREAM,
            logprobs=None
        )
        # Create and store the consumer task
        task = asyncio.create_task(
            stream_processing_worker(chain_id, stream_generator, active_chains[chain_id], all_tasks_done_event),
            name=f"worker_{chain_id}"
        )
        consumer_tasks[chain_id] = task

    # --- Periodic Analysis Loop ---
    analysis_step = 0
    # Loop as long as there are tasks that aren't finished (either active or waiting to be marked inactive)
    while any(not state["finished"] for state in active_chains.values()) and analysis_step < max_analysis_steps:
        analysis_step += 1
        logger.info(f"[bold cyan][Q{iteration} Analysis Interval {analysis_step}] Checking chains...[/bold cyan]")

        # --- Identify chains needing analysis/pruning checks ---
        # These are chains that are still marked active AND whose worker isn't finished yet,
        # AND are still in the reasoning phase.
        chains_eligible_for_pruning_check = {
            cid: state for cid, state in active_chains.items()
            if state["is_active"] and not state["finished"] and not state["reasoning_complete"]
        }

        # --- OPTIMIZATION CHECK ---
        # Stop analysis loop early if:
        # 1. Only one chain is left that is eligible for pruning.
        # 2. OR, if ALL chains that are still running (not finished) have completed their reasoning phase.
        running_chains = {cid: state for cid, state in active_chains.items() if not state["finished"]}
        all_running_chains_reasoning_complete = all(state.get("reasoning_complete", False) for state in running_chains.values()) if running_chains else True # True if no running chains

        if len(chains_eligible_for_pruning_check) <= 1 or all_running_chains_reasoning_complete:
            status_msg = ""
            if len(chains_eligible_for_pruning_check) <= 1:
                 status_msg += f"Only {len(chains_eligible_for_pruning_check)} chain(s) left eligible for pruning."
            if all_running_chains_reasoning_complete:
                 if status_msg: 
                    status_msg += " AND All remaining running chains completed reasoning."

            logger.info(f"[Q{iteration} Analysis Interval {analysis_step}] {status_msg} Halting periodic analysis for pruning decisions.")
            # Break analysis loop, but still need to wait for remaining tasks to truly finish
            break

        # If no chains are left in any state (all finished), break
        if not running_chains: # This check is slightly redundant with the while loop condition but clearer
            logger.info(f"[Q{iteration} Analysis Interval {analysis_step}] No running chains remain. Stopping analysis.")
            break

        # --- Continue with analysis/pruning if conditions met ---
        all_new_thoughts_data: List[Tuple[str, int, str]] = [] # (chain_id, thought_idx, text)

        # Only analyze chains that are still active and in the reasoning phase
        for chain_id, chain_state in chains_eligible_for_pruning_check.items():
             new_segments, updated_boundaries = find_newly_completed_thoughts(
                chain_state["full_text"],
                chain_state["processed_boundaries"],
                tokenizer_path,
                target_phrases=TARGET_PHRASES, # Corrected variable name
                min_segment_tokens=MIN_SEGMENT_TOKENS
             )
             if new_segments:
                logger.debug(f"Chain {chain_id}: Found {len(new_segments)} new completed thoughts in interval {analysis_step}.")
                for start_idx, end_idx, text in new_segments:
                    thought_idx = chain_state["completed_thought_count"]
                    all_new_thoughts_data.append((chain_id, thought_idx, text))
                    chain_state["completed_thought_count"] += 1
                chain_state["processed_boundaries"] = updated_boundaries

        # --- Embed new thoughts AND Store Them ---
        newly_completed_thoughts_for_faiss: List[Tuple[str, int, str, np.ndarray]] = []
        if all_new_thoughts_data:
            texts_only = [t[2] for t in all_new_thoughts_data]
            embeddings = embed_segments(texts_only)
            if embeddings is None or len(embeddings) != len(all_new_thoughts_data):
                logger.error("[red]Embedding failed or returned incorrect number. Skipping analysis this interval.[/red]")
            else:
                embedding_idx = 0
                for chain_id, thought_idx, text in all_new_thoughts_data:
                    # Ensure chain still exists and is active before appending
                    if chain_id in active_chains and active_chains[chain_id]["is_active"]:
                         current_embedding = embeddings[embedding_idx]
                         # STORE the embedding in the chain's state
                         active_chains[chain_id]["embeddings"].append(current_embedding)
                         # Prepare data needed for FAISS check/add
                         newly_completed_thoughts_for_faiss.append((chain_id, thought_idx, text, current_embedding))
                    embedding_idx += 1

        # --- Check Similarity and Prune ---
        chains_to_prune_this_interval: Set[str] = set()
        embeddings_to_add_to_faiss: List[Tuple[np.ndarray, str, int, str]] = []
        # Store pruning step info temporarily
        pruning_info_this_interval: Dict[str, int] = {} # {chain_id: analysis_step}

        # The FAISS index should only contain embeddings from chains that are currently active
        # We don't need to rebuild the index, but the search must filter results
        # based on the current `active_chains` status before comparing similarity scores.
        # The `FaissIndexManager.search_nearest_neighbor` needs to be aware of which chains are active.
        # We remove embeddings for inactive chains from the FAISS index itself using remove_ids.
        # So, we just need to ensure we only search against embeddings that are still logically in the index (represented in metadata_map).
        # The search method already checks `if faiss_id in self.metadata_map`.
        
        num_chains_in_index = index_manager.get_num_active_chains() # Active chains in FAISS metadata

        # Use the newly_completed_thoughts_for_faiss list which contains the actual embeddings
        if newly_completed_thoughts_for_faiss:
            logger.info(f"[Q{iteration} Int {analysis_step}] Checking similarity for {len(newly_completed_thoughts_for_faiss)} new thoughts using '{pruning_strategy}' strategy.")

            # Iterate through thoughts that were successfully embedded
            for chain_id, thought_idx, text, embedding in newly_completed_thoughts_for_faiss:
                # Double-check eligibility (might have been pruned already in this loop by another thought)
                if chain_id not in chains_eligible_for_pruning_check or chain_id in chains_to_prune_this_interval:
                    continue # Skip if chain no longer eligible or already marked for pruning

                chain_state = active_chains[chain_id] # Get current state

                # Pruning check conditions:
                # 1. Thought index is >= 2 (allow first two thoughts)
                # 2. The chain is still in the reasoning phase (checked by chains_eligible_for_pruning_check)
                # 3. There is at least one other chain currently represented in the FAISS index (meaning at least 2 chains in index)
                can_check_pruning = (
                    thought_idx >= 2 and
                    num_chains_in_index >= 2 # Prune only if there's at least one other chain in the index
                )

                if can_check_pruning:
                    logger.debug(f"Chain {chain_id} (T{thought_idx}): Eligible for pruning check (reasoning not complete, T>{thought_idx}, >1 chain in index).")
                    # Search for nearest neighbor EXCLUDING embeddings from the current chain_id
                    neighbor_result = index_manager.search_nearest_neighbor(embedding, chain_id)
                    if neighbor_result:
                        sim_score_faiss, neighbor_chain_id, _, _ = neighbor_result
                        # --- Step 2: If FAISS similarity high, calculate NEW Diversity Efficiency ---
                        if sim_score_faiss > similarity_threshold:
                            logger.warning(f"[bold yellow]PRUNING CONDITION (FAISS)![/bold yellow] Chain {chain_id} (T{thought_idx}) vs {neighbor_chain_id}, FAISS_score={sim_score_faiss:.4f}")

                            neighbor_state = active_chains.get(neighbor_chain_id)
                            if neighbor_state is None or not neighbor_state['is_active']:
                                logger.warning(f"Neighbor chain {neighbor_chain_id} not found or inactive. Skipping diversity check.")
                                embeddings_to_add_to_faiss.append((embedding, chain_id, thought_idx, text))
                                continue

                            prune_target_id = None
                            # Get thought counts (needed for multiple strategies now)
                            current_thought_count = chain_state.get('completed_thought_count', 0)
                            neighbor_thought_count = neighbor_state.get('completed_thought_count', 0)

                            if pruning_strategy == "fewest_thoughts":
                                logger.info(f"Fewest Thoughts Check: Chain {chain_id} (T={current_thought_count}) vs {neighbor_chain_id} (T={neighbor_thought_count})")
                                if current_thought_count <= neighbor_thought_count:
                                    prune_target_id = chain_id
                                    logger.warning(f"--> Pruning Chain {chain_id} (Fewest Thoughts: <= thoughts).")
                                else:
                                    prune_target_id = neighbor_chain_id
                                    logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Fewest Thoughts: fewer thoughts).")

                            elif pruning_strategy == "most_thoughts":
                                logger.info(f"Most Thoughts Check: Chain {chain_id} (T={current_thought_count}) vs {neighbor_chain_id} (T={neighbor_thought_count})")
                                if current_thought_count >= neighbor_thought_count:
                                    prune_target_id = chain_id
                                    logger.warning(f"--> Pruning Chain {chain_id} (Most Thoughts: > thoughts).")
                                else:
                                    prune_target_id = neighbor_chain_id
                                    logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Most Thoughts: > thoughts).")

                            elif pruning_strategy == "diversity":
                                embeddings_A = chain_state.get("embeddings", [])
                                embeddings_B = neighbor_state.get("embeddings", [])
                                num_thoughts_A = len(embeddings_A)
                                num_thoughts_B = len(embeddings_B)

                                mean_sim_A = calculate_mean_pairwise_similarity(embeddings_A)
                                mean_sim_B = calculate_mean_pairwise_similarity(embeddings_B)

                                # InternalSim = MeanPairwiseSim / NumThoughts (Higher is worse)
                                internal_sim_A = (mean_sim_A / num_thoughts_A) if num_thoughts_A > 0 else 0.0
                                internal_sim_B = (mean_sim_B / num_thoughts_B) if num_thoughts_B > 0 else 0.0

                                logger.info(f"Diversity Check: Chain {chain_id} (Thoughts={num_thoughts_A}, MeanSim={mean_sim_A:.4f}, InternalSim={internal_sim_A:.4f}) vs "
                                            f"Chain {neighbor_chain_id} (Thoughts={num_thoughts_B}, MeanSim={mean_sim_B:.4f}, InternalSim={internal_sim_B:.4f})")

                                if internal_sim_A > internal_sim_B:
                                    prune_target_id = chain_id
                                    logger.warning(f"--> Pruning Chain {chain_id} (Diversity: Higher internal_sim).")
                                elif internal_sim_B > internal_sim_A:
                                    prune_target_id = neighbor_chain_id # Neighbor guaranteed eligible here
                                    logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Diversity: Higher internal_sim).")
                                else: # InternalSim are equal - Tie-break with fewer thoughts
                                    logger.warning("Diversity internal_sims equal. Tie-break: fewer thoughts.")
                                    if num_thoughts_A <= num_thoughts_B:
                                        prune_target_id = chain_id
                                        logger.warning(f"--> Pruning Chain {chain_id} (Tie-break: <= thoughts).")
                                    else:
                                        prune_target_id = neighbor_chain_id # Neighbor guaranteed eligible here
                                        logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Tie-break: fewer thoughts).")

                            if prune_target_id:
                                # Check how many chains are currently marked active
                                num_active_before_this_prune = sum(1 for state in active_chains.values() if state['is_active'])
                                # How many are already marked for pruning in this interval
                                num_already_marked_this_interval = len(chains_to_prune_this_interval)

                                # Check if this prune_target_id is already marked (can happen if compared multiple times)
                                target_already_marked = prune_target_id in chains_to_prune_this_interval

                                # Calculate how many would be left if we prune this target
                                # (only decrement count if it's not already marked for pruning)
                                potential_remaining_active = num_active_before_this_prune - num_already_marked_this_interval
                                if not target_already_marked:
                                     potential_remaining_active -= 1

                                # Only add to prune list if doing so leaves AT LEAST ONE active chain
                                if potential_remaining_active >= 1:
                                     if not target_already_marked: # Add only if not already present
                                        chains_to_prune_this_interval.add(prune_target_id)
                                        pruning_info_this_interval[prune_target_id] = analysis_step
                                        logger.warning(f"--> OK to prune Chain {prune_target_id} (based on T{thought_idx} from {chain_id}). Will leave {potential_remaining_active} active.")
                                     # else: chain was already marked for pruning this interval, no action needed
                                else:
                                     # This prune would eliminate the last chain(s), so we skip it.
                                     logger.warning(f"--> Skipped pruning Chain {prune_target_id} (based on T{thought_idx} from {chain_id}). Pruning would leave {potential_remaining_active} active chain(s). Keeping >= 1 active.")
                            else: # Should not happen if logic is correct, but as fallback:
                                if chain_id not in chains_to_prune_this_interval:
                                    embeddings_to_add_to_faiss.append((embedding, chain_id, thought_idx, text))

                        else: # sim_score_faiss <= similarity_threshold
                            if chain_id not in chains_to_prune_this_interval:
                                embeddings_to_add_to_faiss.append((embedding, chain_id, thought_idx, text))
                    else: # No neighbor found
                        if chain_id not in chains_to_prune_this_interval:
                            embeddings_to_add_to_faiss.append((embedding, chain_id, thought_idx, text))
                else: # Not eligible for pruning check
                    if chain_id not in chains_to_prune_this_interval:
                        embeddings_to_add_to_faiss.append((embedding, chain_id, thought_idx, text))

        # --- Apply Pruning and Update FAISS ---
        if chains_to_prune_this_interval:
            for prune_id in chains_to_prune_this_interval:
                # Only prune if it hasn't already been marked inactive (e.g. due to error or earlier prune)
                if active_chains.get(prune_id, {}).get("is_active", False):
                    logger.debug(f"Marking chain {prune_id} as inactive/pruned due to similarity.")
                    active_chains[prune_id]["is_active"] = False
                    # We do not mark 'finished' here. 'finished' means the worker task is done.
                    # We need to cancel the worker task, which will set 'finished' in its finally block.
                    # We also set the finish_reason here for clarity in logs/summary
                    active_chains[prune_id]["finish_reason"] = active_chains[prune_id].get("finish_reason", "pruned_similarity")

                    # Store the pruning step in the chain's state
                    active_chains[prune_id]["pruned_at_step"] = pruning_info_this_interval.get(prune_id)

                    # Cancel the worker task associated with the pruned chain
                    task_to_cancel = consumer_tasks.get(prune_id)
                    if task_to_cancel and not task_to_cancel.done():
                        logger.debug(f"Cancelling worker task for pruned chain {prune_id}")
                        task_to_cancel.cancel()
                    elif not task_to_cancel:
                        logger.warning(f"Worker task for chain {prune_id} not found or already done when trying to cancel.")

                    # Remove from FAISS index immediately to prevent it being a neighbor for future searches
                    index_manager.remove_chain_embeddings(prune_id)
                else:
                    logger.debug(f"Chain {prune_id} was marked for pruning but was already inactive. Skipping cancellation/removal.")


        # Add embeddings for chains that were not pruned in this interval
        # And ensure these chains are still active and not finished
        valid_embeddings_to_add = [
             (emb, cid, tidx, txt) for emb, cid, tidx, txt in embeddings_to_add_to_faiss
             if active_chains.get(cid, {}).get("is_active", False) and not active_chains.get(cid, {}).get("finished", True)
        ]
        for emb, cid, tidx, txt in valid_embeddings_to_add:
            index_manager.add_embedding(emb, cid, tidx, txt)
            # logger.debug(f"Added embedding for Chain {cid} (T{tidx}) to FAISS.") # Very verbose

        # Update the number of chains in the index count after pruning and adding
        num_chains_in_index = index_manager.get_num_active_chains()
        logger.debug(f"FAISS index now contains embeddings from {num_chains_in_index} chains.")

         # --- Wait for next interval or task completion ---
        # Wait until at least one task finishes OR the timeout occurs.
        # This prevents busy-waiting if no new text comes in for a while.
        # It also ensures we don't miss tasks finishing naturally.
        all_tasks_done_event.clear() # Clear the event before waiting

        # We need to wait for tasks that were active at the start of the interval and are still running.
        # It's okay if new tasks become done during the wait. The event mechanism handles this.
        tasks_currently_running = [t for t in consumer_tasks.values() if not t.done()]

        if not tasks_currently_running:
            logger.info("No running consumer tasks left. Exiting analysis loop.")
            break # All tasks are finished, exit the analysis loop

        try:
            # Wait for any of the currently running tasks to complete or for the timeout
            # The all_tasks_done_event will be set by any worker completing.
            # Waiting on the event itself is sufficient to wake up when at least one task finishes.
            # Using asyncio.wait_for with event.wait() is the standard way to do this with a timeout.
            await asyncio.wait_for(all_tasks_done_event.wait(), timeout=ANALYSIS_INTERVAL_SECONDS)
            logger.debug(f"Woke up from wait: Event was set (a task finished).")
        except asyncio.TimeoutError:
            logger.debug(f"Woke up from wait: Timeout reached.")
            # If timeout, clear event again before next wait (handled by loop start) and continue analysis
            pass # Just continue the loop to the next analysis step
        except asyncio.CancelledError:
            logger.info("Analysis loop wait was cancelled.")
            break # Exit loop if analysis itself is cancelled (e.g., Ctrl+C)


    # --- Analysis loop finished (either by max_steps, no chains left, or only one left) ---
    logger.info(f"Analysis loop concluded after {analysis_step} intervals.")

    # --- Ensure all tasks are truly finished ---
    # This is crucial. We need to wait for all consumer tasks to finish their execution,
    # including the ones that were cancelled. Their finally blocks need to run to set `finished=True`.
    logger.info(f"Waiting for all {len(consumer_tasks)} worker tasks to finalize...")
    try:
        # Wait for all tasks to be done. return_exceptions=True ensures we don't crash if one task had an unhandled error.
        await asyncio.gather(*consumer_tasks.values(), return_exceptions=True)
        logger.info("All worker tasks finalized.")
    except asyncio.CancelledError:
        # If this gather is cancelled, something higher up is cancelling.
        logger.warning("[yellow]Final worker task gather was cancelled.[/yellow]")
    except Exception as e:
        logger.exception("[red]Error during final worker task gather.[/red]")

    # Re-evaluate end time after final gather finishes
    end_process_time = time.time()
    total_duration = end_process_time - start_process_time
    logger.info(f"[bold green]Finished processing Q{iteration} in {total_duration:.2f}s.[/bold green]")

    # --- KV Cache Extraction ---
    # Extract KV cache data after all processing for the question is done
    kv_cache_stats = extract_kv_cache_usage_for_question(
        start_process_time, end_process_time, iteration, paths
    )
    avg_kv_usage = kv_cache_stats.get('avg_kv_cache_usage')
    max_kv_usage = kv_cache_stats.get('max_kv_cache_usage')

    # --- Post-Processing, Voting, Saving ---
    # Classify chains based on their final state after all workers finished
    final_completed_chains_for_voting = [] # Chains that ran to completion (stop, length) AND were not pruned or errored
    pruned_chains_data = [] # Chains marked inactive due to pruning
    error_chains_data = [] # Chains where an error occurred

    # Re-evaluate state after all workers have finished
    for chain_id, state in active_chains.items():
        # Check if the task actually finished its run (its state['finished'] is True)
        if not state.get('finished'):
            logger.error(f"[red]Chain {chain_id} worker task did NOT set 'finished=True'. State: {state}. This is unexpected.[/red]")
            if not state.get('error'): state['error'] = 'Worker task did not finish cleanly.'
            # Classify as error if the worker didn't finish
            error_chains_data.append(state)
            continue # Skip further classification for this chain


        # Classify based on final state and reason
        # 1. Prioritize explicit errors
        if state.get("error"):
            error_chains_data.append(state)
        # 2. Check if the chain was marked as inactive (pruned) during the run
        elif not state.get("is_active", True): # If is_active is False
            # This chain was pruned. Ensure the finish_reason reflects this if possible.
            if not state.get("finish_reason"): # If reason is missing, set a default prune reason
                state["finish_reason"] = "pruned_during_run"
            # It's definitely pruned, add to pruned_chains_data
            pruned_chains_data.append(state)
        # 3. If it wasn't errored and wasn't marked inactive, it completed the stream
        else:
            # These are the candidates for voting
            final_completed_chains_for_voting.append(state)

    logger.info(f"Q{iteration} Final Status: {len(final_completed_chains_for_voting)} chains completed stream, {len(pruned_chains_data)} pruned, {len(error_chains_data)} errors/incomplete.")

    # Prepare results for voting - use only chains that completed their stream naturally and have an extracted answer
    successful_chain_results_for_voting = []
    # Aggregate metrics across ALL chains that started, for reporting
    # Prompt tokens assumed uniform across chains
    total_prompt_tokens_agg = active_chains.get(f"q{iteration}_c1", {}).get("prompt_tokens", 0) # Get from the first chain

    # total_completion_tokens_across_all_started_chains: Sum of completion tokens
    # from usage reports for ALL chains that were started and whose worker task finished,
    # including those that were pruned or errored. This captures the total compute used.
    total_completion_tokens_across_all_started_chains = sum(
        state.get('completion_tokens', 0) for state in active_chains.values()
        if state.get('finished', False) and state.get('completion_tokens') is not None
    )

    for chain_state in final_completed_chains_for_voting:
        extracted_answer = extract_answer_gpqa(chain_state["full_text"])
        if extracted_answer is None:
            logger.debug(f"Chain {chain_state['id']} completed but no answer could be extracted. Excluded from voting.")
            continue

        # Calculate the final mean pairwise similarity for this chain
        final_embeddings = chain_state.get("embeddings", [])
        num_thoughts = len(final_embeddings)
        final_mean_similarity = calculate_mean_pairwise_similarity(final_embeddings)

        # Calculate Internal Similarity (lower is better for diversity per thought)
        final_internal_similarity = (final_mean_similarity / num_thoughts) if num_thoughts > 0 else 0.0 # Handle division by zero

        logger.debug(f"Chain {chain_state['id']}: Final mean sim={final_mean_similarity:.4f}, num_thoughts={num_thoughts}, internal sim={final_internal_similarity:.4f}")

        successful_chain_results_for_voting.append({
            "chain_index": int(chain_state['id'].split('_c')[-1]),
            "full_content": chain_state["full_text"],
            "extracted_answer": extracted_answer,
            "finish_reason": chain_state["finish_reason"],
            "prompt_tokens": chain_state.get("prompt_tokens", 0),
            "completion_tokens": chain_state.get("completion_tokens", 0),
            "completed_thought_count": chain_state.get("completed_thought_count", 0),
            "final_mean_pairwise_similarity": final_mean_similarity, # Keep for logging/analysis if needed
            "final_internal_similarity": final_internal_similarity
        })

    # Majority Vote - uses the extracted_answer field we just added.
    if not successful_chain_results_for_voting:
        logger.warning(f"[Q{iteration}] No chains with extracted answers completed streams for majority vote.")
        voted_answer, final_score, all_extracted_answers = None, 0, []
    else:
        voted_answer, final_score, all_extracted_answers = majority_vote_for_sim_prune(
            successful_chain_results_for_voting, correct_answer_letter
        )

    # --- Save Individual Chain Outputs ---
    # Save individual chain outputs for ALL chains that started
    for chain_id, chain_state in active_chains.items():
        chain_idx = int(chain_id.split('_c')[-1])
        final_status = "unknown"
        if chain_state in error_chains_data: final_status = "error"
        elif chain_state in pruned_chains_data: final_status = "pruned"
        elif chain_state in final_completed_chains_for_voting: final_status = "completed"
        else: final_status = "unclassified" # Should not happen with the new logic

        chain_filename = os.path.join(paths["chains"], f"question_{iteration}_chain_{chain_idx}_{final_status}.txt")
        try:
            with open(chain_filename, "w", encoding='utf-8') as f:
                f.write(f"--- Chain {chain_idx} for Question {iteration} ---\n")
                f.write(f"Status: {final_status.upper()}\n")
                f.write(f"Is Active Flag: {chain_state.get('is_active', 'N/A')}\n") # Add is_active flag for debugging
                f.write(f"Finish Reason: {chain_state.get('finish_reason', 'N/A')}\n")
                f.write(f"Reasoning Complete Flag: {chain_state.get('reasoning_complete', 'N/A')}\n")
                f.write(f"Error: {chain_state.get('error', 'N/A')}\n")
                f.write(f"Prompt Tokens: {chain_state.get('prompt_tokens', 'N/A')}\n")
                f.write(f"Completion Tokens: {chain_state.get('completion_tokens', 'N/A')}\n")
                f.write(f"Completed Thoughts: {chain_state.get('completed_thought_count', 0)}\n")
                # Add pruned_at_step if the chain was pruned
                if final_status == "pruned":
                     f.write(f"Pruned at Analysis Step: {chain_state.get('pruned_at_step', 'N/A')}\n")

                # Add final similarity scores if available
                # Find the corresponding entry in successful_chain_results_for_voting
                voting_data = next((vd for vd in successful_chain_results_for_voting if vd.get('chain_index') == chain_idx), None)
                if voting_data:
                    mean_sim = voting_data.get('final_mean_pairwise_similarity', 'N/A')
                    internal_sim = voting_data.get('final_internal_similarity', 'N/A')
                    f.write(f"Final Mean Pairwise Similarity: {mean_sim if isinstance(mean_sim, str) else f'{mean_sim:.4f}'}\n")
                    f.write(f"Final Internal Similarity: {internal_sim if isinstance(internal_sim, str) else f'{internal_sim:.4f}'}\n")

                f.write(f"Final Processed Boundaries: {chain_state.get('processed_boundaries', [])}\n")
                f.write("\n--- Full Content ---\n")
                f.write(chain_state.get('full_text', 'N/A'))
        except IOError as e:
            logger.exception(f"[red]Error writing chain output file {chain_filename}[/red]")

    # --- Save summary JSON ---
    # total_completion_tokens_across_all_started_chains captures the total compute across all attempts for this question.
    # This is used for the primary completion token metric in the CSV and aggregated metrics.
    summary_data: Dict[str, Any] = { # Add type hint for clarity
        "iteration": iteration, "question_id": question_id,
        "status": "SUCCESS" if len(error_chains_data) == 0 else f"PARTIAL_SUCCESS ({len(error_chains_data)}_failed)",
        "n_chains_start": n_chains_start,
        "n_chains_completed_stream_for_voting": len(successful_chain_results_for_voting), # Chains that completed & extracted answers & used for voting
        "n_chains_pruned": len(pruned_chains_data),
        "n_chains_error": len(error_chains_data),
        "similarity_threshold": similarity_threshold,
        "correct_answer_letter": correct_answer_letter,
        "individual_answers_final": all_extracted_answers, # Answers from chains used for voting
        "voted_answer": voted_answer,
        "final_score": final_score,
        "processing_duration_sec": total_duration, # Use float for JSON
        "total_analysis_intervals": analysis_step, # Number of intervals completed
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
        "usage_aggregated": {
             "prompt_tokens": total_prompt_tokens_agg, # Sum across all started chains (assuming same prompt tokens)
             "total_completion_tokens_across_all_started_chains": total_completion_tokens_across_all_started_chains, # Sum across ALL chains whose streams finished (includes pruned)
        },
         "chains_for_voting_details": [ # Save key details per chain used for voting
             {
                "chain_index": cr.get("chain_index"),
                "finish_reason": cr.get("finish_reason"),
                "extracted_answer": cr.get("extracted_answer"),
                "prompt_tokens": cr.get("prompt_tokens"),
                "completion_tokens": cr.get("completion_tokens"),
                "completed_thought_count": cr.get("completed_thought_count"),
                "final_mean_pairwise_similarity": cr.get("final_mean_pairwise_similarity"),
                "final_internal_similarity": cr.get("final_internal_similarity")
             } for cr in successful_chain_results_for_voting
         ],
         "pruned_chain_details": [ # Details for chains explicitly pruned by similarity
              {
                 "chain_index": int(state['id'].split('_c')[-1]),
                 "finish_reason": state.get('finish_reason'),
                 "completion_tokens": state.get('completion_tokens'), # Tokens generated before pruning
                 "completed_thought_count": state.get('completed_thought_count'),
                 "full_text_len": len(state.get('full_text', '')),
                 "pruned_at_step": state.get("pruned_at_step", None)
              } for state in pruned_chains_data
         ],
         "error_chain_details": [ # Details for chains that encountered errors
              {
                 "chain_index": int(state['id'].split('_c')[-1]),
                 "finish_reason": state.get('finish_reason'),
                 "completion_tokens": state.get('completion_tokens'), # Tokens generated before error
                 "error": state.get('error'),
                 "full_text_len": len(state.get('full_text', ''))
              } for state in error_chains_data
         ]
    }
    summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
    try:
        with open(summary_filename, "w", encoding='utf-8') as f:
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
    # Standardized metrics requested: overall accuracy, mean/max completion tokens, mean/max kv cache usage, mean/max processing duration, mean/max chains pruned.
    # The CSV stores PER-QUESTION values, which are then used for overall aggregation.
    # `total_completion_tokens` for the CSV should be the sum across ALL chains that were started
    # and reported tokens via usage, representing the total compute cost for this question attempt.
    prompt_tokens_for_csv = total_prompt_tokens_agg

    return {
        "iteration": iteration,
        "question_id": question_id,
        "n_chains_start": n_chains_start,
        "n_chains_completed_stream_for_voting": len(successful_chain_results_for_voting),
        "n_chains_error": len(error_chains_data),
        "similarity_threshold": similarity_threshold,
        "correct_answer": correct_answer_letter,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "prompt_tokens": prompt_tokens_for_csv, # Sum across all started chains (assuming uniform)
        "total_completion_tokens": total_completion_tokens_across_all_started_chains, # Sum across ALL started chains whose streams finished (includes pruned)
        "total_tokens": prompt_tokens_for_csv + total_completion_tokens_across_all_started_chains, # Sum across ALL started chains
        "individual_answers_str": json.dumps(all_extracted_answers), # Answers from chains used for voting
        "total_analysis_intervals": analysis_step,
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
        "processing_duration_sec": total_duration,
    }