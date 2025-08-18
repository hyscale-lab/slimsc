# slimsc/prune/evaluation/processing_similarity.py
import os
import json
import time
import asyncio
import numpy as np
import random
from typing import Dict, Optional, List, Tuple, Set, AsyncGenerator, Any

# Keep necessary imports
from ..clients import stream_vllm_request, close_aiohttp_session
from ..utils import DatasetHandler, count_tokens
from ..utils.similarity_utils import (
    FaissIndexManager, embed_segments, find_newly_completed_thoughts,
    extract_final_thought, get_embedding_model, MIN_SEGMENT_TOKENS, TARGET_PHRASES
)
from .voting import majority_vote_for_sim_prune, fallback_tie_break_logic
from .kv_cache_extraction import extract_kv_cache_usage_for_question

import logging
logger = logging.getLogger(__name__)

# --- Constants ---
ANALYSIS_INTERVAL_SECONDS = 3 # How often to check for new thoughts/prune
random.seed(42) # For reproducibility


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
                 # Check if prompt_tokens from usage is valid before overwriting
                 server_prompt_tokens = usage.get("prompt_tokens")
                 if server_prompt_tokens is not None and server_prompt_tokens > 0: # Ensure it's a plausible value
                      # Overwrite the pre-calculated value if we get a valid one from the server
                      if chain_state["prompt_tokens"] != server_prompt_tokens:
                           logger.debug(f"Chain {chain_id}: Updating prompt_tokens from {chain_state['prompt_tokens']} (pre-calc/old) to {server_prompt_tokens} (server usage).")
                           chain_state["prompt_tokens"] = server_prompt_tokens
                 else:
                     # Keep the pre-calculated value if server sends None or 0
                     if chain_state["prompt_tokens"] is None: # Should not happen with pre-calculation, but safety check
                          chain_state["prompt_tokens"] = 0 # Set to 0 if pre-calculation failed AND server didn't provide
                          logger.warning(f"Chain {chain_id}: Server did not provide valid prompt_tokens in usage, and pre-calculation missing. Setting to 0.")

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
    threshold_schedule: str,
    dataset_name: str,  # Add dataset_name parameter
    num_steps_to_delay_pruning: int,
    max_analysis_steps: int = 1000 # Limit analysis intervals
) -> Optional[Dict]:
    """
    Processes a question using Similarity Pruning with continuous streams.
    Prunes based on the specified strategy 'fewest_thoughts', 'most_thoughts', 'diversity', or 'random')..
    Pruning only occurs during the reasoning phase (before server sends non-empty 'content').
    The first two thoughts (idx 0, 1) are never pruned but their embeddings are added.
    Stops analysis loop if only one chain remains active.

    Args:
        example: The question example to process
        iteration: Current iteration number (1-indexed)
        n_chains_start: Number of chains to start with
        paths: Dictionary of output paths
        vllm_url: URL of the vLLM server
        model_name: Name/identifier of the model to use
        tokenizer_path: Path to the tokenizer
        similarity_threshold: Threshold for pruning (0.0-1.0)
        pruning_strategy: Strategy for pruning ("fewest_thoughts", "most_thoughts", "diversity", or "random")
        dataset_name: Name of the dataset ("gpqa_diamond", "aime", "math500")
        max_analysis_steps: Maximum number of analysis intervals
    """
    # Validate strategy
    valid_strategies = ["fewest_thoughts", "most_thoughts", "diversity", "random", "prune_farthest", "random_after_delay"]
    if pruning_strategy not in valid_strategies:
        logger.error(f"[red]Invalid pruning strategy: {pruning_strategy}. Must be one of {valid_strategies}.[/red]")
        raise ValueError(f"Invalid pruning strategy: {pruning_strategy}")

    logger.info(f"--- Processing Question {iteration} (N={n_chains_start}, SimPrune-{pruning_strategy} Thresh={similarity_threshold}) ---")
    question_id = example.get("id", f"index_{iteration-1}")
    start_process_time = time.time()

    # --- Setup ---
    prompt_text: str = ""
    correct_answer_for_scoring: Any = None # Will hold correct letter for GPQA, or answer string for AIME/MATH

    try:
        dataset_handler = DatasetHandler(dataset_name=dataset_name)
        # dataset_handler.create_prompt returns (prompt_string, details)
        # For GPQA, details is (choices_list, correct_answer_letter_string).
        # For AIME/MATH, details is correct_answer_string.
        _prompt_text_val, _prompt_output_details = dataset_handler.create_prompt(example)
        prompt_text = _prompt_text_val

        if dataset_name == "gpqa_diamond":
            # _prompt_output_details is (_choices_list, _correct_letter_string)
            # _choices_list = _prompt_output_details[0] # Available if needed
            correct_answer_for_scoring = _prompt_output_details[1] # This is the correct_letter for GPQA
        else: # For AIME and MATH500
            # _prompt_output_details is the correct_answer_string
            correct_answer_for_scoring = _prompt_output_details

        if not correct_answer_for_scoring:
            logger.error(f"[red]Failed to obtain correct answer reference for Q{iteration} from dataset_handler for dataset {dataset_name}. Details: {_prompt_output_details}[/red]")
            raise ValueError("Correct answer reference not found after prompt creation.")

    except Exception as e:
        logger.exception(f"[red]Error creating prompt or obtaining correct answer for Q{iteration}[/red]")
        summary_data: Dict[str, Any] = {
           "iteration": iteration, "question_id": question_id, "status": "PROMPT_ERROR",
           "n_chains_start": n_chains_start, "error_message": str(e),
           "processing_duration_sec": 0.0,
        }
        summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
        try:
           with open(summary_filename, "w", encoding='utf-8') as f:
               json.dump(summary_data, f, indent=2)
        except IOError as e_save:
           logger.exception(f"[red]Error writing prompt error summary file: {e_save}[/red]")
        return None
    
    pre_calculated_prompt_tokens: Optional[int] = None
    try:
        pre_calculated_prompt_tokens = count_tokens(prompt_text, tokenizer_path)
        if pre_calculated_prompt_tokens is None:
             logger.warning(f"[yellow]Pre-calculation of prompt tokens failed for Q{iteration}. Will rely on server usage report.[/yellow]")
             pre_calculated_prompt_tokens = 0 # Default to 0 if calculation fails
        else:
             logger.info(f"Q{iteration}: Pre-calculated prompt tokens: {pre_calculated_prompt_tokens}")
    except Exception as e_tok:
        logger.exception(f"[red]Error during prompt token pre-calculation for Q{iteration}. Setting initial to 0.[/red]")
        pre_calculated_prompt_tokens = 0 # Default to 0 on exception

    try:
        embedding_model = get_embedding_model()
        search_mode = 'dissimilarity' if pruning_strategy == 'prune_farthest' else 'similarity'
        dim = embedding_model.get_sentence_embedding_dimension()
        index_manager = FaissIndexManager(dimension=dim, search_mode=search_mode)
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
            "prompt_tokens": pre_calculated_prompt_tokens,
            "completion_tokens": 0,
            "pruned_count": 0,
            "pruned_by": None,
            "pruned_others": []
        }
        # Create the long-running stream request
        stream_generator = stream_vllm_request(
            prompt=prompt_text,
            vllm_url=vllm_url,
            model_name=model_name,
            request_id=chain_id,
            temperature=0.6,
            max_tokens=32768,
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

        # --- OPTIMIZATION CHECK ---
        # Stop analysis loop early if only one active chain is left.
        # This check is now simpler and more general for all strategies.
        num_active_chains = sum(1 for state in active_chains.values() if state["is_active"] and not state["finished"])
        if num_active_chains <= 1:
            logger.info(f"[Q{iteration} Analysis Interval {analysis_step}] Only {num_active_chains} chain(s) left active. Halting periodic analysis.")
            break

        # If no chains are left in any state (all finished), break
        if not any(not state["finished"] for state in active_chains.values()):
            logger.info(f"[Q{iteration} Analysis Interval {analysis_step}] No running chains remain. Stopping analysis.")
            break
        
        chains_to_prune_this_interval: Set[str] = set()
        pruning_info_this_interval: Dict[str, Tuple[int, str]] = {}
        
        if pruning_strategy == 'random_after_delay':
            # --- New Strategy: Randomly prune one active chain after delay ---
            if analysis_step > num_steps_to_delay_pruning:
                eligible_for_random_prune = [
                    cid for cid, state in active_chains.items()
                    if state["is_active"] and not state["finished"]
                ]
                
                if len(eligible_for_random_prune) > 1:
                    loser_id = random.choice(eligible_for_random_prune)
                    chains_to_prune_this_interval.add(loser_id)
                    
                    # Set info for logging and final summary
                    pruning_info_this_interval[loser_id] = (analysis_step, "random_system")
                    if loser_id in active_chains:
                        active_chains[loser_id]["pruned_by"] = "random_system"

                    logger.warning(
                        f"[bold yellow]RANDOM PRUNING (after delay)[/bold yellow] "
                        f"Step {analysis_step} > {num_steps_to_delay_pruning}. "
                        f"Randomly chose to prune: {loser_id}"
                    )
            else:
                logger.info(f"[Q{iteration} Int {analysis_step}] Delay step <= {num_steps_to_delay_pruning}. Skipping random pruning.")
        
        else:
            chains_eligible_for_pruning_check = {
                cid: state for cid, state in active_chains.items()
                if state["is_active"] and not state["finished"] and not state["reasoning_complete"]
            }
            
            # Additional check: if all remaining chains are out of reasoning phase, stop pruning checks
            if not chains_eligible_for_pruning_check:
                logger.info(f"[Q{iteration} Analysis Interval {analysis_step}] No chains eligible for similarity pruning (all completed reasoning or finished). Halting analysis.")
                break

            all_new_thoughts_data: List[Tuple[str, int, str]] = []
            for chain_id, chain_state in chains_eligible_for_pruning_check.items():
                new_segments, updated_boundaries = find_newly_completed_thoughts(
                    chain_state["full_text"],
                    chain_state["processed_boundaries"],
                    tokenizer_path,
                    target_phrases=TARGET_PHRASES,
                    min_segment_tokens=MIN_SEGMENT_TOKENS
                )
                if new_segments:
                    logger.debug(f"Chain {chain_id}: Found {len(new_segments)} new thoughts in interval {analysis_step}.")
                    for start_idx, end_idx, text in new_segments:
                        thought_idx = chain_state["completed_thought_count"]
                        all_new_thoughts_data.append((chain_id, thought_idx, text))
                        chain_state["completed_thought_count"] += 1
                    chain_state["processed_boundaries"] = updated_boundaries

            newly_completed_thoughts_for_faiss: List[Tuple[str, int, str, np.ndarray]] = []
            if all_new_thoughts_data:
                texts_only = [t[2] for t in all_new_thoughts_data]
                embeddings = embed_segments(texts_only)
                if embeddings is not None and len(embeddings) == len(all_new_thoughts_data):
                    for i, (chain_id, thought_idx, text) in enumerate(all_new_thoughts_data):
                        if chain_id in active_chains and active_chains[chain_id]["is_active"]:
                            current_embedding = embeddings[i]
                            active_chains[chain_id]["embeddings"].append(current_embedding)
                            newly_completed_thoughts_for_faiss.append((chain_id, thought_idx, text, current_embedding))
                else:
                    logger.error("[red]Embedding failed or returned incorrect number. Skipping analysis this interval.[/red]")

            embeddings_to_add_to_faiss: List[Tuple[np.ndarray, str, int, str]] = []
            num_chains_in_index = index_manager.get_num_active_chains()

            if newly_completed_thoughts_for_faiss and pruning_strategy == 'prune_farthest':
                logger.info(f"[Q{iteration} Int {analysis_step}] Finding most dissimilar thought using 'prune_farthest' strategy.")
                for chain_id, thought_idx, text, embedding in newly_completed_thoughts_for_faiss:
                    index_manager.add_embedding(embedding, chain_id, thought_idx, text)
                max_dist_found = -1.0
                most_dissimilar_pair = None
                for chain_id, thought_idx, text, embedding in newly_completed_thoughts_for_faiss:
                    neighbor_result = index_manager.search_farthest_neighbor(embedding, chain_id)
                    if neighbor_result:
                        dist, neighbor_chain_id, _, _ = neighbor_result
                        if dist > max_dist_found:
                            max_dist_found = dist
                            most_dissimilar_pair = (chain_id, neighbor_chain_id)
                if most_dissimilar_pair:
                    loser_candidate_A, loser_candidate_B = most_dissimilar_pair
                    potential_loser_id = random.choice([loser_candidate_A, loser_candidate_B])
                    actual_winner_id = loser_candidate_B if potential_loser_id == loser_candidate_A else loser_candidate_A
                    logger.warning(f"[bold yellow]PRUNING CONDITION (FARTHEST)![/bold yellow] Farthest pair: {loser_candidate_A} and {loser_candidate_B} (Dist={max_dist_found:.4f}). Randomly chose to prune: {potential_loser_id}")
                    num_active_before_this_prune = sum(1 for st in active_chains.values() if st['is_active'])
                    if num_active_before_this_prune > 1:
                        chains_to_prune_this_interval.add(potential_loser_id)
                        pruning_info_this_interval[potential_loser_id] = (analysis_step, actual_winner_id)
                    else:
                        logger.warning(f"--> Skipped pruning Chain {potential_loser_id}. Would leave 0 active chains.")

            elif newly_completed_thoughts_for_faiss:
                logger.info(f"[Q{iteration} Int {analysis_step}] Checking similarity for {len(newly_completed_thoughts_for_faiss)} new thoughts using '{pruning_strategy}' strategy.")
                for chain_id, thought_idx, text, embedding in newly_completed_thoughts_for_faiss:
                    if chain_id not in chains_to_prune_this_interval:
                        chain_state_for_faiss_add = active_chains.get(chain_id)
                        if chain_state_for_faiss_add and chain_state_for_faiss_add.get("is_active") and not chain_state_for_faiss_add.get("finished"):
                            index_manager.add_embedding(embedding, chain_id, thought_idx, text)

                # Pruning check conditions:
                # 1. Thought index is > 20 (allow first 20 thoughts)
                # 2. The chain is still in the reasoning phase (checked by chains_eligible_for_pruning_check)
                # 3. There is at least one other chain currently represented in the FAISS index (meaning at least 2 chains in index)
                can_check_pruning = (
                    thought_idx > 20 and
                    analysis_step > num_steps_to_delay_pruning and
                    num_chains_in_index >= 2 # Prune only if there's at least one other chain in the index
                )

                if can_check_pruning:
                    logger.debug(f"Chain {chain_id} (T{thought_idx}): Eligible for pruning check (reasoning not complete, T>{thought_idx}, >1 chain in index).")
                    # Search for nearest neighbor EXCLUDING embeddings from the current chain_id
                    neighbor_result = index_manager.search_nearest_neighbor(embedding, chain_id)
                    if neighbor_result:
                        sim_score_faiss, neighbor_chain_id, _, _ = neighbor_result
                        if sim_score_faiss > similarity_threshold:
                            logger.warning(f"[bold yellow]PRUNING CONDITION (FAISS)![/bold yellow] Chain {chain_id} (T{thought_idx}) vs {neighbor_chain_id}, FAISS_score={sim_score_faiss:.4f} > Threshold={similarity_threshold:.4f}")

                            neighbor_state = active_chains.get(neighbor_chain_id)
                            if neighbor_state is None or not neighbor_state['is_active']:
                                logger.warning(f"Neighbor chain {neighbor_chain_id} not found or inactive. Skipping diversity check.")
                                embeddings_to_add_to_faiss.append((embedding, chain_id, thought_idx, text))
                                continue

                            potential_loser_id = None
                            # Get thought counts (needed for multiple strategies now)
                            current_thought_count = chain_state.get('completed_thought_count', 0)
                            neighbor_thought_count = neighbor_state.get('completed_thought_count', 0)

                            if pruning_strategy == "random":
                                logger.info(f"Random Pruning Choice between: Chain {chain_id} and Chain {neighbor_chain_id}")
                                # Randomly select one of the two chains involved in the similarity match
                                potential_loser_id = random.choice([chain_id, neighbor_chain_id])
                                logger.warning(f"--> Randomly chose to prune Chain {potential_loser_id}.")

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
                                    potential_loser_id = chain_id
                                    logger.warning(f"--> Pruning Chain {chain_id} (Diversity: Higher internal_sim).")
                                elif internal_sim_B > internal_sim_A:
                                    potential_loser_id = neighbor_chain_id # Neighbor guaranteed eligible here
                                    logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Diversity: Higher internal_sim).")
                                else: # InternalSim are equal - Tie-break with fewer thoughts
                                    logger.warning("Diversity internal_sims equal. Tie-break: fewer thoughts.")
                                    if num_thoughts_A <= num_thoughts_B:
                                        potential_loser_id = chain_id
                                        logger.warning(f"--> Pruning Chain {chain_id} (Tie-break: <= thoughts).")
                                    else:
                                        potential_loser_id = neighbor_chain_id # Neighbor guaranteed eligible here
                                        logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Tie-break: fewer thoughts).")

                            elif pruning_strategy == "fewest_thoughts":
                                logger.info(f"Fewest Thoughts Check: Chain {chain_id} (T={current_thought_count}) vs {neighbor_chain_id} (T={neighbor_thought_count})")
                                if current_thought_count <= neighbor_thought_count:
                                    potential_loser_id = chain_id
                                    logger.warning(f"--> Pruning Chain {chain_id} (Fewest Thoughts: <= thoughts).")
                                else:
                                    potential_loser_id = neighbor_chain_id
                                    logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Fewest Thoughts: fewer thoughts).")

                            elif pruning_strategy == "most_thoughts":
                                logger.info(f"Most Thoughts Check: Chain {chain_id} (T={current_thought_count}) vs {neighbor_chain_id} (T={neighbor_thought_count})")
                                if current_thought_count >= neighbor_thought_count:
                                    potential_loser_id = chain_id
                                    logger.warning(f"--> Pruning Chain {chain_id} (Most Thoughts: > thoughts).")
                                else:
                                    potential_loser_id = neighbor_chain_id
                                    logger.warning(f"--> Pruning Chain {neighbor_chain_id} (Most Thoughts: > thoughts).")

                            if potential_loser_id: # A loser was identified by the strategy
                                # Determine actual winner and loser for this specific pruning event
                                actual_loser_id = potential_loser_id
                                actual_winner_id = neighbor_chain_id if actual_loser_id == chain_id else chain_id

                                proceed_with_prune = True

                                # CHECK 1: Is the designated winner already going to be pruned in this interval?
                                if actual_winner_id in chains_to_prune_this_interval:
                                    logger.warning(f"Chain {actual_winner_id} (intended pruner of {actual_loser_id}) "
                                                   f"is ALREADY in chains_to_prune_this_interval. Skipping this pruning event.")
                                    proceed_with_prune = False
                                
                                # CHECK 2: Is the designated loser already going to be pruned in this interval?
                                elif actual_loser_id in chains_to_prune_this_interval:
                                    logger.warning(f"Chain {actual_loser_id} (intended_loser) "
                                                   f"is ALREADY in chains_to_prune_this_interval by another chain. Skipping this new pruning event.")
                                    proceed_with_prune = False
                                
                                if proceed_with_prune:
                                    # CHECK 3: "leaves >= 1 active"
                                    num_active_before_this_prune = sum(1 for st in active_chains.values() if st['is_active'])
                                    # num_already_marked_this_interval considers those already in the set
                                    num_already_marked_this_interval = len(chains_to_prune_this_interval)
                                    # If we add actual_loser_id (who is not yet in the set), one more chain is marked.
                                    potential_remaining_active_count = num_active_before_this_prune - num_already_marked_this_interval - 1

                                    if potential_remaining_active_count >= 1:
                                        # All checks passed, COMMIT to this pruning action for this interval
                                        loser_state = active_chains.get(actual_loser_id)
                                        winner_state = active_chains.get(actual_winner_id)

                                        if loser_state and winner_state and loser_state['is_active'] and winner_state['is_active']:
                                            loser_pruned_count = loser_state.get("pruned_count", 0)
                                            count_to_transfer = 1 + loser_pruned_count
                                            winner_state["pruned_count"] = winner_state.get("pruned_count", 0) + count_to_transfer
                                            winner_state.setdefault("pruned_others", []).append({
                                                "pruned_chain_id": actual_loser_id,
                                                "transferred_count": count_to_transfer,
                                                "step": analysis_step
                                            })
                                            loser_state["pruned_by"] = actual_winner_id
                                            logger.info(f"Chain {actual_winner_id} will prune chain {actual_loser_id} (strategy: {pruning_strategy}). "
                                                        f"Transferring count: {count_to_transfer}.")

                                            # Add loser to the set of chains to be pruned at the end of this interval
                                            chains_to_prune_this_interval.add(actual_loser_id)
                                            pruning_info_this_interval[actual_loser_id] = (analysis_step, actual_winner_id)
                                        else:
                                            logger.warning(f"Could not perform prune: "
                                                           f"loser ({actual_loser_id}) or winner ({actual_winner_id}) state invalid/inactive at point of transfer.")
                                    else:
                                        logger.warning(f"--> Skipped pruning Chain {actual_loser_id} by {actual_winner_id}. "
                                                       f"Would leave {potential_remaining_active_count} active chain(s). Keeping >= 1 active.")
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
                if active_chains.get(prune_id, {}).get("is_active", False):
                    logger.debug(f"Marking chain {prune_id} as inactive/pruned.")
                    active_chains[prune_id]["is_active"] = False
                    active_chains[prune_id]["finish_reason"] = "pruned"

                    prune_info = pruning_info_this_interval.get(prune_id)
                    if prune_info:
                         active_chains[prune_id]["pruned_at_step"] = prune_info[0]

                    task_to_cancel = consumer_tasks.get(prune_id)
                    if task_to_cancel and not task_to_cancel.done():
                        logger.debug(f"Cancelling worker task for pruned chain {prune_id}")
                        task_to_cancel.cancel()

                    if pruning_strategy != 'random_after_delay':
                        index_manager.remove_chain_embeddings(prune_id)
                else:
                    logger.debug(f"Chain {prune_id} was marked for pruning but was already inactive.")

        # --- Wait for next interval or task completion ---
        all_tasks_done_event.clear()
        tasks_currently_running = [t for t in consumer_tasks.values() if not t.done()]
        if not tasks_currently_running:
            logger.info("No running consumer tasks left. Exiting analysis loop.")
            break
        try:
            await asyncio.wait_for(all_tasks_done_event.wait(), timeout=ANALYSIS_INTERVAL_SECONDS)
            logger.debug(f"Woke up from wait: Event was set (a task finished).")
        except asyncio.TimeoutError:
            logger.debug(f"Woke up from wait: Timeout reached.")
            pass
        except asyncio.CancelledError:
            logger.info("Analysis loop wait was cancelled.")
            break

    # --- Analysis loop finished ---
    logger.info(f"Analysis loop concluded after {analysis_step} intervals.")

    # --- Ensure all tasks are truly finished ---
    logger.info(f"Waiting for all {len(consumer_tasks)} worker tasks to finalize...")
    try:
        await asyncio.gather(*consumer_tasks.values(), return_exceptions=True)
        logger.info("All worker tasks finalized.")
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
    # Aggregate metrics across ALL chains
    # Now, get the prompt tokens from the first chain's final state.
    # It will either be the pre-calculated value or the server-reported one if received.
    # Default to 0 if chain 1 state is somehow missing
    first_chain_state = active_chains.get(f"q{iteration}_c1")
    total_prompt_tokens_agg = first_chain_state.get("prompt_tokens", 0) if first_chain_state else 0
    if total_prompt_tokens_agg == 0:
         logger.warning(f"Q{iteration}: Aggregated prompt tokens is 0. Check if pre-calculation failed and server usage was not received for chain 1.")

    # total_completion_tokens_across_all_started_chains: Sum of completion tokens
    # from usage reports for ALL chains that were started and whose worker task finished,
    # including those that were pruned or errored. This captures the total compute used.
    total_completion_tokens_across_all_started_chains = sum(
        state.get('completion_tokens', 0) for state in active_chains.values()
        if state.get('finished', False) and state.get('completion_tokens') is not None
    )

    for chain_state in final_completed_chains_for_voting:
        extracted_answer = dataset_handler.extract_answer(chain_state["full_text"])
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
            "final_internal_similarity": final_internal_similarity,
            "pruned_count": chain_state.get("pruned_count", 0)
        })

    voted_answer: Optional[str] = None
    final_score: int = 0
    all_extracted_answers_from_initial_vote: List[str] = [] # Keep track of original answers
    llm_tie_break_performed = False
    tie_break_prompt_tokens_additional = 0
    tie_break_completion_tokens_additional = 0

    if not successful_chain_results_for_voting:
        logger.warning(f"[Q{iteration}] No chains with extracted answers completed streams for majority vote.")
        # voted_answer remains None, final_score 0
    else:
        # Call the modified majority vote function
        vote_status, initial_voted_answer, initial_score, extracted_answers_temp, \
        chains_for_llm_tiebreak, tied_answers_list = majority_vote_for_sim_prune(
            successful_chain_results_for_voting,
            correct_answer_for_scoring, # This is correct_answer_reference
            dataset_name=dataset_name
        )
        all_extracted_answers_from_initial_vote = extracted_answers_temp

        if vote_status == "winner":
            voted_answer = initial_voted_answer
            final_score = initial_score
        elif vote_status == "empty":
            voted_answer = None
            final_score = 0
        elif vote_status == "REQUIRES_LLM_TIEBREAK" and chains_for_llm_tiebreak and tied_answers_list:
            llm_tie_break_performed = True
            logger.info(f"Q{iteration}: Tie detected. Attempting LLM-based tie-breaking.")

            # 1. Construct the tie-breaking prompt
            # Extract the core question from the example. This might vary by dataset.
            # Assuming 'example' has a 'Question' field for GPQA-like datasets.
            # For MATH/AIME, the prompt_text itself is the question.
            original_question_text = example.get("Question", "")
            if not original_question_text and "gpqa" in dataset_name.lower(): # Fallback for GPQA
                 original_question_text = example.get("Problem", "").split("\n\nQuestion: ")[-1].split("\n\nChoices:")[0]
            elif not original_question_text: # General fallback
                 original_question_text = "The original problem was previously presented."


            tie_breaker_prompt_parts = [
                "You will be presented with an original question and several reasoning chains that attempt to answer it."
                "These reasoning chains have resulted in a tie for the most common final answer.",
                "Your task is to review all the provided reasoning chains and determine which reasoning leads to the most confident and correct final answer.\n",
                f"The original question was:\n{original_question_text}\n",
                "Here are the reasoning chains that led to a tie:"
            ]
            for idx, chain_data in enumerate(chains_for_llm_tiebreak):
                tie_breaker_prompt_parts.append(f"\n--- Reasoning Chain {idx+1} (Original Answer: {chain_data.get('extracted_answer')}) ---")
                tie_breaker_prompt_parts.append(chain_data.get("full_content", "Content not available."))
                tie_breaker_prompt_parts.append(f"--- End of Reasoning Chain {idx+1} ---")

            tie_breaker_prompt_parts.append(
                "\nAfter reviewing all chains, please provide your final concluded answer. "
                "Your response should ONLY be the thought process and the final answer, "
                "concluding with the final answer in the same format as the reasoning chains provided above."
                "Do not add any preamble like \"Based on my analysis...\" or \"I have chosen...\". "
                "Just provide the reasoning and the final answer.\n\nFinal Answer:"
            )
            tie_breaker_full_prompt = "\n".join(tie_breaker_prompt_parts)

            # 2. Calculate tiebreak_prompt_tokens
            try:
                tie_break_prompt_tokens_additional = count_tokens(tie_breaker_full_prompt, tokenizer_path)
                if tie_break_prompt_tokens_additional is None: tie_break_prompt_tokens_additional = 0
            except Exception as e:
                logger.error(f"Error counting tokens for tie-breaker prompt: {e}")
                tie_break_prompt_tokens_additional = 0
            
            total_prompt_tokens_agg += tie_break_prompt_tokens_additional # Add to question's total

            # 3. Make the LLM call for tie-breaking
            tie_breaker_response_text = ""
            tie_breaker_request_id = f"q{iteration}_tiebreaker_c{len(active_chains) + 1}" # Unique ID
            logger.info(f"Q{iteration}: Sending tie-breaker prompt to LLM (request_id: {tie_breaker_request_id}). Tokens: {tie_break_prompt_tokens_additional}")
            
            try:
                tie_breaker_stream_gen = stream_vllm_request(
                    prompt=tie_breaker_full_prompt,
                    vllm_url=vllm_url,
                    model_name=model_name,
                    request_id=tie_breaker_request_id,
                    temperature=0.1, # Low temperature for judgment
                    logprobs=None
                )
                async for chunk in tie_breaker_stream_gen:
                    if "error" in chunk:
                        logger.error(f"LLM tie-breaker error: {chunk['error']}")
                        tie_breaker_response_text = "" # Mark as failed
                        break
                    if chunk and "choices" in chunk and chunk["choices"]:
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            tie_breaker_response_text += content
                        # Get completion tokens from usage if available in the last chunk
                        if chunk["choices"][0].get("finish_reason") == "stop" and "usage" in chunk and chunk["usage"]:
                            tie_break_completion_tokens_additional = chunk["usage"].get("completion_tokens", 0)


                if not tie_breaker_response_text:
                    logger.warning(f"Q{iteration}: LLM tie-breaker returned no text.")
                else:
                    logger.info(f"Q{iteration}: LLM tie-breaker response received (length {len(tie_breaker_response_text)}).")
                    # If completion tokens weren't in usage, count them manually (less accurate)
                    if tie_break_completion_tokens_additional == 0:
                        try:
                            manual_ct = count_tokens(tie_breaker_response_text, tokenizer_path)
                            tie_break_completion_tokens_additional = manual_ct if manual_ct is not None else 0
                        except Exception as e_ct:
                            logger.error(f"Error counting tokens for tie-breaker response: {e_ct}")
                            tie_break_completion_tokens_additional = 0
                
                total_completion_tokens_across_all_started_chains += tie_break_completion_tokens_additional # Add to question's total

            except Exception as e:
                logger.exception(f"Q{iteration}: Exception during LLM tie-breaker call.")
                tie_breaker_response_text = "" # Ensure it's empty to trigger fallback

            # 4. Extract answer from LLM tie-breaker response
            if tie_breaker_response_text:
                voted_answer = dataset_handler.extract_answer(tie_breaker_response_text)
                if voted_answer:
                    logger.info(f"Q{iteration}: LLM tie-breaker selected answer: {voted_answer}")
                    final_score = dataset_handler.calculate_score(voted_answer, correct_answer_for_scoring)
                else:
                    logger.warning(f"Q{iteration}: Could not extract answer from LLM tie-breaker response. Applying fallback.")
                    voted_answer, final_score = fallback_tie_break_logic(
                        chains_for_llm_tiebreak,
                        tied_answers_list,
                        correct_answer_for_scoring,
                        dataset_name,
                        tokenizer_path
                    )
            else: # LLM call failed or returned empty
                logger.warning(f"Q{iteration}: LLM tie-breaker failed. Applying fallback.")
                voted_answer, final_score = fallback_tie_break_logic(
                    chains_for_llm_tiebreak,
                    tied_answers_list,
                    correct_answer_for_scoring,
                    dataset_name,
                    tokenizer_path
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
                f.write(f"Accumulated Pruned Count: {chain_state.get('pruned_count', 0)}\n")
                # Add pruned_at_step if the chain was pruned
                if final_status == "pruned":
                     f.write(f"Pruned at Analysis Step: {chain_state.get('pruned_at_step', 'N/A')}\n")
                     f.write(f"Pruned By Chain ID: {chain_state.get('pruned_by', 'N/A')}\n")

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
    summary_data: Dict[str, Any] = {
        "iteration": iteration, "question_id": question_id,
        "status": "SUCCESS" if len(error_chains_data) == 0 else f"PARTIAL_SUCCESS ({len(error_chains_data)}_failed)",
        "n_chains_start": n_chains_start,
        "n_chains_completed_stream_for_voting": len(successful_chain_results_for_voting), # Chains that completed & extracted answers & used for voting
        "n_chains_pruned": len(pruned_chains_data),
        "n_chains_error": len(error_chains_data),
        "similarity_threshold": similarity_threshold,
        "num_steps_to_delay_pruning": num_steps_to_delay_pruning,
        "correct_answer_reference": correct_answer_for_scoring,
        "individual_answers_final": all_extracted_answers_from_initial_vote, # Answers from chains used for voting
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
                "final_internal_similarity": cr.get("final_internal_similarity"),
                "pruned_count": cr.get("pruned_count")
             } for cr in successful_chain_results_for_voting
         ],
         "pruned_chain_details": [ # Details for chains explicitly pruned by similarity
              {
                 "chain_index": int(state['id'].split('_c')[-1]),
                 "finish_reason": state.get('finish_reason'),
                 "completion_tokens": state.get('completion_tokens'), # Tokens generated before pruning
                 "completed_thought_count": state.get('completed_thought_count'),
                 "full_text_len": len(state.get('full_text', '')),
                 "pruned_at_step": state.get("pruned_at_step", None),
                 "pruned_by_chain_id": state.get("pruned_by", None),
                 "initial_pruned_count": state.get("pruned_count", 0)
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
    if llm_tie_break_performed:
        summary_data["llm_tie_break_performed"] = True
        summary_data["llm_tie_break_prompt_tokens"] = tie_break_prompt_tokens_additional
        summary_data["llm_tie_break_completion_tokens"] = tie_break_completion_tokens_additional
        summary_data["llm_tie_break_response_text"] = tie_breaker_response_text

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

    return {
        "iteration": iteration,
        "question_id": question_id,
        "n_chains_start": n_chains_start,
        "n_chains_completed_stream_for_voting": len(successful_chain_results_for_voting),
        "n_chains_pruned": len(pruned_chains_data),
        "n_chains_error": len(error_chains_data),
        "similarity_threshold": similarity_threshold,
        "threshold_schedule": threshold_schedule,
        "correct_answer": correct_answer_for_scoring,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "prompt_tokens": total_prompt_tokens_agg,
        "total_completion_tokens": total_completion_tokens_across_all_started_chains, # Sum across ALL started chains whose streams finished (includes pruned)
        "total_tokens": total_prompt_tokens_agg + total_completion_tokens_across_all_started_chains, # Sum across ALL started chains
        "individual_answers_str": json.dumps(all_extracted_answers_from_initial_vote), # Answers from chains used for voting
        "total_analysis_intervals": analysis_step,
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
        "processing_duration_sec": total_duration,
    }