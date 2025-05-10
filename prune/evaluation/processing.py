# slimsc/prune/evaluation/processing.py
import os
import json
import time
import asyncio
from typing import Dict, Optional, AsyncGenerator, Any

from ..clients import stream_vllm_request, process_stream_chunks
from ..utils import DatasetHandler, count_tokens
from .kv_cache_extraction import extract_kv_cache_usage_for_question
from .voting import majority_vote # Voting needs tokenizer path for tie-breaking

import logging

logger = logging.getLogger(__name__)

async def process_single_stream(stream_generator: AsyncGenerator[Dict, None], chain_index: int) -> Dict:
    """Helper coroutine to consume a stream and process its chunks."""
    # We need to collect all chunks before processing them together
    # because usage stats and finish reasons often come in the final chunks.
    chunks = [chunk async for chunk in stream_generator]
    return process_stream_chunks(chunks, chain_index)


async def process_question_sc_stream(
    example: Dict,
    iteration: int,
    n_chains: int,
    paths: Dict[str, str],
    vllm_url: str,
    model_name: str,
    tokenizer_path: Optional[str],
    dataset_name: str
) -> Optional[Dict]:
    """
    Processes a single question using Self-Consistency by consuming parallel streams.

    Args:
        example (Dict): The dataset example.
        iteration (int): The 1-based index of the question.
        n_chains (int): Number of SC chains to run.
        paths (Dict[str, str]): Dictionary of output paths.
        vllm_url (str): URL of the vllm server.
        model_name (str): Name of the model in vllm.
        tokenizer_path (Optional[str]): Path to tokenizer model/files.
        dataset_name (str): Name of the dataset

    Returns:
        Optional[Dict]: Aggregated results for this question, or None on failure.
    """
    logger.info(f"--- Processing Question {iteration} (N={n_chains}, Streaming) ---")
    question_id = example.get("id", f"index_{iteration-1}")

    # Initialize dataset handler
    handler = DatasetHandler(dataset_name=dataset_name)
    
    # Create Prompt
    try:
        if dataset_name == "gpqa_diamond":
            prompt, choices, correct_answer = handler.create_prompt(example)
        else:
            prompt, correct_answer = handler.create_prompt(example)
            choices = None  # other datasets don't use choices
    except Exception as e:
        logger.exception(f"[red]Error creating prompt for question {iteration}[/red]")
        # Save a failure summary for this question
        summary_data: Dict[str, Any] = { # Add type hint for clarity
            "iteration": iteration, "question_id": question_id, "status": "PROMPT_ERROR",
            "n_chains_requested": n_chains, "error_message": str(e),
            "processing_duration_sec": 0.0, # Mark duration as 0 or N/A
        }
        summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
        try:
           with open(summary_filename, "w") as f:
               json.dump(summary_data, f, indent=2)
        except IOError as e_save:
           logger.exception(f"[red]Error writing prompt error summary file[/red]")
        return None # Return None to indicate failure

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
            logprobs=None
        )
        # Create a task that consumes the generator and processes chunks
        stream_consumers.append(process_single_stream(stream_generator, i + 1))

    start_process_time = time.time()
    # Use asyncio.gather to run streams concurrently
    # `return_exceptions=True` is crucial to ensure that if one stream consumer task fails,
    # the others are not cancelled and the gather operation completes.
    processed_chain_results = await asyncio.gather(*stream_consumers, return_exceptions=True)

    end_process_time = time.time()
    gather_duration = end_process_time - start_process_time
    logger.info(f"[bold green]Gathered & Processed {n_chains} streams for Q{iteration} in {gather_duration:.2f}s[/bold green]")

    # Analyze KV Cache Usage
    # Extract KV cache data *after* all processing for the question is done
    kv_cache_stats = extract_kv_cache_usage_for_question(
        start_process_time, end_process_time, iteration, paths
    )
    avg_kv_usage = kv_cache_stats.get('avg_kv_cache_usage')
    max_kv_usage = kv_cache_stats.get('max_kv_cache_usage')


    # Process Gathered Results
    # processed_chain_results contains results from process_single_stream or exceptions/errors
    completed_streams_with_content = [] # Chains that finished processing chunks, yielded a result dict, AND produced some full_content
    error_chains_data = [] # Chains that resulted in a task-level exception or error chunk from the client or model finish error

    # Separate successful results from errors and incomplete ones
    for result in processed_chain_results:
        if isinstance(result, Exception) or ("error" in result and result.get("error",{}).get("status") not in ["ignored", "none"]): # Check for client/task errors
            error_info = result.get("error", str(result)) if isinstance(result, dict) else str(result)
            chain_idx_err = result.get("chain_index", "N/A") if isinstance(result, dict) else "N/A"
            logger.warning(f"Stream consumer task/chain {chain_idx_err} failed: {error_info}")
            # Store error details
            error_data: Dict[str, Any] = {"chain_index": chain_idx_err, "error": error_info}
            # Try to get partial data if available (e.g., if error happened mid-stream)
            if isinstance(result, dict):
                 error_data.update({
                      "full_content": result.get("full_content", ""),
                      "prompt_tokens": result.get("prompt_tokens"),
                      "completion_tokens": result.get("completion_tokens"),
                      "finish_reason": result.get("finish_reason"),
                 })
            error_chains_data.append(error_data)

        else:
            # Successfully processed chain data (may still have a finish_reason error from model)
            chain_data = result
            # Check if the processed result produced any content. Chains without content are not useful for voting.
            if not chain_data.get("full_content"):
                 logger.warning(f"Chain {chain_data.get('chain_index', 'N/A')} finished stream but produced no content. Excluded from voting.")
                 # Store as an incomplete chain
                 error_data: Dict[str, Any] = {"chain_index": chain_data.get("chain_index", "N/A"), "finish_reason": chain_data.get('finish_reason'), "error": "No content generated"}
                 error_data.update({k: chain_data.get(k) for k in ["full_content", "prompt_tokens", "completion_tokens"]})
                 error_chains_data.append(error_data)
            else:
                 # Chain successfully finished its stream and produced content. Add to list for further processing.
                 completed_streams_with_content.append(chain_data)


    # Now, process the chains that completed their streams and produced content: extract answers and count tokens
    successful_chains_for_voting = [] # Chains that produced content AND had an extractable answer
    total_reasoning_tokens_agg = 0 # Sum over successful chains used in voting
    total_non_reasoning_tokens_agg = 0 # Sum over successful chains used in voting
    token_counting_possible = (tokenizer_path is not None) # Flag if counting is attempted

    for chain_data in completed_streams_with_content:
         # Extract answer (assuming it's in final_answer_text, which is full_content for SC)
         extracted_answer = handler.extract_answer(chain_data.get("final_answer_text", chain_data.get("full_content"))) # Use full_content as fallback
         chain_data["extracted_answer"] = extracted_answer # Add extracted answer to the chain data

         if extracted_answer is None:
              logger.warning(f"Chain {chain_data.get('chain_index', 'N/A')} completed stream and produced content but no answer could be extracted. Excluded from voting.")
              # Store as an incomplete chain that couldn't extract an answer
              error_data: Dict[str, Any] = {"chain_index": chain_data.get("chain_index", "N/A"), "finish_reason": chain_data.get('finish_reason'), "error": "Content generated but no answer extracted"}
              error_data.update({k: chain_data.get(k) for k in ["full_content", "prompt_tokens", "completion_tokens"]})
              error_chains_data.append(error_data)
              continue # Skip chains that didn't extract an answer

         # Chain successfully finished, produced content, and had an extractable answer. Add to list for voting.
         successful_chains_for_voting.append(chain_data)


         # **** Count Tokens using Tokenizer (only for successful chains used for voting) ****
         # This needs to be done *after* extracted_answer is determined if you only count for chains used for voting
         chain_reasoning_tokens = None
         chain_non_reasoning_tokens = None
         if token_counting_possible:
             # For SC, 'reasoning_text' and 'final_answer_text' might just be parts of the full text if the model
             # doesn't explicitly separate them using the custom fields. Use the fields populated by process_stream_chunks.
             reasoning_text = chain_data.get("reasoning_text", "")
             final_answer_text = chain_data.get("final_answer_text", "") # This is the 'content' part

             chain_reasoning_tokens = count_tokens(reasoning_text, tokenizer_path)
             chain_non_reasoning_tokens = count_tokens(final_answer_text, tokenizer_path)

             chain_data["reasoning_tokens_counted"] = chain_reasoning_tokens # Store per-chain count (can be None)
             chain_data["non_reasoning_tokens_counted"] = chain_non_reasoning_tokens # Store per-chain count (can be None)

             # Add to aggregate if counting was successful
             if chain_reasoning_tokens is not None:
                 total_reasoning_tokens_agg += chain_reasoning_tokens
             if chain_non_reasoning_tokens is not None:
                 total_non_reasoning_tokens_agg += chain_non_reasoning_tokens
         else:
              chain_data["reasoning_tokens_counted"] = None # Mark as not counted
              chain_data["non_reasoning_tokens_counted"] = None

    # --- Done processing individual chains ---

    n_chains_used_for_voting = len(successful_chains_for_voting) # Number of chains that completed, produced content, AND extracted an answer

    # Aggregate Usage Stats from ALL requested chains (This calculation is correct and should remain here)
    total_completion_tokens_across_all_requested_chains = sum(
        result.get("completion_tokens", 0) for result in processed_chain_results
        if isinstance(result, dict) and result.get("completion_tokens") is not None
    )
    # Assume prompt tokens are same for all chains, take from the first valid result
    first_prompt_tokens = next(
        (result.get("prompt_tokens") for result in processed_chain_results if isinstance(result, dict) and result.get("prompt_tokens") is not None),
        0 # Default to 0 if no result reported prompt tokens
    )
    total_tokens_usage_across_all_requested_chains = first_prompt_tokens + total_completion_tokens_across_all_requested_chains


    # Perform Majority Vote on chains that produced content AND had an extractable answer
    if n_chains_used_for_voting == 0:
         logger.warning(f"[Q{iteration}] No chains with extracted answers completed streams and extracted answers for majority vote.")
         # Save failure summary
         summary_data_no_vote: Dict[str, Any] = { # Add type hint for clarity
            "iteration": iteration, "question_id": question_id, "status": "NO_CHAINS_FOR_VOTING",
            "prompt_len": len(prompt), "choices": choices, "correct_answer_letter": correct_answer,
            "n_chains_requested": n_chains,
            "n_chains_completed_stream_with_content": len(completed_streams_with_content), # Report how many finished streams with content
            "n_chains_completed_stream_for_voting": 0, # Report how many were used for voting (0)
            "error_chains_count": len(error_chains_data), # Report how many had errors/incomplete
            "avg_kv_cache_usage": avg_kv_usage,
            "max_kv_cache_usage": max_kv_usage,
            "processing_duration_sec": gather_duration,
             "usage_aggregated": {
                 "prompt_tokens": first_prompt_tokens,
                 "total_completion_tokens_across_all_requested_chains": total_completion_tokens_across_all_requested_chains,
                 "total_reasoning_tokens_counted": None, # N/A if no successful chains for voting
                 "total_non_reasoning_tokens_counted": None, # N/A if no successful chains for voting
                 "total_tokens_usage": total_tokens_usage_across_all_requested_chains
             },
            "error_chain_details": error_chains_data # Include details for all failed/incomplete chains
         }
         summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
         try:
            with open(summary_filename, "w") as f:
                json.dump(summary_data_no_vote, f, indent=2)
         except IOError as e:
            logger.exception(f"[red]Error writing failure summary file[/red]")
         return { # Return data for CSV to indicate failure
             "iteration": iteration,
             "question_id": question_id,
             "n_chains_requested": n_chains,
             "n_chains_received": len(completed_streams_with_content), # Report how many streams finished (even if not used for voting)
             "correct_answer": correct_answer,
             "voted_answer": None,
             "final_score": 0,
             "prompt_tokens": first_prompt_tokens,
             "total_completion_tokens": total_completion_tokens_across_all_requested_chains,
             "total_reasoning_tokens": None, # Report None for CSV aggregation
             "total_non_reasoning_tokens": None, # Report None for CSV aggregation
             "total_tokens": total_tokens_usage_across_all_requested_chains,
             "avg_kv_cache_usage": avg_kv_usage,
             "max_kv_cache_usage": max_kv_usage,
             "processing_duration_sec": gather_duration,
             "individual_answers_str": json.dumps([]),
         }

    # Perform Majority Vote on successful chains data
    # Pass tokenizer_path for tie-breaking fallback
    voted_answer, final_score, all_extracted_answers = majority_vote(
        successful_chains_for_voting, correct_answer, dataset_name
    )

    # Save individual chain outputs - loop through ALL initial results to save outputs for every chain
    # This loop needs to be separate from the processing/classification loops
    for i, result in enumerate(processed_chain_results):
         chain_idx = i + 1
         chain_filename = os.path.join(paths["chains"], f"question_{iteration}_chain_{chain_idx}_status.txt") # Placeholder status
         status = "processed_ok" # Default status if no explicit error
         chain_data_to_save = {} # Data to write to file

         if isinstance(result, Exception):
              status = "task_error"
              chain_data_to_save = {"error": str(result), "chain_index": chain_idx}
         elif "error" in result and result.get("error",{}).get("status") not in ["ignored", "none", "json_decode_warning"]:
             status = "client_error" # Error reported by client processing
             chain_data_to_save = result
         elif result.get("full_content") is None or result.get("full_content") == "":
              status = "no_content"
              chain_data_to_save = result
         else:
             # This chain produced content. Check if it extracted an answer.
             extracted_answer = result.get("extracted_answer") # Check if already extracted in the loop above
             # If it was added to successful_chains_for_voting, it had an extracted answer
             if any(c.get('chain_index') == chain_idx for c in successful_chains_for_voting):
                 status = "used_for_voting"
             else:
                 # Produced content, but didn't extract answer (or extracted answer was None)
                 status = "content_no_extract"

             chain_data_to_save = result # Use the full processed result data


         try:
             with open(chain_filename.replace("_status", f"_{status}"), "w") as f:
                  f.write(f"--- Chain {chain_idx} for Question {iteration} ---\n")
                  f.write(f"Status: {status.upper()}\n")
                  f.write(f"Finish Reason (Model): {chain_data_to_save.get('finish_reason', 'N/A')}\n")
                  f.write(f"Extracted Answer: {chain_data_to_save.get('extracted_answer', 'N/A')}\n")
                  f.write(f"Prompt Tokens (Usage): {chain_data_to_save.get('prompt_tokens', 'N/A')}\n")
                  f.write(f"Completion Tokens (Usage): {chain_data_to_save.get('completion_tokens', 'N/A')}\n")
                  f.write(f"Reasoning Tokens (Counted): {chain_data_to_save.get('reasoning_tokens_counted', 'N/A')}\n") # Display counted tokens if available
                  f.write(f"Non-Reasoning Tokens (Counted): {chain_data_to_save.get('non_reasoning_tokens_counted', 'N/A')}\n") # Display counted tokens if available
                  f.write(f"Total Tokens (Usage): {chain_data_to_save.get('total_tokens_usage', 'N/A')}\n")
                  if 'error' in chain_data_to_save:
                      f.write(f"Error Details: {chain_data_to_save['error']}\n")

                  f.write("\n--- Reasoning Content ---\n")
                  f.write(chain_data_to_save.get('reasoning_text', 'N/A'))
                  f.write("\n\n--- Final Answer Content ---\n")
                  f.write(chain_data_to_save.get('final_answer_text', 'N/A'))

         except IOError as e:
             logger.exception(f"[red]Error writing chain output file {chain_filename.replace('_status', f'_{status}')}[/red]")


    # --- Save summary JSON ---
    summary_data: Dict[str, Any] = { # Add type hint for clarity
        "iteration": iteration,
        "question_id": question_id,
        "status": "SUCCESS" if len(error_chains_data) == 0 else f"PARTIAL_SUCCESS ({len(error_chains_data)}_failed)",
        "n_chains_requested": n_chains,
        "n_chains_completed_stream_with_content": len(completed_streams_with_content),
        "n_chains_completed_stream_for_voting": n_chains_used_for_voting, # Chains used for voting
        "error_chains_count": len(error_chains_data), # Report how many failed/incomplete
        "prompt_len": len(prompt),
        "correct_answer_letter": correct_answer,
        "individual_answers": all_extracted_answers, # Answers from chains used for voting
        "voted_answer": voted_answer,
        "final_score": final_score,
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
        "processing_duration_sec": gather_duration, # Use float for JSON
        "usage_aggregated": {
             "prompt_tokens": first_prompt_tokens,
             "total_completion_tokens_across_all_requested_chains": total_completion_tokens_across_all_requested_chains,
             "total_reasoning_tokens_counted": total_reasoning_tokens_agg if token_counting_possible else None,
             "total_non_reasoning_tokens_counted": total_non_reasoning_tokens_agg if token_counting_possible else None,
             "total_tokens_usage": total_tokens_usage_across_all_requested_chains
        },
        "chains_for_voting_details": [ # Save key details per chain used for voting
             {
                "chain_index": cr.get("chain_index"),
                "finish_reason": cr.get("finish_reason"),
                "extracted_answer": cr.get("extracted_answer"),
                "prompt_tokens": cr.get("prompt_tokens"),
                "completion_tokens": cr.get("completion_tokens"), # Usage completion tokens
                "reasoning_tokens_counted": cr.get("reasoning_tokens_counted"), # Counted tokens
                "non_reasoning_tokens_counted": cr.get("non_reasoning_tokens_counted"), # Counted tokens
             } for cr in successful_chains_for_voting
         ],
         "error_chain_details": error_chains_data # Include details for all failed chains
    }

    summary_filename = os.path.join(paths["summaries"], f"question_{iteration}_summary.json")
    try:
        with open(summary_filename, "w") as f:
            json.dump(summary_data, f, indent=2)
    except IOError as e:
        logger.exception(f"[red]Error writing summary file {summary_filename}[/red]")

    # --- Return data for final CSV ---
    # Standardized metrics requested: overall accuracy, mean/max completion tokens, mean/max kv cache usage, mean/max processing duration.
    # The CSV stores PER-QUESTION values, which are then used for overall aggregation.
    # total_completion_tokens should be the sum across ALL chains requested, as reported by usage stats.

    return {
        "iteration": iteration,
        "question_id": question_id,
        "n_chains_requested": n_chains,
        "n_chains_received": len(completed_streams_with_content), # Report how many streams finished (even if not used for voting)
        "n_chains_completed_stream_with_content": len(completed_streams_with_content), # New metric for CSV
        "n_chains_completed_stream_for_voting": n_chains_used_for_voting, # New metric for CSV
        "correct_answer": correct_answer,
        "voted_answer": voted_answer,
        "final_score": final_score,
        "prompt_tokens": first_prompt_tokens, # From usage, assuming uniform across chains
        "total_completion_tokens": total_completion_tokens_across_all_requested_chains, # Usage total across ALL requested chains
        "total_reasoning_tokens": total_reasoning_tokens_agg if token_counting_possible else None, # Counted total over successful chains (used for vote)
        "total_non_reasoning_tokens": total_non_reasoning_tokens_agg if token_counting_possible else None, # Counted total over successful chains (used for vote)
        "total_tokens": total_tokens_usage_across_all_requested_chains, # Usage total across ALL requested chains + prompt
        "avg_kv_cache_usage": avg_kv_usage,
        "max_kv_cache_usage": max_kv_usage,
        "processing_duration_sec": gather_duration, # Added duration to CSV
        "individual_answers_str": json.dumps(all_extracted_answers), # Answers from chains used for voting
    }