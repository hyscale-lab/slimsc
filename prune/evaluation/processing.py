import os
import json
import time
import asyncio
from typing import Dict, Optional, AsyncGenerator

from ..clients import stream_vllm_request, process_stream_chunks
from ..utils import create_prompt_gpqa, extract_answer_gpqa, count_tokens
from .kv_cache_extraction import extract_kv_cache_usage_for_question
from .voting import majority_vote

import logging

logger = logging.getLogger(__name__)

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
    logger.info(f"--- Processing Question {iteration} (N={n_chains}, Streaming) ---")
    question_id = example.get("id", f"index_{iteration-1}")

    # Create Prompt
    try:
        prompt, choices, correct_answer_letter = create_prompt_gpqa(example)
    except Exception as e:
        logger.exception(f"[red]Error creating prompt for question {iteration}[/red]")
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
    logger.info(f"[bold green]Gathered & Processed {n_chains} streams for Q{iteration} in {gather_duration:.2f}s[/bold green]")

    # Analyze KV Cache Usage
    kv_cache_stats = extract_kv_cache_usage_for_question(
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
            logger.warning(f"Stream consumer task/chain {chain_idx_err} failed: {error_info}")
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
                  logger.warning(f"Prompt token mismatch in Q{iteration}. Chain {chain_data['chain_index']}: {chain_data['prompt_tokens']}, First: {first_prompt_tokens}")

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
                logger.exception(f"[red]Error writing chain output file {chain_filename}[/red]")


    n_chains_received = len(successful_chains_data)
    if n_chains_received == 0:
         logger.warning(f"All chains failed for question {iteration}. Skipping.")
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
            logger.exception(f"[red]Error writing failure summary file[/red]")
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
        logger.exception(f"[red]Error writing summary file {summary_filename}[/red]")

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