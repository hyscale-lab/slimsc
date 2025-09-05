# slimsc/prune/clients/vllm_client.py
import asyncio
import aiohttp
import json
import time
from typing import Dict, Optional, List, AsyncGenerator
import random

import logging

logger = logging.getLogger(__name__)

# Use a single session for connection pooling
_aiohttp_session = None

async def get_aiohttp_session() -> aiohttp.ClientSession:
    """Initializes and returns a shared aiohttp ClientSession."""
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        timeout_config = aiohttp.ClientTimeout(
            total=None,          # No overall total timeout for the entire request life-cycle
            connect=60,          # Timeout for establishing connection
            sock_connect=60,     # Timeout for socket connection part
            sock_read=3600       # Timeout for waiting for data from server AFTER connection
        )
        connector = aiohttp.TCPConnector(limit_per_host=100)
        _aiohttp_session = aiohttp.ClientSession(timeout=timeout_config, connector=connector)
    return _aiohttp_session

async def close_aiohttp_session():
    """Closes the shared aiohttp session."""
    global _aiohttp_session
    if _aiohttp_session and not _aiohttp_session.closed:
        await _aiohttp_session.close()
    _aiohttp_session = None
    await asyncio.sleep(0.1) # Allow time for connections to close gracefully

async def stream_vllm_request(
    prompt: str,
    vllm_url: str,
    model_name: str,
    request_id: str, # Unique ID for logging/tracking this specific request
    temperature: float = 0.6,
    max_tokens: int = None,
    stop_sequences: Optional[List[str]] = None,
    logprobs: Optional[int] = None, # Logprobs might behave differently in stream
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    max_backoff: float = 10.0
) -> AsyncGenerator[Dict, None]:
    """
    Makes a single asynchronous STREAMING request (n=1) and yields parsed delta chunks.

    Handles the custom `reasoning_content` and `content` fields if present.

    Yields:
        Dict: Parsed JSON data from each stream chunk.
              Includes a final chunk with usage stats if provided by the server.
    """
    api_url = f"{vllm_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "n": 1,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop_sequences,
        "request_id": request_id,
        "stream": True, # Enable streaming
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        # Logprobs in stream might need specific server support / format
        **({"logprobs": True, "top_logprobs": logprobs} if logprobs is not None else {})
    }
    # Fix for payload where logprobs=1 was used instead of True
    if logprobs is not None:
        payload["logprobs"] = True


    session = await get_aiohttp_session()
    start_time = time.time()
    logging.debug(f"Starting stream request {request_id}...")

    retries = 0
    backoff = initial_backoff
    while retries <= max_retries:
        try:
            async with session.post(api_url, headers=headers, json=payload) as response:
                response.raise_for_status() # Check initial HTTP status

                async for line_bytes in response.content:
                    line = line_bytes.decode('utf-8').strip()
                    if line.startswith("data:"):
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            logging.debug(f"Stream {request_id} received [DONE]")
                            break # Stream finished
                        try:
                            chunk_data = json.loads(data_str)
                            yield chunk_data # Yield the parsed chunk
                        except json.JSONDecodeError:
                            logger.exception(f"[red]Warning: Failed to decode JSON chunk for {request_id}: {data_str}[/red]")
                    elif line: # Handle potential non-SSE lines if needed
                        logger.warning(f"[yellow]Warning: Received non-SSE line for {request_id}: {line}[/yellow]")

                elapsed_time = time.time() - start_time
                logging.debug(f"Stream {request_id} finished in {elapsed_time:.2f}s") # Verbose
                return # Exit the retry loop

        except (aiohttp.ClientError, asyncio.TimeoutError) as e: # Catch potentially transient errors
            retries += 1
            if retries > max_retries:
                logging.exception(f"[red]Max retries reached for {request_id}.[/red]")
                yield {"error": {"status": "failed_after_retries", "message": str(e), "request_id": request_id}}
                return # Stop retrying

            logging.warning(f"[yellow]Request {request_id} failed (Attempt {retries}/{max_retries}). Retrying in {backoff:.2f}s.[/yellow]")
            await asyncio.sleep(backoff)
            # Exponential backoff with jitter
            backoff = min(max_backoff, backoff * 2) + random.uniform(0, 1)
            continue # Go to next iteration of the while loop

        except Exception as e: # Catch unexpected errors - maybe don't retry these
             yield {"error": {"status": "unexpected", "message": f"{type(e).__name__}: {e}", "request_id": request_id}}
             logging.exception(f"[red]Unexpected Error during stream {request_id} (not retrying): {type(e).__name__}[/red]")
             return # Stop on non-retryable errors


def process_stream_chunks(chunks: List[Dict], chain_index: int) -> Dict:
    """
    Processes a list of collected stream chunks for a single chain.
    Accumulates reasoning_content and content separately. Extracts final usage.

    Args:
        chunks (List[Dict]): List of parsed JSON chunks received for one stream.
        chain_index (int): The 1-based index of this chain.

    Returns:
        Dict: Processed data including accumulated texts and usage.
    """
    accumulated_reasoning = ""
    accumulated_content = ""
    final_usage = None
    finish_reason = None
    prompt_tokens = None # Usually in the final usage chunk for vLLM stream

    for chunk in chunks:
        if "error" in chunk: # Skip processing if an error chunk was yielded
             return {"error": chunk["error"], "chain_index": chain_index}

        if not chunk or "choices" not in chunk or not chunk["choices"]:
            continue # Skip empty or invalid chunks

        delta = chunk["choices"][0].get("delta", {})
        #finish_reason = chunk["choices"][0].get("finish_reason") or finish_reason # Capture last non-null finish_reason

        # Check for custom reasoning_content field FIRST
        if "reasoning_content" in delta and delta["reasoning_content"] is not None:
            accumulated_reasoning += delta["reasoning_content"]
        # Check for standard content field SECOND
        elif "content" in delta and delta["content"] is not None:
            accumulated_content += delta["content"]

        # Check for finish reason in the main choice object (often in last chunk)
        if chunk["choices"][0].get("finish_reason"):
            finish_reason = chunk["choices"][0].get("finish_reason")


        # Usage stats usually arrive in the last chunk in vLLM's streaming format
        if "usage" in chunk and chunk["usage"] is not None:
            final_usage = chunk["usage"]
            prompt_tokens = final_usage.get("prompt_tokens") # Get prompt tokens from final usage


    # Get completion tokens from final usage if available
    completion_tokens = final_usage.get("completion_tokens") if final_usage else None

    return {
        "chain_index": chain_index,
        "reasoning_text": accumulated_reasoning,
        "final_answer_text": accumulated_content, # Assuming non-reasoning is the final answer part
        "full_content": accumulated_reasoning + accumulated_content, # Combine for potential full analysis
        "logprobs": None, # Logprobs handling in stream needs specific implementation based on server format
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "usage": final_usage # Store the whole usage dict if needed
    }