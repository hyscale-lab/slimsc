# slimsc/prune/evaluation/kv_cache_extraction.py
import os
import pandas as pd
from typing import Dict, Optional

import logging

logger = logging.getLogger(__name__)


def clear_source_kv_cache(source_kv_file: str):
    """Clear the source KV cache file"""
    if source_kv_file:
        try:
            if os.path.exists(source_kv_file):
                logger.info(f"Removing existing KV cache usage file: {source_kv_file}")
                os.remove(source_kv_file)
            else:
                logger.warning(f"[yellow]KV cache usage file {source_kv_file} does not exist, proceeding.[/yellow]")
        except OSError as e:
            logger.exception(f"[red]Could not remove KV cache file {source_kv_file}. Permissions issue?[/red]")
    else:
        logger.warning("[yellow]Source KV cache file path not found in paths configuration.[/yellow]")


def extract_kv_cache_usage_for_question(
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
        logger.warning(f"[yellow][Q{iteration}]: Source KV cache usage file path not configured.[/yellow]")
        return results
    if not save_dir:
        logger.warning(f"[yellow][Q{iteration}]: KV cache usage save directory path not configured.[/yellow]")
        # Continue to calculate aggregates if possible, but cannot save per-question log
    if not os.path.exists(source_file):
        logger.warning(f"[yellow][Q{iteration}]: KV cache usage file not found at {source_file}. Cannot analyze or save.[/yellow]")
        return results

    try:
        # Define expected header names from server log
        header_names = ['timestamp', 'gpu_cache_usage_perc']
        # Try reading, explicitly providing names and setting header=0 if file has header
        # Adjust header=0/None based on whether your server log *actually* writes a header row
        try:
            df = pd.read_csv(source_file) # Try reading normally first
            if list(df.columns) != header_names:
                 logger.warning(f"[yellow][Q{iteration}]: Header mismatch in {source_file}. Expected {header_names}, got {list(df.columns)}. Attempting recovery.[/yellow]")
                 # Reread forcing names, skipping the actual header row if present
                 df = pd.read_csv(source_file, names=header_names, header=0, skiprows=[0] if os.path.getsize(source_file)>0 else None)
        except pd.errors.ParserError:
             # Fallback if parsing fails, maybe due to bad lines or no header
             logger.warning(f"[yellow][Q{iteration}]: ParserError reading {source_file}. Attempting read with explicit names and no header.[/yellow]")
             df = pd.read_csv(source_file, names=header_names, header=None)


        if df.empty:
             logger.warning(f"[yellow][Q{iteration}]: KV cache usage file {source_file} is empty after reading.[/yellow]")
             # Optionally save an empty file with header
             if save_dir:
                  empty_save_path = os.path.join(save_dir, f"question_{iteration}_kvcache_usage.csv")
                  try:
                       pd.DataFrame(columns=header_names).to_csv(empty_save_path, index=False)
                       logger.info(f"Saved empty KV cache log for Q{iteration} to {empty_save_path}")
                  except IOError as e_save:
                       logger.exception(f"[red]Error saving empty KV log for Q{iteration} to {empty_save_path}[/red]")
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
                logger.info(f"Saved filtered KV cache log for Q{iteration} {status_msg} to {saved_log_path}")
            except IOError as e_save:
                logger.exception(f"[red]Error saving filtered KV log for Q{iteration} to {saved_log_path}[/red]")
                saved_log_path = None # Indicate saving failed
        else:
            logger.warning(f"[yellow]Info [Q{iteration}]: Save directory not configured. Skipping save of filtered KV log.[/yellow]")


        # Calculate aggregates from filtered data
        if df_filtered.empty:
            logger.warning(f"[yellow][Q{iteration}]: No KV cache usage data found within the time window[/yellow] "
                  f"({start_time:.2f} - {end_time:.2f}) in {source_file}.")
            # Results remain None
        else:
            avg_usage = df_filtered['gpu_cache_usage_perc'].mean()
            max_usage = df_filtered['gpu_cache_usage_perc'].max()

            results['avg_kv_cache_usage'] = float(avg_usage) if pd.notna(avg_usage) else None
            results['max_kv_cache_usage'] = float(max_usage) if pd.notna(max_usage) else None

            logger.info(f"KV Cache Usage % Aggregates [Q{iteration}]: "
                  f"Avg={results['avg_kv_cache_usage']:.4f}, Max={results['max_kv_cache_usage']:.4f}")

    except pd.errors.EmptyDataError:
         logger.exception(f"[red]Warning [Q{iteration}]: KV cache usage file {source_file} is empty or header-only.[/red]")
    except FileNotFoundError:
         logger.exception(f"[red]Warning [Q{iteration}]: KV cache usage file not found at {source_file}.[/red]")
    except Exception as e:
        logger.exception(f"[red]Error processing KV cache usage for Question {iteration} from {source_file}[/red]")
        import traceback
        traceback.print_exc() # More detailed error

    # Add saved path to results for potential logging elsewhere if needed (optional)
    # results['saved_kv_log_path'] = saved_log_path
    return results