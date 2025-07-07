# ~/slimsc/prune/utils/recalculate_simprune_scores.py

import os
import re
import sys
import json
import pandas as pd
import argparse
import logging
import string
import tempfile
from typing import Any
from tqdm import tqdm
from rich.logging import RichHandler

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
try:
    from prune.utils.dataset_handler import DatasetHandler
    from prune.evaluation.voting import majority_vote_for_sim_prune, fallback_tie_break_logic
except ImportError as e:
    print(f"Error: Could not import 'slimsc' library modules: {e}")
    exit(1)

# --- Logger Setup ---
logger = logging.getLogger(__name__)


# --- Sanitization Functions ---
def sanitize_path_strict(path: str) -> str:
    """Removes any character not explicitly known to be safe for file paths."""
    safe_chars = string.ascii_letters + string.digits + "_-./~"
    sanitized_path = ''.join(filter(lambda char: char in safe_chars, path))
    if path != sanitized_path:
        logger.warning(f"[yellow]Sanitized input path. Original: '{repr(path)}', Cleaned: '{repr(sanitized_path)}'[/yellow]")
    return sanitized_path

def sanitize_string(s: str) -> str:
    """Sanitizes a single string by removing non-printable characters."""
    if not isinstance(s, str):
        return s
    cleaned_s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    return cleaned_s.encode('utf-8', 'replace').decode('utf-8')

def sanitize_data_recursive(data: Any) -> Any:
    """Recursively sanitizes all string values in a data structure."""
    if isinstance(data, dict):
        return {k: sanitize_data_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_data_recursive(item) for item in data]
    elif isinstance(data, str):
        return sanitize_string(data)
    else:
        return data


# --- File I/O and Parsing Functions ---
def safe_write_json(data: Any, final_path: str):
    """Safely writes a JSON file by using a temporary file and renaming."""
    temp_dir = os.path.dirname(final_path)
    # Use a file-like object from tempfile to handle creation and cleanup
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=temp_dir, delete=False, suffix='.tmp') as tmp:
            json.dump(data, tmp, indent=2)
            temp_path = tmp.name
        # Atomically rename the temporary file to the final destination
        os.rename(temp_path, final_path)
    except Exception as e:
        logger.error(f"Failed during safe write to {final_path}")
        # Clean up the temporary file if it still exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e # Re-raise the exception to be caught by the calling function

def safe_write_csv(dataframe: pd.DataFrame, final_path: str):
    """Safely writes a DataFrame to a CSV file using a temporary file."""
    temp_dir = os.path.dirname(final_path)
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=temp_dir, delete=False, suffix='.tmp') as tmp:
            dataframe.to_csv(tmp.name, index=False)
            temp_path = tmp.name
        os.rename(temp_path, final_path)
    except Exception as e:
        logger.error(f"Failed during safe CSV write to {final_path}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

def parse_chain_file_content_last_five(filepath: str) -> str:
    """Parses a chain output file by reading only its last five lines."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        return "".join(lines[-5:]).strip()
    except Exception as e:
        logger.exception(f"Error parsing chain file {filepath}: {e}")
        return ""


def get_chain_index_from_filename(filename: str) -> int:
    """Extracts the chain index from a filename."""
    match = re.search(r'_chain_(\d+)_', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract chain index from filename: {filename}")


# --- Main Recalculation Logic ---
def recalculate_run(results_dir: str, dataset_name: str, tokenizer_path: str):
    """Main function to recalculate scores for a similarity pruning run."""
    sanitized_results_dir = sanitize_path_strict(results_dir)
    logger.info(f"Starting recalculation for SimPrune run at: {sanitized_results_dir}")
    
    if not tokenizer_path:
        logger.error("A --tokenizer_path is required for fallback tie-breaking logic.")
        return

    chains_dir = os.path.join(sanitized_results_dir, "individual_chains")
    summaries_dir = os.path.join(sanitized_results_dir, "summaries")
    csv_path = os.path.join(sanitized_results_dir, "evaluation_summary.csv")
    aggregated_metrics_path = os.path.join(sanitized_results_dir, "aggregated_metrics.json")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV at {csv_path}")
        logger.exception(e)
        return
    
    dataset_handler = DatasetHandler(dataset_name=dataset_name)
    updated_rows = []
    
    logger.info(f"Processing {len(df)} questions from the evaluation summary...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Recalculating Questions"):
        row_dict = row.to_dict()
        iteration = int(row_dict['iteration'])
        correct_answer = row_dict['correct_answer']

        chain_files = [f for f in os.listdir(chains_dir) if f.startswith(f"question_{iteration}_chain_") and f.endswith("_completed.txt")]
        
        if not chain_files:
            updated_rows.append(row_dict)
            continue

        successful_chain_results_for_voting = []
        for chain_filename in chain_files:
            content = parse_chain_file_content_last_five(os.path.join(chains_dir, chain_filename))
            sanitized_content = sanitize_string(content)
            new_extracted_answer = dataset_handler.extract_answer(sanitized_content)
            if new_extracted_answer is None:
                continue
            successful_chain_results_for_voting.append({
                "chain_index": get_chain_index_from_filename(chain_filename), "extracted_answer": new_extracted_answer,
                "full_content": sanitized_content, "pruned_count": 0,
                "final_internal_similarity": 0.0, "completion_tokens": 0,
            })

        successful_chain_results_for_voting.sort(key=lambda x: x['chain_index'])
        vote_status, initial_voted_answer, initial_score, all_extracted_answers, \
        chains_for_llm_tiebreak, tied_answers_list = majority_vote_for_sim_prune(
            successful_chain_results_for_voting, correct_answer, dataset_name=dataset_name)

        voted_answer, final_score = None, 0
        if vote_status == "winner":
            voted_answer, final_score = initial_voted_answer, initial_score
        elif vote_status == "REQUIRES_LLM_TIEBREAK":
            voted_answer, final_score = fallback_tie_break_logic(
                chains_for_llm_tiebreak, tied_answers_list, correct_answer, dataset_name, tokenizer_path)

        row_dict['voted_answer'], row_dict['final_score'], row_dict['individual_answers_str'] = \
            voted_answer, final_score, json.dumps(all_extracted_answers)
        updated_rows.append(row_dict)

        summary_path = os.path.join(summaries_dir, f"question_{iteration}_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8', errors='replace') as f:
                    summary_data = json.load(f)
                
                summary_data.update({
                    'voted_answer': voted_answer, 'final_score': final_score,
                    'individual_answers_final': all_extracted_answers
                })
                
                sanitized_summary_data = sanitize_data_recursive(summary_data)
                safe_write_json(sanitized_summary_data, summary_path)

            except Exception as e:
                logger.warning(f"Failed to update summary for Q{iteration}. Path: {summary_path}")
                logger.exception(e)

    if not updated_rows:
        return
        
    new_df = pd.DataFrame(updated_rows)
    try:
        new_df = new_df[df.columns]
        safe_write_csv(new_df, csv_path)
        logger.info(f"Successfully updated and saved 'evaluation_summary.csv'")
    except Exception as e:
        logger.error(f"Failed to save the final CSV. Path: {csv_path}")
        logger.exception(e)
        return

    logger.info("Recalculating aggregated metrics...")
    try:
        with open(aggregated_metrics_path, 'r', encoding='utf-8', errors='replace') as f:
            aggregated_metrics = json.load(f)

        overall_accuracy = new_df['final_score'].dropna().mean() if not new_df['final_score'].dropna().empty else 0.0
        aggregated_metrics['metrics'].update({
            'overall_accuracy': float(overall_accuracy),
            'num_qns_with_score': int(new_df['final_score'].notna().sum())
        })
        
        sanitized_aggregated_metrics = sanitize_data_recursive(aggregated_metrics)
        safe_write_json(sanitized_aggregated_metrics, aggregated_metrics_path)
        logger.info(f"Successfully updated and saved 'aggregated_metrics.json'")
        logger.info(f"New Overall Accuracy: {overall_accuracy:.4f}")

    except Exception as e:
        logger.error(f"Failed to update aggregated_metrics.json. Path: {aggregated_metrics_path}")
        logger.exception(e)

    logger.info("[bold green]Recalculation complete![/bold green]")


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="[%X]",
                        handlers=[RichHandler(markup=True, rich_tracebacks=True, log_time_format="[%X] ")])

def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Recalculate scores for a Similarity Pruning run.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--results_dir', type=str, required=True,
                        help="Path to the specific SimPrune run directory.")
    parser.add_argument('--dataset_name', type=str, default='hmmt',
                        help="The name of the dataset used in the run (e.g., 'hmmt').")
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to HuggingFace tokenizer directory (REQUIRED for tie-breaking).')
    args = parser.parse_args()
    recalculate_run(args.results_dir, args.dataset_name, args.tokenizer_path)

if __name__ == "__main__":
    main()