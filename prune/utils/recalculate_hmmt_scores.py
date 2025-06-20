# ~/slimsc/prune/utils/recalculate_hmmt_scores.py

import os
import re
import sys
import json
import pandas as pd
import argparse
import logging
import string
from tqdm import tqdm
from rich.logging import RichHandler

# --- Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Imports ---
try:
    from prune.utils.dataset_handler import DatasetHandler
    from prune.evaluation.voting import majority_vote
except ImportError as e:
    print(f"Error: Could not import 'slimsc' library modules: {e}")
    print("Please ensure you are running this script from its intended location ('~/slimsc/prune/utils/')")
    print("and that the necessary __init__.py files exist in the directories.")
    exit(1)

# --- Logger Setup ---
logger = logging.getLogger(__name__)

# --- NEW: Stricter, Whitelist-based Sanitization function ---
def sanitize_path_strict(path: str) -> str:
    """
    Removes any character not explicitly known to be safe for file paths.
    This is a much stricter whitelist approach.
    """
    # Whitelist of safe characters
    safe_chars = string.ascii_letters + string.digits + "_-./~"
    
    sanitized_path = ''.join(filter(lambda char: char in safe_chars, path))
    
    if path != sanitized_path:
        logger.warning(f"[yellow]Sanitized input path with strict filter. Original: '{repr(path)}', Cleaned: '{repr(sanitized_path)}'[/yellow]")
    return sanitized_path


def parse_chain_file_content_last_five(filepath: str) -> str:
    """
    Parses a chain output file by reading only its last five lines.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        last_lines = lines[-5:]
        content = "".join(last_lines)
        return content.strip()
    
    except FileNotFoundError:
        logger.error(f"Chain file not found: {filepath}")
        return ""
    except Exception as e:
        logger.exception(f"Error parsing chain file {filepath}: {e}")
        return ""


def get_chain_index_from_filename(filename: str) -> int:
    """Extracts the chain index (e.g., 3 from 'question_10_chain_3_...txt')."""
    match = re.search(r'_chain_(\d+)_', filename)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract chain index from filename: {filename}")


def recalculate_run(results_dir: str, dataset_name: str):
    """
    Main function to recalculate scores for a given HMMT evaluation run.
    """
    # --- Use the new, stricter sanitizer on the input path ---
    sanitized_results_dir = sanitize_path_strict(results_dir)

    if dataset_name != "hmmt":
        logger.error(f"This script is specifically designed for the 'hmmt' dataset, but got '{dataset_name}'. Aborting.")
        return

    logger.info(f"Starting recalculation for HMMT run at: {sanitized_results_dir}")

    # --- 1. Define paths using the SANITIZED base directory ---
    chains_dir = os.path.join(sanitized_results_dir, "individual_chains")
    summaries_dir = os.path.join(sanitized_results_dir, "summaries")
    csv_path = os.path.join(sanitized_results_dir, "evaluation_summary.csv")
    aggregated_metrics_path = os.path.join(sanitized_results_dir, "aggregated_metrics.json")

    if not os.path.isdir(sanitized_results_dir):
         logger.error(f"The provided results directory does not exist or is not a directory: {sanitized_results_dir}")
         return

    required_paths = [chains_dir, summaries_dir, csv_path, aggregated_metrics_path]
    if not all(os.path.exists(p) for p in required_paths):
        logger.error("One or more required directories/files are missing. Please check the results directory.")
        for p in required_paths:
            logger.error(f"Path: {p} (Exists: {os.path.exists(p)})")
        return

    # --- 2. Load existing data and initialize handlers ---
    try:
        df = pd.read_csv(csv_path)
    except OSError as e:
        logger.error(f"OSError reading CSV file. Path may be invalid. Path repr: {repr(csv_path)}")
        logger.exception(e)
        return
    
    dataset_handler = DatasetHandler(dataset_name=dataset_name)
    updated_rows = []
    
    # --- 3. Loop through each question and re-evaluate ---
    logger.info(f"Processing {len(df)} questions from the evaluation summary...")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Recalculating Questions"):
        iteration = int(row['iteration'])
        correct_answer = row['correct_answer']

        chain_files = [
            f for f in os.listdir(chains_dir) 
            if f.startswith(f"question_{iteration}_chain_") and f.endswith("_used_for_voting.txt")
        ]
        
        if not chain_files:
            logger.warning(f"[yellow]No chains with suffix '_used_for_voting.txt' found for question {iteration}. Keeping original result.[/yellow]")
            updated_rows.append(row.to_dict())
            continue

        re_evaluated_chains = []
        for chain_filename in chain_files:
            chain_filepath = os.path.join(chains_dir, chain_filename)
            last_five_lines_content = parse_chain_file_content_last_five(chain_filepath)
            
            new_extracted_answer = None
            if last_five_lines_content:
                new_extracted_answer = dataset_handler.extract_answer(last_five_lines_content)

            chain_index = get_chain_index_from_filename(chain_filename)
            re_evaluated_chains.append({
                "chain_index": chain_index,
                "extracted_answer": new_extracted_answer,
                "full_content": last_five_lines_content
            })

        re_evaluated_chains.sort(key=lambda x: x['chain_index'])

        voted_answer, final_score, all_extracted_answers = majority_vote(
            chain_results=re_evaluated_chains,
            correct_answer_letter=correct_answer,
            dataset_name=dataset_name,
        )

        row['voted_answer'] = voted_answer
        row['final_score'] = final_score
        row['individual_answers_str'] = json.dumps(all_extracted_answers)
        updated_rows.append(row.to_dict())

        summary_path = os.path.join(summaries_dir, f"question_{iteration}_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                
                summary_data['voted_answer'] = voted_answer
                summary_data['final_score'] = final_score
                summary_data['individual_answers'] = all_extracted_answers

                if 'chains_for_voting_details' in summary_data:
                    summary_data['chains_for_voting_details'].sort(key=lambda x: x.get('chain_index', float('inf')))
                    
                    chain_answer_map = {chain['chain_index']: chain['extracted_answer'] for chain in re_evaluated_chains}
                    for detail in summary_data['chains_for_voting_details']:
                        chain_idx = detail.get('chain_index')
                        if chain_idx in chain_answer_map:
                            detail['extracted_answer'] = chain_answer_map[chain_idx]

                # Diagnostic logging before writing
                logger.debug(f"Attempting to write summary. Path repr: {repr(summary_path)}")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(summary_data, f, indent=2)

            except OSError as e:
                logger.warning(f"Could not update summary for question {iteration}. Path repr: {repr(summary_path)}")
                logger.exception(e)
            except Exception as e:
                logger.warning(f"A non-OS error occurred while updating summary for question {iteration}")
                logger.exception(e)


    # --- 4. Save the updated CSV ---
    new_df = pd.DataFrame(updated_rows)
    if not new_df.empty:
        try:
            new_df = new_df[df.columns]
            # Diagnostic logging before writing
            logger.debug(f"Attempting to write CSV. Path repr: {repr(csv_path)}")
            new_df.to_csv(csv_path, index=False)
            logger.info(f"Successfully updated and saved 'evaluation_summary.csv'")
        except OSError as e:
            logger.error(f"Failed to save the final CSV. Path repr: {repr(csv_path)}")
            logger.exception(e)
            return # Exit if we can't save the main results

    # --- 5. Recalculate and save aggregated metrics ---
    logger.info("Recalculating aggregated metrics...")
    try:
        with open(aggregated_metrics_path, 'r', encoding='utf-8') as f:
            aggregated_metrics = json.load(f)

        num_questions_with_score = new_df['final_score'].dropna().shape[0]
        overall_accuracy = new_df['final_score'].dropna().mean() if num_questions_with_score > 0 else 0.0

        aggregated_metrics['metrics']['overall_accuracy'] = f'{overall_accuracy:.2f}'
        aggregated_metrics['metrics']['num_questions_with_score'] = num_questions_with_score
        
        # Diagnostic logging before writing
        logger.debug(f"Attempting to write aggregated metrics. Path repr: {repr(aggregated_metrics_path)}")
        with open(aggregated_metrics_path, 'w', encoding='utf-8') as f:
            json.dump(aggregated_metrics, f, indent=2)
        logger.info(f"Successfully updated and saved 'aggregated_metrics.json'")
        logger.info(f"New Overall Accuracy: {overall_accuracy:.4f}")

    except OSError as e:
        logger.error(f"Failed to update aggregated_metrics.json. Path repr: {repr(aggregated_metrics_path)}")
        logger.exception(e)
    except Exception as e:
        logger.error(f"A non-OS error occurred while updating aggregated_metrics.json")
        logger.exception(e)


    logger.info("[bold green]Recalculation complete![/bold green]")


def configure_logging():
    """Sets up rich logging for better console output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(markup=True, rich_tracebacks=True, log_time_format="[%X] ")]
    )


def main():
    """Main entry point for the script."""
    configure_logging()
    parser = argparse.ArgumentParser(
        description="Recalculate scores for an HMMT evaluation run after fixing the parser/grader.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True,
        help="Path to the specific run directory containing the evaluation files.\n"
             "Example: ~/slimsc/prune/results/model-name/hmmt/sc_10_control"
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='hmmt',
        choices=['hmmt'],
        help="The name of the dataset. Must be 'hmmt' for this script."
    )
    
    args = parser.parse_args()
    
    recalculate_run(args.results_dir, args.dataset_name)


if __name__ == "__main__":
    main()