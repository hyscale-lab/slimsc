import json
import logging
import numpy as np
from tqdm import tqdm
import time # Potentially needed if adding delays

import config
from data_loader import load_yaml_data, extract_thoughts_from_section
# Import from the scorers package
from scorers import (
    calculate_tfidf_cosine_similarity,
    calculate_sbert_cosine_similarity,
    calculate_llm_judge_similarity,
    calculate_gemini_cosine_similarity,
    llm_client,
    gemini_client
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_similarity_analysis(input_filepath, output_prefix=None):
    """Loads data, runs all similarity checks, and saves results."""
    logging.info(f"Starting semantic similarity analysis for: {input_filepath}")
    if output_prefix:
        logging.info(f"Using output prefix: {output_prefix}")

    # --- Determine Output File Paths ---
    if output_prefix:
        # Add underscore if prefix doesn't end with one and isn't empty
        prefix = output_prefix if output_prefix.endswith(('_', '-')) or not output_prefix else output_prefix + "_"
        section_scores_file = os.path.join(config.RESULTS_DIR, f"{prefix}section_scores.json")
        average_scores_file = os.path.join(config.RESULTS_DIR, f"{prefix}average_scores.json")
    else:
        # Use default paths from config if no prefix
        section_scores_file = config.SECTION_SCORES_FILE
        average_scores_file = config.AVERAGE_SCORES_FILE
    logging.info(f"Section scores will be saved to: {section_scores_file}")
    logging.info(f"Average scores will be saved to: {average_scores_file}")


    # 1. Load Data
    questions_data = load_yaml_data(input_filepath) # Use argument
    if not questions_data:
        logging.error(f"Failed to load data from {input_filepath}. Exiting.")
        return

    # 2. Prepare for Results Storage (No change in structure)
    section_results = {}
    all_scores_by_method = {
        config.METHOD_TFIDF: [],
        config.METHOD_LLM: [],
        config.METHOD_GEMINI: [],
    }
    for sbert_cfg in config.SBERT_CONFIGS:
        all_scores_by_method[sbert_cfg["method_key"]] = []

    # 3. Process Each Question and Section (Loop structure remains similar)
    logging.info("Processing questions and sections...")
    for question in tqdm(questions_data, desc="Processing Questions"):
        q_id = question.get("question", "UnknownQuestion")
        if "sections" not in question or not isinstance(question["sections"], list):
            logging.warning(f"Skipping question '{q_id}': No 'sections' list found.")
            continue

        for section in question["sections"]:
            s_title = section.get("section_title", "UnknownSection")
            section_key = f"{q_id}_{s_title}"
            logging.debug(f"--- Processing Section: {section_key} ---")

            thoughts = extract_thoughts_from_section(section)

            if len(thoughts) < 2:
                logging.info(f"Skipping similarity calculation for section {section_key}: Needs at least 2 thoughts ({len(thoughts)} found).")
                section_results[section_key] = {"num_thoughts": len(thoughts), "scores": {}, "error": "Insufficient thoughts"}
                continue

            logging.info(f"Processing section {section_key} with {len(thoughts)} thoughts.")
            section_scores = {}

            # --- Method 1: TF-IDF ---
            try:
                tfidf_score = calculate_tfidf_cosine_similarity(thoughts)
                section_scores[config.METHOD_TFIDF] = tfidf_score
                if not np.isnan(tfidf_score):
                    all_scores_by_method[config.METHOD_TFIDF].append(tfidf_score)
                logging.debug(f"[{section_key}] TF-IDF Score: {tfidf_score:.4f}")
            except Exception as e:
                 logging.error(f"[{section_key}] Unhandled error in TF-IDF scoring: {e}")
                 section_scores[config.METHOD_TFIDF] = np.nan


            # --- Method 2: Sentence-BERT ---
            for model_config in config.SBERT_CONFIGS: # Iterate through the list of dicts
                method_key = model_config["method_key"] # Get the unique key
                logging.debug(f"Calculating SBERT similarity with config: {method_key}")
                try:
                    sbert_score = calculate_sbert_cosine_similarity(thoughts, model_config)
                    section_scores[method_key] = sbert_score # Use method_key for results
                    if not np.isnan(sbert_score):
                        all_scores_by_method[method_key].append(sbert_score)
                    logging.debug(f"[{section_key}] SBERT ({method_key}) Score: {sbert_score:.4f}")
                except Exception as e:
                    # Catch potential loading errors if not caught deeper
                    logging.error(f"[{section_key}] Unhandled error in SBERT scoring ({method_key}): {e}")
                    section_scores[method_key] = np.nan


            # --- Method 3: LLM-as-a-Judge ---
            if llm_client:
                logging.debug(f"Calculating LLM similarity with {config.OPENAI_MODEL}...")
                try:
                    llm_score = calculate_llm_judge_similarity(thoughts)
                    section_scores[config.METHOD_LLM] = llm_score
                    if not np.isnan(llm_score):
                        all_scores_by_method[config.METHOD_LLM].append(llm_score)
                    logging.debug(f"[{section_key}] LLM Judge ({config.OPENAI_MODEL}) Score: {llm_score:.4f}")
                except Exception as e:
                    logging.error(f"[{section_key}] Unhandled error in LLM scoring: {e}")
                    section_scores[config.METHOD_LLM] = np.nan
            else:
                 section_scores[config.METHOD_LLM] = np.nan
                 logging.debug(f"[{section_key}] Skipping LLM Judge scoring (client not available).")


            section_results[section_key] = {
                "num_thoughts": len(thoughts),
                "scores": section_scores
                }
            # Optional delay might be useful if hitting HF rate limits during download
            # time.sleep(0.5)


            # --- Method 4: Gemini Embeddings ---
            if gemini_client: # Check if client was initialized successfully
                # logging.debug(f"Calculating Gemini similarity with {config.GEMINI_EMBEDDING_MODEL_ID}...") # Moved inside scorer
                try:
                    # Call the new scorer function
                    gemini_score = calculate_gemini_cosine_similarity(thoughts)
                    section_scores[config.METHOD_GEMINI] = gemini_score
                    if not np.isnan(gemini_score):
                        all_scores_by_method[config.METHOD_GEMINI].append(gemini_score)
                    logging.debug(f"[{section_key}] Gemini ({config.GEMINI_EMBEDDING_MODEL_ID}) Score: {gemini_score:.4f}")
                except Exception as e:
                    logging.error(f"[{section_key}] Unhandled error in Gemini scoring: {e}")
                    section_scores[config.METHOD_GEMINI] = np.nan
            else:
                 section_scores[config.METHOD_GEMINI] = np.nan # Mark as not run/calculated
                 logging.debug(f"[{section_key}] Skipping Gemini scoring (client not available).")


            section_results[section_key] = {
                "num_thoughts": len(thoughts),
                "scores": section_scores
                }
            # Consider adding a small delay here if hitting multiple API limits frequently
            # if llm_client or gemini_client:
            #    time.sleep(0.2) # Small delay between sections


    # 4. Calculate Average Scores (Identical logic)
    logging.info("Calculating average scores across all sections...")
    average_scores = {}
    num_total_eligible_sections = 0
    for section_key, section_data in section_results.items():
        if section_data.get("error") != "Insufficient thoughts":
            num_total_eligible_sections += 1
    logging.info(f"Total number of sections eligible for scoring (>= 2 thoughts): {num_total_eligible_sections}")

    # Now calculate averages using the correct eligible count
    for method, successful_scores_list in all_scores_by_method.items():
        # successful_scores_list already contains only non-NaN scores from step 3
        valid_scores = successful_scores_list

        if valid_scores:
            avg = np.mean(valid_scores)
            num_scored_successfully = len(valid_scores)
            average_scores[method] = {
                "average_score": avg,
                "num_sections_scored_successfully": num_scored_successfully,
                "num_sections_eligible": num_total_eligible_sections
            }
            logging.info(f"Average score for {method}: {avg:.4f} (from {num_scored_successfully}/{num_total_eligible_sections} eligible sections)")
            if num_scored_successfully < num_total_eligible_sections:
                 logging.warning(f"  Method {method}: {num_total_eligible_sections - num_scored_successfully} eligible section(s) failed scoring (resulted in NaN).")
        else:
            average_scores[method] = {
                "average_score": np.nan,
                "num_sections_scored_successfully": 0,
                # Use the pre-calculated total eligible count
                "num_sections_eligible": num_total_eligible_sections
            }
            if num_total_eligible_sections > 0:
                logging.warning(f"No valid scores successfully calculated for method: {method} (out of {num_total_eligible_sections} eligible sections)")
            else:
                 logging.info(f"No sections were eligible for scoring for method: {method}.")


    # 5. Save Results (Use determined file paths)
    def default_serializer(obj):
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj) if not np.isnan(obj) else None # Convert NaN to None for JSON
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
             return int(obj)
        elif np.isnan(obj): # Catch standalone NaN
            return None
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    logging.info(f"Saving section scores to {section_scores_file}")
    try:
        with open(section_scores_file, 'w', encoding='utf-8') as f:
            json.dump(section_results, f, indent=4, default=default_serializer, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save section scores to {section_scores_file}: {e}")

    logging.info(f"Saving average scores to {average_scores_file}")
    try:
        with open(average_scores_file, 'w', encoding='utf-8') as f:
            json.dump(average_scores, f, indent=4, default=default_serializer, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save average scores to {average_scores_file}: {e}")

    logging.info(f"Analysis complete for {input_filepath}.")


# --- Add Argument Parsing ---
if __name__ == "__main__":
    import argparse
    import os # Make sure os is imported here if not globally

    parser = argparse.ArgumentParser(description="Run semantic similarity analysis on thoughts from a YAML file.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=config.YAML_FILE_PATH, # Default to the path in config.py
        help=f"Path to the input YAML file (default: {config.YAML_FILE_PATH})"
    )
    parser.add_argument(
        "-o", "--output_prefix",
        type=str,
        default=None,
        help="Optional prefix for output JSON files (e.g., 'different'). If provided, files will be named '<prefix>_section_scores.json' and '<prefix>_average_scores.json'."
    )

    args = parser.parse_args()

    # Ensure input file exists before running
    if not os.path.exists(args.input):
        logging.error(f"Input file not found: {args.input}")
    else:
        if not config.OPENAI_API_KEY:
             logging.warning("OpenAI API key not found. LLM Judge scoring will be skipped.")
        if not config.GEMINI_API_KEY:
             logging.warning("Gemini API key not found. Gemini embedding scoring will be skipped.")
        run_similarity_analysis(args.input, args.output_prefix)