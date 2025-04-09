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
    calculate_sbert_cosine_similarity, # Function name remains the same
    calculate_llm_judge_similarity,
    llm_client
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_similarity_analysis():
    """Loads data, runs all similarity checks, and saves results."""
    logging.info("Starting semantic similarity analysis...")

    # 1. Load Data (No change)
    questions_data = load_yaml_data(config.YAML_FILE_PATH)
    if not questions_data:
        logging.error("Failed to load data. Exiting.")
        return

    # 2. Prepare for Results Storage
    section_results = {}
    all_scores_by_method = {
        config.METHOD_TFIDF: [],
        config.METHOD_LLM: []
    }
    # *** CHANGE: Initialize based on method_key from SBERT_CONFIGS ***
    for sbert_cfg in config.SBERT_CONFIGS:
        all_scores_by_method[sbert_cfg["method_key"]] = []

    # 3. Process Each Question and Section (Loop structure remains similar)
    logging.info("Processing questions and sections...")
    for question in tqdm(questions_data, desc="Processing Questions"):
        q_id = question.get("question_id", "UnknownQuestion")
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

            # --- Method 1: TF-IDF (No change in call) ---
            try:
                tfidf_score = calculate_tfidf_cosine_similarity(thoughts)
                section_scores[config.METHOD_TFIDF] = tfidf_score
                if not np.isnan(tfidf_score):
                    all_scores_by_method[config.METHOD_TFIDF].append(tfidf_score)
                logging.debug(f"[{section_key}] TF-IDF Score: {tfidf_score:.4f}")
            except Exception as e:
                 logging.error(f"[{section_key}] Unhandled error in TF-IDF scoring: {e}")
                 section_scores[config.METHOD_TFIDF] = np.nan


            # --- Method 2: Sentence-BERT *** CHANGE: Iterate SBERT_CONFIGS *** ---
            for model_config in config.SBERT_CONFIGS: # Iterate through the list of dicts
                method_key = model_config["method_key"] # Get the unique key
                logging.debug(f"Calculating SBERT similarity with config: {method_key}")
                try:
                    # *** CHANGE: Pass the entire model_config dictionary ***
                    sbert_score = calculate_sbert_cosine_similarity(thoughts, model_config)
                    section_scores[method_key] = sbert_score # Use method_key for results
                    if not np.isnan(sbert_score):
                        all_scores_by_method[method_key].append(sbert_score)
                    logging.debug(f"[{section_key}] SBERT ({method_key}) Score: {sbert_score:.4f}")
                except Exception as e:
                    # Catch potential loading errors if not caught deeper
                    logging.error(f"[{section_key}] Unhandled error in SBERT scoring ({method_key}): {e}")
                    section_scores[method_key] = np.nan


            # --- Method 3: LLM-as-a-Judge (No change in call) ---
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

    # 4. Calculate Average Scores (Identical logic)
    logging.info("Calculating average scores across all sections...")
    average_scores = {}
    total_sections_processed = len(section_results)
    eligible_sections = {method: [] for method in all_scores_by_method}

    for section_data in section_results.values():
        if section_data.get("error") == "Insufficient thoughts":
            continue # Skip sections that didn't have enough thoughts for any scoring
        for method, score in section_data.get("scores", {}).items():
             eligible_sections[method].append(score) # Add score (could be NaN if calc failed)

    for method, scores in all_scores_by_method.items():
        valid_scores = [s for s in scores if not np.isnan(s)] # Scores where calc succeeded
        num_eligible = len(eligible_sections[method]) # Sections that *should* have been scored

        if valid_scores:
            avg = np.mean(valid_scores)
            average_scores[method] = {
                "average_score": avg,
                "num_sections_scored_successfully": len(valid_scores),
                "num_sections_eligible": num_eligible
            }
            logging.info(f"Average score for {method}: {avg:.4f} (from {len(valid_scores)}/{num_eligible} eligible sections)")
        else:
            average_scores[method] = {
                "average_score": np.nan,
                "num_sections_scored_successfully": 0,
                "num_sections_eligible": num_eligible
            }
            logging.warning(f"No valid scores successfully calculated for method: {method} (out of {num_eligible} eligible sections)")


    # 5. Save Results (Identical logic, but improved NaN/Numpy handling)
    def default_serializer(obj):
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj) if not np.isnan(obj) else None # Convert NaN to None for JSON
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
             return int(obj)
        elif np.isnan(obj): # Catch standalone NaN
            return None
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    logging.info(f"Saving section scores to {config.SECTION_SCORES_FILE}")
    try:
        with open(config.SECTION_SCORES_FILE, 'w', encoding='utf-8') as f:
            json.dump(section_results, f, indent=4, default=default_serializer, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save section scores: {e}")

    logging.info(f"Saving average scores to {config.AVERAGE_SCORES_FILE}")
    try:
        with open(config.AVERAGE_SCORES_FILE, 'w', encoding='utf-8') as f:
            json.dump(average_scores, f, indent=4, default=default_serializer, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save average scores: {e}")

    logging.info("Analysis complete.")


if __name__ == "__main__":
    run_similarity_analysis()