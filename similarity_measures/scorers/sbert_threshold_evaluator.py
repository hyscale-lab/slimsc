# slimsc/similarity_measures/scorers/sbert_threshold_evaluator.py
import os
import logging
import itertools
from typing import List, Tuple, Dict, Any
import sys
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import config
from data_loader import load_yaml_data, extract_thoughts_from_section
from .sbert_scorer import _get_sbert_model_and_tokenizer, _mean_pooling

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__) # Use a specific logger for this module

# --- Configuration ---
SCRIPT_DIR_ABS = os.path.dirname(os.path.abspath(__file__))
SIM_MEASURES_DIR_ABS = os.path.dirname(SCRIPT_DIR_ABS)
PROJECT_ROOT_ABS = os.path.dirname(SIM_MEASURES_DIR_ABS)
MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
TOKENIZER_ID = "sentence-transformers/all-mpnet-base-v2"
METHOD_KEY = "sbert_eval_mpnet_base_v2" # Unique key for this evaluation setup
SIMILAR_FILE = os.path.join(SIM_MEASURES_DIR_ABS, 'similar.yml') # Path relative to slimsc root
DIFFERENT_FILE = os.path.join(SIM_MEASURES_DIR_ABS, 'different.yml') # Path relative to slimsc root
# RESULTS_DIR comes from config, assumed relative to execution dir (project root)
RESULTS_FILE = os.path.join(config.RESULTS_DIR, "sbert_thresholds.json")

THRESHOLDS = [0.65, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.8, 0.85, 0.9]
DEVICE = config.DEVICE

# Create a model config dict matching the structure used in sbert_scorer
SBERT_EVAL_CONFIG = {
    "method_key": METHOD_KEY,
    "model_id": MODEL_ID,
    "tokenizer_id": TOKENIZER_ID,
    "model_kwargs": {},
    "tokenizer_kwargs": {},
}

def generate_pairs_from_data(filepath: str, label: int) -> List[Tuple[str, str, int]]:
    """Loads data and generates pairs of thoughts with the given label."""
    pairs = []
    questions_data = load_yaml_data(filepath)
    if not questions_data:
        log.error(f"Could not load data from {filepath}")
        return []

    log.info(f"Generating pairs from {filepath} with label {label}...")
    for question in questions_data:
        q_id = question.get("question", "UnknownQ")
        if "sections" not in question or not isinstance(question["sections"], list):
            continue
        for section in question["sections"]:
            s_title = section.get("section_title", "UnknownS")
            thoughts = extract_thoughts_from_section(section)
            if len(thoughts) >= 2:
                # Generate all unique pairs within this section
                for thought1, thought2 in itertools.combinations(thoughts, 2):
                    # Ensure thoughts are non-empty strings before adding
                    if thought1 and thought2:
                         pairs.append((thought1, thought2, label))
                    else:
                        log.warning(f"Skipping pair with empty thought in Q{q_id}_{s_title}")

    log.info(f"Generated {len(pairs)} pairs from {filepath}.")
    return pairs

def get_embeddings_for_thoughts(unique_thoughts: List[str], model_config: Dict[str, Any], device: str) -> Dict[str, np.ndarray] | None:
    """Generates embeddings for a list of unique thoughts."""
    log.info(f"Getting embeddings for {len(unique_thoughts)} unique thoughts using {model_config['method_key']}...")
    try:
        tokenizer, model = _get_sbert_model_and_tokenizer(model_config, device)

        embeddings_dict = {}
        batch_size = 32 # Process in batches for efficiency
        for i in tqdm(range(0, len(unique_thoughts), batch_size), desc="Generating Embeddings"):
            batch_texts = unique_thoughts[i:i+batch_size]
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
            with torch.no_grad():
                model_output = model(**encoded_input)

            sentence_embeddings = _mean_pooling(model_output, encoded_input['attention_mask'])
            # Normalize embeddings (helpful for cosine similarity)
            # embeddings_normalized = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            batch_embeddings_np = sentence_embeddings.cpu().numpy()

            for text, embedding in zip(batch_texts, batch_embeddings_np):
                embeddings_dict[text] = embedding
        log.info("Finished generating embeddings.")
        return embeddings_dict

    except Exception as e:
        log.error(f"Error generating SBERT embeddings with {model_config['method_key']}: {e}", exc_info=True)
        return None

def calculate_pairwise_similarity(pair: Tuple[str, str, int], embeddings_dict: Dict[str, np.ndarray]) -> float | None:
    """Calculates cosine similarity for a single pair using precomputed embeddings."""
    thought1, thought2, _ = pair
    emb1 = embeddings_dict.get(thought1)
    emb2 = embeddings_dict.get(thought2)

    if emb1 is None or emb2 is None:
        log.warning(f"Could not find precomputed embedding for one or both thoughts in pair: ('{thought1[:50]}...', '{thought2[:50]}...')")
        return None

    # Reshape for cosine_similarity function (expects 2D arrays)
    emb1 = emb1.reshape(1, -1)
    emb2 = emb2.reshape(1, -1)

    # Calculate cosine similarity
    sim_score = cosine_similarity(emb1, emb2)[0, 0]

    # Clip score to [0, 1] range if needed, though normalized embeddings should be [-1, 1]
    # Cosine similarity naturally ranges from -1 to 1. Clipping to [0, 1] might discard useful info
    # but aligns with some interpretations of similarity.
    sim_score = np.clip(sim_score, 0, 1)
    return float(sim_score)


def evaluate_thresholds(all_pairs_with_scores: List[Tuple[int, float]], thresholds: List[float]):
    """Evaluates classification performance at different thresholds."""
    true_labels = np.array([label for label, score in all_pairs_with_scores])
    scores = np.array([score for label, score in all_pairs_with_scores])

    results = {}
    print("\n--- Evaluation Results ---")
    for threshold in thresholds:
        predicted_labels = (scores >= threshold).astype(int) # 1 if similar, 0 if different

        acc = accuracy_score(true_labels, predicted_labels)
        # Calculate precision and f1 for the "similar" class (label=1)
        # zero_division=0 means precision is 0 if there are no positive predictions
        prec = precision_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, pos_label=1, zero_division=0)
        cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]) # Ensure labels are [Different, Similar]

        results[threshold] = {
            "accuracy": acc,
            "precision_similar": prec,
            "f1_similar": f1,
            "confusion_matrix": cm.tolist() # Convert numpy array to list for printing/saving
        }

        print(f"\nThreshold: {threshold:.2f}")
        print(f"  Accuracy:          {acc:.3f}")
        print(f"  Precision (Similar): {prec:.3f}")
        print(f"  F1 Score:          {f1:.3f}")
        print(f"  Confusion Matrix (Rows: Actual, Cols: Predicted):")
        print(f"      [TN, FP]")
        print(f"      [FN, TP]")
        print(f"    Predict Neg (0) Predict Pos (1)")
        print(f"Actual Neg (0)  {cm[0, 0]:<6}        {cm[0, 1]:<6}")
        print(f"Actual Pos (1)  {cm[1, 0]:<6}        {cm[1, 1]:<6}")

    return results


# --- Main Execution Function ---
def run_sbert_evaluation():
    """Runs the full SBERT classification evaluation process."""
    log.info(f"Starting SBERT classification evaluation using model: {MODEL_ID}")
    log.info(f"Using device: {DEVICE}")
    log.info(f"Similar data: {SIMILAR_FILE}")
    log.info(f"Different data: {DIFFERENT_FILE}")
    log.info(f"Results will be saved to: {RESULTS_FILE}")

    # Ensure results directory exists
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # 1. Load data and generate pairs
    similar_pairs = generate_pairs_from_data(SIMILAR_FILE, label=1)
    different_pairs = generate_pairs_from_data(DIFFERENT_FILE, label=0)

    if not similar_pairs and not different_pairs:
        log.error("No pairs generated from either file. Exiting.")
        return # Return instead of exit

    all_pairs = similar_pairs + different_pairs
    log.info(f"Total pairs to evaluate: {len(all_pairs)}")

    # 2. Get unique thoughts
    unique_thoughts = set(t for pair in all_pairs for t in pair[:2])
    unique_thoughts_list = list(unique_thoughts)
    log.info(f"Found {len(unique_thoughts_list)} unique thoughts.")

    # 3. Calculate embeddings
    embeddings_map = get_embeddings_for_thoughts(unique_thoughts_list, SBERT_EVAL_CONFIG, DEVICE)
    if embeddings_map is None:
        log.error("Failed to generate embeddings. Aborting evaluation.")
        return

    # 4. Calculate similarity scores
    all_pairs_with_scores = []
    log.info("Calculating similarity scores for all pairs...")
    for pair in tqdm(all_pairs, desc="Calculating Pair Similarities"):
        score = calculate_pairwise_similarity(pair, embeddings_map)
        if score is not None:
            all_pairs_with_scores.append((pair[2], score))
        else:
            # Warning already logged in calculate_pairwise_similarity
            pass

    if not all_pairs_with_scores:
        log.error("No similarity scores could be calculated. Aborting evaluation.")
        return

    log.info(f"Successfully calculated scores for {len(all_pairs_with_scores)} pairs.")

    # 5. Evaluate thresholds
    evaluation_results = evaluate_thresholds(all_pairs_with_scores, THRESHOLDS)

    # 6. Save results
    log.info(f"Attempting to save results to: {RESULTS_FILE}")
    try:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        output_data = {
            "model_id": MODEL_ID,
            "thresholds_evaluated": THRESHOLDS,
            "results": evaluation_results
        }
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4)
        log.info(f"Evaluation results saved successfully.")
    except Exception as e:
        log.error(f"Failed to save evaluation results: {e}", exc_info=True)

    log.info("SBERT evaluation finished.")


# Allow running this script directly for testing/debugging if needed
if __name__ == "__main__":
    print("Running sbert_threshold_evaluator.py directly...")
    run_sbert_evaluation()