# slimsc/prune/analysis/analyze_hypothesis_from_sc.py
import os
import pandas as pd
import argparse
from tqdm import tqdm
import json
from typing import List, Dict, Optional, Set, Tuple, Any
import numpy as np
import glob
import re

# Ensure correct relative imports if running as part of the package
try:
    # Attempt relative imports for package structure
    from ..utils import extract_final_thought
    from ..utils.similarity_utils import (
        FaissIndexManager, embed_segments, find_newly_completed_thoughts,
        get_embedding_model, MIN_SEGMENT_TOKENS, TARGET_PHRASES
    )
except (ImportError, ValueError): # ValueError added for cases like "attempted relative import beyond top-level package"
     # Fallback for running script directly
     print("Running in script mode. Attempting fallback imports.")
     import sys
     # Navigate up two levels to the project root (assuming script is in slimsc/prune/analysis)
     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
     if project_root not in sys.path:
         sys.path.insert(0, project_root)
     try:
         from slimsc.prune.utils import extract_final_thought
         from slimsc.prune.utils.similarity_utils import (
            FaissIndexManager, embed_segments, find_newly_completed_thoughts,
            get_embedding_model, MIN_SEGMENT_TOKENS, TARGET_PHRASES
         )
         print("Fallback imports successful.")
     except ImportError as e:
         print(f"Fallback imports failed: {e}")
         print(f"sys.path: {sys.path}")
         print(f"Current working directory: {os.getcwd()}")
         # Optionally raise the error if imports are critical
         # raise

from rich.logging import RichHandler
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(markup=True, rich_tracebacks=True)]
)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING) # Silence PIL logs often triggered by HF


DEFAULT_SEED = 4 # Keep consistent if sampling needed later


def setup_output_directories_analysis(
        base_output_dir: str,
        model_name: str,
        dataset_name: str,
        n_start: int) -> Dict[str, str]:
    """Creates directories for storing hypothesis analysis results."""
    run_name = f"hypothesis_analysis_n{n_start}" # Analysis results for N
    model_dataset_dir = os.path.join(base_output_dir, model_name, dataset_name, run_name)
    results_csv_path = os.path.join(model_dataset_dir, "hypothesis_analysis_summary.csv")
    aggregated_metrics_path = os.path.join(model_dataset_dir, "aggregated_metrics.json")
    per_question_dir = os.path.join(model_dataset_dir, "per_question_details") # Store detailed JSONs here

    os.makedirs(model_dataset_dir, exist_ok=True)
    os.makedirs(per_question_dir, exist_ok=True)

    return {
        "base": model_dataset_dir,
        "csv": results_csv_path,
        "aggregated_metrics_json": aggregated_metrics_path,
        "per_question_details_dir": per_question_dir,
    }

def load_sc_chains_for_question(
    sc_chains_dir: str,
    iteration: int,
    n_chains: int
) -> Optional[Dict[str, Tuple[str, str]]]:
    """
    Loads chain data for a given question iteration from SC run files.

    Returns:
        Optional[Dict[str, Tuple[str, str]]]:
            Map from chain_id to a tuple of (full_text, reasoning_text),
            or None on critical failure.
            Returns data only for chains 1 to n_chains found.
    """
    chain_data: Dict[str, Tuple[str, str]] = {}
    essential_chains_found = 0

    for chain_idx in range(1, n_chains + 1):
        chain_id = f"q{iteration}_c{chain_idx}"
        # Find the specific file for this chain index, ignoring status suffix
        pattern = os.path.join(sc_chains_dir, f"question_{iteration}_chain_{chain_idx}_*.txt")
        found_files = glob.glob(pattern)

        if not found_files:
            logger.warning(f"Chain file not found for {chain_id} (pattern: {pattern}).")
            continue # Skip this chain index, check others

        if len(found_files) > 1:
            logger.warning(f"Multiple files found for {chain_id} (pattern: {pattern}). Using first one: {found_files[0]}")

        fpath = found_files[0]
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                file_content = f.read()

            # Find markers
            reasoning_marker = "--- Reasoning Content ---"
            final_answer_marker = "--- Final Answer Content ---"

            reasoning_start_pos = file_content.find(reasoning_marker)
            final_answer_start_pos = file_content.find(final_answer_marker)

            reasoning_text = ""
            final_answer_text = ""
            full_text = ""

            if reasoning_start_pos != -1:
                # Adjust start position past the marker line itself
                reasoning_start_content = reasoning_start_pos + len(reasoning_marker)
                # Find the first newline after the marker to start content
                newline_after_reasoning_marker = file_content.find('\n', reasoning_start_content)
                if newline_after_reasoning_marker != -1:
                    reasoning_start_content = newline_after_reasoning_marker + 1

                if final_answer_start_pos != -1:
                    # Reasoning is between the two markers
                    reasoning_text = file_content[reasoning_start_content:final_answer_start_pos].strip()
                else:
                    # No final answer marker, reasoning is rest of file
                    logger.debug(f"Final answer marker not found in {fpath}. Assuming reasoning is till end.")
                    reasoning_text = file_content[reasoning_start_content:].strip()
            else:
                logger.warning(f"Reasoning marker not found in {fpath}. Cannot extract reasoning text.")
                # Cannot proceed reliably without reasoning text for length check

            if final_answer_start_pos != -1:
                # Adjust start position past the marker line itself
                final_answer_start_content = final_answer_start_pos + len(final_answer_marker)
                # Find the first newline after the marker
                newline_after_final_marker = file_content.find('\n', final_answer_start_content)
                if newline_after_final_marker != -1:
                    final_answer_start_content = newline_after_final_marker + 1
                final_answer_text = file_content[final_answer_start_content:].strip()

            # Combine for full text used in thought finding
            # Include reasoning even if final answer is missing, as thoughts might be there.
            # Ensure separation if both exist
            if reasoning_text and final_answer_text:
                full_text = f"{reasoning_text}\n{final_answer_text}"
            elif reasoning_text:
                full_text = reasoning_text
            elif final_answer_text:
                full_text = final_answer_text # Unlikely case, but possible
            else:
                 logger.warning(f"Could not extract any content for {chain_id} from {fpath}.")
                 # We might still have a final answer from summary, so don't skip chain entirely yet
                 # But we need full_text for thought finding. If full_text is empty, we can't analyze it.
                 if not full_text: continue # Skip if we couldn't derive full_text

            chain_data[chain_id] = (full_text, reasoning_text)
            essential_chains_found += 1

        except Exception as e:
            logger.error(f"Error reading or parsing chain file {fpath}: {e}")
            # If reading one file fails, maybe skip it or abort? Abort is safer.
            logger.error(f"Aborting loading for Q{iteration} due to error in file {fpath}.")
            return None

    if essential_chains_found == 0:
        logger.error(f"No chain files could be successfully read for Q{iteration}. Cannot proceed.")
        return None

    if essential_chains_found < n_chains:
        logger.warning(f"Found only {essential_chains_found}/{n_chains} chain files for Q{iteration}. Proceeding with available chains.")
        # The calling function will use len(chain_data) as the actual number analyzed

    return chain_data


def load_sc_summary_data(sc_summary_dir: str, iteration: int) -> Optional[Dict]:
    """Loads the summary JSON for a given question iteration from the SC run."""
    summary_filepath = os.path.join(sc_summary_dir, f"question_{iteration}_summary.json")
    if not os.path.exists(summary_filepath):
        logger.warning(f"SC Summary file not found: {summary_filepath}")
        return None
    try:
        with open(summary_filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading SC summary file {summary_filepath}: {e}")
        return None


def is_thought_eligible_for_detection(
    thought_idx: int,
    thought_end_char: int,
    chain_reasoning_len: int
) -> bool:
    """
    Checks if a thought meets the criteria to be considered for
    similarity detection recording.
    - Thought index must be 2 or greater (meaning at least the second detected segment, 0-indexed internally).
    - Thought must end within the chain's reasoning phase.
    """
    # Internally, thought finding returns 0-indexed segments.
    # So, "thought index 2 or greater" means the 3rd segment onwards.
    # Thus, we check if internal index `thought_idx >= 2`.
    if thought_idx < 2:
        return False
    if thought_end_char < 0: # Handle case where end char is unknown
        logger.warning(f"Thought index {thought_idx} has invalid end char {thought_end_char}. Marking ineligible.")
        return False
    # Strict check: thought must END within reasoning.
    if thought_end_char > chain_reasoning_len:
        return False
    return True


def analyze_question_hypothesis_offline(
    iteration: int,
    question_id: str,
    n_chains: int,
    chain_data: Dict[str, Tuple[str, str]], # Map chain_id -> (full_text, reasoning_text)
    final_answers: Dict[str, Optional[str]],
    tokenizer_path: str,
    similarity_thresholds: List[float],
) -> Optional[Dict]:
    """
    Performs the hypothesis analysis for a single question using pre-generated text.
    Hypothesis: Once a pair of similar thoughts (> threshold) are detected between
    two chains (A and B), their final answers will be the same.
    Constraint: After detecting the *first* eligible similar thought pair between chains
    A and B (above a given threshold), do not consider *any further* thought pairs
    between A and B for similarity detection *at that threshold*.
    """
    logger.info(f"Analyzing Q{iteration} for thresholds: {similarity_thresholds}...")
    if n_chains < 2:
         logger.warning(f"Q{iteration}: Need at least 2 chains for analysis, found {n_chains}. Skipping.")
         return None

    # Extract reasoning lengths and full texts from the loaded data
    chain_reasoning_lengths: Dict[str, int] = {
        chain_id: len(texts[1]) for chain_id, texts in chain_data.items()
    }
    chain_full_texts: Dict[str, str] = {
        chain_id: texts[0] for chain_id, texts in chain_data.items()
    }

    try:
        embedding_model = get_embedding_model()
        index_manager = FaissIndexManager(dimension=embedding_model.get_sentence_embedding_dimension())

        # --- Data Structures ---
        # Store detailed info for detected pairs per threshold:
        detected_similar_pairs_by_thresh: Dict[float, List[Dict]] = {t: [] for t in similarity_thresholds}
        # Store which pairs of *chains* have already been marked for a given threshold:
        processed_chain_pairs_by_thresh: Dict[float, Set[frozenset[str]]] = {t: set() for t in similarity_thresholds}

        # --- Thought Finding and Embedding ---
        # Store: (chain_id, thought_idx, text, start_char, end_char)
        all_thoughts_detailed: List[Tuple[str, int, str, int, int]] = []
        processed_boundaries_per_chain: Dict[str, List[int]] = {}

        for chain_id, full_text in chain_full_texts.items():
            completed_thought_count = 0
            # Initialize processed boundaries for this chain
            current_processed_boundaries: List[int] = []
            
            # Simulate iterative discovery for robustness, though full_text is available
            segments, updated_boundaries = find_newly_completed_thoughts(
                full_text, [], tokenizer_path, TARGET_PHRASES, MIN_SEGMENT_TOKENS
            )
            for start_char, end_char, text in segments:
                 all_thoughts_detailed.append((chain_id, completed_thought_count, text, start_char, end_char))
                 completed_thought_count += 1
            current_processed_boundaries = updated_boundaries # Keep track of boundary starts

            # Extract final thought segment if chain finished (offline equivalent)
            final_thought = extract_final_thought(
                full_text, current_processed_boundaries, tokenizer_path, MIN_SEGMENT_TOKENS
            )
            if final_thought:
                 start_char, end_char, text = final_thought
                 # Check if this final segment's start wasn't already the start of the last regular segment
                 is_new_segment = True
                 if segments and segments[-1][0] == start_char:
                      is_new_segment = False # It was already found as the last 'newly completed'

                 if is_new_segment:
                      all_thoughts_detailed.append((chain_id, completed_thought_count, text, start_char, end_char))
                      completed_thought_count += 1
                 # else: Already added

            processed_boundaries_per_chain[chain_id] = current_processed_boundaries # Store for reference if needed

        if not all_thoughts_detailed:
             logger.warning(f"Q{iteration}: No thoughts found across all chains. Skipping analysis.")
             results_by_threshold = { t: {"detected_pairs": [], "num_detected": 0, "num_same_answer": 0, "same_answer_perc": None} for t in similarity_thresholds }
             return {"iteration": iteration, "question_id": question_id, "results_by_threshold": results_by_threshold}

        texts_only = [t[2] for t in all_thoughts_detailed]
        embeddings = embed_segments(texts_only)
        if embeddings is None or len(embeddings) != len(all_thoughts_detailed):
            logger.error(f"Q{iteration}: Embedding failed. Skipping analysis.")
            return None

        # Combine details with embeddings: (chain_id, t_idx, text, s_char, e_char, embedding)
        all_thoughts_with_embeddings = [
            (chain_id, t_idx, text, s_char, e_char, embedding)
            for (chain_id, t_idx, text, s_char, e_char), embedding in zip(all_thoughts_detailed, embeddings)
        ]
        # Create a map for easy lookup of thought details (e.g., end char for eligibility)
        thought_details_map: Dict[Tuple[str, int], Dict] = {
            (cid, tidx): {"text": txt, "start": sc, "end": ec, "embedding": emb}
            for cid, tidx, txt, sc, ec, emb in all_thoughts_with_embeddings
        }

        # Add all embeddings to FAISS
        for cid, tidx, txt, sc, ec, emb in all_thoughts_with_embeddings:
             index_manager.add_embedding(emb, cid, tidx, txt)

        # --- Similarity Checking Loop ---
        # Sort thoughts primarily by index, then chain_id for consistent processing order
        all_thoughts_with_embeddings.sort(key=lambda x: (x[1], x[0]))

        for chain_id_A, thought_idx_A, text_A, start_char_A, end_char_A, embedding_A in all_thoughts_with_embeddings:

            # 1. Check Eligibility of Current Thought (Thought A)
            reasoning_len_A = chain_reasoning_lengths.get(chain_id_A, 0)
            thought_A_eligible = is_thought_eligible_for_detection(
                thought_idx=thought_idx_A,
                thought_end_char=end_char_A,
                chain_reasoning_len=reasoning_len_A
            )

            if not thought_A_eligible:
                continue # Skip similarity search if current thought (A) is ineligible

            # 2. Search for Nearest Neighbor (Potential Thought B) from OTHER chains
            neighbor_result = index_manager.search_nearest_neighbor(embedding_A, chain_id_A)

            if neighbor_result:
                sim_score, chain_id_B, thought_idx_B, text_B = neighbor_result

                # 3. Check Conditions for Recording Pair (Iterate through thresholds)
                for thresh in similarity_thresholds:
                    processed_pairs_for_thresh = processed_chain_pairs_by_thresh[thresh]
                    # Use frozenset for chain pair to handle order invariance (A,B) vs (B,A)
                    current_chain_pair = frozenset({chain_id_A, chain_id_B})

                    # Condition 0: Check if score meets threshold
                    if sim_score > thresh:

                         # Condition 1: Have we already processed this CHAIN PAIR for this THRESHOLD?
                         if current_chain_pair in processed_pairs_for_thresh:
                             # logger.debug(f"[Thresh={thresh:.2f}] Skipping pair ({chain_id_A} T{thought_idx_A}, {chain_id_B} T{thought_idx_B}) - Chain pair {current_chain_pair} already processed.")
                             continue # Skip this pair for this threshold

                         # If score is high enough AND chain pair is new for this threshold,
                         # check the eligibility of this specific thought pair (A and B).

                         # Condition 2: Neighbor thought index <= Current thought index
                         # This ensures we record a pair based on the "earlier" thought's perspective
                         # if both thoughts trigger a similarity search later.
                         if not (thought_idx_B <= thought_idx_A):
                              record_this_pair = False
                         else:
                              record_this_pair = True

                         # Condition 3: Neighbor Thought (B) Eligibility (check only if needed)
                         if record_this_pair:
                              neighbor_details = thought_details_map.get((chain_id_B, thought_idx_B))
                              if neighbor_details:
                                  end_char_B = neighbor_details.get("end", -1)
                                  reasoning_len_B = chain_reasoning_lengths.get(chain_id_B, 0)
                                  thought_B_eligible = is_thought_eligible_for_detection(
                                      thought_idx=thought_idx_B,
                                      thought_end_char=end_char_B,
                                      chain_reasoning_len=reasoning_len_B
                                  )
                                  if not thought_B_eligible:
                                      record_this_pair = False
                              else:
                                  # Should not happen if thought_details_map is populated correctly
                                  logger.warning(f"Could not find details for neighbor thought ({chain_id_B}, T{thought_idx_B}). Marking pair ineligible.")
                                  record_this_pair = False

                         # --- Recording Logic ---
                         # If this specific THOUGHT PAIR is eligible AND the CHAIN PAIR hasn't been processed yet for this threshold:
                         if record_this_pair:
                             # Mark this CHAIN PAIR as processed for this threshold
                             # to prevent future thoughts between A and B being recorded at this threshold.
                             processed_pairs_for_thresh.add(current_chain_pair)

                             # Record the details of this detected similar thought pair
                             detected_similar_pairs_by_thresh[thresh].append({
                                 "chain1_id": chain_id_A,
                                 "chain2_id": chain_id_B,
                                 "thought1_idx": thought_idx_A,
                                 "thought2_idx": thought_idx_B,
                                 "text1": text_A,
                                 "text2": text_B,
                                 "score": float(sim_score) # Ensure score is standard float
                             })
                             # logger.info(f"[Thresh={thresh:.2f}] Q{iteration} Similarity RECORDED: {chain_id_A}(T{thought_idx_A}) vs {chain_id_B}(T{thought_idx_B}), Score={sim_score:.4f}")

                    # Else (sim_score <= thresh): Continue to next threshold or next thought


        # --- Calculate Hypothesis Results per Threshold ---
        results_by_threshold = {}
        for thresh in similarity_thresholds:
            detected_pairs = detected_similar_pairs_by_thresh[thresh]
            num_detected = len(detected_pairs) # Number of thought pairs recorded
            num_same_answer = 0

            if num_detected > 0:
                for pair_info in detected_pairs:
                    # Get final answers for the chains involved in this specific recorded pair
                    ans1 = final_answers.get(pair_info["chain1_id"])
                    ans2 = final_answers.get(pair_info["chain2_id"])

                    # Check if both answers are non-None and equal
                    if ans1 is not None and ans2 is not None and ans1 == ans2:
                        num_same_answer += 1

                # Calculate percentage based on the number of detected thought pairs
                perc = (num_same_answer / num_detected) * 100 if num_detected > 0 else None
            else:
                perc = None # Avoid division by zero if no pairs were detected at this threshold

            results_by_threshold[thresh] = {
                "detected_pairs": detected_pairs, # Keep the list of pairs with texts and score
                "num_detected": num_detected,
                "num_same_answer": num_same_answer,
                "same_answer_perc": perc
            }
            logger.info(f"[Thresh={thresh:.2f}] Q{iteration} Hypothesis Result: {num_same_answer}/{num_detected} ({perc if perc is not None else 'N/A'}%) detected similar pairs had same final answer.")


        return {"iteration": iteration, "question_id": question_id, "results_by_threshold": results_by_threshold}

    except Exception as e:
        logger.exception(f"Unexpected error analyzing Q{iteration}: {e}")
        return None
    finally:
        # Clean up FAISS index if needed (optional, depends on memory usage)
        if 'index_manager' in locals() and hasattr(index_manager, 'index'):
             del index_manager.index # Release FAISS index object
             del index_manager


# --- Main Runner ---
def run_analysis(
    n_start: int,
    similarity_thresholds: List[float],
    sc_results_base_dir: str,
    model_name: str,
    dataset_name: str,
    analysis_output_base_dir: str,
    tokenizer_path: str,
    start_iteration: int = 1,
    end_iteration: Optional[int] = None,
    specific_iterations: Optional[List[int]] = None
):
    """Runs the offline hypothesis analysis for a single n_start value."""
    logger.info(f"Starting Offline Hypothesis Analysis for N={n_start}, Thresh={similarity_thresholds}")
    logger.info(f"Reading SC results from: {sc_results_base_dir}")
    logger.info(f"Saving analysis results to: {analysis_output_base_dir}")

    sc_run_name = f"sc_{n_start}_control"
    sc_run_dir = os.path.join(sc_results_base_dir, model_name, dataset_name, sc_run_name)
    sc_chains_dir = os.path.join(sc_run_dir, "individual_chains")
    sc_summaries_dir = os.path.join(sc_run_dir, "summaries")

    if not os.path.isdir(sc_run_dir): logger.error(f"SC results directory not found: {sc_run_dir}"); return
    if not os.path.isdir(sc_chains_dir): logger.error(f"SC chains directory not found: {sc_chains_dir}"); return
    if not os.path.isdir(sc_summaries_dir): logger.error(f"SC summaries directory not found: {sc_summaries_dir}"); return

    paths = setup_output_directories_analysis(
        analysis_output_base_dir, model_name, dataset_name, n_start
    )
    logger.info(f"Analysis outputs will be saved in: {paths['base']}")

    # --- Determine Iterations to Process ---
    processed_sc_iterations = set()
    try:
        for fname in os.listdir(sc_summaries_dir):
            match = re.match(r"question_(\d+)_summary\.json", fname)
            if match: processed_sc_iterations.add(int(match.group(1)))
    except Exception as e: logger.error(f"Error listing SC summary files: {e}"); return
    if not processed_sc_iterations: logger.error(f"No processed question summaries found in {sc_summaries_dir}."); return
    logger.info(f"Found {len(processed_sc_iterations)} processed questions in the SC run.")

    target_iterations_set: Set[int] = set()
    if specific_iterations is not None:
        valid_specific_iterations = sorted([i for i in specific_iterations if i in processed_sc_iterations])
        if len(valid_specific_iterations) != len(specific_iterations):
            skipped = set(specific_iterations) - set(valid_specific_iterations)
            logger.warning(f"[yellow]Specified iterations {skipped} not found in SC run {sc_run_name}. Skipping.[/yellow]")
        target_iterations_set = set(valid_specific_iterations)
        logger.info(f"Targeting {len(target_iterations_set)} specific iteration(s) for N={n_start}.")
    else:
        start = max(1, start_iteration)
        end = end_iteration if end_iteration is not None else max(processed_sc_iterations) if processed_sc_iterations else 0
        target_iterations_set = {i for i in range(start, end + 1) if i in processed_sc_iterations}
        logger.info(f"Targeting iterations from {start} to {end} (within available SC iterations) for N={n_start}.")

    iterations_to_analyze = sorted(list(target_iterations_set))
    if not iterations_to_analyze: logger.info(f"No iterations selected for analysis for N={n_start}."); return
    logger.info(f"Analyzing {len(iterations_to_analyze)} iterations for N={n_start}.")

    # --- Main Analysis Loop ---
    all_results_list = []
    pbar = tqdm(iterations_to_analyze, desc=f"Analyzing N={n_start}")
    for i in pbar:
        pbar.set_description(f"Analyzing N={n_start} Q={i}")

        # 1. Load chain data (full_text, reasoning_text)
        chain_data = load_sc_chains_for_question(sc_chains_dir, i, n_start)
        if not chain_data:
            logger.error(f"Failed to load sufficient chain data for Q{i}. Skipping.");
            continue
        current_n_chains = len(chain_data) # Actual number of chains loaded and analyzed

        # 2. Load summary to get final answers
        sc_summary = load_sc_summary_data(sc_summaries_dir, i)
        if not sc_summary: logger.error(f"Failed to load SC summary for Q{i}. Skipping."); continue

        question_id = sc_summary.get("question_id", f"index_{i-1}") # Fallback qid
        final_answers: Dict[str, Optional[str]] = {}

        # Extract final answers from summary
        # Combine details from successful and error chains as both might have answers
        details_list = sc_summary.get("chains_for_voting_details", []) + \
                       sc_summary.get("error_chain_details", [])
        details_map = {f"q{i}_c{cd['chain_index']}": cd for cd in details_list if 'chain_index' in cd}

        # Populate final_answers only for chains successfully loaded and present in the summary
        for chain_id in chain_data.keys():
             detail = details_map.get(chain_id)
             if detail:
                 # Use extracted_answer, potentially fall back to raw if needed? For now, just extracted.
                 final_answers[chain_id] = detail.get("extracted_answer")
             else:
                 logger.warning(f"No summary details found for successfully loaded chain {chain_id} in Q{i}. Final answer will be None.")
                 final_answers[chain_id] = None # Mark as None if details missing

        # 3. Run offline analysis
        analysis_result = analyze_question_hypothesis_offline(
            iteration=i,
            question_id=question_id,
            n_chains=current_n_chains, # Pass actual number analyzed
            chain_data=chain_data,
            final_answers=final_answers,
            tokenizer_path=tokenizer_path,
            similarity_thresholds=similarity_thresholds,
        )

        # 4. Store results
        if analysis_result:
            q_detail_path = os.path.join(paths["per_question_details_dir"], f"q_{i}_analysis.json")
            try:
                 # Define a serializer that handles numpy types
                 def default_serializer(obj):
                     if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                     if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                     if isinstance(obj, np.ndarray): return obj.tolist() # Convert arrays to lists
                     if isinstance(obj, frozenset): return list(obj) # Convert frozenset to list for JSON
                     raise TypeError(f"Type {type(obj)} not serializable")

                 # Save the full analysis result, including texts/scores in detected_pairs
                 with open(q_detail_path, "w", encoding='utf-8') as f:
                     json.dump(analysis_result, f, indent=2, default=default_serializer)
            except Exception as e:
                logger.exception(f"Error saving per-question detail JSON for Q{i}: {e}")
                # Log the problematic structure if possible
                # logger.error(f"Problematic data structure: {analysis_result}")


            # Prepare row for the summary CSV
            row = {"iteration": i, "question_id": question_id}
            for thresh, data in analysis_result.get("results_by_threshold", {}).items():
                 # Format threshold string consistently (e.g., 0.70)
                 thresh_str = f"{thresh:.2f}"
                 row[f"num_detected_t{thresh_str}"] = data["num_detected"]
                 row[f"num_same_answer_t{thresh_str}"] = data["num_same_answer"]
                 row[f"same_answer_perc_t{thresh_str}"] = data["same_answer_perc"]
            all_results_list.append(row)
        else:
            logger.error(f"Analysis failed for Q{i}.")


    # --- Final Aggregation and Saving ---
    if not all_results_list:
        logger.warning(f"No questions were successfully analyzed for N={n_start}.")
        return

    final_df = pd.DataFrame(all_results_list)
    csv_cols = ["iteration", "question_id"]
    agg_metrics_thresholds = {}
    formatted_thresholds = [f"{t:.2f}" for t in similarity_thresholds] # Consistent formatting

    for thresh_str in formatted_thresholds:
        num_det_col = f"num_detected_t{thresh_str}"
        num_same_col = f"num_same_answer_t{thresh_str}"
        perc_col = f"same_answer_perc_t{thresh_str}"
        csv_cols.extend([num_det_col, num_same_col, perc_col])

        # Ensure columns exist and fill NaNs for calculations
        if num_det_col not in final_df.columns: final_df[num_det_col] = 0
        else: final_df[num_det_col] = final_df[num_det_col].fillna(0).astype(int)

        if num_same_col not in final_df.columns: final_df[num_same_col] = 0
        else: final_df[num_same_col] = final_df[num_same_col].fillna(0).astype(int)

        if perc_col not in final_df.columns: final_df[perc_col] = pd.NA
        else: final_df[perc_col] = pd.to_numeric(final_df[perc_col], errors='coerce') # Ensure numeric, coerce errors to NaT/NaN

        # Aggregate calculations
        total_det = final_df[num_det_col].sum()
        total_same = final_df[num_same_col].sum()
        # Overall percentage: total same / total detected across all questions
        overall_perc = (total_same / total_det * 100) if total_det > 0 else None
        # Mean percentage: average of the per-question percentages (where calculation was possible)
        mean_perc = final_df[perc_col].mean(skipna=True) # Use skipna=True explicitly
        if pd.isna(mean_perc): mean_perc = None # Convert pandas NaN to None if needed


        agg_metrics_thresholds[thresh_str] = {
             "total_similar_pairs_detected": int(total_det),
             "total_similar_pairs_same_answer": int(total_same),
             "overall_same_answer_percentage": float(overall_perc) if overall_perc is not None else None,
             "mean_same_answer_percentage_per_question": float(mean_perc) if mean_perc is not None else None,
        }

    try:
        # Ensure columns in the DataFrame match the desired order for CSV
        final_df_sorted = final_df[csv_cols].sort_values(by="iteration")
        final_df_sorted.to_csv(paths["csv"], index=False, float_format='%.2f') # Format floats in CSV
        logger.info(f"Analysis summary saved to {paths['csv']}")
    except Exception as e:
        logger.exception(f"Error saving final analysis CSV: {e}")
        logger.error(f"Columns expected: {csv_cols}")
        logger.error(f"Columns available: {final_df.columns.tolist()}")


    aggregated_data = {
        "analysis_config": {
            "n_start": n_start,
            "similarity_thresholds_tested": similarity_thresholds, # Original list
            "source_sc_run_directory": sc_run_dir,
            "tokenizer_path_provided": tokenizer_path is not None,
            "detection_conditions": "thought_idx>=2 AND end_char<=reasoning_len (for both); neighbor_idx<=current_idx; first_eligible_pair_per_chain_pair_per_threshold",
        },
        "num_questions_analyzed": len(final_df),
        "results_by_threshold": agg_metrics_thresholds # Uses formatted threshold strings as keys
    }
    try:
        # Use the same robust serializer for the aggregated JSON
        def default_serializer(obj):
             if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
             if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
             if isinstance(obj, np.ndarray): return obj.tolist()
             if isinstance(obj, frozenset): return list(obj)
             raise TypeError(f"Type {type(obj)} not serializable")

        with open(paths["aggregated_metrics_json"], "w", encoding='utf-8') as f:
            json.dump(aggregated_data, f, indent=2, default=default_serializer)
        logger.info(f"Aggregated analysis metrics saved to {paths['aggregated_metrics_json']}")
    except Exception as e: logger.exception(f"Error saving aggregated analysis JSON: {e}")

    logger.info(f"Offline Hypothesis Analysis Finished for N={n_start}.")


# --- Main Entry Point ---
def main():
    # Logging configured globally at the start
    home = os.path.expanduser("~")
    parser = argparse.ArgumentParser(description='Run Offline Similarity Hypothesis Analysis on completed SC runs.')

    # Inputs
    parser.add_argument('--n_start_values', type=str, required=True,
                        help='Comma-separated list of N values for the SC runs to analyze (e.g., "2,4,8,16,32").')
    parser.add_argument('--thresholds', type=str, default="0.65,0.7,0.75,0.8,0.85,0.9,0.95",
                        help='Comma-separated list of similarity thresholds to test (e.g., "0.7,0.8,0.9"). Default="0.65,0.7,0.75,0.8,0.85,0.9,0.95"')
    parser.add_argument('--sc_results_dir', type=str, required=True,
                        help='Base directory containing the SC results (e.g., ~/slimsc/prune/results). The script expects .../model/dataset/sc_N_control inside this.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Short name of the model (used to find SC directory).')
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='Name of the dataset (used to find SC directory).')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help='Path to HuggingFace tokenizer directory (REQUIRED for finding thoughts).')
    parser.add_argument('--analysis_output_dir', type=str, default=os.path.join(home, "slimsc/prune/analysis"), # Changed default output location
                        help='Base directory to SAVE the analysis results.')

    # Iteration selection (optional, defaults to all available in SC run)
    parser.add_argument('--start', type=int, default=1, help='Starting iteration (1-indexed) to analyze.')
    parser.add_argument('--end', type=int, default=None, help='Ending iteration (inclusive) to analyze.')
    parser.add_argument('--iterations', type=str, default=None, help='Comma-separated list/ranges of specific iterations to analyze (e.g., "1,5,10-12"). Overrides --start/--end.')

    args = parser.parse_args()

    # Parse n_start values
    try:
        n_start_values = [int(n.strip()) for n in args.n_start_values.split(',') if n.strip()]
        if not n_start_values: raise ValueError("No valid N values provided.")
        if not all(n >= 2 for n in n_start_values): raise ValueError("All N values must be >= 2.")
        n_start_values = sorted(list(set(n_start_values))) # Sort unique
    except ValueError as e:
        logger.error(f"[red]Invalid --n_start_values argument: {e}[/red]")
        return

    # Parse thresholds
    try:
        similarity_thresholds = [float(t.strip()) for t in args.thresholds.split(',') if t.strip()]
        if not similarity_thresholds: raise ValueError("No valid thresholds provided.")
        if not all(0.0 < t <= 1.0 for t in similarity_thresholds):
             raise ValueError("Thresholds must be between 0.0 (exclusive) and 1.0 (inclusive).")
        similarity_thresholds = sorted(list(set(similarity_thresholds))) # Sort unique
    except ValueError as e:
        logger.error(f"[red]Invalid --thresholds argument: {e}[/red]")
        return

    if not args.tokenizer_path or not os.path.isdir(args.tokenizer_path):
         logger.error(f"[red]Invalid --tokenizer_path: '{args.tokenizer_path}'. Directory not found or not specified. Required.[/red]")
         return

    # --- Iteration List Parsing Logic ---
    specific_iterations_list: Optional[List[int]] = None
    if args.iterations:
        specific_iterations_list = []
        try:
            for part in args.iterations.split(','):
                part = part.strip();
                if not part: continue
                if '-' in part:
                    start_range, end_range = map(int, part.split('-'))
                    if start_range <= 0 or end_range <= 0: raise ValueError("Iteration numbers must be positive.")
                    if start_range <= end_range: specific_iterations_list.extend(range(start_range, end_range + 1))
                    else: logger.warning(f"Invalid range in --iterations: {start_range}-{end_range}. Skipping.")
                else:
                    iter_num = int(part)
                    if iter_num <= 0: raise ValueError("Iteration numbers must be positive.")
                    specific_iterations_list.append(iter_num)
            specific_iterations_list = sorted(list(set(specific_iterations_list)))
            if not specific_iterations_list: logger.error("[red]No valid iterations parsed from --iterations.[/red]"); return
            logger.info(f"Will attempt to analyze specific iterations: {specific_iterations_list} (if available in each SC run).")
        except ValueError as e: logger.error(f"[red]Invalid format in --iterations argument: {e}[/red]"); return
    # --- End Iteration List Parsing ---


    try:
        # Preload embedding model once
        logger.info("Preloading embedding model...")
        get_embedding_model()
        logger.info("Embedding model preloaded.")

        # Loop through each specified n_start value
        for n_start in n_start_values:
            logger.info(f"--- Running Analysis for N={n_start} ---")
            run_analysis(
                n_start=n_start,
                similarity_thresholds=similarity_thresholds,
                sc_results_base_dir=args.sc_results_dir,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                analysis_output_base_dir=args.analysis_output_dir,
                tokenizer_path=args.tokenizer_path,
                start_iteration=args.start,
                end_iteration=args.end,
                specific_iterations=specific_iterations_list # Pass the parsed list
            )
            logger.info(f"--- Finished Analysis for N={n_start} ---")

    except KeyboardInterrupt:
        logger.exception("[yellow]\nAnalysis interrupted by user.[/yellow]")
    except Exception as e:
        logger.exception(f"[red]An unexpected error occurred during the analysis loop: {e}[/red]")

if __name__ == "__main__":
    main()