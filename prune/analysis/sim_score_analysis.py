import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from transformers import AutoTokenizer
from termcolor import colored
from typing import Dict, List, Tuple
import argparse
import os
import pandas as pd
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import concurrent.futures

print(colored("Importing modules...", "green"))
try:
    from slimsc.prune.utils.similarity_utils import (
        TARGET_PHRASES, MIN_SEGMENT_TOKENS, get_embedding_model, FaissIndexManager, find_thought_boundaries,
        embed_segments
    )
    from slimsc.prune.utils import DatasetHandler
except ImportError:
     import sys
     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
     from slimsc.prune.utils.similarity_utils import (
        TARGET_PHRASES, MIN_SEGMENT_TOKENS, get_embedding_model, FaissIndexManager, find_thought_boundaries,
        embed_segments
    )
     from slimsc.prune.utils import DatasetHandler

TOKENIZER_PATH = "/home/users/ntu/colinhon/scratch/r1-distill"
_tokenizer_cache_offline = [None, None, None, None, None]
tokenizer_cache_max_size = 5

EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_embedding_model_cache = [None, None, None, None, None]
embedding_model_cache_max_size = 5

def get_tokenizer_offline(tokenizer_path: str, tokenizer_idx: int):
    if tokenizer_idx > tokenizer_cache_max_size:
        raise ValueError(f"Tokenizer index {tokenizer_idx} exceeds max cache size {tokenizer_cache_max_size}")
    if _tokenizer_cache_offline[tokenizer_idx] is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            _tokenizer_cache_offline[tokenizer_idx] = tokenizer
            print(f"Tokenizer {tokenizer_path} loaded for offline analysis.")
            return _tokenizer_cache_offline[tokenizer_idx]
        except Exception as e:
            print(f"Error loading tokenizer {tokenizer_path}: {e}")
            raise
    return _tokenizer_cache_offline[tokenizer_idx]

def get_embedding_model_optimized(model_idx: int):
    if _embedding_model_cache[model_idx] is None:
        _embedding_model_cache[model_idx] = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
    return _embedding_model_cache[model_idx]

def find_newly_completed_thoughts_optimized(
    full_text: str,
    processed_boundaries: List[int], # Start indices of thoughts already processed
    worker_idx: int,
    target_phrases: List[str],
    min_segment_tokens: int = MIN_SEGMENT_TOKENS
) -> Tuple[List[Tuple[int, int, str]], List[int]]:
    """
    Identifies newly completed thought segments based on detected boundaries.
    
    A segment is considered 'completed' when its start boundary is detected,
    and the next boundary in the text has also appeared.

    Returns:
        Tuple: (new_segments, updated_processed_boundaries)
        - new_segments: List of (start_idx, end_idx, segment_text) for valid new thoughts.
        - updated_processed_boundaries: The list of start indices including newly processed ones.
    """
    if not full_text:
        return [], processed_boundaries

    # Find all potential boundaries in the current full text
    all_current_boundaries = find_thought_boundaries(full_text, target_phrases)

    newly_completed_segments = []
    # Track starts found and processed as a start of a newly completed segment in *this* function call
    new_segment_starts_processed_this_call = []

    # Ensure processed_boundaries is sorted and unique for efficient checking
    current_processed_boundaries = sorted(list(set(processed_boundaries)))
    # The index of the boundary that starts the LAST segment we finished processing previously.
    # We only look for new segment boundaries starting >= after this.
    last_processed_start = current_processed_boundaries[-1] if current_processed_boundaries else -1

    # Iterate through pairs of boundaries (start, end)
    # A segment is defined by `all_current_boundaries[i]` (start) and `all_current_boundaries[i+1]` (end)
    for i in range(len(all_current_boundaries) - 1):
        boundary_start = all_current_boundaries[i]
        boundary_end = all_current_boundaries[i+1]

        # Check if this segment's START boundary is NEW relative to what we've processed in PREVIOUS calls
        # AND has not been marked as a new segment start in *this* call yet.
        if boundary_start >= last_processed_start and boundary_start not in new_segment_starts_processed_this_call:

             # This boundary_start defines the start of a potential new segment.
             # Check if this specific boundary_start was already processed in any previous call.
             # We only process boundary_start if it's strictly greater than the last start boundary
             # we added to `processed_boundaries` in a previous call, or if it's a boundary
             # that appeared for the first time now.
             # The logic `boundary_start >= last_processed_start` covers most cases,
             # but `boundary_start not in current_processed_boundaries` is the most robust check
             # to see if this *specific* start index has ever defined a processed segment boundary before.

             if boundary_start not in current_processed_boundaries:
                  segment_text = full_text[boundary_start:boundary_end].strip()

                  if not segment_text: # Skip empty segments
                       continue

                  # Check token length using the shared utility function
                  num_tokens = len(get_tokenizer_offline(TOKENIZER_PATH, worker_idx).encode(segment_text, add_special_tokens=False))
                  # num_tokens = count_tokens(segment_text, tokenizer_path)

                  if num_tokens is None:
                       logger.warning(f"Could not tokenize segment [{boundary_start}:{boundary_end}] for length check. Skipping segment.")
                       continue # Skip segment if tokenization fails

                  if num_tokens >= min_segment_tokens:
                      # logger.debug(f"Found NEW completed thought segment [{boundary_start}:{boundary_end}], tokens={num_tokens}.") # Text: '{segment_text[:80]}...'")
                      newly_completed_segments.append((boundary_start, boundary_end, segment_text))
                      new_segment_starts_processed_this_call.append(boundary_start) # Mark this start as processed NOW
                  # else: logger.debug(f"New segment [{boundary_start}:{boundary_end}] too short (tokens={num_tokens}).")

    # Combine old and newly processed start boundaries found in this call
    # Add the starts found *in this call* to the existing processed boundaries.
    # Sort and make unique to maintain a clean list.
    updated_processed_boundaries = sorted(list(set(current_processed_boundaries + new_segment_starts_processed_this_call)))

    return newly_completed_segments, updated_processed_boundaries

def get_text_up_to_n_tokens(full_text: str, token_limit: int, tokenizer_idx: int) -> str:
    if not full_text:
        return ""
    tokenizer = get_tokenizer_offline(TOKENIZER_PATH, tokenizer_idx)
    token_ids = tokenizer.encode(full_text, add_special_tokens=False) # Often better for segments
    if len(token_ids) <= token_limit:
        return full_text
    truncated_token_ids = token_ids[:token_limit+1]
    return tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

def process_single_question_offline_sync(
    question_info: Dict,
    chain_contents: Dict[str, str],
    chain_correctness: Dict[str, bool],
    n_chains: int,
    token_step_size: int,
    worker_idx: int,
    chain_extracted_answers: Dict[str, str],
) -> List[float]:
    """
    Process a single question offline and return a list of similarity scores.
    """
    embedding_model = get_embedding_model_optimized(model_idx=worker_idx)
    index_manager = FaissIndexManager(dimension=embedding_model.get_sentence_embedding_dimension())
    chain_states = {}
    max_tokens_across_all_chains = 0

    for i in range(n_chains):
        chain_id = f"q{question_info['iteration']}_c{i+1}"
        full_text = chain_contents.get(chain_id, "")
        is_eventually_correct = chain_correctness.get(chain_id, False)
        extracted_answer = chain_extracted_answers.get(chain_id, "")
        num_tokens_in_chain = len(get_tokenizer_offline(TOKENIZER_PATH, worker_idx).encode(full_text, add_special_tokens=False))
        if num_tokens_in_chain is None: 
            num_tokens_in_chain = 0
        max_tokens_across_all_chains = max(max_tokens_across_all_chains, num_tokens_in_chain)

        chain_states[chain_id] = {
            "id": chain_id, "full_text_original": full_text,
            "is_eventually_correct": is_eventually_correct,
            "current_text_for_step": "", "processed_boundaries": [0],
            "completed_thought_count": 0, "embeddings": [],
            "max_tokens_in_chain": num_tokens_in_chain,
            "extracted_answer": extracted_answer,
        }

    results_at_each_step_counts = []

    for current_token_limit in range(token_step_size, max_tokens_across_all_chains + token_step_size, token_step_size):
        all_new_thoughts_this_step_data = []
        all_sim_scores_this_step = []
        correct_sim_scores_this_step = []
        incorrect_sim_scores_this_step = []
        correct_to_correct_scores = []
        correct_to_incorrect_scores = []
        incorrect_to_correct_scores = []
        incorrect_to_incorrect_scores = []
        same_answer_chains_scores = []
        same_answer_chains_correct_scores = []

        for chain_id, state in chain_states.items():
            text_for_this_chain_at_limit = get_text_up_to_n_tokens(state['full_text_original'], current_token_limit, worker_idx)
            if not text_for_this_chain_at_limit:
                state['current_text_for_step'] = ""
                continue
            state['current_text_for_step'] = text_for_this_chain_at_limit
            
            new_segments, updated_boundaries = find_newly_completed_thoughts_optimized(
                full_text=state['current_text_for_step'],
                processed_boundaries=state['processed_boundaries'],
                worker_idx=worker_idx,
                target_phrases=TARGET_PHRASES,
                min_segment_tokens=1
            )

            if new_segments:
                for _s_idx, _e_idx, text_c in new_segments:
                    thought_idx_for_chain = state['completed_thought_count']
                    all_new_thoughts_this_step_data.append({
                        'chain_id': chain_id, 
                        'thought_idx': thought_idx_for_chain, 
                        'text': text_c,
                        'is_correct': state['is_eventually_correct'],
                        'extracted_answer': state['extracted_answer']
                    })
                    state['completed_thought_count'] += 1
                state['processed_boundaries'] = updated_boundaries

        if not all_new_thoughts_this_step_data:
            continue

        texts_to_embed = [item['text'] for item in all_new_thoughts_this_step_data]
        embeddings_for_new_thoughts = embed_segments(texts_to_embed)

        if embeddings_for_new_thoughts is None or len(embeddings_for_new_thoughts) != len(all_new_thoughts_this_step_data):
            continue

        for i, item in enumerate(all_new_thoughts_this_step_data): 
            item['embedding'] = embeddings_for_new_thoughts[i]

        candidate_embeddings_for_faiss = []

        for thought_data in all_new_thoughts_this_step_data:
            chain_id, thought_idx, embedding, text, is_correct, extracted_answer = (
                thought_data['chain_id'], 
                thought_data['thought_idx'], 
                thought_data['embedding'], 
                thought_data['text'],
                thought_data['is_correct'],
                thought_data['extracted_answer']
            )

            can_potentially_prune = (thought_idx >= 2 and index_manager.get_num_embeddings() > 0)

            if can_potentially_prune:
                neighbor_result = index_manager.search_nearest_neighbor(embedding, chain_id)
                if neighbor_result:
                    sim_score, neighbor_chain_id, _, _ = neighbor_result
                    identifier = [chain_id, neighbor_chain_id]
                    identifier.sort()
                    identifier = tuple(identifier)
                    all_sim_scores_this_step.append((identifier, sim_score))
                    
                    # Get the correctness of the neighbor chain
                    neighbor_is_correct = chain_states[neighbor_chain_id]['is_eventually_correct']
                    neighbor_extracted_answer = chain_states[neighbor_chain_id]['extracted_answer']

                    if extracted_answer == neighbor_extracted_answer:
                        same_answer_chains_scores.append((identifier, sim_score))
                        if is_correct:
                            same_answer_chains_correct_scores.append((identifier, sim_score))
                    
                    # Track all combinations of correctness
                    if is_correct:
                        correct_sim_scores_this_step.append((identifier, sim_score))
                        if neighbor_is_correct:
                            correct_to_correct_scores.append((identifier, sim_score))
                        else:
                            correct_to_incorrect_scores.append((identifier, sim_score))
                    else:
                        incorrect_sim_scores_this_step.append((identifier, sim_score))
                        if neighbor_is_correct:
                            incorrect_to_correct_scores.append((identifier, sim_score))
                        else:
                            incorrect_to_incorrect_scores.append((identifier, sim_score))

            candidate_embeddings_for_faiss.append({
                'embedding': embedding, 
                'chain_id': chain_id, 
                'thought_idx': thought_idx, 
                'text': text
            })
                    
        for item_to_add in candidate_embeddings_for_faiss:
            cid_add = item_to_add['chain_id']
            index_manager.add_embedding(item_to_add['embedding'], cid_add, item_to_add['thought_idx'], item_to_add['text'])
            chain_states[cid_add]['embeddings'].append(item_to_add['embedding'])

        results_at_each_step_counts.append({
            'all': all_sim_scores_this_step,
            'correct': correct_sim_scores_this_step,
            'incorrect': incorrect_sim_scores_this_step,
            'correct_to_correct': correct_to_correct_scores,
            'correct_to_incorrect': correct_to_incorrect_scores,
            'incorrect_to_correct': incorrect_to_correct_scores,
            'incorrect_to_incorrect': incorrect_to_incorrect_scores,
            'same_answer_chains': same_answer_chains_scores,
            'same_answer_chains_correct': same_answer_chains_correct_scores
        })

    return results_at_each_step_counts

def process_question_worker(chosen_question_iterations, worker_idx, sampled_df, control_summaries_dir, control_chains_dir, args):
    from tqdm import tqdm
    import os
    import json
    worker_results = []
    per_question_results = {}
    from slimsc.prune.utils import DatasetHandler
    dataset_handler = DatasetHandler(dataset_name=args.dataset_name)
    for iteration_num in tqdm(chosen_question_iterations, desc=f"Processing Sampled Questions for Worker {worker_idx}"):
        question_row = sampled_df[sampled_df['iteration'] == iteration_num].iloc[0]
        correct_answer_ref = question_row['correct_answer']
        chain_contents, chain_correctness = {}, {}
        summary_json_path = os.path.join(control_summaries_dir, f"question_{iteration_num}_summary.json")
        individual_chain_answers = {}
        if os.path.exists(summary_json_path):
            with open(summary_json_path, 'r') as f: q_summary = json.load(f)
            if "chains_for_voting_details" in q_summary:
                for detail in q_summary["chains_for_voting_details"]:
                    if detail.get("chain_index") is not None : # Ensure chain_index is not None
                        individual_chain_answers[detail["chain_index"]] = detail["extracted_answer"]
        n_chains_sc = args.n_chains  # Number of chains in the control run, e.g., 16
        for chain_idx_1based in range(1, n_chains_sc + 1):
            chain_id = f"q{iteration_num}_c{chain_idx_1based}"
            found_file = None
            for status_suffix in ["used_for_voting"]:
                potential_fname = f"question_{iteration_num}_chain_{chain_idx_1based}_{status_suffix}.txt"
                if os.path.exists(os.path.join(control_chains_dir, potential_fname)):
                    found_file = potential_fname
                    break
            if found_file:
                with open(os.path.join(control_chains_dir, found_file), 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    content_start_line = 0
                    for idx_line, line_txt in enumerate(lines):
                        if line_txt.strip() == "--- Reasoning Content ---":
                            content_start_line = idx_line + 1
                            break
                    chain_contents[chain_id] = "".join(lines[content_start_line:]).strip()
            else: chain_contents[chain_id] = ""
            extracted_ans_from_summary = individual_chain_answers.get(chain_idx_1based)
            is_correct = False
            if extracted_ans_from_summary is not None:
                is_correct = dataset_handler.calculate_score(extracted_ans_from_summary, correct_answer_ref) == 1
            elif chain_contents[chain_id]:
                extracted_ans = dataset_handler.extract_answer(chain_contents[chain_id])
                is_correct = dataset_handler.calculate_score(extracted_ans, correct_answer_ref) == 1
            chain_correctness[chain_id] = is_correct
        question_info_dict = {'iteration': iteration_num, 'correct_answer': correct_answer_ref}
        question_results = process_single_question_offline_sync(
            question_info=question_info_dict, chain_contents=chain_contents,
            chain_correctness=chain_correctness, n_chains=n_chains_sc,
            token_step_size=args.token_step_size,
            worker_idx=worker_idx,
            chain_extracted_answers=individual_chain_answers
        )
        worker_results.append(question_results)
        per_question_results[str(iteration_num)] = question_results
    return per_question_results

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def main_offline_analysis(args):
    # Create output directory at the start
    output_dir = args.output_dir if hasattr(args, 'output_dir') else 'sim_score_results'
    os.makedirs(output_dir, exist_ok=True)

    base_results_dir = os.path.join(args.base_slimsc_dir, "prune/results", args.model_arch, args.dataset_name)
    control_run_name = args.control_run_name
    control_dir = os.path.join(base_results_dir, control_run_name)
    control_eval_summary_path = os.path.join(control_dir, "evaluation_summary.csv")
    control_chains_dir = os.path.join(control_dir, "individual_chains")
    control_summaries_dir = os.path.join(control_dir, "summaries")

    if not os.path.exists(control_eval_summary_path):
        print(f"Control evaluation summary not found at: {control_eval_summary_path}")
        return
    if not os.path.exists(control_chains_dir):
        print(f"Control chains directory not found at: {control_chains_dir}")
        return
    if not os.path.exists(control_summaries_dir):
        print(f"Control summaries directory not found at: {control_summaries_dir}")
        return
        
    df_full_control = pd.read_csv(control_eval_summary_path)
    sample_size = args.num_questions
    seed = args.seed
    sampled_df = df_full_control.sample(n=min(sample_size, len(df_full_control)), random_state=seed, replace=False)
    sampled_question_iterations = sampled_df['iteration'].tolist()
    print(f"Sampled {len(sampled_question_iterations)} question iterations using seed {seed}.")

    dataset_handler = DatasetHandler(dataset_name=args.dataset_name)
    all_results_data = {} # { (strategy, threshold) : [ [q1_percents], [q2_percents], ... ] }
    # print("Loading embedding model...")
    # get_embedding_model() # Pre-load
    # print("Embedding model loaded.")

    # Dictionary to store combined scores by processing step
    combined_scores_by_step = {
        'all': [],      # List of lists, each inner list contains scores for one step
        'correct': [],  # Same structure for correct chains
        'incorrect': [], # Same structure for incorrect chains
        'correct_to_correct': [],    # New metric: correct chains similar to correct chains
        'correct_to_incorrect': [],  # New metric: correct chains similar to incorrect chains
        'incorrect_to_correct': [],  # New metric: incorrect chains similar to correct chains
        'incorrect_to_incorrect': [], # New metric: incorrect chains similar to incorrect chains
        'same_answer_chains': [], # New metric: chains with the same answer
        'same_answer_chains_correct': [] # New metric: correct chains with the same answer
    }

    num_workers = min(5, len(sampled_question_iterations))
    workers = list(range(num_workers))
    worker_args = []
    for worker_idx in workers:
        chosen_question_iterations = sampled_question_iterations[worker_idx::len(workers)]
        worker_args.append((chosen_question_iterations, worker_idx, sampled_df, control_summaries_dir, control_chains_dir, args))

    # Use multiprocessing to run workers in parallel
    all_worker_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(workers)) as executor:
        futures = [executor.submit(process_question_worker, *args) for args in worker_args]
        for future in concurrent.futures.as_completed(futures):
            worker_result = future.result()
            all_worker_results.append(worker_result)

    # Combine results from all workers
    for worker_result in all_worker_results:
        for question_id, question_results in worker_result.items():
            for step_idx, step_results in enumerate(question_results):
                while len(combined_scores_by_step['all']) <= step_idx:
                    for key in combined_scores_by_step:
                        combined_scores_by_step[key].append([])
                combined_scores_by_step['all'][step_idx].extend(step_results['all'])
                combined_scores_by_step['correct'][step_idx].extend(step_results['correct'])
                combined_scores_by_step['incorrect'][step_idx].extend(step_results['incorrect'])
                combined_scores_by_step['correct_to_correct'][step_idx].extend(step_results['correct_to_correct'])
                combined_scores_by_step['correct_to_incorrect'][step_idx].extend(step_results['correct_to_incorrect'])
                combined_scores_by_step['incorrect_to_correct'][step_idx].extend(step_results['incorrect_to_correct'])
                combined_scores_by_step['incorrect_to_incorrect'][step_idx].extend(step_results['incorrect_to_incorrect'])
                combined_scores_by_step['same_answer_chains'][step_idx].extend(step_results['same_answer_chains'])
                combined_scores_by_step['same_answer_chains_correct'][step_idx].extend(step_results['same_answer_chains_correct'])
    # Calculate statistics for each step
    steps = np.array(range(len(combined_scores_by_step['all'])))
    stats = {
        'all': {'median': [], 'p25': [], 'p75': []},
        'correct': {'median': [], 'p25': [], 'p75': []},
        'incorrect': {'median': [], 'p25': [], 'p75': []},
        'correct_to_correct': {'median': [], 'p25': [], 'p75': []},
        'correct_to_incorrect': {'median': [], 'p25': [], 'p75': []},
        'incorrect_to_correct': {'median': [], 'p25': [], 'p75': []},
        'incorrect_to_incorrect': {'median': [], 'p25': [], 'p75': []}
    }

    for step_idx in range(len(combined_scores_by_step['all'])):
        for category in stats:
            scores = combined_scores_by_step[category][step_idx]
            if scores:
                stats[category]['median'].append(np.median([item[1] for item in scores]))
                stats[category]['p25'].append(np.percentile([item[1] for item in scores], 25))
                stats[category]['p75'].append(np.percentile([item[1] for item in scores], 75))
            else:
                stats[category]['median'].append(np.nan)
                stats[category]['p25'].append(np.nan)
                stats[category]['p75'].append(np.nan)

    # Create stacked bar chart for same answer chains
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate data for each threshold
    thresholds = np.arange(0.8, 0.98 + 0.02, 0.02)  # Stop at 0.98 inclusive
    # print('same_answer_chains', combined_scores_by_step['same_answer_chains'])
    same_answer_counts = []
    same_answer_correct_counts = []
    flat_same_answer_chains_correct = set()
    flat_same_answer_chains = set()
    all_sim_chains = set()
    for step_results in combined_scores_by_step['same_answer_chains_correct']:
        for identifier, _ in step_results:
            flat_same_answer_chains_correct.add(identifier)
    for step_results in combined_scores_by_step['same_answer_chains']:
        for identifier, _ in step_results:
            flat_same_answer_chains.add(identifier)
    for step_results in combined_scores_by_step['all']:
        for item in step_results:
            all_sim_chains.add(item)
    
    # print('all_sim_chains', all_sim_chains)
    # print('flat_same_answer_chains', flat_same_answer_chains)
    # print('flat_same_answer_chains_correct', flat_same_answer_chains_correct)

    total_similar_pairs_counts = []
    different_answer_counts = []

    for threshold in thresholds:
        num_same_answer_chains = 0
        num_same_answer_chains_correct = 0
        total_similar_pairs = 0
        seen_pairs = set()
        for identifier, sim_score in all_sim_chains:
            if identifier in seen_pairs:
                continue
            seen_pairs.add(identifier)
            if sim_score >= threshold:
                total_similar_pairs += 1
                if identifier in flat_same_answer_chains:
                    num_same_answer_chains += 1
                    if identifier in flat_same_answer_chains_correct:
                        num_same_answer_chains_correct += 1
        same_answer_counts.append(num_same_answer_chains)
        same_answer_correct_counts.append(num_same_answer_chains_correct)
        total_similar_pairs_counts.append(total_similar_pairs)
        different_answer_counts.append(total_similar_pairs - num_same_answer_chains)
    
    # Create the stacked bars
    x = np.arange(len(thresholds))
    width = 0.35
    
    # Plot the stacked bars - now with 4 levels
    ax.bar(x, same_answer_correct_counts, width, label='Same Answer (Correct)', color='tab:blue')
    ax.bar(x, [a - b for a, b in zip(same_answer_counts, same_answer_correct_counts)], 
           width, bottom=same_answer_correct_counts, label='Same Answer (Incorrect)', color='tab:orange')
    ax.bar(x, different_answer_counts, width, 
           bottom=same_answer_counts, label='Different Answer', color='tab:gray')
    # ax.bar(x, [a - b for a, b in zip(total_similar_pairs_counts, same_answer_counts + different_answer_counts)], 
    #        width, bottom=[a + b for a, b in zip(same_answer_counts, different_answer_counts)], 
    #        label='Not Similar', color='tab:gray')
    
    # Customize the plot
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('No. of Similar Pairs of Chains', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the stacked bar chart
    stacked_bar_path = os.path.join(output_dir, f'same_answer_chains_stacked_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(stacked_bar_path)
    plt.close()

    # Create normalized stacked bar chart (percentages)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate percentages
    same_answer_correct_percentages = []
    same_answer_incorrect_percentages = []
    different_answer_percentages = []
    not_similar_percentages = []
    
    for total, same, diff, correct in zip(total_similar_pairs_counts, same_answer_counts, 
                                        different_answer_counts, same_answer_correct_counts):
        if total > 0:
            correct_pct = (correct / total) * 100
            incorrect_pct = ((same - correct) / total) * 100
            diff_pct = (diff / total) * 100
            not_similar_pct = 100 - (correct_pct + incorrect_pct + diff_pct)
        else:
            correct_pct = incorrect_pct = diff_pct = not_similar_pct = 0
        same_answer_correct_percentages.append(correct_pct)
        same_answer_incorrect_percentages.append(incorrect_pct)
        different_answer_percentages.append(diff_pct)
        not_similar_percentages.append(not_similar_pct)
    
    # Create the normalized stacked bars
    x = np.arange(len(thresholds))
    width = 0.35
    
    # Plot the stacked bars - now with 4 levels
    ax.bar(x, same_answer_correct_percentages, width, label='Same Answer (Correct)', color='tab:blue')
    ax.bar(x, same_answer_incorrect_percentages, width, 
           bottom=same_answer_correct_percentages, label='Same Answer (Incorrect)', color='tab:orange')
    ax.bar(x, different_answer_percentages, width, 
           bottom=[a + b for a, b in zip(same_answer_correct_percentages, same_answer_incorrect_percentages)], 
           label='Different Answer', color='tab:gray')
    # ax.bar(x, not_similar_percentages, width, 
    #        bottom=[a + b + c for a, b, c in zip(same_answer_correct_percentages, 
    #                                           same_answer_incorrect_percentages, 
    #                                           different_answer_percentages)], 
    #        label='Not Similar', color='tab:gray')
    
    # Customize the plot
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Similar Pairs of Chains', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    
    # Set y-axis to show percentages from 0 to 100
    ax.set_ylim(0, 100)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the normalized stacked bar chart
    normalized_stacked_bar_path = os.path.join(output_dir, f'same_answer_chains_stacked_normalized_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(normalized_stacked_bar_path)
    plt.close()

    # Create the main categories plot
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot main categories (all, correct, incorrect)
    ax.plot(steps, stats['all']['median'], color='tab:purple', label='All Chains (median)', alpha=0.7)
    ax.fill_between(steps, stats['all']['p25'], stats['all']['p75'], color='tab:purple', alpha=0.2, label='All Chains (25-75%)')

    ax.plot(steps, stats['correct']['median'], color='tab:blue', label='Correct Chains (median)', alpha=0.7)
    ax.fill_between(steps, stats['correct']['p25'], stats['correct']['p75'], color='tab:blue', alpha=0.2, label='Correct Chains (25-75%)')

    ax.plot(steps, stats['incorrect']['median'], color='tab:orange', label='Incorrect Chains (median)', alpha=0.7)
    ax.fill_between(steps, stats['incorrect']['p25'], stats['incorrect']['p75'], color='tab:orange', alpha=0.2, label='Incorrect Chains (25-75%)')

    ax.set_xlabel('Processing Step', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    # Save the main categories plot
    main_plot_path = os.path.join(output_dir, f'main_category_similarity_scores_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(main_plot_path)
    plt.close()

    # Create and save the cross-category plot
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Plot cross-category similarities using tab10 colors
    ax.plot(steps, stats['correct_to_correct']['median'], color='tab:blue', label='Correct→Correct (median)', alpha=0.7)
    ax.plot(steps, stats['correct_to_incorrect']['median'], color='tab:orange', label='Correct→Incorrect (median)', alpha=0.7)
    ax.plot(steps, stats['incorrect_to_correct']['median'], color='tab:purple', label='Incorrect→Correct (median)', alpha=0.7)
    ax.plot(steps, stats['incorrect_to_incorrect']['median'], color='tab:cyan', label='Incorrect→Incorrect (median)', alpha=0.7)

    ax.set_xlabel('Processing Step', fontsize=12)
    ax.set_ylabel('Similarity Score', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    
    # Save the cross-category plot
    cross_plot_path = os.path.join(output_dir, f'cross_category_similarity_scores_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(cross_plot_path)
    plt.close()

    # Create threshold analysis plots (proportions)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Define thresholds
    thresholds = np.arange(0.8, 0.98 + 0.02, 0.02)  # Stop at 0.98 inclusive
    
    # Calculate proportions and counts for each threshold
    correct_proportions = []
    incorrect_proportions = []
    correct_counts = []
    incorrect_counts = []
    
    for threshold in thresholds:
        # Get all similarity scores for correct and incorrect chains
        all_correct_scores = []
        all_incorrect_scores = []
        
        for step_scores in combined_scores_by_step['correct']:
            all_correct_scores.extend([item[1] for item in step_scores])
        for step_scores in combined_scores_by_step['incorrect']:
            all_incorrect_scores.extend([item[1] for item in step_scores])
            
        # Calculate proportions and counts above threshold
        if all_correct_scores:
            correct_prop = sum(score >= threshold for score in all_correct_scores) / len(all_correct_scores)
            correct_count = sum(score >= threshold for score in all_correct_scores)
        else:
            correct_prop = 0
            correct_count = 0
            
        if all_incorrect_scores:
            incorrect_prop = sum(score >= threshold for score in all_incorrect_scores) / len(all_incorrect_scores)
            incorrect_count = sum(score >= threshold for score in all_incorrect_scores)
        else:
            incorrect_prop = 0
            incorrect_count = 0
            
        correct_proportions.append(correct_prop)
        incorrect_proportions.append(incorrect_prop)
        correct_counts.append(correct_count)
        incorrect_counts.append(incorrect_count)
    
    # Set width of bars
    barWidth = 0.35
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(thresholds))
    r2 = [x + barWidth for x in r1]
    
    # Create the proportion bars
    ax.bar(r1, correct_proportions, width=barWidth, color='tab:blue', label='Correct Chains')
    ax.bar(r2, incorrect_proportions, width=barWidth, color='tab:orange', label='Incorrect Chains')
    
    # Add labels for proportion plot
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Proportion of Chains Above Threshold', fontsize=12)
    
    # Set x-axis ticks for proportion plot
    ax.set_xticks([r + barWidth/2 for r in range(len(thresholds))])
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend for proportion plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    
    # Add grid for proportion plot
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the proportion plot
    threshold_prop_plot_path = os.path.join(output_dir, f'threshold_analysis_proportions_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(threshold_prop_plot_path)
    plt.close()

    # Create threshold analysis plot (counts)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Create the count bars
    ax.bar(r1, correct_counts, width=barWidth, color='tab:blue', label='Correct Chains')
    ax.bar(r2, incorrect_counts, width=barWidth, color='tab:orange', label='Incorrect Chains')
    
    # Add labels for count plot
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Number of Chains Above Threshold', fontsize=12)
    
    # Set x-axis ticks for count plot
    ax.set_xticks([r + barWidth/2 for r in range(len(thresholds))])
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend for count plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    
    # Add grid for count plot
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the count plot
    threshold_count_plot_path = os.path.join(output_dir, f'threshold_analysis_counts_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(threshold_count_plot_path)
    plt.close()

    # Create cross-category threshold analysis plot (proportions)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate proportions and counts for each threshold
    correct_to_correct_props = []
    correct_to_incorrect_props = []
    incorrect_to_correct_props = []
    incorrect_to_incorrect_props = []
    correct_to_correct_counts = []
    correct_to_incorrect_counts = []
    incorrect_to_correct_counts = []
    incorrect_to_incorrect_counts = []
    
    for threshold in thresholds:
        # Get all similarity scores for each category
        all_correct_to_correct = []
        all_correct_to_incorrect = []
        all_incorrect_to_correct = []
        all_incorrect_to_incorrect = []
        
        for step_scores in combined_scores_by_step['correct_to_correct']:
            all_correct_to_correct.extend([item[1] for item in step_scores])
        for step_scores in combined_scores_by_step['correct_to_incorrect']:
            all_correct_to_incorrect.extend([item[1] for item in step_scores])
        for step_scores in combined_scores_by_step['incorrect_to_correct']:
            all_incorrect_to_correct.extend([item[1] for item in step_scores])
        for step_scores in combined_scores_by_step['incorrect_to_incorrect']:
            all_incorrect_to_incorrect.extend([item[1] for item in step_scores])
            
        # Calculate proportions and counts above threshold
        if all_correct_to_correct:
            correct_to_correct_prop = sum(score >= threshold for score in all_correct_to_correct) / len(all_correct_to_correct)
            correct_to_correct_count = sum(score >= threshold for score in all_correct_to_correct)
        else:
            correct_to_correct_prop = 0
            correct_to_correct_count = 0
            
        if all_correct_to_incorrect:
            correct_to_incorrect_prop = sum(score >= threshold for score in all_correct_to_incorrect) / len(all_correct_to_incorrect)
            correct_to_incorrect_count = sum(score >= threshold for score in all_correct_to_incorrect)
        else:
            correct_to_incorrect_prop = 0
            correct_to_incorrect_count = 0
            
        if all_incorrect_to_correct:
            incorrect_to_correct_prop = sum(score >= threshold for score in all_incorrect_to_correct) / len(all_incorrect_to_correct)
            incorrect_to_correct_count = sum(score >= threshold for score in all_incorrect_to_correct)
        else:
            incorrect_to_correct_prop = 0
            incorrect_to_correct_count = 0
            
        if all_incorrect_to_incorrect:
            incorrect_to_incorrect_prop = sum(score >= threshold for score in all_incorrect_to_incorrect) / len(all_incorrect_to_incorrect)
            incorrect_to_incorrect_count = sum(score >= threshold for score in all_incorrect_to_incorrect)
        else:
            incorrect_to_incorrect_prop = 0
            incorrect_to_incorrect_count = 0
            
        correct_to_correct_props.append(correct_to_correct_prop)
        correct_to_incorrect_props.append(correct_to_incorrect_prop)
        incorrect_to_correct_props.append(incorrect_to_correct_prop)
        incorrect_to_incorrect_props.append(incorrect_to_incorrect_prop)
        correct_to_correct_counts.append(correct_to_correct_count)
        correct_to_incorrect_counts.append(correct_to_incorrect_count)
        incorrect_to_correct_counts.append(incorrect_to_correct_count)
        incorrect_to_incorrect_counts.append(incorrect_to_incorrect_count)
    
    # Set width of bars
    barWidth = 0.2
    
    # Set positions of the bars on X axis
    r1 = np.arange(len(thresholds))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]
    
    # Create the proportion bars
    ax.bar(r1, correct_to_correct_props, width=barWidth, color='tab:blue', label='Correct→Correct')
    ax.bar(r2, correct_to_incorrect_props, width=barWidth, color='tab:orange', label='Correct→Incorrect')
    ax.bar(r3, incorrect_to_correct_props, width=barWidth, color='tab:purple', label='Incorrect→Correct')
    ax.bar(r4, incorrect_to_incorrect_props, width=barWidth, color='tab:cyan', label='Incorrect→Incorrect')
    
    # Add labels for proportion plot
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Proportion of Chains Above Threshold', fontsize=12)
    
    # Set x-axis ticks for proportion plot
    ax.set_xticks([r + barWidth*1.5 for r in range(len(thresholds))])
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend for proportion plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    
    # Add grid for proportion plot
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the cross-category proportion plot
    cross_threshold_prop_plot_path = os.path.join(output_dir, f'cross_category_threshold_analysis_proportions_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(cross_threshold_prop_plot_path)
    plt.close()

    # Create cross-category threshold analysis plot (counts)
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Create the count bars
    ax.bar(r1, correct_to_correct_counts, width=barWidth, color='tab:blue', label='Correct→Correct')
    ax.bar(r2, correct_to_incorrect_counts, width=barWidth, color='tab:orange', label='Correct→Incorrect')
    ax.bar(r3, incorrect_to_correct_counts, width=barWidth, color='tab:purple', label='Incorrect→Correct')
    ax.bar(r4, incorrect_to_incorrect_counts, width=barWidth, color='tab:cyan', label='Incorrect→Incorrect')
    
    # Add labels for count plot
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Number of Chains Above Threshold', fontsize=12)
    
    # Set x-axis ticks for count plot
    ax.set_xticks([r + barWidth*1.5 for r in range(len(thresholds))])
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend for count plot
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, framealpha=0.9, edgecolor='black', fontsize=12)
    
    # Add grid for count plot
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save the cross-category count plot
    cross_threshold_count_plot_path = os.path.join(output_dir, f'cross_category_threshold_analysis_counts_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    plt.savefig(cross_threshold_count_plot_path)
    plt.close()

    correct_correct_props = []
    correct_incorrect_props = []
    incorrect_incorrect_props = []
    correct_correct_counts = []
    correct_incorrect_counts = []
    incorrect_incorrect_counts = []
    
    for threshold in thresholds:
        # Get all similarity scores for each category
        all_correct_correct = []
        all_correct_incorrect = []
        all_incorrect_incorrect = []
        
        seen_pairs = set()
        for step_scores in combined_scores_by_step['correct_to_correct'][20:]:
            for item in step_scores:
                if item[0] in seen_pairs:
                    continue
                seen_pairs.add(item[0])
                all_correct_correct.append(item[1])
        for step_scores in combined_scores_by_step['correct_to_incorrect'][20:]:
            for item in step_scores:
                if item[0] in seen_pairs:
                    continue
                seen_pairs.add(item[0])
                all_correct_incorrect.append(item[1])
        for step_scores in combined_scores_by_step['incorrect_to_correct'][20:]:
            for item in step_scores:
                if item[0] in seen_pairs:
                    continue
                seen_pairs.add(item[0])
                all_correct_incorrect.append(item[1])
        for step_scores in combined_scores_by_step['incorrect_to_incorrect'][20:]:
            for item in step_scores:
                if item[0] in seen_pairs:
                    continue
                seen_pairs.add(item[0])
                all_incorrect_incorrect.append(item[1])
            
        # Calculate proportions and counts above threshold
        if all_correct_correct:
            correct_correct_count = sum(score >= threshold for score in all_correct_correct)
        else:
            correct_correct_count = 0
            
        if all_correct_incorrect:
            correct_incorrect_count = sum(score >= threshold for score in all_correct_incorrect)
        else:
            correct_incorrect_count = 0
            
        if all_incorrect_incorrect:
            incorrect_incorrect_count = sum(score >= threshold for score in all_incorrect_incorrect)
        else:
            incorrect_incorrect_count = 0

        total_count = correct_correct_count + correct_incorrect_count + incorrect_incorrect_count

        if total_count == 0:
            correct_correct_prop = 0
            correct_incorrect_prop = 0
            incorrect_incorrect_prop = 0
        else:
            correct_correct_prop = (correct_correct_count / total_count) * 100
            correct_incorrect_prop = (correct_incorrect_count / total_count) * 100
            incorrect_incorrect_prop = (incorrect_incorrect_count / total_count) * 100
            
        correct_correct_props.append(correct_correct_prop)
        correct_incorrect_props.append(correct_incorrect_prop)
        incorrect_incorrect_props.append(incorrect_incorrect_prop)
        correct_correct_counts.append(correct_correct_count)
        correct_incorrect_counts.append(correct_incorrect_count)
        incorrect_incorrect_counts.append(incorrect_incorrect_count)

    # make stacked bar plots. set axislabel and fontsize
    fig, ax = plt.subplots(figsize=(5, 4))
    # Stacked bars, not grouped
    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(x, correct_correct_props, width, label='Correct-Correct', color='tab:blue')
    ax.bar(x, correct_incorrect_props, width, bottom=correct_correct_props, label='Correct-Incorrect', color='tab:purple')
    ax.bar(x, incorrect_incorrect_props, width, bottom=[a + b for a, b in zip(correct_correct_props, correct_incorrect_props)], label='Incorrect-Incorrect', color='tab:orange')
    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Similar Chains', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'stacked_bar_plot_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png'))
    plt.close()


    # Save per-question and combined results to a single JSON file
    output_json = {
        # 'per_question_results': all_worker_results,
        # 'combined_scores_by_step': combined_scores_by_step,
        # 'statistics': stats,
        'plot_data': {
            # Main categories plot data
            'main_categories': {
                'steps': steps.tolist(),
                'all_median': stats['all']['median'],
                'all_p25': stats['all']['p25'],
                'all_p75': stats['all']['p75'],
                'correct_median': stats['correct']['median'],
                'correct_p25': stats['correct']['p25'],
                'correct_p75': stats['correct']['p75'],
                'incorrect_median': stats['incorrect']['median'],
                'incorrect_p25': stats['incorrect']['p25'],
                'incorrect_p75': stats['incorrect']['p75']
            },
            # Cross-category plot data
            'cross_category': {
                'steps': steps.tolist(),
                'correct_to_correct_median': stats['correct_to_correct']['median'],
                'correct_to_incorrect_median': stats['correct_to_incorrect']['median'],
                'incorrect_to_correct_median': stats['incorrect_to_correct']['median'],
                'incorrect_to_incorrect_median': stats['incorrect_to_incorrect']['median']
            },
            # Threshold analysis data
            'threshold_analysis': {
                'thresholds': thresholds.tolist(),
                'correct_proportions': correct_proportions,
                'incorrect_proportions': incorrect_proportions,
                'correct_counts': correct_counts,
                'incorrect_counts': incorrect_counts,
                'correct_to_correct_props': correct_to_correct_props,
                'correct_to_incorrect_props': correct_to_incorrect_props,
                'incorrect_to_correct_props': incorrect_to_correct_props,
                'incorrect_to_incorrect_props': incorrect_to_incorrect_props,
                'correct_to_correct_counts': correct_to_correct_counts,
                'correct_to_incorrect_counts': correct_to_incorrect_counts,
                'incorrect_to_correct_counts': incorrect_to_correct_counts,
                'incorrect_to_incorrect_counts': incorrect_to_incorrect_counts
            },
            # Same answer chains analysis data
            'same_answer_chains': {
                'thresholds': thresholds.tolist(),
                'same_answer_counts': same_answer_counts,
                'same_answer_correct_counts': same_answer_correct_counts,
                'total_similar_pairs_counts': total_similar_pairs_counts,
                'different_answer_counts': different_answer_counts,
                'same_answer_correct_percentages': same_answer_correct_percentages,
                'same_answer_incorrect_percentages': same_answer_incorrect_percentages,
                'different_answer_percentages': different_answer_percentages
            },
            'stacked_bar_plot': {
                'thresholds': thresholds.tolist(),
                'correct_correct_props': correct_correct_props,
                'correct_incorrect_props': correct_incorrect_props,
                'incorrect_incorrect_props': incorrect_incorrect_props
            }
        }
    }
    
    # Convert numpy types to Python native types before JSON serialization
    output_json = convert_numpy_types(output_json)
    
    with open(os.path.join(output_dir, 'sim_score_results.json'), 'w') as f:
        json.dump(output_json, f, indent=2)

def test():
    # test loading and using 2 tokenizers concurrently
    tokenizer_1 = get_tokenizer_offline(TOKENIZER_PATH, 0)
    tokenizer_2 = get_tokenizer_offline(TOKENIZER_PATH, 1)
    print(tokenizer_1.encode("Hello, world!"))
    print(tokenizer_2.encode("Hello, world!"))
        
    # test get_text_up_to_n_tokens
    print(get_text_up_to_n_tokens("Hello, world!", 1, 0))
    print(get_text_up_to_n_tokens("Hello, world!", 1, 1))

    # test process_single_question_offline_sync
    question_info = {
        "iteration": 0
    }
    chain_contents = {
        "q0_c1": "Hello, world! Alternative, we can say hello to the world. And then go get some pizze or something at the pizza place. Alternative, we can say bye to the world. But we could also not do that and we could say something else. alternative, hi there how are you doing today on this fine day? And then how are you going to do this thing that you want to do? Some text to make this thought longer than it needs to be. alternative, I will tell you this story about a time when I was in the world and I did this thing that I wanted to do.",
        "q0_c2": "Hello, world! Alternative, we can say hello to the world. And then go get some pizze or something at the pizza place. Alternative, we can say bye to the world. But we could also not do that and we could say something else. alternative, hi there how are you doing today on this fine day? And then how are you going to do this thing that you want to do? Some text to make this thought longer than it needs to be. alternative, I will tell you this story about a time when I was in the world and I did this thing that I wanted to do."
    }
    chain_correctness = {
        "q0_c1": True,
        "q0_c2": False
    }
    process_single_question_offline_sync(question_info, chain_contents, chain_correctness, 2, 5, 0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Self-Consistency Similarity Score Analysis")
    parser.add_argument('--base_slimsc_dir', type=str, default="slimsc", help="Base directory of the slimsc project.")
    parser.add_argument('--model_arch', type=str, default="QwQ-32B", help="Model architecture name (e.g., QwQ-32B).")
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help="Dataset name (e.g., gpqa_diamond).")
    parser.add_argument('--control_run_name', type=str, default="sc_16_control", help="Name of the control run directory.")
    parser.add_argument('--n_chains', type=int, default=16, help="Number of chains in the control run.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the HuggingFace tokenizer.")
    parser.add_argument('--num_questions', type=int, default=50, help="Number of questions to sample.")
    parser.add_argument('--seed', type=int, default=7, help="Random seed for sampling questions.")
    parser.add_argument('--token_step_size', type=int, default=100, help="Token interval for pruning simulation steps.")
    parser.add_argument('--output_dir', type=str, default='sim_score_results', help="Directory to save similarity score results.")

    cli_args = parser.parse_args()

    main_offline_analysis(cli_args)

    """
        sample command
        python sim_score_analysis.py --model_arch R1-Distill-Qwen-14B --base_slimsc_dir "/home/users/ntu/colinhon/slimsc" --dataset_name aime --control_run_name sc_64_control --n_chains 64 --tokenizer_path /home/users/ntu/colinhon/scratch/r1-distill --num_questions 30 --seed 7 --token_step_size 100 --output_dir aime_sim_score_results
    """


    