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

TOKENIZER_PATH = "/home/users/ntu/aqui0001/scratch/r1-distill"
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
    worker_idx: int
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
        
        num_tokens_in_chain = len(get_tokenizer_offline(TOKENIZER_PATH, worker_idx).encode(full_text, add_special_tokens=False))
        if num_tokens_in_chain is None: 
            num_tokens_in_chain = 0
        max_tokens_across_all_chains = max(max_tokens_across_all_chains, num_tokens_in_chain)

        chain_states[chain_id] = {
            "id": chain_id, "full_text_original": full_text,
            "is_eventually_correct": is_eventually_correct,
            "current_text_for_step": "", "processed_boundaries": [0],
            "completed_thought_count": 0, "embeddings": [],
            "max_tokens_in_chain": num_tokens_in_chain
        }

    results_at_each_step_counts = []

    for current_token_limit in range(token_step_size, max_tokens_across_all_chains + token_step_size, token_step_size):
        all_new_thoughts_this_step_data = []
        all_sim_scores_this_step = []
        correct_sim_scores_this_step = []
        incorrect_sim_scores_this_step = []

        for chain_id, state in chain_states.items():
            text_for_this_chain_at_limit = get_text_up_to_n_tokens(state['full_text_original'], current_token_limit, worker_idx)
            if not text_for_this_chain_at_limit:
                state['current_text_for_step'] = ""
                continue
            state['current_text_for_step'] = text_for_this_chain_at_limit
            
            # to do: need to apply multi tokenizer approach here
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
                    all_new_thoughts_this_step_data.append({'chain_id': chain_id, 'thought_idx': thought_idx_for_chain, 'text': text_c})
                    state['completed_thought_count'] += 1
                state['processed_boundaries'] = updated_boundaries

        if not all_new_thoughts_this_step_data:
            continue

        texts_to_embed = [item['text'] for item in all_new_thoughts_this_step_data]
        embeddings_for_new_thoughts = embed_segments(texts_to_embed)

        if embeddings_for_new_thoughts is None or len(embeddings_for_new_thoughts) != len(all_new_thoughts_this_step_data):
            continue

        for i, item in enumerate(all_new_thoughts_this_step_data): item['embedding'] = embeddings_for_new_thoughts[i]

        candidate_embeddings_for_faiss = []

        for thought_data in all_new_thoughts_this_step_data:
            chain_id, thought_idx, embedding, text = thought_data['chain_id'], thought_data['thought_idx'], thought_data['embedding'], thought_data['text']

            can_potentially_prune = (thought_idx >= 2 and index_manager.get_num_embeddings() > 0)

            # print("chain_id: ", chain_id)
            # print("thought_idx: ", thought_idx)
            # print("index_manager.get_num_embeddings(): ", index_manager.get_num_embeddings())
            # print("can_potentially_prune: ", can_potentially_prune)
            # print("\n\n")

            if can_potentially_prune:
                neighbor_result = index_manager.search_nearest_neighbor(embedding, chain_id)
                if neighbor_result:
                    sim_score, _, _, _ = neighbor_result
                    all_sim_scores_this_step.append(sim_score)
                    if chain_states[chain_id]['is_eventually_correct']:
                        correct_sim_scores_this_step.append(sim_score)
                    else:
                        incorrect_sim_scores_this_step.append(sim_score)
            candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
                    
        for item_to_add in candidate_embeddings_for_faiss:
            # print('running add_embedding')
            cid_add = item_to_add['chain_id']
            index_manager.add_embedding(item_to_add['embedding'], cid_add, item_to_add['thought_idx'], item_to_add['text'])
            chain_states[cid_add]['embeddings'].append(item_to_add['embedding'])

        results_at_each_step_counts.append({'all': all_sim_scores_this_step, 'correct': correct_sim_scores_this_step, 'incorrect': incorrect_sim_scores_this_step})

    # print('results_at_each_step_counts: ', results_at_each_step_counts)
    
    # Print thoughts for each chain
    total_thoughts = 0
    for chain_id, state in chain_states.items():
        # print(f"\nChain {chain_id}: {state['completed_thought_count']} thoughts")
        total_thoughts += state['completed_thought_count']
        
        # Get the text segments for this chain
        text_for_chain = state['current_text_for_step']
        segments, _ = find_newly_completed_thoughts_optimized(
            full_text=text_for_chain,
            processed_boundaries=[0],
            worker_idx=worker_idx,
            target_phrases=TARGET_PHRASES,
            min_segment_tokens=1
        )
        
        # # Print each thought
        # for i, (start_idx, end_idx, thought_text) in enumerate(segments):
        #     print(f"  Thought {i+1}: {thought_text}")
    
    print(f"\nTotal thoughts across all chains: {total_thoughts}")
    
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
                        if line_txt.strip() == "--- Full Content ---":
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
            worker_idx=worker_idx
        )
        worker_results.append(question_results)
        per_question_results[str(iteration_num)] = question_results
    return per_question_results

def main_offline_analysis(args):
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
        'incorrect': [] # Same structure for incorrect chains
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
                    combined_scores_by_step['all'].append([])
                    combined_scores_by_step['correct'].append([])
                    combined_scores_by_step['incorrect'].append([])
                combined_scores_by_step['all'][step_idx].extend(step_results['all'])
                combined_scores_by_step['correct'][step_idx].extend(step_results['correct'])
                combined_scores_by_step['incorrect'][step_idx].extend(step_results['incorrect'])

    # Improved box plot visualization
    all_scores = combined_scores_by_step['all']
    correct_scores = combined_scores_by_step['correct']
    incorrect_scores = combined_scores_by_step['incorrect']
    num_steps = len(all_scores)
    
    # Create plots for every 20 steps
    chunk_size = 20
    for chunk_idx in range(0, num_steps, chunk_size):
        end_idx = min(chunk_idx + chunk_size, num_steps)
        chunk_all_scores = all_scores[chunk_idx:end_idx]
        chunk_correct_scores = correct_scores[chunk_idx:end_idx]
        chunk_incorrect_scores = incorrect_scores[chunk_idx:end_idx]
        
        positions = np.arange(len(chunk_all_scores))
        width = 0.25

        plt.figure(figsize=(15, 6))

        # Plot each category with an offset and color
        box1 = plt.boxplot(
            chunk_all_scores,
            positions=positions - width,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor='blue', color='blue', alpha=0.5),
            medianprops=dict(color='black'),
            showfliers=False,
            labels=['' for _ in range(len(chunk_all_scores))]
        )
        box2 = plt.boxplot(
            chunk_correct_scores,
            positions=positions,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor='green', color='green', alpha=0.5),  # GREEN for correct
            medianprops=dict(color='black'),
            showfliers=False,
            labels=['' for _ in range(len(chunk_all_scores))]
        )
        box3 = plt.boxplot(
            chunk_incorrect_scores,
            positions=positions + width,
            widths=width,
            patch_artist=True,
            boxprops=dict(facecolor='red', color='red', alpha=0.5),  # RED for incorrect
            medianprops=dict(color='black'),
            showfliers=False,
            labels=['' for _ in range(len(chunk_all_scores))]
        )

        # Set x-ticks every 5 steps (or all if < 20 steps)
        step_ticks = np.arange(0, len(chunk_all_scores), 5 if len(chunk_all_scores) > 20 else 1)
        plt.xticks(positions[step_ticks], [f'Step {i+chunk_idx+1}' for i in step_ticks], rotation=45)

        plt.xlabel('Processing Step')
        plt.ylabel('Similarity Score')
        
        # Create a more descriptive title with additional information
        model_name = args.model_arch
        dataset_name = args.dataset_name
        n_chains = args.n_chains
        n_questions = args.num_questions
        seed = args.seed
        plt.title(
            f'Similarity Score Distribution\n'
            f'{model_name} on {dataset_name}\n'
            f'Chains: {n_chains}, Questions: {n_questions}, Seed: {seed}\n'
            f'Steps {chunk_idx+1}-{end_idx}',
            pad=20
        )
        
        # Add legend with better positioning and formatting
        legend = plt.legend(
            [box1["boxes"][0], box2["boxes"][0], box3["boxes"][0]],
            ['All Chains', 'Correct Chains', 'Incorrect Chains'],
            loc='upper right',
            bbox_to_anchor=(1.0, 1.0),
            frameon=True,
            framealpha=0.9,
            edgecolor='black'
        )
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot to the output directory
        output_dir = args.output_dir if hasattr(args, 'output_dir') else 'sim_score_results'
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'box_plot_of_similarity_scores_by_processing_step_{args.model_arch}_{args.dataset_name}_{args.control_run_name}_steps_{chunk_idx+1}-{end_idx}.png')
        plt.savefig(plot_path)
        plt.close()

    # Add argument for output directory
    output_dir = args.output_dir if hasattr(args, 'output_dir') else 'sim_score_results'
    os.makedirs(output_dir, exist_ok=True)

    # Save per-question and combined results to a single JSON file
    output_json = {
        'per_question_results': all_worker_results,
        'combined_scores_by_step': combined_scores_by_step
    }
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
        python sim_score_analysis.py --model_arch QwQ-32B --base_slimsc_dir "/home/users/ntu/aqui0001/slimsc" --dataset_name gpqa_diamond --control_run_name sc_16_control --n_chains 16 --tokenizer_path /home/users/ntu/aqui0001/scratch/qwq --num_questions 50 --seed 7 --token_step_size 100 --output_dir testing
    """


    