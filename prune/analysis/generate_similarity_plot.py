import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from termcolor import colored
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Start: Utility Imports ---
print(colored("Importing utility modules...", "green"))
try:
    from prune.utils import DatasetHandler
    from prune.utils.similarity_utils import (
        MIN_SEGMENT_TOKENS, TARGET_PHRASES, find_thought_boundaries)
except ImportError:
    print(colored("Could not import from slimsc.prune.utils. Falling back to project structure.", "yellow"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from slimsc.prune.utils import DatasetHandler
    from slimsc.prune.utils.similarity_utils import (
        MIN_SEGMENT_TOKENS, TARGET_PHRASES, find_thought_boundaries)
# --- End: Utility Imports ---

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    print(colored("Multiprocessing context already set.", "yellow"))

# --- Globals and Caching ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
_tokenizer_cache_offline = [None] * 256
_embedding_model_cache = [None] * 256

def get_tokenizer_offline(tokenizer_path: str, tokenizer_idx: int):
    """Loads and caches a tokenizer instance for a given worker."""
    if tokenizer_idx >= len(_tokenizer_cache_offline):
        raise ValueError(f"Tokenizer index {tokenizer_idx} exceeds max cache size.")
    if _tokenizer_cache_offline[tokenizer_idx] is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            _tokenizer_cache_offline[tokenizer_idx] = tokenizer
        except Exception as e:
            print(f"Error loading tokenizer {tokenizer_path} for worker {tokenizer_idx}: {e}")
            raise
    return _tokenizer_cache_offline[tokenizer_idx]

def get_embedding_model(model_idx: int, device: str):
    """Loads and caches a SentenceTransformer model for a given worker on a specific device."""
    if model_idx >= len(_embedding_model_cache):
        raise ValueError(f"Model index {model_idx} exceeds max cache size.")
    if _embedding_model_cache[model_idx] is None:
        print(colored(f"Worker {model_idx}: Loading embedding model onto device '{device}'...", "cyan"))
        _embedding_model_cache[model_idx] = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    return _embedding_model_cache[model_idx]

def embed_segments(segments: list[str], model: SentenceTransformer) -> np.ndarray:
    """Encodes a list of text segments into embeddings using the provided model."""
    if not segments:
        return np.array([])
    return model.encode(segments, convert_to_tensor=False, show_progress_bar=False)

def find_newly_completed_thoughts_optimized(
    full_text: str,
    processed_boundaries: list[int],
    tokenizer_path: str,
    worker_idx: int,
    target_phrases: list[str] = TARGET_PHRASES,
    min_segment_tokens: int = MIN_SEGMENT_TOKENS
) -> tuple[list[tuple[int, int, str]], list[int]]:
    """Identifies new, complete thought segments in text based on specified boundary phrases."""
    if not full_text:
        return [], processed_boundaries

    all_current_boundaries = find_thought_boundaries(full_text, target_phrases)
    newly_completed_segments = []
    new_segment_starts_processed_this_call = []
    current_processed_boundaries = sorted(list(set(processed_boundaries)))
    last_processed_start = current_processed_boundaries[-1] if current_processed_boundaries else -1

    for i in range(len(all_current_boundaries) - 1):
        boundary_start, boundary_end = all_current_boundaries[i], all_current_boundaries[i+1]
        
        if boundary_start >= last_processed_start and boundary_start not in new_segment_starts_processed_this_call:
             if boundary_start not in current_processed_boundaries:
                  segment_text = full_text[boundary_start:boundary_end].strip()
                  if not segment_text:
                       continue
                  
                  tokenizer = get_tokenizer_offline(tokenizer_path, worker_idx)
                  num_tokens = len(tokenizer.encode(segment_text, add_special_tokens=False))
                  
                  if num_tokens >= min_segment_tokens:
                      newly_completed_segments.append((boundary_start, boundary_end, segment_text))
                      new_segment_starts_processed_this_call.append(boundary_start)

    updated_processed_boundaries = sorted(list(set(current_processed_boundaries + new_segment_starts_processed_this_call)))
    return newly_completed_segments, updated_processed_boundaries

def get_text_up_to_n_tokens(full_text: str, token_limit: int, tokenizer_path: str, tokenizer_idx: int) -> str:
    """Truncates text to a specified maximum number of tokens."""
    if not full_text:
        return ""
    tokenizer = get_tokenizer_offline(tokenizer_path, tokenizer_idx)
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if len(token_ids) <= token_limit:
        return full_text
    truncated_token_ids = token_ids[:token_limit+1]
    return tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

def process_single_question(
    question_info: dict,
    chain_contents: dict[str, str],
    chain_correctness: dict[str, bool],
    chain_extracted_answers: dict[str, str],
    n_chains: int,
    token_step_size: int,
    tokenizer_path: str,
    worker_idx: int,
    device: str,
) -> list[dict]:
    """Analyzes a single question's chains to compute similarity scores over time."""
    from slimsc.prune.utils.similarity_utils import FaissIndexManager
    
    embedding_model = get_embedding_model(model_idx=worker_idx, device=device)
    index_manager = FaissIndexManager(dimension=embedding_model.get_sentence_embedding_dimension())
    
    chain_states = {}
    max_tokens_across_all_chains = 0

    for i in range(n_chains):
        chain_id = f"q{question_info['iteration']}_c{i+1}"
        full_text = chain_contents.get(chain_id, "")
        tokenizer = get_tokenizer_offline(tokenizer_path, worker_idx)
        num_tokens_in_chain = len(tokenizer.encode(full_text, add_special_tokens=False))
        max_tokens_across_all_chains = max(max_tokens_across_all_chains, num_tokens_in_chain)

        chain_states[chain_id] = {
            "id": chain_id,
            "full_text_original": full_text,
            "is_eventually_correct": chain_correctness.get(chain_id, False),
            "extracted_answer": chain_extracted_answers.get(chain_id, ""),
            "processed_boundaries": [0],
            "completed_thought_count": 0,
        }

    results_at_each_step = []

    for current_token_limit in range(token_step_size, max_tokens_across_all_chains + token_step_size, token_step_size):
        all_new_thoughts_this_step = []
        for chain_id, state in chain_states.items():
            text_for_step = get_text_up_to_n_tokens(state['full_text_original'], current_token_limit, tokenizer_path, worker_idx)
            if not text_for_step:
                continue
            
            new_segments, updated_boundaries = find_newly_completed_thoughts_optimized(
                text_for_step, state['processed_boundaries'], tokenizer_path, worker_idx, min_segment_tokens=1
            )

            if new_segments:
                for _, _, text_content in new_segments:
                    all_new_thoughts_this_step.append({
                        'chain_id': chain_id,
                        'thought_idx': state['completed_thought_count'],
                        'text': text_content,
                        'is_correct': state['is_eventually_correct'],
                        'extracted_answer': state['extracted_answer']
                    })
                    state['completed_thought_count'] += 1
                state['processed_boundaries'] = updated_boundaries

        if not all_new_thoughts_this_step:
            continue

        texts_to_embed = [item['text'] for item in all_new_thoughts_this_step]
        new_embeddings = embed_segments(texts_to_embed, model=embedding_model)

        step_scores = {key: [] for key in ['correct_to_correct', 'correct_to_incorrect', 'incorrect_to_correct', 'incorrect_to_incorrect']}
        
        for i, thought_data in enumerate(all_new_thoughts_this_step):
            thought_data['embedding'] = new_embeddings[i]
            
            can_potentially_prune = (thought_data['thought_idx'] >= 2 and index_manager.get_num_embeddings() > 0)

            if can_potentially_prune:
                neighbor = index_manager.search_nearest_neighbor(thought_data['embedding'], thought_data['chain_id'])
                if neighbor:
                    sim_score, neighbor_chain_id, _, _ = neighbor
                    pair_identifier = tuple(sorted((thought_data['chain_id'], neighbor_chain_id)))
                    
                    is_correct = thought_data['is_correct']
                    neighbor_is_correct = chain_states[neighbor_chain_id]['is_eventually_correct']

                    if is_correct and neighbor_is_correct:
                        step_scores['correct_to_correct'].append((pair_identifier, sim_score))
                    elif is_correct and not neighbor_is_correct:
                        step_scores['correct_to_incorrect'].append((pair_identifier, sim_score))
                    elif not is_correct and neighbor_is_correct:
                        step_scores['incorrect_to_correct'].append((pair_identifier, sim_score))
                    else:
                        step_scores['incorrect_to_incorrect'].append((pair_identifier, sim_score))

            index_manager.add_embedding(thought_data['embedding'], thought_data['chain_id'], thought_data['thought_idx'], thought_data['text'])
        
        results_at_each_step.append(step_scores)

    return results_at_each_step

def process_question_worker(args_tuple):
    """Worker function to process a subset of questions."""
    chosen_question_iterations, worker_idx, sampled_df, control_summaries_dir, control_chains_dir, args = args_tuple
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    per_question_results = {}
    dataset_handler = DatasetHandler(dataset_name=args.dataset_name)
    
    pbar = tqdm(chosen_question_iterations, desc=f"Worker {worker_idx} on device '{device}'", position=worker_idx)
    for iteration_num in pbar:
        question_row = sampled_df[sampled_df['iteration'] == iteration_num].iloc[0]
        correct_answer_ref = question_row['correct_answer']
        
        chain_contents, chain_correctness, individual_chain_answers = {}, {}, {}
        
        summary_path = os.path.join(control_summaries_dir, f"question_{iteration_num}_summary.json")
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                q_summary = json.load(f)
            if "chains_for_voting_details" in q_summary:
                for detail in q_summary["chains_for_voting_details"]:
                    if detail.get("chain_index") is not None:
                        individual_chain_answers[detail["chain_index"]] = detail.get("extracted_answer", "")

        for i in range(1, args.n_chains + 1):
            chain_id = f"q{iteration_num}_c{i}"
            chain_file_path = os.path.join(control_chains_dir, f"question_{iteration_num}_chain_{i}_used_for_voting.txt")
            
            if os.path.exists(chain_file_path):
                with open(chain_file_path, 'r', encoding='utf-8') as f:
                    lines = f.read().split("--- Reasoning Content ---")
                    chain_contents[chain_id] = lines[1].strip() if len(lines) > 1 else ""
            else:
                chain_contents[chain_id] = ""

            extracted_ans = individual_chain_answers.get(i)
            is_correct = dataset_handler.calculate_score(extracted_ans, correct_answer_ref) == 1.0 if extracted_ans is not None else False
            chain_correctness[chain_id] = is_correct

        question_info = {'iteration': iteration_num}
        question_results = process_single_question(
            question_info, chain_contents, chain_correctness, individual_chain_answers,
            args.n_chains, args.token_step_size, args.tokenizer_path, worker_idx, device
        )
        per_question_results[str(iteration_num)] = question_results
        
    return per_question_results

def plot_final_stacked_bar(data: dict, output_path: str):
    """Generates and saves the final stacked bar plot."""
    thresholds = data['thresholds']
    correct_correct_props = np.array(data['correct_correct_props'])
    correct_incorrect_props = np.array(data['correct_incorrect_props'])
    incorrect_incorrect_props = np.array(data['incorrect_incorrect_props'])

    fig, ax = plt.subplots(figsize=(6, 4.5)) # Slightly wider for more labels
    x = np.arange(len(thresholds))
    width = 0.6

    ax.bar(x, correct_correct_props, width, label='Correct-Correct', color='tab:blue')
    ax.bar(x, correct_incorrect_props, width, bottom=correct_correct_props, label='Correct-Incorrect', color='tab:purple')
    ax.bar(x, incorrect_incorrect_props, width, bottom=[a + b for a, b in zip(correct_correct_props, correct_incorrect_props)], label='Incorrect-Incorrect', color='tab:orange')

    ax.set_xlabel('Similarity Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Similar Pairs', fontsize=12)
    ax.set_xticks(x)
    # Use str(t) to handle mixed precision labels correctly
    ax.set_xticklabels([str(t) for t in thresholds], rotation=45, ha="right", fontsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=11, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 100)

    plt.tight_layout(pad=1.0)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(colored(f"Stacked bar plot saved successfully to: {output_path}", "green"))

def main(args):
    """Main function to run the analysis and generate the plot."""
    # Get the absolute path to the directory where the script is located.
    # If the script is at ~/slimsc/prune/analysis/generate_similarity_plot.py,
    # script_dir will be ~/slimsc/prune/analysis/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the output directory path relative to the script's directory.
    # This ensures the base is always ~/slimsc/prune/analysis/
    output_dir = os.path.join(script_dir, args.model_arch, args.dataset_name, args.control_run_name)
    
    os.makedirs(output_dir, exist_ok=True)
    stacked_bar_plot_path = os.path.join(output_dir, 'stacked_bar_plot_granular.png')

    control_eval_summary_path = os.path.join(args.control_run_dir, "evaluation_summary.csv")
    control_chains_dir = os.path.join(args.control_run_dir, "individual_chains")
    control_summaries_dir = os.path.join(args.control_run_dir, "summaries")

    for path in [control_eval_summary_path, control_chains_dir, control_summaries_dir]:
        if not os.path.exists(path):
            print(colored(f"Error: Required input path not found: {path}", "red"))
            sys.exit(1)

    df_full_control = pd.read_csv(control_eval_summary_path)
    if args.num_questions is None:
        sample_size = len(df_full_control)
    else:
        sample_size = min(args.num_questions, len(df_full_control))
    sampled_df = df_full_control.sample(n=sample_size, random_state=args.seed)
    sampled_question_iterations = sampled_df['iteration'].tolist()
    print(f"Sampled {len(sampled_question_iterations)} questions (seed={args.seed}) from {args.control_run_dir}.")

    if args.num_workers is None:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            num_workers = torch.cuda.device_count()
            print(colored(f"Auto-detected {num_workers} GPUs. Using {num_workers} worker processes.", "green"))
        else:
            num_workers = max(1, os.cpu_count() // 2)
            print(colored(f"No GPUs detected. Using {num_workers} worker processes.", "yellow"))
    else:
        num_workers = args.num_workers

    # num_workers = min(num_workers, len(sampled_question_iterations))
    
    worker_args_list = [
        (sampled_question_iterations[i::num_workers], i, sampled_df, control_summaries_dir, control_chains_dir, args)
        for i in range(num_workers)
    ]

    all_worker_results = []
    print(f"Starting analysis with {num_workers} parallel workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_question_worker, arg_tuple) for arg_tuple in worker_args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                all_worker_results.append(future.result())
            except Exception as e:
                print(colored(f"A worker process failed: {e}", "red"))

    combined_scores_by_step = {key: [] for key in ['correct_to_correct', 'correct_to_incorrect', 'incorrect_to_correct', 'incorrect_to_incorrect']}
    
    max_steps = 0
    for worker_result in all_worker_results:
        for question_results in worker_result.values():
            max_steps = max(max_steps, len(question_results))
            
    for key in combined_scores_by_step:
        combined_scores_by_step[key] = [[] for _ in range(max_steps)]

    for worker_result in all_worker_results:
        for question_results in worker_result.values():
            for step_idx, step_results in enumerate(question_results):
                for key in step_results:
                    combined_scores_by_step[key][step_idx].extend(step_results[key])

    # MODIFICATION: Use the specified granular thresholds.
    thresholds = [0.90, 0.92, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999]
    
    correct_correct_props, correct_incorrect_props, incorrect_incorrect_props = [], [], []
    start_step_for_analysis = 20
    
    for threshold in thresholds:
        seen_pairs = set()
        
        all_cc_pairs = {}
        for step_scores in combined_scores_by_step['correct_to_correct'][start_step_for_analysis:]:
            for pair, score in step_scores:
                if pair not in seen_pairs:
                    all_cc_pairs[pair] = score
        
        all_ci_pairs = {}
        for step_scores_list in [combined_scores_by_step['correct_to_incorrect'][start_step_for_analysis:], combined_scores_by_step['incorrect_to_correct'][start_step_for_analysis:]]:
            for step_scores in step_scores_list:
                for pair, score in step_scores:
                    if pair not in seen_pairs:
                         all_ci_pairs[pair] = score

        all_ii_pairs = {}
        for step_scores in combined_scores_by_step['incorrect_to_incorrect'][start_step_for_analysis:]:
            for pair, score in step_scores:
                if pair not in seen_pairs:
                    all_ii_pairs[pair] = score
        
        seen_pairs.update(all_cc_pairs.keys(), all_ci_pairs.keys(), all_ii_pairs.keys())

        cc_count = sum(1 for score in all_cc_pairs.values() if score >= threshold)
        ci_count = sum(1 for score in all_ci_pairs.values() if score >= threshold)
        ii_count = sum(1 for score in all_ii_pairs.values() if score >= threshold)
        
        total_count = cc_count + ci_count + ii_count
        
        if total_count == 0:
            correct_correct_props.append(0)
            correct_incorrect_props.append(0)
            incorrect_incorrect_props.append(0)
        else:
            correct_correct_props.append((cc_count / total_count) * 100)
            correct_incorrect_props.append((ci_count / total_count) * 100)
            incorrect_incorrect_props.append((ii_count / total_count) * 100)

    plot_data = {
        'thresholds': thresholds,
        'correct_correct_props': correct_correct_props,
        'correct_incorrect_props': correct_incorrect_props,
        'incorrect_incorrect_props': incorrect_incorrect_props
    }
    plot_final_stacked_bar(plot_data, stacked_bar_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run similarity analysis and generate a stacked bar plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--control_run_dir', type=str, required=True,
                        help="Path to the folder containing evaluation_summary.csv, individual_chains/, and summaries/.")
    parser.add_argument('--model_arch', type=str, required=True,
                        help="Model architecture name (e.g., 'R1-Distill-Qwen-14B'). Used for output path.")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Dataset name (e.g., 'aime'). Used for output path.")
    parser.add_argument('--control_run_name', type=str, required=True,
                        help="Name of the control run (e.g., 'sc_16_control'). Used for output path.")
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help="Path to the HuggingFace tokenizer directory.")
    parser.add_argument('--n_chains', type=int, required=True,
                        help="Number of self-consistency chains in the control run.")
    parser.add_argument('--num_questions', type=int,
                        help="Number of questions to sample for the analysis.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for sampling questions.")
    parser.add_argument('--token_step_size', type=int, default=100,
                        help="Token interval for each analysis step.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="Number of parallel worker processes. If not set, defaults to the number of GPUs, or half the CPU cores if no GPUs are available.")

    cli_args = parser.parse_args()
    main(cli_args)