# offline_pruning_analyzer.py
import os
import pandas as pd
import numpy as np
import json
import random
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple, Any

# Ensure correct relative imports if running as part of the package
try:
    from slimsc.prune.utils import DatasetHandler, count_tokens
    from slimsc.prune.utils.similarity_utils import (
        FaissIndexManager, embed_segments, find_newly_completed_thoughts,
        get_embedding_model, MIN_SEGMENT_TOKENS, TARGET_PHRASES
    )
    from slimsc.prune.evaluation.processing_similarity import calculate_mean_pairwise_similarity
    from slimsc.prune.clients import close_aiohttp_session
except ImportError:
     # Fallback for running script directly
     import sys
     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
     from slimsc.prune.utils import DatasetHandler, count_tokens
     from slimsc.prune.utils.similarity_utils import (
        FaissIndexManager, embed_segments, find_newly_completed_thoughts,
        get_embedding_model, MIN_SEGMENT_TOKENS, TARGET_PHRASES
    )
     from slimsc.prune.evaluation.processing_similarity import calculate_mean_pairwise_similarity
     from slimsc.prune.clients import close_aiohttp_session


from transformers import AutoTokenizer # For get_text_up_to_n_tokens

# --- Global Variables & Configuration ---
_tokenizer_cache_offline = {}

def get_tokenizer_offline(tokenizer_path: str):
    global _tokenizer_cache_offline
    if tokenizer_path not in _tokenizer_cache_offline:
        try:
            _tokenizer_cache_offline[tokenizer_path] = AutoTokenizer.from_pretrained(tokenizer_path)
            print(f"Tokenizer {tokenizer_path} loaded for offline analysis.")
        except Exception as e:
            print(f"Error loading tokenizer {tokenizer_path}: {e}")
            raise
    return _tokenizer_cache_offline[tokenizer_path]

def get_text_up_to_n_tokens(full_text: str, token_limit: int, tokenizer_path: str) -> str:
    if not full_text:
        return ""
    tokenizer = get_tokenizer_offline(tokenizer_path)
    token_ids = tokenizer.encode(full_text, add_special_tokens=False) # Often better for segments
    if len(token_ids) <= token_limit:
        return full_text
    truncated_token_ids = token_ids[:token_limit]
    return tokenizer.decode(truncated_token_ids, skip_special_tokens=True)

def determine_prune_target(
    chain_id_A: str, chain_id_B: str,
    pruning_strategy: str,
    chain_states: Dict[str, Dict]
) -> Optional[str]:
    state_A = chain_states[chain_id_A]
    state_B = chain_states[chain_id_B]
    prune_target_id = None

    current_thought_count_A = state_A.get('completed_thought_count', 0)
    current_thought_count_B = state_B.get('completed_thought_count', 0)

    if pruning_strategy == "random":
        prune_target_id = random.choice([chain_id_A, chain_id_B])
    elif pruning_strategy == "fewest_thoughts":
        if current_thought_count_A <= current_thought_count_B:
            prune_target_id = chain_id_A
        else:
            prune_target_id = chain_id_B
    elif pruning_strategy == "most_thoughts":
        if current_thought_count_A >= current_thought_count_B:
            prune_target_id = chain_id_A
        else:
            prune_target_id = chain_id_B
    elif pruning_strategy == "diversity":
        embeddings_A_list = state_A.get("embeddings", [])
        embeddings_B_list = state_B.get("embeddings", [])
        num_thoughts_A = len(embeddings_A_list)
        num_thoughts_B = len(embeddings_B_list)

        # calculate_mean_pairwise_similarity returns 0.0 if < 2 embeddings
        mean_sim_A = calculate_mean_pairwise_similarity(embeddings_A_list)
        mean_sim_B = calculate_mean_pairwise_similarity(embeddings_B_list)

        # Higher internal_sim is worse (less diverse per thought)
        # Penalize chains with 0 or 1 thought by giving them high internal similarity
        if num_thoughts_A < 2: internal_sim_A = float('inf') 
        else: internal_sim_A = (mean_sim_A / num_thoughts_A)
        
        if num_thoughts_B < 2: internal_sim_B = float('inf')
        else: internal_sim_B = (mean_sim_B / num_thoughts_B)

        if internal_sim_A == float('inf') and internal_sim_B == float('inf'): # Both have <2 thoughts
             # Tie-break: prune the one with fewer thoughts (0 vs 1, or both 0, or both 1)
            if num_thoughts_A <= num_thoughts_B: prune_target_id = chain_id_A
            else: prune_target_id = chain_id_B
        elif internal_sim_A > internal_sim_B:
            prune_target_id = chain_id_A
        elif internal_sim_B > internal_sim_A:
            prune_target_id = chain_id_B
        else: # InternalSim are equal (and not both inf) - Tie-break with fewer thoughts
            if num_thoughts_A <= num_thoughts_B:
                prune_target_id = chain_id_A
            else:
                prune_target_id = chain_id_B
    else:
        raise ValueError(f"Unknown pruning strategy: {pruning_strategy}")
    return prune_target_id

def process_single_question_offline_sync(
    question_info: Dict,
    chain_contents: Dict[str, str],
    chain_correctness: Dict[str, bool],
    n_chains_start: int,
    tokenizer_path_global: str,
    similarity_threshold: float,
    pruning_strategy: str,
    dataset_name: str, # For context, not directly used in core logic if not needed by utils
    token_step_size: int = 100,
    min_segment_tokens_val: int = MIN_SEGMENT_TOKENS,
    target_phrases_val: List[str] = TARGET_PHRASES
) -> List[float]:
    embedding_model = get_embedding_model()
    index_manager = FaissIndexManager(dimension=embedding_model.get_sentence_embedding_dimension())
    chain_states = {}
    max_tokens_across_all_chains = 0

    for i in range(n_chains_start):
        chain_id = f"q{question_info['iteration']}_c{i+1}"
        full_text = chain_contents.get(chain_id, "")
        is_eventually_correct = chain_correctness.get(chain_id, False)
        
        num_tokens_in_chain = count_tokens(full_text, tokenizer_path_global) if full_text else 0
        if num_tokens_in_chain is None: num_tokens_in_chain = 0
        max_tokens_across_all_chains = max(max_tokens_across_all_chains, num_tokens_in_chain)

        chain_states[chain_id] = {
            "id": chain_id, "full_text_original": full_text,
            "is_eventually_correct": is_eventually_correct,
            "current_text_for_step": "", "processed_boundaries": [0],
            "completed_thought_count": 0, "embeddings": [],
            "is_active": True, "pruned_at_token_step": -1,
            "max_tokens_in_chain": num_tokens_in_chain
        }

        # print("DEBUG: Chain states:", chain_states[chain_id])
    
    if max_tokens_across_all_chains == 0: return []
    
    results_at_each_step_counts = []

    for current_token_limit in range(token_step_size, max_tokens_across_all_chains + token_step_size, token_step_size):
        all_new_thoughts_this_step_data = []
        for chain_id, state in chain_states.items():
            if not state['is_active']: continue
            
            text_for_this_chain_at_limit = get_text_up_to_n_tokens(state['full_text_original'], current_token_limit, tokenizer_path_global)
            if not text_for_this_chain_at_limit:
                state['current_text_for_step'] = ""
                continue
            state['current_text_for_step'] = text_for_this_chain_at_limit

            new_segments, updated_boundaries = find_newly_completed_thoughts(
                full_text=state['current_text_for_step'],
                processed_boundaries=state['processed_boundaries'],
                tokenizer_path=tokenizer_path_global,
                target_phrases=target_phrases_val,
                min_segment_tokens=min_segment_tokens_val
            )

            if new_segments:
                for _s_idx, _e_idx, text_c in new_segments:
                    thought_idx_for_chain = state['completed_thought_count']
                    all_new_thoughts_this_step_data.append({'chain_id': chain_id, 'thought_idx': thought_idx_for_chain, 'text': text_c})
                    state['completed_thought_count'] += 1
                state['processed_boundaries'] = updated_boundaries

        if not all_new_thoughts_this_step_data:
            current_total_unpruned = sum(1 for s_in in chain_states.values() if s_in['is_active'])
            current_correct_unpruned = sum(1 for s_in in chain_states.values() if s_in['is_active'] and s_in['is_eventually_correct'])
            results_at_each_step_counts.append({'total': current_total_unpruned, 'correct': current_correct_unpruned})
            continue

        texts_to_embed = [item['text'] for item in all_new_thoughts_this_step_data]
        embeddings_for_new_thoughts = embed_segments(texts_to_embed)

        if embeddings_for_new_thoughts is None or len(embeddings_for_new_thoughts) != len(all_new_thoughts_this_step_data):
            current_total_unpruned = sum(1 for s_in in chain_states.values() if s_in['is_active'])
            current_correct_unpruned = sum(1 for s_in in chain_states.values() if s_in['is_active'] and s_in['is_eventually_correct'])
            results_at_each_step_counts.append({'total': current_total_unpruned, 'correct': current_correct_unpruned})
            continue

        for i, item in enumerate(all_new_thoughts_this_step_data): item['embedding'] = embeddings_for_new_thoughts[i]

        chains_to_prune_ids_this_step = set()
        candidate_embeddings_for_faiss = []

        for thought_data in all_new_thoughts_this_step_data:
            chain_id, thought_idx, embedding, text = thought_data['chain_id'], thought_data['thought_idx'], thought_data['embedding'], thought_data['text']
            if chain_id in chains_to_prune_ids_this_step or not chain_states[chain_id]['is_active']: continue

            can_potentially_prune = (thought_idx >= 2 and index_manager.get_num_embeddings() > 0)
            if can_potentially_prune:
                neighbor_result = index_manager.search_nearest_neighbor(embedding, chain_id)
                if neighbor_result:
                    sim_score, neighbor_chain_id, _, _ = neighbor_result
                    if sim_score > similarity_threshold:
                        if not chain_states[neighbor_chain_id]['is_active'] or neighbor_chain_id in chains_to_prune_ids_this_step:
                            candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
                            continue
                        prune_target_id = determine_prune_target(chain_id, neighbor_chain_id, pruning_strategy, chain_states)
                        if prune_target_id:
                            num_active_now = sum(1 for s_glob in chain_states.values() if s_glob['is_active'])
                            num_marked_to_prune = len(chains_to_prune_ids_this_step)
                            would_add_new_prune = 1 if prune_target_id not in chains_to_prune_ids_this_step else 0
                            if (num_active_now - num_marked_to_prune - would_add_new_prune) >= 1:
                                chains_to_prune_ids_this_step.add(prune_target_id)
                                if prune_target_id != chain_id:
                                    candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
                            else: candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
                        else: candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
                    else: candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
                else: candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})
            else: candidate_embeddings_for_faiss.append({'embedding': embedding, 'chain_id': chain_id, 'thought_idx': thought_idx, 'text': text})

        for pid_to_prune in chains_to_prune_ids_this_step:
            if chain_states[pid_to_prune]['is_active']:
                chain_states[pid_to_prune]['is_active'] = False
                chain_states[pid_to_prune]['pruned_at_token_step'] = current_token_limit
                index_manager.remove_chain_embeddings(pid_to_prune)

        for item_to_add in candidate_embeddings_for_faiss:
            cid_add = item_to_add['chain_id']
            if chain_states[cid_add]['is_active']:
                index_manager.add_embedding(item_to_add['embedding'], cid_add, item_to_add['thought_idx'], item_to_add['text'])
                chain_states[cid_add]['embeddings'].append(item_to_add['embedding'])
        
        current_total_unpruned = sum(1 for s_final in chain_states.values() if s_final['is_active'])
        current_correct_unpruned = sum(1 for s_final in chain_states.values() if s_final['is_active'] and s_final['is_eventually_correct'])
        results_at_each_step_counts.append({'total': current_total_unpruned, 'correct': current_correct_unpruned})

    # print(f"[DEBUG Q{question_info['iteration']}] Final results_at_each_step_counts: {results_at_each_step_counts}")

    percentages_over_steps = [(res['correct'] / res['total'] * 100.0) if res['total'] > 0 else 0.0 for res in results_at_each_step_counts]

    # print(f"[DEBUG Q{question_info['iteration']}] Percentages over steps: {percentages_over_steps}")
    return percentages_over_steps

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
    
    print("Loading embedding model...")
    get_embedding_model() # Pre-load
    print("Embedding model loaded.")
    # Load tokenizer for get_text_up_to_n_tokens and count_tokens
    print(f"Loading tokenizer from {args.tokenizer_path}...")
    get_tokenizer_offline(args.tokenizer_path) # Pre-load
    _ = count_tokens("test", args.tokenizer_path) # Ensure count_tokens also loads its tokenizer
    print("Tokenizer loaded.")


    for iteration_num in tqdm(sampled_question_iterations, desc="Processing Sampled Questions"):
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

        for strategy in args.pruning_strategies:
            for threshold in args.similarity_thresholds:
                percentages_for_this_q_config = process_single_question_offline_sync(
                    question_info=question_info_dict, chain_contents=chain_contents,
                    chain_correctness=chain_correctness, n_chains_start=n_chains_sc,
                    tokenizer_path_global=args.tokenizer_path, similarity_threshold=threshold,
                    pruning_strategy=strategy, dataset_name=args.dataset_name,
                    token_step_size=args.token_step_size
                )
                # print(f"[MAIN_DEBUG Q{iteration_num}] For {strategy} T{threshold}, got percentages: {percentages_for_this_q_config}")

                result_key = (strategy, threshold)
                if result_key not in all_results_data: all_results_data[result_key] = []
                if percentages_for_this_q_config: all_results_data[result_key].append(percentages_for_this_q_config)
    
    # print("\n--- Final Aggregation Phase ---")
    # print(f"[MAIN_DEBUG] all_results_data keys: {list(all_results_data.keys())}")

    max_steps_observed = 0
    for key in all_results_data:
        for q_result_list in all_results_data[key]:
            max_steps_observed = max(max_steps_observed, len(q_result_list))
    
    if max_steps_observed == 0:
        print("No data to plot. All questions might have been empty or processing failed.")
        if args.close_session_after_run: close_aiohttp_session() # Ensure cleanup
        return

    averaged_results = {}
    for key, list_of_q_results in all_results_data.items():
        if not list_of_q_results: continue
        padded_q_results = []
        for q_res_list in list_of_q_results:
            if not q_res_list: padded_q_results.append([np.nan] * max_steps_observed)
            else:
                padding_needed = max_steps_observed - len(q_res_list)
                padded_list = q_res_list + ([q_res_list[-1]] * padding_needed if q_res_list else [np.nan] * padding_needed) # Pad with last or NaN
                padded_q_results.append(padded_list)
        if not padded_q_results: continue
        try:
            np_array_results = np.array(padded_q_results, dtype=float)
            mean_percentages_per_step = np.nanmean(np_array_results, axis=0)
            averaged_results[key] = mean_percentages_per_step.tolist()
        except Exception as e:
            print(f"Error averaging results for {key}: {e}")
            averaged_results[key] = [np.nan] * max_steps_observed

    plt.figure(figsize=(14, 8))
    token_steps_x_axis = [(s+1) * args.token_step_size for s in range(max_steps_observed)]

    for (strategy, threshold), avg_percentages in averaged_results.items():
        if not avg_percentages or all(np.isnan(p) for p in avg_percentages): continue
        plt.plot(token_steps_x_axis, avg_percentages, marker='o', linestyle='-', markersize=4, label=f'{strategy} T={threshold:.2f}')

    plt.xlabel(f"Tokens Processed per Chain (Step Size: {args.token_step_size})")
    plt.ylabel("Avg. % of Correct Chains Remaining Unpruned")
    plt.title(f"Offline Pruning: {args.model_arch} on {args.dataset_name} ({sample_size} Qs, Seed {seed})")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust for legend
    
    plot_filename = f"offline_pruning_{args.model_arch}_{args.dataset_name}_s{seed}_n{sample_size}.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")
    # plt.show()

    output_data_for_csv = []
    for (strategy, threshold), avg_percentages in averaged_results.items():
        for i, percentage in enumerate(avg_percentages):
            token_step_val = (i + 1) * args.token_step_size
            output_data_for_csv.append({
                'strategy': strategy, 'threshold': threshold, 'token_step': token_step_val,
                'avg_percent_correct_unpruned': percentage if not np.isnan(percentage) else None
            })
    if output_data_for_csv:
        df_output = pd.DataFrame(output_data_for_csv)
        csv_filename = f"offline_pruning_metrics_{args.model_arch}_{args.dataset_name}_s{seed}_n{sample_size}.csv"
        df_output.to_csv(csv_filename, index=False)
        print(f"Aggregated metrics saved to {csv_filename}")
    
    if args.close_session_after_run: # Good practice for aiohttp if used by get_embedding_model
        import asyncio
        asyncio.run(close_aiohttp_session())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Self-Consistency Pruning Analysis")
    parser.add_argument('--base_slimsc_dir', type=str, default="slimsc", help="Base directory of the slimsc project.")
    parser.add_argument('--model_arch', type=str, default="QwQ-32B", help="Model architecture name (e.g., QwQ-32B).")
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help="Dataset name (e.g., gpqa_diamond).")
    parser.add_argument('--control_run_name', type=str, default="sc_16_control", help="Name of the control run directory.")
    parser.add_argument('--n_chains', type=int, default=16, help="Number of chains in the control run.")
    parser.add_argument('--tokenizer_path', type=str, required=True, help="Path to the HuggingFace tokenizer.")
    parser.add_argument('--num_questions', type=int, default=50, help="Number of questions to sample.")
    parser.add_argument('--seed', type=int, default=7, help="Random seed for sampling questions.")
    parser.add_argument('--token_step_size', type=int, default=100, help="Token interval for pruning simulation steps.")
    parser.add_argument('--pruning_strategies', nargs='+', default=['fewest_thoughts', 'most_thoughts', 'diversity', 'random'], help="List of pruning strategies to test.")
    parser.add_argument('--similarity_thresholds', nargs='+', type=float, default=[0.70, 0.80, 0.90, 0.95], help="List of similarity thresholds to test.")
    parser.add_argument('--close_session_after_run', action='store_true', help="Close aiohttp session after run (if embedding model uses it).")

    cli_args = parser.parse_args()
    main_offline_analysis(cli_args)