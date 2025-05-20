import argparse
import os
import sys
import pandas as pd
import json
from tqdm import tqdm

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from prune.utils import DatasetHandler

def main_offline_analysis(args):
    print(f"Running offline analysis for {args.model_arch} on {args.dataset_name} with base slimsc dir {args.base_slimsc_dir}")

    output_dir = os.path.join(args.base_slimsc_dir, "prune/results", args.model_arch, args.dataset_name, 'ESC')
    os.makedirs(output_dir, exist_ok=True)

    base_results_dir = os.path.join(args.base_slimsc_dir, "prune/results", args.model_arch, args.dataset_name)

    # get all control runs in the format of "sc_<n_chains>_control" in the base_results_dir
    control_run_names = [f for f in os.listdir(base_results_dir) if f.startswith("sc_") and f.endswith("_control")]

    dataset_handler = DatasetHandler(args.dataset_name)

    # create a results directory

    for control_run_name in tqdm(control_run_names):
        result_df = pd.DataFrame(columns=["iteration", "question_id", "correct_answer", "final_answer", "score", "num_chains_generated", "total_tokens_generated", "window_size", "individual_answers_str", "correct_answer_in_chains"])

        control_dir = os.path.join(base_results_dir, control_run_name)
        control_eval_summary_path = os.path.join(control_dir, "evaluation_summary.csv")
        control_summaries_dir = os.path.join(control_dir, "summaries")

        if not os.path.exists(control_eval_summary_path):
            print(f"Control evaluation summary not found at: {control_eval_summary_path}")
            continue
        if not os.path.exists(control_summaries_dir):
            print(f"Control summaries directory not found at: {control_summaries_dir}")
            continue

        # get window size. W = min(2, n_chains / 8)
        n_chains = int(control_run_name.split("_")[1])
        window_size = max(2, int(n_chains / 8))  # Ensure window_size is at least 1 and is an integer
        if n_chains == 1:
            window_size = 1

        # read in csv file
        control_eval_summary = pd.read_csv(control_eval_summary_path)

        # Flag to track if we should skip this control run
        skip_control_run = False

        for index, row in control_eval_summary.iterrows():
            # get the individual_answers_str
            summary_path = os.path.join(control_summaries_dir, f"question_{row['iteration']}_summary.json")
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                    individual_answers = summary["individual_answers"]
                    # print(f"\nProcessing file: {summary_path}")
                    # print(f"num_chains_generated: {num_chains_generated}")
                    # print(f"available chains: {len(summary['chains_for_voting_details'])}")

                    final_answer = None
                    num_chains_generated = 0
                    for i in range(0, len(individual_answers), window_size):
                        window = individual_answers[i:i+window_size]
                        num_chains_generated = i + len(window)  # Count up to the current window
                        if len(set(window)) == 1:
                            final_answer = window[0]
                            break
                    if final_answer is None:
                        final_answer = row["voted_answer"]
                        num_chains_generated = len(individual_answers)

                    # get the answer from the dataset
                    score = dataset_handler.calculate_score(final_answer, row["correct_answer"])

                    # get total tokens generated
                    total_tokens_generated = 0
                    for i in range(num_chains_generated):
                        total_tokens_generated += summary["chains_for_voting_details"][i]["completion_tokens"]

                    result_df = result_df._append({
                        "iteration": row["iteration"],
                        "question_id": row["question_id"],
                        "correct_answer": row["correct_answer"],
                        "final_answer": final_answer,
                        "score": score,
                        "num_chains_generated": num_chains_generated,
                        "total_tokens_generated": total_tokens_generated,
                        "window_size": window_size,
                        "individual_answers_str": individual_answers,
                        "correct_answer_in_chains": row["correct_answer"] in individual_answers[:num_chains_generated]
                    }, ignore_index=True)

            except (FileNotFoundError, KeyError) as e:
                print(f"Error processing summary file {summary_path}: {str(e)}")
                print(f"Skipping entire control run: {control_run_name}")
                skip_control_run = True
                break

        if not skip_control_run:
            result_df.to_csv(os.path.join(output_dir, f"esc_sc_n{n_chains}_w{window_size}_results.csv"), index=False)

            # compute aggregated metrics
            accuracy = result_df["score"].mean()
            avg_total_tokens_generated = result_df["total_tokens_generated"].mean()
            avg_num_chains_generated = result_df["num_chains_generated"].mean()
            correct_answer_in_chains = result_df["correct_answer_in_chains"].mean()
            voting_efficiency = accuracy / correct_answer_in_chains if correct_answer_in_chains > 0 else 0.0

            # save aggregated metrics
            with open(os.path.join(output_dir, f"esc_sc_n{n_chains}_w{window_size}_aggregated_metrics.json"), "w") as f:
                json.dump({
                    "accuracy": accuracy,
                    "avg_total_tokens_generated": avg_total_tokens_generated,
                    "avg_num_chains_generated": avg_num_chains_generated,
                    "correct_answer_in_chains": correct_answer_in_chains,
                    "voting_efficiency": voting_efficiency,
                    "window_size": window_size,
                    "n_chains": n_chains
                }, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Self-Consistency Similarity Score Analysis")
    parser.add_argument('--base_slimsc_dir', type=str, default="slimsc", help="Base directory of the slimsc project.")
    parser.add_argument('--model_arch', type=str, default="QwQ-32B", help="Model architecture name (e.g., QwQ-32B).")
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help="Dataset name (e.g., gpqa_diamond).")

    cli_args = parser.parse_args()

    main_offline_analysis(cli_args)