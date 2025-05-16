import argparse


def main_offline_analysis(args):
    base_results_dir = os.path.join(args.base_slimsc_dir, "prune/results", args.model_arch, args.dataset_name)

    # get all control runs in the format of "sc_<n_chains>_control" in the base_results_dir
    control_run_names = [f for f in os.listdir(base_results_dir) if f.startswith("sc_") and f.endswith("_control")]

    for control_run_name in control_run_names:
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
        window_size = min(2, n_chains / 8)

        # read in csv file
        control_eval_summary = pd.read_csv(control_eval_summary_path)

        
        
    
        

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline Self-Consistency Similarity Score Analysis")
    parser.add_argument('--base_slimsc_dir', type=str, default="slimsc", help="Base directory of the slimsc project.")
    parser.add_argument('--model_arch', type=str, default="QwQ-32B", help="Model architecture name (e.g., QwQ-32B).")
    parser.add_argument('--dataset_name', type=str, default="gpqa_diamond", help="Dataset name (e.g., gpqa_diamond).")
    parser.add_argument('--output_dir', type=str, default='sim_score_results', help="Directory to save similarity score results.")

    cli_args = parser.parse_args()

    main_offline_analysis(cli_args)
