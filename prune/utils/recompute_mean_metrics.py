import os
import pandas as pd
import numpy as np
import json
import argparse
import glob
import logging
import collections.abc

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[logging.StreamHandler()]
)

def flatten_dict(d, parent_key='', sep='_'):
    """ Flattens a nested dictionary. """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def calculate_and_save_mean_stats(base_run_dir: str):
    """
    Finds all aggregated_metrics.json files in run* subdirectories,
    calculates the mean and std dev, and saves to mean_aggregated_metrics.json.
    """
    logging.info(f"Recalculating mean stats for base directory: {base_run_dir}")
    if not os.path.isdir(base_run_dir):
        logging.error(f"Provided path is not a directory: {base_run_dir}")
        return

    run_dirs = glob.glob(os.path.join(base_run_dir, "run*"))
    if not run_dirs:
        logging.warning(f"No 'run*' subdirectories found in {base_run_dir}. Nothing to do.")
        return
    
    all_metrics_data = []
    first_run_config = None
    for run_dir in sorted(run_dirs):
        metrics_file = os.path.join(run_dir, "aggregated_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'metrics' in data and isinstance(data['metrics'], dict):
                        # Capture config from the first valid run file
                        if first_run_config is None and 'config' in data:
                            first_run_config = data.get('config')
                        
                        flat_metrics = flatten_dict(data['metrics'])
                        all_metrics_data.append(flat_metrics)
                        logging.info(f"Successfully processed metrics from {metrics_file}")
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not read or parse {metrics_file}: {e}")
    
    if not all_metrics_data:
        logging.warning("No valid aggregated_metrics.json files found to average. Skipping.")
        return

    df = pd.DataFrame(all_metrics_data)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols:
        logging.warning("No numeric metrics found to average. Skipping.")
        return

    mean_stats = df[numeric_cols].mean().to_dict()
    std_stats = df[numeric_cols].std().to_dict()
    
    # This renaming logic is specific to the original script, so we replicate it
    if 'mean_mean_kv_cache_usage_per_question_perc' in mean_stats:
        mean_stats['mean_mean_mean_kv_cache_usage_per_question_perc'] = mean_stats.pop('mean_mean_kv_cache_usage_per_question_perc')
    if 'mean_max_kv_cache_usage_per_question_perc' in mean_stats:
        mean_stats['mean_mean_max_kv_cache_usage_per_question_perc'] = mean_stats.pop('mean_max_kv_cache_usage_per_question_perc')
    
    final_mean_metrics = {
        "num_runs_aggregated": len(df),
        "mean": {k: v for k, v in mean_stats.items() if pd.notna(v)},
        "std_dev": {f"{k}_std": v for k, v in std_stats.items() if pd.notna(v)},
        "config": first_run_config
    }

    output_path = os.path.join(base_run_dir, "mean_aggregated_metrics.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_mean_metrics, f, indent=2)
        logging.info(f"Successfully saved mean stats for {len(df)} runs to {output_path}")
    except (IOError, TypeError) as e:
        logging.error(f"Failed to save mean stats to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recompute the mean_aggregated_metrics.json for a base run directory from all its run*/aggregated_metrics.json files."
    )
    parser.add_argument(
        '--base_run_dir',
        type=str,
        required=True,
        help='The path to the base run directory containing run* subdirectories (e.g., "results/model/dataset/config").'
    )
    args = parser.parse_args()

    calculate_and_save_mean_stats(args.base_run_dir)