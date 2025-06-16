import json
import os
import ast # For safely evaluating string representations of lists
# Removed 'random' as sampling is no longer used
from pathlib import Path # Use Pathlib for cleaner path handling

# Data analysis and visualization imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm # For accessing color maps
import numpy as np

# Define base path (as requested)
user = os.environ.get("USER", "chong032") # Default to chong032 if USER env var not set
BASE_RESULTS_PATH = Path(f"/home/users/ntu/{user}/slimsc/prune/results")

# Define the specific SC runs to analyze for each model and dataset
# Structure: {model_name: {dataset_name: [run_subdirectories]}}
SC_RUNS_TO_ANALYZE = {
    "QwQ-32B": {
        "aime": [
            "sc_2_control",
            "sc_8_control",
            "sc_16_control",
        ],
        "gpqa_diamond": [
            "sc_2_control",
            "sc_8_control",
            "sc_16_control",
            "sc_32_control",
        ],
        "aqua_rat": [
            "sc_2_control",
            "sc_8_control",
            "sc_16_control",
            "sc_32_control",
        ],
    },
    "R1-Distill-Qwen-14B": {
        "aime": [
            "sc_2_control",
            "sc_8_control",
            "sc_16_control",
            "sc_32_control",
            "sc_64_control",
        ],
        "gpqa_diamond": [
            "sc_2_control",
            "sc_8_control",
            "sc_16_control",
            "sc_32_control",
            "sc_64_control",
        ],
        "aqua_rat": [
            "sc_2_control",
            "sc_8_control",
            "sc_16_control",
            "sc_32_control",
            "sc_64_control",
        ],
    },
}

# Output directory for plots
PLOTS_OUTPUT_DIR = Path(f"/home/users/ntu/{user}/slimsc/prune/visualization/plots")


# --- Helper Function for Full Data Metric Calculation ---

def calculate_metrics_from_full_data(csv_path: Path) -> dict:
    """
    Calculates % Correct Answer Voted (Overall Accuracy) (from mean of final_score),
    % Correct in Individual Answer Candidates
    from the full data in an evaluation_summary.csv.

    Args:
        csv_path: Path to the evaluation_summary.csv file.

    Returns:
        A dictionary containing 'overall_accuracy',
        'perc_correct_in_individual_answers'.
        Returns dict with NaN values if file or data is missing or errors occur.
    """
    results = {
        "overall_accuracy": float("NaN"),
        "perc_correct_in_individual_answers": float("NaN"),
    }

    def clean_answer(ans):
        s = str(ans).strip()
        # Attempt to convert to float if it looks like a number
        try:
            # Using float handles both integers (33 -> 33.0) and floats (33.0 -> 33.0)
            return float(s)
        except ValueError:
             # If not a number, strip common artifacts if it's a letter choice
            if len(s) > 1 and s[0].isalpha() and not s[1].isalnum(): # e.g. "A.", "A)"
                return s[0].upper()
            return s.upper() # General case, convert to upper for case-insensitivity


    if not csv_path.exists():
        print(f"Warning: Evaluation summary file not found at {csv_path}")
        return results
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Warning: {csv_path} is empty.")
            return results

        total_rows = len(df)

        # --- Calculate % Correct Answer Voted (Overall Accuracy) (using final_score mean) ---
        if 'final_score' in df.columns and total_rows > 0:
            try:
                # Convert to numeric, coercing errors to NaN, then calculate the mean
                # Mean of 0s and 1s is the proportion of 1s (accuracy)
                accuracy_from_final_score = pd.to_numeric(df['final_score'], errors='coerce').mean()
                if pd.notna(accuracy_from_final_score):
                     results['overall_accuracy'] = accuracy_from_final_score * 100
                else:
                    print(f"Warning: Final score column in {csv_path.name} contains non-numeric values preventing mean calculation.")
                    print(f"Debug: Overall accuracy set to NaN for {csv_path.name} due to non-numeric final_score.")

            except Exception as e:
                 print(f"Error calculating overall accuracy from final_score mean for {csv_path.name}: {e}")
                 print(f"Debug: Overall accuracy set to NaN for {csv_path.name} due to calculation error.")
        else:
            print(f"Warning: 'final_score' column not found or data is empty in {csv_path.name}. Cannot calculate overall accuracy.")
            print(f"Debug: Overall accuracy set to NaN for {csv_path.name} due to missing column or empty data.")


        # --- Calculate % Correct in Individual Answer Candidates ---
        correct_present_count = 0
        total_rows_parsable_individual = 0

        if 'individual_answers_str' in df.columns and 'correct_answer' in df.columns:
             for _, row in df.iterrows():
                try:
                    # Use the clean_answer function defined above
                    current_correct_answer_cleaned = clean_answer(row['correct_answer'])

                    individual_answers_str = row['individual_answers_str']

                    if pd.isna(individual_answers_str) or not str(individual_answers_str).strip():
                        individual_answers_list_cleaned = []
                    else:
                        raw_list = ast.literal_eval(str(individual_answers_str))
                        # Clean every answer in the individual list
                        individual_answers_list_cleaned = [clean_answer(ans) for ans in raw_list]

                    if current_correct_answer_cleaned in individual_answers_list_cleaned:
                        correct_present_count += 1
                    total_rows_parsable_individual += 1

                except (SyntaxError, ValueError) as e:
                     # print(f"Warning: Could not parse individual_answers_str in data from {csv_path.name} for iteration: {row.get('iteration', 'N/A')}. Error: {e}")
                     pass # Suppress excessive warnings; rows with errors won't count towards the denominator for this metric
                except Exception as e:
                    print(f"Warning: Error processing row for individual answers in {csv_path.name} for iteration: {row.get('iteration', 'N/A')}: {e}")

             if total_rows_parsable_individual > 0:
                  results['perc_correct_in_individual_answers'] = (correct_present_count / total_rows_parsable_individual) * 100
             # else: It remains NaN from initialization if no rows were parsable
             else:
                 print(f"Debug: Individual percentage set to NaN for {csv_path.name} due to 0 parsable rows.")
        else:
            print(f"Warning: Missing 'individual_answers_str' or 'correct_answer' for individual answer calculation in {csv_path.name}")


        # # --- Calculate Voting Efficiency ---
        # acc = results.get('overall_accuracy')
        # perc_ind = results.get('perc_correct_in_individual_answers')

        # # Check if both values are numeric and valid (handle potential NaNs from previous steps)
        # if pd.notna(acc) and pd.notna(perc_ind):
        #      if perc_ind > 0:
        #           results['voting_efficiency'] = acc / perc_ind * 100 # Efficiency is ratio of percentages
        #      else:
        #           # If %_ind is 0, efficiency is 0.0
        #           results['voting_efficiency'] = 0.0
        # else:
        #      print(f"Debug: Efficiency set to NaN for {csv_path.name} because accuracy ({acc}) or individual percentage ({perc_ind}) was NaN.")


    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: No data in file {csv_path}")
    except Exception as e:
        print(f"An unexpected error occurred while processing {csv_path}: {e}")
        
    return results


all_sc_data = []

print("--- Starting Data Collection for SC Runs (Full Data) ---")

for model, datasets in SC_RUNS_TO_ANALYZE.items():
    print(f"\nProcessing model: {model}")
    for dataset, run_names in datasets.items():
        print(f"  Processing dataset: {dataset}")
        for run_name in run_names:
            print(f"    Processing run: {run_name}...")
            run_path = BASE_RESULTS_PATH / model / dataset / run_name
            eval_summary_path = run_path / "evaluation_summary.csv"

            # Extract n_chains from run name
            # Example: "sc_16_control" -> 16
            try:
                parts = run_name.split('_')
                if len(parts) >= 2 and parts[0] == 'sc':
                    n_chains = int(parts[1])
                else:
                    print(f"Warning: Could not extract n_chains from run name: {run_name}. Skipping.")
                    continue
            except PercentageError:
                 print(f"Warning: Could not parse n_chains from run name: {run_name}. Skipping.")
                 continue

            # Calculate metrics from full data
            metrics = calculate_metrics_from_full_data(eval_summary_path)

            # Append results
            run_data = {
                "Model": model,
                "Dataset": dataset,
                "n_chains": n_chains,
                "Correct Answer Voted": metrics.get("overall_accuracy", float("NaN")),
                "Correct Answer Present in Candidates": metrics.get("perc_correct_in_individual_answers", float("NaN")),
            }
            all_sc_data.append(run_data)
            # print(f"      Collected data: {run_data}") # Uncomment for detailed logging

print("\n--- Finished Data Collection ---")



df_sc = pd.DataFrame(all_sc_data)

if not df_sc.empty:
    # Sort by n_chains for consistent plotting order
    df_sc = df_sc.sort_values(by=['Model', 'Dataset', 'n_chains']).reset_index(drop=True)

    print("\nFull Data SC Metrics Summary:")
    # display(df_sc.style.format({
    #     "% Correct Answer Voted (Overall Accuracy)": "{:.2f}%",
    #     "% Correct Answer Present in Final Candidates": "{:.2f}%",
    # }, na_rep="N/A"))

    # Melt the DataFrame for plotting
    df_sc_melted = df_sc.melt(
        id_vars=['Model', 'Dataset', 'n_chains'],
        value_vars=['Correct Answer Voted', 'Correct Answer Present in Candidates'],
        var_name='Metric',
        value_name='Percentage'
    )

    # Add a column to distinguish between percentage metrics and efficiency for formatting
    def get_format_type(metric_name):
        if '%' in metric_name:
            return 'percentage'
        else:
            return 'efficiency'

    df_sc_melted['MetricType'] = df_sc_melted['Metric'].apply(get_format_type)

else:
    print("\nNo data collected for SC experiments.")


# --- Plotting Version 2 ---

if not df_sc_melted.empty:
    print("\n--- Generating Plots ---")

    for (model, dataset), group_df in df_sc_melted.groupby(['Model', 'Dataset']):
        print(f"  Generating plot for {model} / {dataset}...")

        group_df = group_df.sort_values(by='n_chains')
        metrics = group_df['Metric'].unique()
        n_metrics = len(metrics)
        n_chains = group_df['n_chains'].unique()
        x = np.arange(len(n_chains))
        width = 0.35 if n_metrics == 2 else 0.8 / n_metrics  # Adjust width for more metrics

        fig, ax = plt.subplots(figsize=(5, 2.4))

        # Plot each metric as a separate bar group (side by side)
        for i, metric in enumerate(metrics):
            metric_df = group_df[group_df['Metric'] == metric]
            # Align bars side by side
            ax.bar(x + (i - (n_metrics-1)/2)*width, metric_df['Percentage'], width=width, label=metric)

        ax.set_xlabel('Number of Chains', fontsize=11.5)
        ax.set_ylabel('Percentage of Questions', fontsize=11.5)
        ax.yaxis.set_label_coords(-0.1, 0.27)
        ax.set_xticks(x)
        ax.set_xticklabels(n_chains, fontsize=10.8)
        ax.tick_params(axis='y', labelsize=10.8)

        # Wrap the second legend label to two lines
        handles, labels = ax.get_legend_handles_labels()
        labels = [label.replace("Correct Answer Present in Candidates", "Correct Answer Present\nin Candidates") for label in labels]
        ax.legend(handles, labels, loc='lower right', fontsize=10.8, frameon=True, framealpha=0.9, edgecolor='black')

        ax.grid(axis='y', linestyle='--', alpha=0.7)

        max_val = group_df['Percentage'].max()
        ax.set_ylim(0, max_val * 1.15 if pd.notna(max_val) else 1)

        fig.tight_layout(pad=1.0)

        plot_filename = f"version_2_{model.replace('/', '_')}_{dataset}_sc_ideal_vs_actual_acc.png"
        plot_path = PLOTS_OUTPUT_DIR / plot_filename
        fig.savefig(plot_path)
        print(f"    Saved plot to {plot_path}")

        plt.show()
else:
    print("\nNo data to plot.")

print("\n--- Plotting Complete ---")