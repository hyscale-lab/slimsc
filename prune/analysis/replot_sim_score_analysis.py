import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def plot_stacked_bar(data, output_path):
    # Extract data from the JSON
    thresholds = np.array(data['plot_data']['stacked_bar_plot']['thresholds'])
    correct_correct_props = np.array(data['plot_data']['stacked_bar_plot']['correct_correct_props'])
    correct_incorrect_props = np.array(data['plot_data']['stacked_bar_plot']['correct_incorrect_props'])
    incorrect_incorrect_props = np.array(data['plot_data']['stacked_bar_plot']['incorrect_incorrect_props'])

    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 2.4))
    
    # Stacked bars
    x = np.arange(len(thresholds))
    width = 0.35
    
    # Plot the stacked bars
    ax.bar(x, correct_correct_props, width, label='Correct-Correct', color='tab:blue')
    ax.bar(x, correct_incorrect_props, width, bottom=correct_correct_props, label='Correct-Incorrect', color='tab:purple')
    ax.bar(x, incorrect_incorrect_props, width, 
           bottom=[a + b for a, b in zip(correct_correct_props, correct_incorrect_props)], 
           label='Incorrect-Incorrect', color='tab:orange')
    
    # Customize the plot
    ax.set_xlabel('Similarity Threshold', fontsize=11.5)
    ax.set_ylabel('Percentage of Similar Chains', fontsize=11.5)
    ax.yaxis.set_label_coords(-0.1, 0.25)  # (x, y) in axis coordinates; adjust as needed
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds], rotation=45, fontsize=10.8)
    ax.tick_params(axis='y', labelsize=10.8)
    
    # Move legend outside the plot to save space
    ax.legend(fontsize=10.8, loc='upper left', frameon=True, framealpha=0.9, edgecolor='black')
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout with specific padding to prevent label cutoff
    plt.tight_layout(pad=1.0)
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot stacked bar chart from similarity score results")
    parser.add_argument('--input_json', type=str, required=True, help="Path to the input JSON file")
    parser.add_argument('--output_dir', type=str, default='plots', help="Directory to save the plot")
    parser.add_argument('--model_arch', type=str, required=True, help="Model architecture name")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name")
    parser.add_argument('--control_run_name', type=str, required=True, help="Name of the control run")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read the JSON file
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # Generate output path
    output_path = os.path.join(args.output_dir, 
                              f'stacked_bar_plot_{args.model_arch}_{args.dataset_name}_{args.control_run_name}.png')
    
    # Create and save the plot
    plot_stacked_bar(data, output_path)
    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    main()

    """
        python replot_sim_score_analysis.py --input_json="aime_n64/sim_score_results.json" --output_dir="replots" --"model_arch"="R1" --"dataset_name"="gpqa_diamond" --control_run_name="sc_64"
    """