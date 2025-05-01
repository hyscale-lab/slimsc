#!/usr/bin/env python
# prune/visualization/get_model_dataset_plots.py

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

def collect_metrics_data(results_dir: str = "../results") -> pd.DataFrame:
    """
    Scans the results directory and collects metrics from all aggregated_metrics.json files.
    
    Directory structure expected:
    results/
        model_name_1/
            dataset_1/
                strategy_config_1/
                    aggregated_metrics.json
                strategy_config_2/
                    aggregated_metrics.json
                ...
            dataset_2/
                ...
        model_name_2/
            ...
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        DataFrame containing all metrics with model, dataset, and strategy information
    """
    data_rows = []
    
    # Convert to absolute path if provided as relative
    results_path = Path(results_dir).resolve()
    print(f"Scanning directory: {results_path}")
    
    # Walk through the directory structure
    for model_dir in results_path.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        print(f"Processing model: {model_name}")
        
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
                
            dataset_name = dataset_dir.name
            print(f"  Dataset: {dataset_name}")
            
            for strategy_dir in dataset_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue
                    
                strategy_name = strategy_dir.name
                metrics_file = strategy_dir / "aggregated_metrics.json"
                
                if metrics_file.exists():
                    print(f"    Found metrics in: {strategy_name}")
                    
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                            
                        # Create a row with metadata and metrics
                        row = {
                            'model': model_name,
                            'dataset': dataset_name,
                            'strategy': strategy_name,
                            'strategy_path': str(strategy_dir)
                        }
                        
                        # Extract pruning method from the folder name
                        if 'fewest' in strategy_name:
                            row['pruning_method'] = 'fewest_thoughts'
                        elif 'most' in strategy_name:
                            row['pruning_method'] = 'most_thoughts'
                        elif 'diversity' in strategy_name:
                            row['pruning_method'] = 'diversity'
                        elif 'sc_' in strategy_name:
                            # Extract the number from sc_X_control or sc_X variants
                            sc_match = re.search(r'sc_(\d+)', strategy_name)
                            if sc_match:
                                sc_num = sc_match.group(1)
                                row['pruning_method'] = f'sc_control_{sc_num}'
                            else:
                                row['pruning_method'] = 'sc_control'
                        else:
                            row['pruning_method'] = 'unknown'
                        
                        # Extract threshold from the folder name
                        # Format: sim_prune_METHOD_nX_threshY.YY
                        threshold_match = re.search(r'thresh(\d+\.\d+)', strategy_name)
                        if threshold_match:
                            row['threshold'] = float(threshold_match.group(1))
                        else:
                            # Fallback for other formats
                            threshold_match = re.search(r'_(\d+\.\d+)$', strategy_name)
                            if threshold_match:
                                row['threshold'] = float(threshold_match.group(1))
                        
                        # Handle nested metrics structure
                        if 'metrics' in metrics_data and isinstance(metrics_data['metrics'], dict):
                            # Extract overall_accuracy from nested structure
                            if 'overall_accuracy' in metrics_data['metrics']:
                                try:
                                    row['overall_accuracy'] = float(metrics_data['metrics']['overall_accuracy'])
                                except (ValueError, TypeError):
                                    print(f"    Warning: Invalid overall_accuracy in {metrics_file}")
                                    row['overall_accuracy'] = None
                            
                            # Add all other metrics from the nested structure
                            for metric_key, metric_value in metrics_data['metrics'].items():
                                if metric_key != 'overall_accuracy':  # Already handled above
                                    try:
                                        if isinstance(metric_value, (int, float, str, bool)):
                                            row[metric_key] = metric_value
                                        elif isinstance(metric_value, (dict, list)):
                                            # Store complex structures as JSON strings
                                            row[metric_key] = json.dumps(metric_value)
                                    except Exception as e:
                                        print(f"    Warning: Could not process metric {metric_key}: {e}")
                        
                        # Also add top-level metrics for backward compatibility
                        for key, value in metrics_data.items():
                            if key != 'metrics':  # Avoid duplicating the metrics we already processed
                                row[f"top_{key}"] = value
                        
                        data_rows.append(row)
                    except Exception as e:
                        print(f"    Error reading {metrics_file}: {e}")
    
    # Create DataFrame from collected data
    if data_rows:
        df = pd.DataFrame(data_rows)
        print(f"Collected data from {len(df)} strategy configurations")
        return df
    else:
        print("No metrics data found")
        return pd.DataFrame()

def save_metrics_data(df: pd.DataFrame, output_file: str = "metrics_summary.csv") -> None:
    """
    Saves the collected metrics data to a CSV file.
    
    Args:
        df: DataFrame containing the metrics data
        output_file: Path to save the CSV file
    """
    if df.empty:
        print("No data to save")
        return
        
    df.to_csv(output_file, index=False)
    print(f"Metrics data saved to {output_file}")

def plot_threshold_vs_accuracy(df: pd.DataFrame, 
                               model_name: Optional[str] = None, 
                               dataset_name: Optional[str] = None,
                               save_path: Optional[str] = None,
                               results_dir: str = "../results") -> None:
    """
    Creates a plot showing threshold vs accuracy for different pruning methods.
    
    Args:
        df: DataFrame containing the metrics data
        model_name: Optional filter for a specific model
        dataset_name: Optional filter for a specific dataset
        save_path: Optional explicit path to save the plot
        results_dir: Path to the results directory for auto-saving plots
    """
    if df.empty:
        print("No data to plot")
        return
    
    # Filter data if model or dataset specified
    plot_df = df.copy()
    if model_name:
        plot_df = plot_df[plot_df['model'] == model_name]
    if dataset_name:
        plot_df = plot_df[plot_df['dataset'] == dataset_name]
    
    if plot_df.empty:
        print(f"No data after filtering for model={model_name}, dataset={dataset_name}")
        return
    
    # Check for required columns
    if 'threshold' not in plot_df.columns or 'pruning_method' not in plot_df.columns:
        print("Required columns 'threshold' or 'pruning_method' not found in data")
        return
    
    if 'overall_accuracy' not in plot_df.columns:
        print(f"Accuracy column 'overall_accuracy' not found in data for {model_name}/{dataset_name}")
        print(f"Available columns: {plot_df.columns.tolist()}")
        return
    
    # Ensure overall_accuracy is numeric
    plot_df['overall_accuracy'] = pd.to_numeric(plot_df['overall_accuracy'], errors='coerce')
    
    # Drop rows with missing accuracy values
    if plot_df['overall_accuracy'].isna().any():
        print(f"Warning: Dropping {plot_df['overall_accuracy'].isna().sum()} rows with missing accuracy values")
        plot_df = plot_df.dropna(subset=['overall_accuracy'])
        
    if plot_df.empty:
        print("No data with valid accuracy values to plot")
        return
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Filter for similarity methods
    similarity_df = plot_df[plot_df['pruning_method'].isin(['fewest_thoughts', 'most_thoughts', 'diversity'])]
    
    # Define color map for SC control lines
    sc_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Plot a line for each SC control version
    sc_control_methods = [method for method in plot_df['pruning_method'].unique() 
                          if method.startswith('sc_control')]
    
    for i, sc_method in enumerate(sc_control_methods):
        sc_control_df = plot_df[plot_df['pruning_method'] == sc_method]
        if not sc_control_df.empty:
            sc_control_accuracy = sc_control_df['overall_accuracy'].mean()
            color_idx = i % len(sc_colors)
            
            # Extract number from the SC control method name for display
            sc_num = sc_method.replace('sc_control_', '')
            label = f'SC Control {sc_num}: {sc_control_accuracy:.2f}'
            
            plt.axhline(y=sc_control_accuracy, color=sc_colors[color_idx], 
                       linestyle='--', label=label)
            print(f"Plotting {sc_method} with accuracy {sc_control_accuracy:.4f}")
    
    # Plot each pruning method
    if not similarity_df.empty:
        for method, color in zip(['fewest_thoughts', 'most_thoughts', 'diversity'], ['blue', 'green', 'orange']):
            method_df = similarity_df[similarity_df['pruning_method'] == method]
            if not method_df.empty:
                # Group by threshold and calculate mean accuracy
                grouped = method_df.groupby('threshold')['overall_accuracy'].mean().reset_index()
                # Sort by threshold for proper line plot
                grouped = grouped.sort_values('threshold')
                
                # Format method name for display (make it more readable)
                display_name = method.replace('_', ' ').title()
                
                plt.plot(grouped['threshold'], grouped['overall_accuracy'], 
                        marker='o', linestyle='-', color=color, 
                        label=f'{display_name}: {grouped["overall_accuracy"].max():.2f}')
    
    # Add labels and title
    model_title = f"Model: {model_name}" if model_name else "All Models"
    dataset_title = f"Dataset: {dataset_name}" if dataset_name else "All Datasets"
    plt.title(f'Threshold vs Accuracy for Different Pruning Methods\n{model_title}, {dataset_title}')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust axis for better visualization
    plt.ylim(bottom=max(0, plt.ylim()[0] * 0.9))  # Start y-axis from 0 or 90% of minimum, whichever is higher
    
    # Generate a filename or use provided save_path
    if save_path:
        plot_save_path = save_path
    else:
        # Create a save path within the results directory structure
        results_path = Path(results_dir).resolve()
        if model_name and dataset_name:
            # Save in model/dataset directory
            plot_dir = results_path / model_name / dataset_name
            if not plot_dir.exists():
                print(f"Warning: Directory {plot_dir} doesn't exist. Creating it...")
                plot_dir.mkdir(parents=True, exist_ok=True)
            
            plot_filename = f"threshold_vs_accuracy_{model_name}_{dataset_name}.png"
            plot_save_path = plot_dir / plot_filename
        else:
            # Save in results directory for overall plots
            plot_filename = "threshold_vs_accuracy_overall.png"
            plot_save_path = results_path / plot_filename
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(plot_save_path, dpi=300)
    print(f"Plot saved to {plot_save_path}")
    
    # Display the plot
    plt.show()

def main():
    # Set the path to your results directory
    results_dir = "../results"
    
    # Collect metrics data
    metrics_df = collect_metrics_data(results_dir)
    
    if not metrics_df.empty:
        # Generate overall plot combining all data
        # plot_threshold_vs_accuracy(metrics_df, results_dir=results_dir)
        
        # Generate plots for each model/dataset combination
        unique_models = metrics_df['model'].unique()
        unique_datasets = metrics_df['dataset'].unique()
        
        for model in unique_models:
            for dataset in unique_datasets:
                plot_threshold_vs_accuracy(
                    metrics_df, 
                    model_name=model, 
                    dataset_name=dataset,
                    results_dir=results_dir
                )

if __name__ == "__main__":
    main()
