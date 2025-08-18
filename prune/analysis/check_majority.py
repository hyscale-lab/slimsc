import csv
import json
import sys
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import argparse

def analyze_csv_file(csv_filepath):
    """
    Analyzes a single CSV file to get counts.
    Returns (total_questions, correct_answer_present_count, correct_answer_highest_proportion_count)
    or None if the file cannot be processed.
    """
    total_questions = 0
    correct_answer_present_count = 0
    correct_answer_highest_proportion_count = 0

    try:
        with open(csv_filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            required_columns = ['correct_answer', 'individual_answers_str']
            if not all(col in reader.fieldnames for col in required_columns):
                # print(f"Warning: File '{csv_filepath}' missing required columns. Skipping.") # Keep print for individual file skip
                return None

            for row in reader:
                total_questions += 1

                correct_answer = row.get('correct_answer')
                individual_answers_str = row.get('individual_answers_str')

                # Basic validation for crucial data points
                if not correct_answer or individual_answers_str is None:
                     # print(f"Warning: Missing 'correct_answer' or 'individual_answers_str' in row from {csv_filepath}. Skipping row.")
                     continue # Skip rows with missing crucial data

                try:
                    individual_answers_list = json.loads(individual_answers_str)
                except (json.JSONDecodeError, TypeError):
                    # print(f"Warning: Could not parse 'individual_answers_str' in row from {csv_filepath}. Skipping row.")
                    continue

                if not isinstance(individual_answers_list, list):
                    # print(f"Warning: 'individual_answers_str' did not parse into a list in row from {csv_filepath}. Skipping row.")
                    continue

                # Check if correct_answer is present
                is_present = correct_answer in individual_answers_list
                if is_present:
                    correct_answer_present_count += 1

                # Check if correct_answer has the highest proportion (new majority definition)
                is_highest_proportion = False
                if individual_answers_list: # Only check if the list is not empty
                    counts = Counter(individual_answers_list)

                    # Find the maximum count among all answers
                    # If counts is not empty, max(counts.values()) is safe
                    max_count = 0
                    if counts:
                         max_count = max(counts.values())

                    # Check if the correct_answer is in the counts AND its count equals the max_count
                    if correct_answer in counts and counts.get(correct_answer, 0) == max_count:
                         is_highest_proportion = True

                if is_highest_proportion:
                    correct_answer_highest_proportion_count += 1 # Count questions where correct answer was highest freq (majority)

    except FileNotFoundError:
        print(f"Error: File not found '{csv_filepath}'. Skipping.")
        return None
    except Exception as e:
        print(f"Error processing file '{csv_filepath}': {e}. Skipping.")
        return None

    # Return the raw counts
    return (total_questions, correct_answer_present_count, correct_answer_highest_proportion_count)


def find_and_group_csvs(root_dir):
    """
    Finds CSV files ONLY in IMMEDIATE subdirectories of root_dir and groups them
    based on predefined patterns: 'sc_64', 'diversity_n64_thresh*', 'random_n64_thresh*'.
    Returns {'sc_64': [list of paths], 'diversity_n64_thresh': [list of paths], 'random_n64_thresh': [list of paths]}
    """
    # Define the target patterns and their corresponding group names
    target_patterns = {
        'sc_64': 'sc_64',
        'diversity_n64_thresh': 'diversity_n64_thresh', # Group all dirs starting with this
        'random_n64_thresh': 'random_n64_thresh'     # Group all dirs starting with this
    }

    grouped_csvs = {group_name: [] for group_name in target_patterns.values()}
    ignored_dirs = []

    print(f"Scanning directories under: {root_dir}")

    if not os.path.isdir(root_dir):
        print(f"Error: Root directory not found or is not a directory: {root_dir}")
        return grouped_csvs # Return empty lists for all groups

    try:
        # Get immediate items in the root directory
        items_in_root = os.listdir(root_dir)
    except OSError as e:
        print(f"Error listing directory contents for {root_dir}: {e}")
        return grouped_csvs # Return empty lists for all groups

    processed_subdirs_count = 0

    for item_name in items_in_root:
        item_path = os.path.join(root_dir, item_name)

        # Check if the item is a directory and not a symbolic link pointing elsewhere
        if os.path.isdir(item_path) and not os.path.islink(item_path):
            processed_subdirs_count += 1
            assigned_group = None

            # Check against the target patterns
            if item_name == target_patterns['sc_64']:
                assigned_group = target_patterns['sc_64']
            elif item_name.startswith(target_patterns['diversity_n64_thresh']):
                 assigned_group = target_patterns['diversity_n64_thresh']
            elif item_name.startswith(target_patterns['random_n64_thresh']):
                 assigned_group = target_patterns['random_n64_thresh']
            else:
                # Directory doesn't match any target pattern
                ignored_dirs.append(item_name)
                continue # Skip processing this directory

            print(f"Processing subdirectory: {item_name} (Group: {assigned_group})")
            try:
                # Find CSV files *within* this subdirectory
                csv_files_in_subdir = [
                    os.path.join(item_path, f)
                    for f in os.listdir(item_path)
                    if f.endswith('.csv') and os.path.isfile(os.path.join(item_path, f))
                ]
                grouped_csvs[assigned_group].extend(csv_files_in_subdir)
                print(f"  Found {len(csv_files_in_subdir)} CSVs in {item_name}.")
            except OSError as e:
                print(f"Error listing directory contents for {item_path}: {e}")
                # Continue to the next item/directory

    if processed_subdirs_count == 0:
        print(f"Warning: No subdirectories found directly under {root_dir}.")
    if ignored_dirs:
         print(f"\nIgnored {len(ignored_dirs)} subdirectories that didn't match target patterns: {', '.join(ignored_dirs)}")


    print(f"\nScan complete.")
    for group_name, file_list in grouped_csvs.items():
        print(f"  Found {len(file_list)} CSVs for group '{group_name}'.")

    return grouped_csvs


def aggregate_results(csv_groups):
    """
    Analyzes CSVs in groups and aggregates counts.
    Calculates percentages for each group.
    Returns:
        tuple: (aggregated_raw_counts, results_percentages)
            aggregated_raw_counts (dict): {'group_name': {'total_questions': N, ...}, ...}
            results_percentages (dict): {'group_name': {'%present': P, ...}, ...}
    """
    aggregated = {}
    results_percentages = {}

    # Initialize dictionaries for all potential groups from csv_groups keys
    for group_name in csv_groups.keys():
        aggregated[group_name] = {'total_questions': 0, 'present_count': 0, 'highest_prop_count': 0, 'num_files': 0}
        results_percentages[group_name] = {'%present': 0, '%highest_prop_overall': 0, '%efficiency': 0}


    # Now process the groups that actually exist and have files
    for group_name, file_list in csv_groups.items():
        if not file_list:
            print(f"\nSkipping aggregation for group '{group_name}': No CSV files found.")
            # Keep the zero-initialized entry in both dictionaries
            continue

        print(f"\nAggregating results for group: {group_name.upper()} ({len(file_list)} files)")
        group_stats = aggregated[group_name] # Use the entry initialized above

        for filepath in file_list:
            counts = analyze_csv_file(filepath)
            if counts:
                total, present, highest_prop = counts
                group_stats['total_questions'] += total
                group_stats['present_count'] += present
                group_stats['highest_prop_count'] += highest_prop
                group_stats['num_files'] += 1
            else:
                # analyze_csv_file already prints a warning for the skipped file
                pass

        # Calculate percentages after aggregating counts for the group
        total_q = group_stats['total_questions']
        present_c = group_stats['present_count']
        highest_prop_c = group_stats['highest_prop_count']

        if total_q > 0:
            results_percentages[group_name]['%present'] = (present_c / total_q) * 100
            results_percentages[group_name]['%highest_prop_overall'] = (highest_prop_c / total_q) * 100

        # Voting Efficiency calculated among questions where it was present
        if present_c > 0:
             results_percentages[group_name]['%efficiency'] = (highest_prop_c / present_c) * 100
        # Else: efficiency remains 0 if present_c is 0

        print(f"Finished aggregation for {group_name.upper()}. Total questions processed: {total_q} from {group_stats['num_files']} file(s).")


    # Return both the raw aggregated counts and the calculated percentages
    return aggregated, results_percentages


def plot_comparison(results_percentages, output_filename=None):
    """
    Generates a bar plot comparing specific groups ('sc_64', 'diversity_n64_thresh', 'random_n64_thresh')
    across metrics. Saves the plot to output_filename if provided, otherwise displays it.
    """
    labels = ['Presence', 'Highest Proportion\n(Overall)', 'Voting Efficiency\n(Given Present)']
    # Use the keys from the results_percentages dictionary to get the group names that actually have data
    group_names = list(results_percentages.keys())
    num_groups = len(group_names)

    if num_groups == 0:
        print("No data available for any specified groups to plot.")
        return

    # Ensure the order of groups in the plot is consistent if possible
    group_names.sort() # Sort alphabetically for consistent bar order

    # Extract values for plotting
    plot_values = {name: [results_percentages[name]['%present'],
                          results_percentages[name]['%highest_prop_overall'],
                          results_percentages[name]['%efficiency']]
                   for name in group_names}

    x = np.arange(len(labels))  # the label locations
    # Adjust bar width and spacing based on the number of groups
    total_width = 0.7 # Total width allocated for a set of bars for one metric
    group_width = total_width / num_groups if num_groups > 0 else 0.35
    # spacing = (1 - total_width) / (num_groups - 1) if num_groups > 1 else 0 # This spacing calculation is tricky with few bars
    # A simpler approach: total width divided equally + a small gap
    bar_spacing_within_group = 0.05 # Small gap between bars for the same metric
    group_width = (total_width - (num_groups - 1) * bar_spacing_within_group) / num_groups if num_groups > 0 else 0.35
    start_pos = x - (total_width / 2)


    fig, ax = plt.subplots(figsize=(12, 7)) # Adjust figure size as needed

    # Plot bars for each group
    rects = []
    for i, group_name in enumerate(group_names):
        bar_position = start_pos + i * (group_width + bar_spacing_within_group)
        rect = ax.bar(bar_position, plot_values[group_name], group_width, label=group_name)
        rects.append(rect)


    # Add some text for labels, title and axes ticks
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Comparison of Correct Answer Metrics by Directory Pattern')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.set_ylim(0, 105) # Ensure space for text labels above bars

    def autolabel(all_rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect_set in all_rects:
            for rect in rect_set:
                height = rect.get_height()
                # Only annotate if height is > 0, otherwise it clutters the plot
                if height > 0.1: # Use a small threshold
                    ax.annotate(f'{height:.1f}%',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom')

    autolabel(rects)

    fig.tight_layout()

    if output_filename:
        try:
            plt.savefig(output_filename)
            print(f"\nPlot saved successfully to '{output_filename}'")
        except Exception as e:
            print(f"\nError saving plot to '{output_filename}': {e}")
        finally:
            plt.close(fig) # Close the figure to free memory
    else:
        # If no output filename, display interactively
        plt.show()


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze CSV results in immediate subdirectories matching specific patterns ('sc_64', 'diversity_n64_thresh*', 'random_n64_thresh*'). Aggregates results per pattern and generates a comparison plot.")
    parser.add_argument("root_path", help="The root directory containing the immediate subdirectories to analyze (e.g., /<user home path>/slimsc/prune/results/R1-Distill-Qwen-14B/gpqa_diamond)")
    parser.add_argument("-o", "--output", help="Filename to save the plot (e.g., comparison_plot.png). If not provided, the plot will be displayed interactively.", default=None)


    args = parser.parse_args()
    root_directory_to_scan = args.root_path
    output_plot_filename = args.output

    # 1. Find and group CSV files by pattern
    csv_groups = find_and_group_csvs(root_directory_to_scan)

    # Check if any relevant files were found at all
    total_relevant_files = sum(len(files) for files in csv_groups.values())
    if total_relevant_files == 0:
         print("\nNo relevant CSV files found matching the specified directory patterns. Exiting.")
         sys.exit(0)


    # 2. Aggregate results for each group
    # Capture both the raw counts and the percentages
    aggregated_raw_counts, aggregated_percentage_results = aggregate_results(csv_groups)

    # Filter out groups that had no files/data processed (total_questions > 0)
    # We need to iterate through the keys of the raw counts to check if any data was aggregated for that group
    plot_data_results = {}
    for group_name in aggregated_raw_counts.keys():
        if aggregated_raw_counts[group_name]['total_questions'] > 0:
             plot_data_results[group_name] = aggregated_percentage_results[group_name]
        # Also include the group if it has 0 total questions but was requested, maybe for completeness,
        # but the plot will show 0%. Let's stick to only groups with actual data for clarity in the plot.
        # If you want 0% bars for groups with no files, remove this filtering step.


    # 3. Print numerical results
    print("\n--- Aggregated Percentage Results ---")
    if not plot_data_results:
        print("No valid data points found across specified groups.")
    else:
        # Iterate through the groups that have data
        for group_name, percentages in plot_data_results.items():
            raw_counts = aggregated_raw_counts[group_name] # Get raw counts for printing detail
            print(f"\nGroup: {group_name.upper()}")
            print(f"  Percentage Correct Answer Present (Overall): {percentages['%present']:.2f}% ({raw_counts['present_count']}/{raw_counts['total_questions']})")
            print(f"  Percentage Correct Answer Highest Proportion (Overall): {percentages['%highest_prop_overall']:.2f}% ({raw_counts['highest_prop_count']}/{raw_counts['total_questions']})")
            # For efficiency, use the raw present count in the denominator printout
            efficiency_denominator = raw_counts['present_count'] if raw_counts['present_count'] > 0 else 0
            print(f"  Voting Efficiency (Highest Proportion | Present): {percentages['%efficiency']:.2f}% ({raw_counts['highest_prop_count']}/{efficiency_denominator})")
    print("-------------------------------------")


    # 4. Generate and save/show plot (only if there's data to plot)
    if not plot_data_results:
         print("\nNo valid data found across specified groups to generate a meaningful plot.")
    else:
        # Pass only the dictionary containing data for the groups that will be plotted
        plot_comparison(plot_data_results, output_plot_filename)