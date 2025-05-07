import pandas as pd
import numpy as np

# Configuration
FILE_PATH = '/home/users/ntu/chong032/slimsc/prune/results/R1-Distill-Qwen-14B/gpqa_diamond/sc_10_control/evaluation_summary.csv'
SAMPLE_SIZE = 50
MIN_SEEDS_TO_TRY = 1000   # Minimum number of seeds to try
MAX_SEEDS_TO_TRY = 10000  # Maximum number of seeds to try if no early stop
EARLY_STOP_TOLERANCE = 1e-9 # Stop if abs(accuracy_diff) < this (effectively zero for practical purposes)

def calculate_accuracy(df_subset):
    """Calculates accuracy based on the 'final_score' column."""
    if df_subset.empty:
        return 0.0
    return df_subset['final_score'].mean()

def find_best_sample_seed(df_full, sample_size, min_seeds, max_seeds, early_stop_tolerance):
    """
    Finds the seed that produces a sample with accuracy closest to the full dataset.
    Tries at least `min_seeds` and up to `max_seeds`, stopping early if a near-perfect match is found.
    """
    if len(df_full) < sample_size:
        raise ValueError(f"Full dataset size ({len(df_full)}) is smaller than desired sample size ({sample_size}).")

    full_accuracy = calculate_accuracy(df_full)
    print(f"Full dataset accuracy ({len(df_full)} questions): {full_accuracy:.6f}")

    best_seed = -1
    min_accuracy_diff = float('inf')
    best_sample_accuracy = -1.0
    
    for i in range(max_seeds):
        seed = i # Using iteration number as seed for simplicity and reproducibility
        
        # Ensure a different sample for each seed by using the seed in random_state
        sample_df = df_full.sample(n=sample_size, random_state=seed, replace=False)
        current_sample_accuracy = calculate_accuracy(sample_df)
        accuracy_diff = abs(current_sample_accuracy - full_accuracy)
        
        if accuracy_diff < min_accuracy_diff:
            min_accuracy_diff = accuracy_diff
            best_seed = seed
            best_sample_accuracy = current_sample_accuracy
            print(f"Iter {i+1}/{max_seeds}: New best: Seed {seed}, Sample Acc: {current_sample_accuracy:.6f}, Diff: {min_accuracy_diff:.6f}")
        elif (i + 1) % (max_seeds // 20) == 0 : # Print progress occasionally
             print(f"Iter {i+1}/{max_seeds}: Current best diff: {min_accuracy_diff:.6f} (Seed {best_seed})")


        # Check for early stopping condition after minimum seeds have been tried
        if (i + 1) >= min_seeds and min_accuracy_diff < early_stop_tolerance:
            print(f"\nEarly stopping at iteration {i+1} (seed {seed}).")
            print(f"Minimum accuracy difference ({min_accuracy_diff:.6f}) is below tolerance ({early_stop_tolerance}).")
            break
            
    return best_seed, best_sample_accuracy, full_accuracy, min_accuracy_diff

# --- Main Script ---
if __name__ == "__main__":
    try:
        df = pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file '{FILE_PATH}' was not found.")
        # Create a dummy DataFrame for demonstration if the file is not present
        print("Creating a dummy DataFrame for demonstration purposes.")
        data = {
            'iteration': [i % 2 + 1 for i in range(198)],
            'question_id': [f'index_{i}' for i in range(198)],
            'n_chains_requested': [10] * 198,
            'n_chains_received': [10] * 198,
            'correct_answer': np.random.choice(['A', 'B', 'C', 'D'], 198),
            'final_score': np.random.choice([0, 1], 198, p=[0.25, 0.75]), # Approx 75% accuracy
            'prompt_tokens': np.random.randint(100, 200, 198),
            'total_completion_tokens': np.random.randint(30000, 40000, 198),
        }
        df = pd.DataFrame(data)
        df['voted_answer'] = df.apply(lambda row: row['correct_answer'] if row['final_score'] == 1 else np.random.choice([ans for ans in ['A','B','C','D'] if ans != row['correct_answer']]), axis=1)
        df.to_csv(FILE_PATH, index=False)
        print(f"Dummy '{FILE_PATH}' created with {len(df)} rows. Please run the script again.")
        exit()

    if 'final_score' not in df.columns:
        print(f"Error: 'final_score' column not found in '{FILE_PATH}'.")
        exit()

    print(f"Loaded {len(df)} questions from '{FILE_PATH}'.")
    
    current_sample_size = SAMPLE_SIZE
    if len(df) < SAMPLE_SIZE:
        print(f"Warning: The dataset has only {len(df)} questions, which is less than the desired sample size of {SAMPLE_SIZE}.")
        print("Proceeding with all available questions as the 'sample'.")
        current_sample_size = len(df)
        # If sample size is full size, only 1 "seed" makes sense, or min/max seeds become irrelevant
        # For simplicity, let's adjust seed counts if full dataset is smaller than min_seeds to try
        # Though find_best_sample_seed already raises an error if len(df) < sample_size, this is more of a conceptual adjustment.
        # The ValueError in find_best_sample_seed will handle the actual impossibility.

    if len(df) == 0:
        print("Error: The CSV file is empty.")
        exit()

    best_seed, best_sample_acc, full_acc, min_diff = find_best_sample_seed(
        df, current_sample_size, MIN_SEEDS_TO_TRY, MAX_SEEDS_TO_TRY, EARLY_STOP_TOLERANCE
    )

    print("\n--- Results ---")
    print(f"Full dataset accuracy ({len(df)} questions): {full_acc:.6f}")
    if best_seed != -1:
        print(f"Best seed found: {best_seed}")
        print(f"Sample accuracy with this seed ({current_sample_size} questions): {best_sample_acc:.6f}")
        print(f"Absolute difference from full accuracy: {min_diff:.6f}")
    else:
        print("Could not find a suitable sample (this should not happen with valid inputs).")