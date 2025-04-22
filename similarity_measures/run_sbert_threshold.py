import logging
import os
import sys

# --- Path Setup (Ensure the package root is in sys.path) ---
SCRIPT_DIR_ABS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_ABS = os.path.dirname(SCRIPT_DIR_ABS)
# Add project root to Python path
if PROJECT_ROOT_ABS not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_ABS)

# --- Import the Evaluation Function ---
# Now that __init__.py is updated, we can import directly from scorers
try:
    from similarity_measures.scorers import run_sbert_evaluation
except ImportError as e:
     # Provide a helpful error message if imports fail
     logging.error(f"ImportError: {e}. Make sure you are running this script from the 'slimsc' project root, e.g., 'python similarity_measures/run_sbert_threshold.py'")
     sys.exit(1)

# --- Basic Logging Setup for the Runner Script ---
# The evaluator module also sets up logging, but setting it here ensures
# messages from this runner script itself are captured.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    logging.info("Executing SBERT classifier evaluation via run_sbert_threshold.py...")
    try:
        # Call the imported function
        run_sbert_evaluation()
        logging.info("Evaluation process initiated by run_sbert_threshold completed.")
    except Exception as e:
        logging.exception("An unexpected error occurred during evaluation run.")
        sys.exit(1) # Exit with error status