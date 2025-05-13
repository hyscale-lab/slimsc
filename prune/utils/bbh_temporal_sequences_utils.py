# slimsc/prune/utils/bbh_temporal_sequences_utils.py
import random
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Define the specific dataset details
BBH_DATASET_PATH = "lukaemon/bbh"
BBH_SUBSET_NAME = "temporal_sequences"
BBH_SPLIT = "test"


def load_data_bbh_temporal() -> List[Dict]:
    """Load and prepare the BBH temporal_sequences test dataset."""
    try:
        # Load the specific subset and split
        dataset = load_dataset(BBH_DATASET_PATH, name=BBH_SUBSET_NAME, split=BBH_SPLIT)

        # Convert to list of dictionaries
        examples = [row for row in dataset]

        # No permutation needed as options are part of the 'input'
        # No column renaming needed if we use 'input' and 'target' directly

        logger.info(f"Loaded {len(examples)} examples from BBH {BBH_SUBSET_NAME} split {BBH_SPLIT}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading BBH dataset '{BBH_DATASET_PATH}' subset '{BBH_SUBSET_NAME}'. Check dataset details and internet connection.[/red]")
        raise e # Re-raise to stop execution if dataset loading fails


def create_prompt_bbh_temporal(example: Dict) -> Tuple[str, str]:
    """Create a prompt and extract the correct answer letter for BBH temporal_sequences.

    Args:
        example (Dict): A dictionary representing one row from the dataset.
                        Expected keys: 'input', 'target'.

    Returns:
        Tuple[str, str]: A tuple containing:
            - The formatted prompt string.
            - The correct answer letter ('A', 'B', 'C', or 'D').
    """
    question_block = example.get('input', 'N/A')
    correct_answer_label = example.get('target', 'N/A') # e.g., "(A)"

    # Extract the letter from the target label (e.g., "(A)" -> "A")
    correct_answer_letter = "E" # Default to E (error)
    if isinstance(correct_answer_label, str) and len(correct_answer_label) == 3 and correct_answer_label.startswith("(") and correct_answer_label.endswith(")"):
        letter = correct_answer_label[1].upper()
        if letter in ['A', 'B', 'C', 'D']:
            correct_answer_letter = letter
        else:
             logger.error(f"[red]Invalid letter extracted '{letter}' from target '{correct_answer_label}'. Example: {example}. Correct answer letter set to 'E'.[/red]")
    else:
        logger.error(f"[red]Invalid target format '{correct_answer_label}'. Expected format like '(A)'. Example: {example}. Correct answer letter set to 'E'.[/red]")


    # Build the multiple-choice prompt with clear instructions
    # The 'input' field already contains the question and options.
    prompt = f"""Answer the following multiple-choice question.
Think step-by-step to reach the solution.
Conclude your response with a single line containing the answer letter formatted exactly as 'Answer: $LETTER'.

{question_block}
"""
    return prompt, correct_answer_letter


def extract_answer_bbh_temporal(content: Optional[str]) -> Optional[str]:
    """Extracts the final answer letter (A, B, C, or D) from the content.
       Reuses the logic from gpqa_utils as the format requirement is the same.
    """
    if content is None:
        return None

    # Regex to find "Answer: $LETTER" potentially with spaces, $, case-insensitive
    # It captures only the letter A, B, C, or D
    # Using raw string r"" is safer for regex patterns
    ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

    # Search the entire content, but prioritize the last match if multiple exist
    matches = list(re.finditer(ANSWER_PATTERN, content))

    if matches:
        # Get the last match found
        last_match = matches[-1]
        extracted_letter = last_match.group(1).upper()
        return extracted_letter
    else:
        # Fallback: Check if the very last non-whitespace character is A, B, C, or D
        # This is less robust but can catch cases where formatting is slightly off
        stripped_content = content.strip()
        if stripped_content and stripped_content[-1].upper() in ['A', 'B', 'C', 'D']:
             # Check if preceded by something like "Answer:" or ":" to reduce false positives
             context_window = stripped_content[-15:] # Look at last few chars
             if "answer" in context_window.lower() or ":" in context_window:
                  # Added check to ensure it's not part of the options list like "D)"
                  if not stripped_content[-2:].endswith(')'):
                      return stripped_content[-1].upper()
        return None # No answer found


def calculate_score_bbh_temporal(extracted_answer: Optional[str], correct_answer_letter: str) -> int:
    """Calculates the score (1 for correct, 0 for incorrect/missing).
       Reuses the logic from gpqa_utils.
    """
    # Ensure correct_answer_letter is a valid letter before comparison
    if not isinstance(correct_answer_letter, str) or correct_answer_letter.upper() not in ['A', 'B', 'C', 'D']:
         # Error should have been logged during prompt creation, but warning here is useful
         logger.warning(f"Invalid correct answer letter provided for scoring: '{correct_answer_letter}'. Score is 0.")
         return 0
    if extracted_answer is None:
        logger.debug(f"Extracted answer is None. Correct: {correct_answer_letter}. Score: 0")
        return 0

    is_correct = extracted_answer.upper() == correct_answer_letter.upper()
    logger.debug(f"Extracted: {extracted_answer}, Correct: {correct_answer_letter}, Score: {int(is_correct)}")
    return 1 if is_correct else 0