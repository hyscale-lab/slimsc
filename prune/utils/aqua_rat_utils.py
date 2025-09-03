# slimsc/prune/utils/aqua_rat_utils.py
import random
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Define the specific dataset details
AQUA_RAT_DATASET_PATH = "deepmind/aqua_rat"
AQUA_RAT_SUBSET_NAME = "raw" # As requested
AQUA_RAT_SPLIT = "test"      # As requested

# Define the number of options. AQUA-RAT typically has 5 options (A-E).
# The example provided shows 5 options ("A) ...", "B) ...", "C) ...", "D) ...", "E) ...")
# The 'correct' column gives the letter.
N_OPTIONS_AQUA_RAT = 5
OPTION_LABELS = "ABCDE" # Up to E for 5 options

def load_data_aqua_rat() -> List[Dict]:
    """Load and prepare the AQUA-RAT dataset (raw subset, test split)."""
    try:
        dataset = load_dataset(AQUA_RAT_DATASET_PATH, name=AQUA_RAT_SUBSET_NAME, split=AQUA_RAT_SPLIT)

        # The dataset columns are 'question', 'options', 'correct'.
        # No renaming needed if we use these directly.
        # 'options' is a list of strings like ["A) ...", "B) ..."]
        # 'correct' is a string like "A"

        examples = []
        for i, row in enumerate(dataset):
            # Add a unique ID for easier tracking if not present
            if 'id' not in row:
                row['id'] = f"aqua_rat_test_{i}"
            examples.append(row)

        logger.info(f"Loaded {len(examples)} examples from AQUA-RAT {AQUA_RAT_SUBSET_NAME} split {AQUA_RAT_SPLIT}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading AQUA-RAT dataset '{AQUA_RAT_DATASET_PATH}' subset '{AQUA_RAT_SUBSET_NAME}'. Check dataset details and internet connection.[/red]")
        raise e

def create_prompt_aqua_rat(example: Dict) -> Tuple[str, str]:
    """Create a prompt and extract the correct answer letter for AQUA-RAT.

    Args:
        example (Dict): A dictionary representing one row from the dataset.
                        Expected keys: 'question', 'options', 'correct'.
                        'options' is a list of strings, e.g., ["A) text A", "B) text B", ...].
                        'correct' is a string, e.g., "A".

    Returns:
        Tuple[str, str]: A tuple containing:
            - The formatted prompt string.
            - The correct answer letter ('A', 'B', 'C', 'D', or 'E').
    """
    question_text = example.get('question', 'N/A')
    options_list = example.get('options', []) # This is a list of strings like "A) option text"
    correct_answer_letter = example.get('correct', 'Z').upper() # Z as an error/default

    if not (isinstance(correct_answer_letter, str) and correct_answer_letter in OPTION_LABELS[:len(options_list)]):
        logger.error(
            f"[red]Invalid or missing 'correct' answer letter '{correct_answer_letter}' "
            f"for {len(options_list)} options. Example ID: {example.get('id', 'N/A')}. "
            f"Setting to 'Z'.[/red]"
        )
        correct_answer_letter = "Z" # Indicate error

    # Build the multiple-choice prompt
    # The 'options' list already contains the letter prefixes.
    options_str = "\n".join(options_list)

    prompt = f"""Answer the following multiple-choice question.
Think step-by-step to reach the solution.
Conclude your response with a single line containing the answer letter formatted exactly as 'Answer: $LETTER'.

Question: {question_text}

Options:
{options_str}
"""
    # The second element returned is just the correct answer letter string.
    # Unlike GPQA, we don't need to return the permuted choices separately because
    # the 'options' field in AQUA-RAT is already structured.
    return prompt, correct_answer_letter

def extract_answer_aqua_rat(content: Optional[str]) -> Optional[str]:
    """
    Extracts the final answer letter (A, B, C, D, or E) from the content.
    Reuses logic similar to GPQA but extends to E.
    """
    if content is None:
        return None

    # Regex to find answer patterns like "Answer: A", "Final answer: B", etc.
    # Case insensitive and allows for various spacings
    ANSWER_PATTERNS = [
        rf"(?i)Answer[ \t]*:[ \t]*\$?([{OPTION_LABELS}])\$?",  # Answer: A
        rf"(?i)Final[ \t]+Answer[ \t]*:[ \t]*\$?([{OPTION_LABELS}])\$?",  # Final Answer: A
        rf"(?i)answer[ \t]*is[ \t]*([{OPTION_LABELS}])[\.!?,;:]*$",  # answer is A
    ]
    
    # Try each pattern and return the last match
    all_matches = []
    for pattern in ANSWER_PATTERNS:
        matches = list(re.finditer(pattern, content))
        all_matches.extend(matches)
    
    if all_matches:
        # Always take the last match
        last_match = all_matches[-1]
        extracted_letter = last_match.group(1).upper()
        return extracted_letter
    
    return None

def calculate_score_aqua_rat(extracted_answer: Optional[str], correct_answer_letter: str) -> int:
    """
    Calculates the score (1 for correct, 0 for incorrect/missing).
    Reuses logic similar to GPQA.
    """
    if not isinstance(correct_answer_letter, str) or correct_answer_letter.upper() not in OPTION_LABELS:
        logger.warning(f"Invalid correct answer letter '{correct_answer_letter}' for AQUA-RAT. Score is 0.")
        return 0
    if extracted_answer is None:
        logger.debug(f"Extracted answer is None. Correct: {correct_answer_letter}. Score: 0")
        return 0

    is_correct = extracted_answer.upper() == correct_answer_letter.upper()
    logger.debug(f"Extracted: {extracted_answer}, Correct: {correct_answer_letter}, Score: {int(is_correct)}")
    return 1 if is_correct else 0