# gpqa_utils.py
import random
import re
from datasets import load_dataset
# import huggingface_hub # Keep commented unless login is strictly needed now
from typing import List, Dict, Tuple, Optional
import transformers

import logging

logger = logging.getLogger(__name__)


def load_data_gpqa(dataset_name: str = "gpqa_diamond", split: str = "train") -> List[Dict]:
    """Load and prepare the GPQA dataset."""
    # api_key = os.environ.get("HUGGINGFACE_API_KEY")
    # if api_key:
    #     huggingface_hub.login(token=api_key)

    try:
        # Use specific subset name directly if that's how it's identified
        df = load_dataset("Idavidrein/gpqa", name=dataset_name, split=split)
        df = df.rename_column("Question", "question") # Standardize column name
        rng = random.Random(0) # Seed RNG for reproducible permutations

        examples = [row for row in df]
        # Attach a random permutation (0..3) to each example
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        logger.info(f"Loaded {len(examples)} examples from GPQA {dataset_name} split {split}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading GPQA dataset[/red]")
        # Consider raising the exception or returning empty list based on desired behavior
        raise e # Re-raise to stop execution if dataset loading fails


def create_prompt_gpqa(example: Dict) -> Tuple[str, List[str], str]:
    """Create a prompt, choices list, and correct answer letter from the example for GPQA."""
    # Randomly permute the four options using the pre-assigned permutation
    choices_pool = [
        example["Correct Answer"],
        example["Incorrect Answer 1"],
        example["Incorrect Answer 2"],
        example["Incorrect Answer 3"],
    ]
    permutation = example.get("permutation")
    if not permutation or len(permutation) != 4:
         # Fallback if permutation missing (shouldn't happen with load_data_gpqa)
        permutation = random.sample(range(4), 4)

    choices = [choices_pool[idx] for idx in permutation]

    # Identify which choice is correct
    correct_index = choices.index(example["Correct Answer"])
    correct_answer_letter = "ABCD"[correct_index]

    # Build the multiple-choice prompt with clear instructions
    prompt = f"""Answer the following multiple-choice science question.
Think step-by-step to reach the solution.
Conclude your response with a single line containing the answer letter formatted exactly as 'Answer: $LETTER'.

Question: {example['question']}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
"""

    return prompt, choices, correct_answer_letter


def extract_answer_gpqa(content: Optional[str]) -> Optional[str]:
    """Extracts the final answer letter (A, B, C, or D) from the content."""
    if content is None:
        return None

    # Regex to find "Answer: $LETTER" potentially with spaces, $, case-insensitive
    # It captures only the letter A, B, C, or D
    ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?" # Matches end of string

    # Search the entire content, but prioritize the last match if multiple exist
    matches = list(re.finditer(ANSWER_PATTERN, content))

    if matches:
        # Get the last match found
        last_match = matches[-1]
        extracted_letter = last_match.group(1).upper()
        return extracted_letter
    else:
        # Fallback: Check if the *very last non-whitespace character* is A, B, C, or D
        # This is less robust but can catch cases where formatting is slightly off
        stripped_content = content.strip()
        if stripped_content and stripped_content[-1].upper() in ['A', 'B', 'C', 'D']:
             # Check if preceded by something like "Answer:" to reduce false positives
             context_window = stripped_content[-10:] # Look at last few chars
             if "answer" in context_window.lower() or ":" in context_window:
                  return stripped_content[-1].upper()
        return None # No answer found


def calculate_score_gpqa(extracted_answer: Optional[str], correct_answer_letter: str) -> int:
    """Calculates the score (1 for correct, 0 for incorrect/missing)."""
    if extracted_answer is None:
        return 0
    return 1 if extracted_answer.upper() == correct_answer_letter.upper() else 0


# --- Optional: Token counting utility (Ensure tokenizer_path is handled) ---
_tokenizer = None
_tokenizer_path_loaded = None

def count_tokens(text: str, tokenizer_path: Optional[str] = None) -> Optional[int]:
    """
    Counts tokens using Hugging Face tokenizer. Loads tokenizer on first call or if path changes.

    Args:
        text (str): The text to tokenize.
        tokenizer_path (Optional[str]): Path to the Hugging Face tokenizer directory.

    Returns:
        Optional[int]: The number of tokens, or None if tokenization fails or tokenizer is unavailable.
    """
    global _tokenizer, _tokenizer_path_loaded
    if not text:
        return 0

    # Load or reload tokenizer if path is provided and different from loaded one, or if not loaded yet
    if tokenizer_path and (tokenizer_path != _tokenizer_path_loaded or _tokenizer is None):
        try:
            # logger.info(f"Loading tokenizer from {tokenizer_path}...") # Verbose
            _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            _tokenizer_path_loaded = tokenizer_path
            # logger.info("Tokenizer loaded.") # Verbose
        except Exception as e:
            logger.exception(f"[red]ERROR: Failed to load tokenizer from {tokenizer_path}. Token counting disabled.[/red]")
            _tokenizer = None
            _tokenizer_path_loaded = None
            return None # Indicate failure

    if _tokenizer:
        try:
            return len(_tokenizer.encode(text))
        except Exception as e:
            logger.exception(f"[red]ERROR: Failed to encode text with tokenizer[/red]")
            return None # Indicate failure
    else:
         # Only print warning if path was ever provided but failed, or never provided
         if tokenizer_path or _tokenizer_path_loaded is None:
              # This warning might be noisy if path is intentionally omitted
              # logger.warning("Tokenizer not available. Cannot count tokens.")
              pass
         return None # Indicate unavailability/failure