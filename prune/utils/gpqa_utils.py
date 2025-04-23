# slimsc/prune/utils/gpqa_utils.py
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
        logger.exception(f"[red]Error loading GPQA dataset '{dataset_name}'. Check dataset name and internet connection.[/red]")
        raise e # Re-raise to stop execution if dataset loading fails


def create_prompt_gpqa(example: Dict) -> Tuple[str, List[str], str]:
    """Create a prompt, choices list, and correct answer letter from the example for GPQA."""
    # Randomly permute the four options using the pre-assigned permutation
    choices_pool = [
        example.get("Correct Answer", "N/A"),
        example.get("Incorrect Answer 1", "N/A"),
        example.get("Incorrect Answer 2", "N/A"),
        example.get("Incorrect Answer 3", "N/A"),
    ]
    permutation = example.get("permutation")
    if not permutation or len(permutation) != 4:
         # Fallback if permutation missing (shouldn't happen with load_data_gpqa)
        permutation = random.sample(range(4), 4)

    try:
        choices = [choices_pool[idx] for idx in permutation]
    except IndexError:
        logger.error(f"[red]Index error when applying permutation {permutation} to choices pool {choices_pool}. Example ID: {example.get('id', 'N/A')}[/red]")
        choices = [choices_pool[i] for i in range(4)] # Use default order as fallback

    # Identify which choice is correct
    correct_answer_text = example.get("Correct Answer", "N/A")
    correct_answer_letter = "E" # Default to E (error)
    try:
        correct_index = choices.index(correct_answer_text)
        correct_answer_letter = "ABCD"[correct_index]
    except ValueError:
        # This can happen if "Correct Answer" is not found in the permuted choices.
        # This shouldn't happen if choices_pool was constructed correctly from the example.
        # But if choices_pool failed or correct_answer_text was missing/invalid, this will occur.
        logger.error(f"[red]Correct answer text '{correct_answer_text}' not found in generated choices {choices}. Example ID: {example.get('id', 'N/A')}. Correct answer letter set to '{correct_answer_letter}'.[/red]")
    except IndexError:
         logger.error(f"[red]Index error when getting correct answer letter from permutation index. Example ID: {example.get('id', 'N/A')}[/red]")

    # Build the multiple-choice prompt with clear instructions
    prompt = f"""Answer the following multiple-choice science question.
Think step-by-step to reach the solution.
Conclude your response with a single line containing the answer letter formatted exactly as 'Answer: $LETTER'.

Question: {example.get('question', 'N/A')}

Options:
A) {choices[0] if len(choices) > 0 else 'N/A'}
B) {choices[1] if len(choices) > 1 else 'N/A'}
C) {choices[2] if len(choices) > 2 else 'N/A'}
D) {choices[3] if len(choices) > 3 else 'N/A'}
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
    # Ensure correct_answer_letter is a valid letter before comparison
    if not isinstance(correct_answer_letter, str) or correct_answer_letter.upper() not in ['A', 'B', 'C', 'D']:
         logger.warning(f"Invalid correct answer letter '{correct_answer_letter}'. Score is 0.")
         return 0
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

    # If tokenizer_path was not provided at all, _tokenizer remains None.
    # If loading failed, _tokenizer is None.
    if _tokenizer:
        try:
            # Encode the text. Handle potential errors during encoding.
            # add_special_tokens=False is typical for counting content tokens
            tokens = _tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.exception(f"[red]ERROR: Failed to encode text with tokenizer for counting[/red]")
            return None # Indicate failure
    else:
         # Only print a warning about tokenizer not being available if the path was expected (provided)
         # If tokenizer_path is None, the caller likely knew counting wasn't possible.
         # If loading failed previously (_tokenizer_path_loaded is not None but _tokenizer is None),
         # a warning was already printed on the first attempt.
         # Avoid repeated warnings here during every count call.
         # logger.debug("Tokenizer not available. Cannot count tokens.")
         return None # Indicate unavailability/failure