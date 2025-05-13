# slimsc/prune/utils/bbeh_temporal_sequences_utils.py
import json
import os
import re # Make sure to have this import
import logging
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Path to your specific BBEH temporal sequences JSON file
BBEH_TEMPORAL_SEQUENCES_DATA_PATH = "slimsc/prune/utils/bbeh_temporal_sequences.json"

# --- Helper functions (inspired by/copied from bbeh.evaluate) ---
# These are made internal to this module to keep it self-contained.

def _strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response

def _bbeh_lib_extract_answer(sample: str) -> str:
    """Extracts the final answer from the sample using bbeh lib logic."""
    answer_prefixes = [
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is "
    ]
    answer = sample
    for answer_prefix in answer_prefixes:
        if answer_prefix in answer:
            answer = answer.split(answer_prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return _strip_latex(answer)

def _fuzzy_match(prediction: str, reference: str) -> bool:
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False

def _preprocess_sample_from_bbeh_lib(sample_text: str) -> str:
    # This 'sample_text' is what `extract_answer_bbeh_temporal` provides.
    # `_bbeh_lib_extract_answer` will be applied to it.
    prediction = _bbeh_lib_extract_answer(sample_text.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    return prediction

def _preprocess_reference_from_bbeh_lib(reference: str) -> str:
    reference = reference.strip().lower()
    reference = reference.replace(", ", ",")
    return reference

def _evaluate_correctness_from_bbeh_lib(
    prediction_to_evaluate: str, # Output from extract_answer_bbeh_temporal
    reference_answer: str
) -> bool:
    processed_prediction = _preprocess_sample_from_bbeh_lib(prediction_to_evaluate)
    processed_reference = _preprocess_reference_from_bbeh_lib(reference_answer)
    return _fuzzy_match(processed_prediction, processed_reference)

# --- Dataset specific functions for bbeh_temporal_sequences ---

def load_data_bbeh_temporal() -> List[Dict]:
    """Load and prepare the BBEH temporal_sequences dataset."""
    try:
        with open(BBEH_TEMPORAL_SEQUENCES_DATA_PATH, 'r', encoding='utf-8') as f:
            # Assuming the file directly contains a list of examples,
            # or a dict with an "examples" key.
            data_content = json.load(f)

        if isinstance(data_content, dict) and "examples" in data_content:
            raw_examples = data_content["examples"]
        elif isinstance(data_content, list):
            raw_examples = data_content
        else:
            logger.error(
                f"Unexpected JSON structure in {BBEH_TEMPORAL_SEQUENCES_DATA_PATH}. "
                "Expected a list of examples or a dictionary with an 'examples' key."
            )
            return []

        examples = []
        for i, ex_data in enumerate(raw_examples):
            if not isinstance(ex_data, dict) or "input" not in ex_data or "target" not in ex_data:
                logger.warning(
                    f"Example {i} in {BBEH_TEMPORAL_SEQUENCES_DATA_PATH} is malformed or "
                    f"missing 'input' or 'target'. Skipping. Data: {ex_data}"
                )
                continue
            examples.append({
                "id": ex_data.get("id", f"bbeh_temporal_{i}"), # Ensure an ID exists
                "question": ex_data["input"], # Map 'input' to 'question'
                "target": ex_data["target"]
            })
        logger.info(f"Loaded {len(examples)} examples from {BBEH_TEMPORAL_SEQUENCES_DATA_PATH}.")
        return examples
    except FileNotFoundError:
        logger.exception(f"[red]Error: BBEH temporal sequences data file not found: {BBEH_TEMPORAL_SEQUENCES_DATA_PATH}.[/red]")
        raise
    except json.JSONDecodeError:
        logger.exception(f"[red]Error: Invalid JSON in BBEH temporal sequences data file: {BBEH_TEMPORAL_SEQUENCES_DATA_PATH}.[/red]")
        raise
    except Exception as e:
        logger.exception(f"[red]An unexpected error occurred while loading BBEH temporal_sequences data: {e}[/red]")
        raise

def create_prompt_bbeh_temporal(example: Dict) -> Tuple[str, str]:
    """
    Create a prompt for a BBEH temporal_sequences example.
    Args:
        example (Dict): Dict with 'question' (from 'input') and 'target'.
    Returns:
        Tuple[str, str]: (prompt string, correct answer string).
    """
    question_content = example.get('question', 'N/A')
    correct_answer = example.get('target', 'N/A')

    prompt = f"""Answer the following question.
The last line of your response should be of the format: 'Answer: $ANSWER'
{question_content}
Think step by step before answering.
"""
    return prompt, correct_answer

def extract_answer_bbeh_temporal(content: Optional[str]) -> Optional[str]:
    """
    Extracts the answer from response using 'Answer: $ANSWER' format.
    Prioritizes the last occurrence.
    """
    if content is None:
        return None

    # Regex: Case-insensitive "answer:", optional spaces, capture everything up to EOL.
    # We look for lines starting with "answer:" (potentially indented).
    pattern = r"^\s*answer:\s*(.*)"
    try:
        matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
    except TypeError as e:
        # This can happen if content is not a string (e.g. bytes)
        logger.error(f"TypeError in re.finditer for BBEH answer extraction. Content type: {type(content)}. Error: {e}")
        return None


    if matches:
        # Get the text from the last matched "Answer:" line
        extracted_text = matches[-1].group(1).strip()
        if extracted_text: # Return if non-empty
            return extracted_text
        else:
            logger.debug(f"BBEH: Found 'Answer:' pattern but no subsequent text on the line: '{matches[-1].group(0)}'")
            return None

    logger.debug(f"Could not extract BBEH answer using 'Answer: $ANSWER' pattern from content: '{content[:200]}...'")
    return None

def calculate_score_bbeh_temporal(extracted_answer: Optional[str], correct_answer_str: str) -> int:
    """
    Calculate the score for BBEH temporal_sequences.
    Args:
        extracted_answer (Optional[str]): Model's extracted answer.
        correct_answer_str (str): Correct answer string.
    Returns:
        int: 1 if correct, 0 otherwise.
    """
    if extracted_answer is None:
        logger.debug(f"Extracted answer is None. Correct: {correct_answer_str}. Score: 0")
        return 0

    # `extracted_answer` is the string obtained by `extract_answer_bbeh_temporal`.
    # This string will be further processed by `_evaluate_correctness_from_bbeh_lib`.
    is_correct = _evaluate_correctness_from_bbeh_lib(extracted_answer, correct_answer_str)
    score = 1 if is_correct else 0
    logger.debug(f"BBEH Temporal Scoring: Extracted='{extracted_answer}', Correct='{correct_answer_str}', ProcessedCorrect={is_correct}, Score={score}")
    return score