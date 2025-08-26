# slimsc/prune/utils/math_dapo_utils.py

import re
import signal
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def load_data_math_dapo(split: str = "train") -> List[Dict]:
    """
    Loads the BytedTsinghua-SIA/DAPO-Math-17k dataset from Hugging Face.
    """
    logger.info("Loading BytedTsinghua-SIA/DAPO-Math-17k dataset...")
    try:
        dataset = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split=split)
        return list(dataset)
    except Exception as e:
        logger.error(f"Failed to load math_dapo dataset: {e}")
        return []

def create_prompt_math_dapo(example: Dict) -> Tuple[str, str]:
    """
    Creates a prompt from a math_dapo example.
    """
    if not isinstance(example.get("prompt"), list) or not example["prompt"]:
        raise ValueError("Example format is invalid: 'prompt' key is not a non-empty list.")
        
    prompt_text = example["prompt"][0].get("content", "")
    correct_answer = example.get("reward_model", {}).get("ground_truth", "")
    
    full_prompt = f"{prompt_text}<think>\n"
    
    return full_prompt, str(correct_answer)

def extract_answer_math_dapo(content: Optional[str]) -> Optional[str]:
    """
    Extracts the final answer from the model's response, which is expected
    to be in the format 'Answer: $ANSWER'.
    """
    if not content:
        return None
    
    # This is a good generic pattern that works for many math datasets.
    # The new evaluation functions below will handle the fine-grained cleaning.
    match = re.search(r"Answer:\s*([\s\S]*?)\s*$", content.strip(), re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
        
    # As a fallback, try to find the last boxed answer if "Answer:" is missing
    return remove_boxed(last_boxed_only_string(content)) if last_boxed_only_string(content) else None

# --- Evaluation functions provided by the DAPO-Math dataset authors ---
# (Source: https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None

def remove_boxed(s: str) -> str:
    left = "\\boxed{"
    if not s.startswith(left) or not s.endswith("}"):
        return s
    return s[len(left):-1]

SUBSTITUTIONS = [
    ("an ", ""), ("a ", ""), (".$", "$"), ("\\$", ""), (r"\ ", ""), (" ", ""),
    ("mbox", "text"), (",\\text{and}", ","), ("\\text{and}", ","), ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question."""
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def calculate_score_math_dapo(extracted_answer: Optional[str], correct_answer: str) -> int:
    """
    Scores the extracted answer using the dataset's provided normalization logic.
    """
    if extracted_answer is None:
        return 0
    
    # Use the powerful normalization function on both the prediction and the ground truth
    # to handle formatting differences, units, etc.
    normalized_prediction = normalize_final_answer(extracted_answer)
    normalized_ground_truth = normalize_final_answer(correct_answer)
    
    is_correct = (normalized_prediction == normalized_ground_truth)
    
    if not is_correct:
        logger.debug(f"Incorrect. Normed Pred: '{normalized_prediction}', Normed GT: '{normalized_ground_truth}'")

    return 1 if is_correct else 0