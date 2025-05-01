# slimsc/prune/utils/math500_utils.py
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


def load_data_math500(dataset_name: str = "math500", split: str = "train") -> List[Dict]:
    """Load and prepare the MATH-500 dataset."""
    try:
        # Load the dataset
        df = load_dataset("HuggingFaceH4/MATH-500", split=split)
        
        # Optionally rename columns for consistency with other datasets
        if "problem" in df.column_names:
            df = df.rename_column("problem", "question")
        
        examples = [row for row in df]
        logger.info(f"Loaded {len(examples)} examples from MATH-500 split {split}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading MATH-500 dataset. Check internet connection.[/red]")
        raise e  # Re-raise to stop execution if dataset loading fails


def create_prompt_math500(example: Dict) -> Tuple[str, str]:
    """Create a prompt and correct answer from the example for MATH-500.
    
    Args:
        example (Dict): Dictionary containing the MATH-500 problem data
        
    Returns:
        Tuple[str, str]: A tuple containing:
            - The formatted prompt with the question and instructions
            - The correct answer 
    """
    # Extract question and other fields from the example
    question = example.get('question', example.get('problem', 'N/A'))
    
    # Build the prompt with clear instructions
    prompt = f"""Answer the following math problem.\nThe last line of your response should be your answer within \\boxed{{}}.\n\n{question}\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering."""

    # Get the correct answer
    correct_answer = example.get('answer', 'N/A')
    
    return prompt, correct_answer


def extract_answer_math500(content: Optional[str]) -> Optional[str]:
    """Extracts the answer from the model's response.
    
    Args:
        content (Optional[str]): The model's response text
        
    Returns:
        Optional[str]: The extracted answer,
                      or None if no valid answer is found
    """
    if content is None:
        logger.error("[red]MATH-500 extract_answer: Content is None.[/red]")
        return None

    # Regex to find \boxed{ANSWER} pattern
    BOXED_PATTERN = r'\\boxed\{([^{}]+)\}'

    # Search the entire content, but prioritize the last match if multiple exist
    matches = list(re.finditer(BOXED_PATTERN, content))
    
    # Alternative regex patterns to try if the first one doesn't work
    if not matches and content:
        # Try with different patterns for robustness
        alt_patterns = [
            r'boxed\{([^{}]+)\}',        # In case the backslash is missing
            r'\\boxed\{\s*([^{}]+)\s*\}', # Allow spaces inside braces
            r'boxed\{\s*([^{}]+)\s*\}'    # Missing backslash + spaces
        ]
        
        for pattern in alt_patterns:
            matches = list(re.finditer(pattern, content))
            if matches:
                break

    if not matches:
        logger.error(f"[red]No \\boxed{{}} answer found in content.[/red]")
        return None

    # Get the last match found
    last_match = matches[-1]
    extracted_answer = last_match.group(1).strip()
    
    return extracted_answer


def calculate_score_math500(extracted_answer: Optional[str], correct_answer: str) -> int:
    """Calculates the score for a MATH-500 problem (1 for correct, 0 for incorrect/missing).
    
    Args:
        extracted_answer (Optional[str]): The extracted answer from the model's response
        correct_answer (str): The correct answer for the problem
        
    Returns:
        int: 1 if the answers match exactly, 0 otherwise
    """
    # Handle None or invalid extracted answer
    if extracted_answer is None:
        logger.error(f"[red]No valid MATH-500 answer found in extracted_answer.[/red]")
        return 0
        
    # Normalize answers for comparison by removing whitespace and converting to lowercase
    extracted_norm = extracted_answer.strip().lower()
    correct_norm = correct_answer.strip().lower()
    
    # Compare the normalized answers
    return 1 if extracted_norm == correct_norm else 0


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Load dataset
    print("\n=== Testing Dataset Loading ===")
    try:
        examples = load_data_math500(split="train")
        print(f"Successfully loaded {len(examples)} examples")
        sample_problem = examples[0]
        print(f"\nSample problem: {sample_problem['question']}")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        sample_problem = {
            "question": "What is the sum of the first 10 positive integers?",
            "answer": "55"  # Example answer
        }
    
    # Test 2: Create prompt
    print("\n=== Testing Prompt Creation ===")
    prompt, correct_answer = create_prompt_math500(sample_problem)
    print(f"Prompt:\n{prompt}")
    print(f"Correct answer: {correct_answer}")
    
    # Test 3: Answer extraction
    print("\n=== Testing Answer Extraction ===")
    test_responses = [
        "After solving step by step, \\boxed{55}",
        "The final answer is \\boxed{55}",
        "Therefore, \\boxed{55}",
        "Invalid response with no answer",
        None,
        "\\boxed{x^2 + 5x + 6}",  # Algebraic expression
        "Multiple answers: \\boxed{45} but actually \\boxed{55}", # Should take last one
        "Answer without braces: \\boxed55",  # Invalid format
        "Answer with spaces: \\boxed{ 55 }"  # Should handle spaces
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nTest {i+1}:")
        print(f"Response: {response}")
        extracted = extract_answer_math500(response)
        print(f"Extracted: {extracted}")
        
        # Test 4: Score calculation
        score = calculate_score_math500(extracted, "55")
        print(f"Score: {score}")
