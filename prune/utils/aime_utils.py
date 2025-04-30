# slimsc/prune/utils/aime_utils.py
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


def load_data_aime(dataset_name: str = "aime2024", split: str = "train") -> List[Dict]:
    """Load and prepare the AIME dataset."""
    try:
        # Use specific subset name directly if that's how it's identified
        df = load_dataset("Maxwell-Jia/AIME_2024", split=split)
        df = df.rename_column("Problem", "question")  # Standardize column name
        
        examples = [row for row in df]
        logger.info(f"Loaded {len(examples)} examples from AIME 2024 split {split}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading AIME dataset. Check internet connection.[/red]")
        raise e  # Re-raise to stop execution if dataset loading fails 

def create_prompt_aime(example: Dict) -> Tuple[str, str]:
    """Create a prompt and correct answer from the example for AIME.
    
    Args:
        example (Dict): Dictionary containing the AIME problem data
        
    Returns:
        Tuple[str, str]: A tuple containing:
            - The formatted prompt with the question and instructions
            - The correct answer as a 3-digit string (e.g., '042')
    """
    # Build the prompt with clear instructions for AIME format
    prompt = f"""Answer the following math problem.\nThe last line of your response should be your integer answer within \\boxed{{}}.\n\n{example.get('question', 'N/A')}\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering."""

    # Get the correct answer and ensure it's a 3-digit string
    correct_answer = str(example.get('Answer', 0))
    
    return prompt, correct_answer 

def extract_answer_aime(content: Optional[str]) -> Optional[str]:
    """Extracts the final three-digit answer from the model's response.
    
    Args:
        content (Optional[str]): The model's response text
        
    Returns:
        Optional[str]: The extracted answer as a three-digit string (e.g., '042'),
                      or None if no valid answer is found
    """
    if content is None:
        logger.error("[red]AIME extract_answer: Content is None.[/red]")
        return None

    # Regex to find \boxed{NUMBER} pattern, accepting any number
    # In a string literal, \\boxed means \boxed in the actual content
    BOXED_PATTERN = r'\\boxed\{(\d+)\}'

    # Search the entire content, but prioritize the last match if multiple exist
    matches = list(re.finditer(BOXED_PATTERN, content))
    
    # Alternative regex patterns to try if the first one doesn't work
    if not matches and content:
        # Try with different patterns for robustness
        alt_patterns = [
            r'boxed\{(\d+)\}',        # In case the backslash is missing
            r'\\boxed\{\s*(\d+)\s*\}', # Allow spaces inside braces
            r'boxed\{\s*(\d+)\s*\}'    # Missing backslash + spaces
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
    extracted_number = last_match.group(1)
    
    # Convert to integer and format as string
    try:
        num = int(extracted_number)
        return str(num)
    except ValueError:
        logger.error(f"[red]Invalid number format in AIME answer: {extracted_number}.[/red]")
        return None

def calculate_score_aime(extracted_answer: Optional[str], correct_answer: str) -> int:
    """Calculates the score for an AIME problem (1 for correct, 0 for incorrect/missing).
    
    Args:
        extracted_answer (Optional[str]): The extracted answer from the model's response
        correct_answer (str): The correct answer for the problem
        
    Returns:
        int: 1 if the answers match exactly, 0 otherwise
    """
    # Handle None or invalid extracted answer
    if extracted_answer is None:
        logger.error(f"[red]No valid AIME answer found in extracted_answer.[/red]")
        return 0
        
    try:
        # Convert both to integers for comparison (strips leading zeros)
        extracted_num = int(extracted_answer)
        correct_num = int(correct_answer)

        # Compare the numbers
        return 1 if extracted_num == correct_num else 0
        
    except ValueError:
        # Handle case where either string isn't a valid integer
        logger.error(f"[red]Invalid AIME numbers: extracted={extracted_answer}, correct={correct_answer}.[/red]")
        return 0 

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Load dataset
    print("\n=== Testing Dataset Loading ===")
    try:
        examples = load_data_aime(split="train")
        print(f"Successfully loaded {len(examples)} examples")
        sample_problem = examples[0]
        print(f"\nSample problem: {sample_problem['question']}")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        sample_problem = {
            "question": "What is the sum of all positive integers n less than 1000 for which n^2 + 3n + 2 is divisible by n + 1?",
            "Answer": 42  # Example answer
        }
    
    # Test 2: Create prompt
    print("\n=== Testing Prompt Creation ===")
    prompt, correct_answer = create_prompt_aime(sample_problem)
    print(f"Prompt:\n{prompt}")
    print(f"Correct answer: {correct_answer}")
    
    # Test 3: Answer extraction
    print("\n=== Testing Answer Extraction ===")
    test_responses = [
        "After solving step by step, \\boxed{042}",
        "The final answer is \\boxed{42}",
        "Therefore, \\boxed{42}",
        "Invalid response with no answer",
        None,
        "\\boxed{1000}",  # Invalid AIME answer (too large)
        "\\boxed{-5}",    # Invalid AIME answer (negative)
        "Multiple answers: \\boxed{24} but actually \\boxed{42}", # Should take last one
        "Answer without braces: \\boxed42",  # Invalid format
        "Answer with spaces: \\boxed{ 42 }"  # Invalid format
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nTest {i+1}:")
        print(f"Response: {response}")
        extracted = extract_answer_aime(response)
        print(f"Extracted: {extracted}")
        
        # Test 4: Score calculation
        score = calculate_score_aime(extracted, "042")
        print(f"Score: {score}") 