# slimsc/prune/utils/aime_utils.py
import random
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import transformers

import logging

logger = logging.getLogger(__name__)


def load_data_aime(dataset_name: str = "aime2024", split: str = "train") -> List[Dict]:
    """Load and prepare the AIME dataset."""
    try:
        # Use specific subset name directly if that's how it's identified
        df = load_dataset("Maxwell-Jia/AIME_2024", name=dataset_name, split=split)
        df = df.rename_column("Problem", "question")  # Standardize column name
        
        examples = [row for row in df]
        logger.info(f"Loaded {len(examples)} examples from AIME {dataset_name} split {split}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading AIME dataset '{dataset_name}'. Check dataset name and internet connection.[/red]")
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
    prompt = f"""Answer the following AIME (American Invitational Mathematics Examination) problem.
                Think step-by-step to reach the solution.
                Your answer should be a three-digit integer between 000 and 999 inclusive.
                Conclude your response with a single line containing the answer formatted exactly as 'Answer: $NUMBER'.

                Problem: {example.get('question', 'N/A')}"""

    # Get the correct answer and ensure it's a 3-digit string
    correct_answer = str(example.get('Answer', 0)).zfill(3)
    
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
        return None

    # Regex to find "Answer: $NUMBER" pattern, where NUMBER is 1-3 digits
    ANSWER_PATTERN = r"(?i)Answer[ \t]*:[ \t]*\$?(\d{1,3})\$?"

    # Search the entire content, but prioritize the last match if multiple exist
    matches = list(re.finditer(ANSWER_PATTERN, content))

    if matches:
        # Get the last match found
        last_match = matches[-1]
        extracted_number = last_match.group(1)
        
        # Validate the number is in valid AIME range (0-999)
        try:
            num = int(extracted_number)
            if 0 <= num <= 999:
                return str(num).zfill(3)  # Return as 3-digit string
        except ValueError:
            pass
        
        logger.error(f"[red]Invalid number found in AIME answer: {extracted_number}.[/red]")
        return None
    
    # Fallback: Check if the last line contains just a number between 0-999
    lines = content.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        try:
            num = int(line)
            if 0 <= num <= 999:
                logger.info(f"[yellow]Found valid fallback AIME answer: {num}.[/yellow]")
                return str(num).zfill(3)
        except ValueError:
            continue
        
    logger.error(f"[red]No valid AIME answer found in content: {content}.[/red]")
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
        
        # Validate both numbers are in valid AIME range
        if not (0 <= extracted_num <= 999 and 0 <= correct_num <= 999):
            logger.error(f"[red]Invalid AIME numbers: extracted={extracted_num}, correct={correct_num}.[/red]")
            return 0
            
        # Compare the numbers
        return 1 if extracted_num == correct_num else 0
        
    except ValueError:
        # Handle case where either string isn't a valid integer
        logger.error(f"[red]Invalid AIME numbers: extracted={extracted_answer}, correct={correct_answer}.[/red]")
        return 0 