# slimsc/prune/utils/hmmt_utils.py
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import logging
import sympy

logger = logging.getLogger(__name__)

def load_data_hmmt(dataset_name: str = "hmmt", split: str = "train") -> List[Dict]:
    """Load and prepare the HMMT dataset."""
    try:
        # Load the dataset from the specified split
        df = load_dataset("MathArena/hmmt_feb_2025", split=split)
        
        # Rename the 'problem' column to 'question' for consistency
        if "problem" in df.column_names:
            df = df.rename_column("problem", "question")
        
        examples = [row for row in df]
        logger.info(f"Loaded {len(examples)} examples from HMMT split {split}.")
        return examples
    except Exception as e:
        logger.exception(f"[red]Error loading HMMT dataset. Check internet connection and dataset name.[/red]")
        raise e


def create_prompt_hmmt(example: Dict) -> Tuple[str, str]:
    """Create a prompt and correct answer from the example for HMMT.
    
    Args:
        example (Dict): Dictionary containing the HMMT problem data.
                        Expected keys: 'question' (or 'problem'), 'answer'.
        
    Returns:
        Tuple[str, str]: A tuple containing:
            - The formatted prompt with the question and instructions.
            - The correct answer string.
    """
    # Extract question from the example dictionary
    question = example.get('question', example.get('problem', 'N/A'))
    
    # Build the prompt with clear instructions for the language model
    prompt = f"""Answer the following math problem.\nThe last line of your response should be your answer within \\boxed{{}}. Your answer should be left in the most appropriate LaTeX format.\n\n{question}\n\nPut your final answer within \\boxed{{}}\nThink step by step before answering."""

    # Extract the correct answer from the example
    correct_answer = example.get('answer', 'N/A')
    
    return prompt, correct_answer


def extract_answer_hmmt(content: Optional[str]) -> Optional[str]:
    """
    Extracts the answer(s) from the model's response.
    Handles single or multiple \boxed{} expressions by finding all instances.
    """
    if content is None:
        logger.error("[red]HMMT extract_answer: Content is None.[/red]")
        return None

    # Use a brace-matching algorithm to properly extract nested LaTeX from all \boxed{} instances
    matches = []
    i = 0
    while i < len(content):
        boxed_start = content.find('\\boxed{', i)
        if boxed_start == -1:
            break

        open_brace_pos = boxed_start + len('\\boxed{')
        brace_counter = 1
        close_brace_pos = -1
        
        for j in range(open_brace_pos, len(content)):
            if content[j] == '{':
                brace_counter += 1
            elif content[j] == '}':
                brace_counter -= 1
                if brace_counter == 0:
                    close_brace_pos = j
                    break
        
        if close_brace_pos != -1:
            extracted = content[open_brace_pos:close_brace_pos].strip()
            matches.append(extracted)
            i = close_brace_pos + 1
        else:
            # Avoid infinite loop if a closing brace is not found
            i = boxed_start + 1

    if not matches:
        logger.error(f"[red]No \\boxed{{}} answer found in content for HMMT.[/red]")
        return None
    
    # Join multiple answers with a comma, return single answer as is.
    return ", ".join(matches)


def normalize_latex_expression(latex_expr: str) -> str:
    """
    Normalizes LaTeX expressions to enable fair comparison regardless of styling.
    """
    if not latex_expr:
        return ""
    
    # Basic normalization: strip whitespace
    normalized = latex_expr.strip()

    # Remove comma separators in numbers (e.g., 10,080 -> 10080)
    normalized = re.sub(r'(?<=\d),(?=\d{3})', '', normalized)
    
    # Remove LaTeX-specific spacing commands
    normalized = re.sub(r'\\,|\\:|\\;|\\!|\\quad|\\qquad|~', '', normalized)
    
    # Standardize fraction commands
    normalized = re.sub(r'\\dfrac', r'\\frac', normalized)
    normalized = re.sub(r'\\tfrac', r'\\frac', normalized)
    
    # Normalize \left and \right commands
    normalized = re.sub(r'\\left\(', '(', normalized)
    normalized = re.sub(r'\\right\)', ')', normalized)
    normalized = re.sub(r'\\left\[', '[', normalized)
    normalized = re.sub(r'\\right\]', ']', normalized)
    normalized = re.sub(r'\\left\\{', '{', normalized)
    normalized = re.sub(r'\\right\\}', '}', normalized)

    # Remove text formatting commands but keep the content
    text_commands = [
        r'\\text', r'\\textrm', r'\\textbf', r'\\textit', r'\\mathbf', r'\\mathrm'
    ]
    for cmd in text_commands:
        normalized = re.sub(cmd + r'\{([^{}]+)\}', r'\1', normalized)
    
    # Normalize multiplication symbols
    normalized = re.sub(r'\\cdot', r'\\times', normalized)
    
    # Remove all whitespace
    normalized = re.sub(r'\s+', '', normalized)

    return normalized


def to_sympy(latex_expr: str) -> Optional[sympy.Expr]:
    """Convert a LaTeX expression to a sympy expression."""
    try:
        parsable_str = latex_expr

        # Iteratively apply replacements for nested LaTeX functions.
        for _ in range(10):  # Limit iterations to prevent infinite loops
            original_str = parsable_str
            parsable_str = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', parsable_str)
            parsable_str = re.sub(r'\\sqrt\{([^{}]+)\}', r'sqrt(\1)', parsable_str)
            if parsable_str == original_str:
                break
        else:
            logger.warning(f"Expression might not be fully parsed: {latex_expr}")

        parsable_str = re.sub(r'\^\{?([^{}]+)\}?', r'**(\1)', parsable_str)
        parsable_str = re.sub(r'\\cdot|\\times', '*', parsable_str)
        parsable_str = re.sub(r'\\pi', 'pi', parsable_str)
        
        parsable_str = parsable_str.replace('{', '').replace('}', '')

        parsable_str = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', parsable_str)
        parsable_str = re.sub(r'(\))([a-zA-Z(])', r'\1*\2', parsable_str)

        expr = sympy.sympify(parsable_str, locals={"sqrt": sympy.sqrt, "pi": sympy.pi})
        return expr
    except (sympy.SympifyError, SyntaxError, TypeError, Exception) as e:
        logger.debug(f"Failed to convert '{latex_expr}' (as '{parsable_str}') to sympy expression: {e}")
        return None

def is_equivalent(expr1: str, expr2: str) -> bool:
    """Check if two LaTeX expressions are mathematically equivalent."""
    norm_expr1 = normalize_latex_expression(expr1)
    norm_expr2 = normalize_latex_expression(expr2)

    if ',' in norm_expr1 or ',' in norm_expr2:
        parts1 = sorted([p.strip() for p in norm_expr1.split(',')])
        parts2 = sorted([p.strip() for p in norm_expr2.split(',')])
        if len(parts1) != len(parts2):
            return False
        return all(is_equivalent(p1, p2) for p1, p2 in zip(parts1, parts2))

    if norm_expr1 == norm_expr2:
        return True

    sympy_expr1 = to_sympy(norm_expr1)
    sympy_expr2 = to_sympy(norm_expr2)

    if sympy_expr1 is not None and sympy_expr2 is not None:
        try:
            # Expand and simplify the difference. If it's zero, they are equivalent.
            if sympy.simplify(sympy.expand(sympy_expr1) - sympy.expand(sympy_expr2)) == 0:
                return True
        except (TypeError, AttributeError, Exception) as e:
            logger.error(f"Error comparing sympy expressions '{sympy_expr1}' and '{sympy_expr2}': {e}")
    
    return False

def calculate_score_hmmt(extracted_answer: Optional[str], correct_answer: str) -> int:
    """
    Calculates the score for a HMMT problem (1 for correct, 0 for incorrect)
    using mathematical equivalence.
    """
    if extracted_answer is None:
        logger.error(f"[red]No valid HMMT answer found in extracted_answer.[/red]")
        return 0
        
    if is_equivalent(extracted_answer, correct_answer):
        return 1
            
    return 0


if __name__ == "__main__":
    from rich.logging import RichHandler
    
    logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(markup=True, rich_tracebacks=True)] # Enable tracebacks
    )
    
    # Test 1: Load dataset
    print("\n=== Testing HMMT Dataset Loading ===")
    try:
        examples = load_data_hmmt(split="train")
        print(f"Successfully loaded {len(examples)} examples from HMMT train split.")
        if examples:
            sample_problem = examples[0]
            print(f"\nSample problem: {sample_problem.get('question', 'N/A')}")
            print(f"Sample answer: {sample_problem.get('answer', 'N/A')}")
        else:
            raise ValueError("No examples loaded.")
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        sample_problem = {
            "question": "Find all real solutions to $x^2+x-4=0$.",
            "answer": "\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}"
        }
    
    # Test 2: Create prompt
    print("\n=== Testing HMMT Prompt Creation ===")
    prompt, correct_answer = create_prompt_hmmt(sample_problem)
    print(f"Prompt:\n{prompt}")
    print(f"Correct answer: {correct_answer}")
    
    # Test 3: Answer extraction with examples from the prompt
    print("\n=== Testing HMMT Answer Extraction ===")
    test_responses = [
        "The answer is \\boxed{\\frac{-1+\\sqrt{17}}{2}}",
        "My final answer is \\boxed{\\frac{-1-\\sqrt{17}}{2}, \\frac{-1+\\sqrt{17}}{2}}.",
        "The solution is \\boxed{1-\\frac{2}{\\pi}}.",
        "So we get \\boxed{2^{25} \\cdot 26!}",
        "So the answer must be \\boxed{20}.",
        "Invalid response",
        r"The solutions are \boxed{6} \quad \text{and} \quad \boxed{-9}",
        r"The roots are \boxed{\dfrac{-1 + \sqrt{17}}{2}} \quad \text{and} \quad \boxed{\dfrac{-1 - \sqrt{17}}{2}}",
    ]
    
    for i, response in enumerate(test_responses):
        print(f"\nTest {i+1}:")
        print(f"Response: '{response}'")
        extracted = extract_answer_hmmt(response)
        print(f"Extracted: '{extracted}'")
        
    # Test 4: Scoring and normalization
    print("\n=== Testing HMMT Scoring & Normalization ===")
    test_pairs = [
        # Normalization cases
        ("\\frac{-1+\\sqrt{17}}{2}", "\\dfrac{-1+\\sqrt{17}}{2}", 1),
        ("1 - \\frac{2}{\\pi}", "1-\\frac{2}{\\pi}", 1),
        ("2^{25} \\cdot 26!}", "2^{25} \\times 26!", 1),
        # Order-agnostic multiple answers
        ("\\frac{-1-\\sqrt{17}}{2}, \\frac{-1+\\sqrt{17}}{2}", "\\frac{-1+\\sqrt{17}}{2},\\frac{-1-\\sqrt{17}}{2}", 1),
        # Sympy equivalence
        ("x*y", "y*x", 1),
        ("1/2", "0.5", 1),
        ("\\frac{\\sqrt{4}}{2}", "1", 1),
        # Multiple boxed answers equivalence
        (r"\dfrac{-1 + \sqrt{17}}{2}, \dfrac{-1 - \sqrt{17}}{2}", "\\frac{-1-\\sqrt{17}}{2}, \\frac{-1+\\sqrt{17}}{2}", 1),
        ("6, -9", "-9, 6", 1),
        # Incorrect case
        ("20", "21", 0)
    ]
    
    for i, (ans1, ans2, expected_score) in enumerate(test_pairs):
        print(f"\nTest {i+1}: Comparing '{ans1}' and '{ans2}'")
        score = calculate_score_hmmt(ans1, ans2)
        print(f"Calculated Score: {score}, Expected: {expected_score}")
        assert score == expected_score, f"Test failed for pair {i+1}"
        print("✓ Test Passed")

    print("\nAll HMMT util tests completed!")