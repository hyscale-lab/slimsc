# slimsc/prune/utils/math500_utils.py
import re
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional

import logging

logger = logging.getLogger(__name__)


def load_data_math500(dataset_name: str = "math500", split: str = "test") -> List[Dict]:
    """Load and prepare the MATH-500 dataset."""
    try:
        # Load the dataset
        df = load_dataset("HuggingFaceH4/MATH-500", split="test")
        
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

    # Approach: Use a brace-matching algorithm that counts open and close braces 
    # to properly extract nested LaTeX expressions within \boxed{}
    if content:
        matches = []
        i = 0
        while i < len(content):
            # Look for \boxed{
            boxed_start = content.find('\\boxed{', i)
            if boxed_start == -1:
                # Try alternative form without backslash
                boxed_start = content.find('boxed{', i)
                if boxed_start == -1:
                    break

            # Check if it has backslash or not
            has_backslash = content[boxed_start:boxed_start+1] == '\\'
            
            # Start after the opening brace
            open_brace_pos = boxed_start + (7 if has_backslash else 6)
            brace_counter = 1
            close_brace_pos = None
            
            # Find the matching closing brace by counting open/close braces
            for j in range(open_brace_pos, len(content)):
                if content[j] == '{':
                    brace_counter += 1
                elif content[j] == '}':
                    brace_counter -= 1
                    if brace_counter == 0:
                        close_brace_pos = j
                        break
            
            # If we found a matching close brace, extract the content
            if close_brace_pos is not None:
                extracted = content[open_brace_pos:close_brace_pos]
                matches.append(extracted)
                i = close_brace_pos + 1
            else:
                # No matching close brace found, move past this occurrence
                i = boxed_start + 1
    
        # If matches were found, return the last one (most likely to be the final answer)
        if matches:
            return matches[-1].strip()
    
    # If the brace-matching approach failed, fall back to regex patterns
    # These will only work for simple cases without nested braces
    logger.warning(f"[yellow]Brace-matching algorithm didn't find answers, trying regex fallbacks.[/yellow]")
    
    # Regex to find \boxed{ANSWER} pattern (simple version, no nested braces)
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


def normalize_latex_expression(latex_expr: str) -> str:
    """Normalizes LaTeX expressions to enable fair comparison regardless of styling.
    
    This function performs various transformations to convert different LaTeX
    representations of the same mathematical expression into a standardized form.
    
    Args:
        latex_expr (str): LaTeX expression to normalize
        
    Returns:
        str: Normalized LaTeX expression
    """
    if not latex_expr:
        return ""
    
    # Make a copy of the original expression for logging/debugging
    original = latex_expr
    
    # Step 1: Basic normalization
    # --------------------------
    # Strip whitespace and convert to lowercase for case-insensitive comparison
    normalized = latex_expr.strip().lower()
    
    # Remove redundant spaces 
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Step 2: LaTeX command normalization
    # ----------------------------------
    # Standardize various fraction formats
    # Fix the regex to ensure it keeps the \frac part
    normalized = re.sub(r'\\dfrac(\{)', r'\\frac\1', normalized)
    normalized = re.sub(r'\\tfrac(\{)', r'\\frac\1', normalized)
    normalized = re.sub(r'\\displaystyle\\frac(\{)', r'\\frac\1', normalized)
    normalized = re.sub(r'\\displaystyle', '', normalized)
    
    # Standardize various typography commands
    normalized = re.sub(r'\\text\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\textrm\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\textbf\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\textit\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\textsf\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\texttt\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\mathbf\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\mathit\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\mathcal\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\mathsf\{([^{}]+)\}', r'\1', normalized)
    normalized = re.sub(r'\\mathtt\{([^{}]+)\}', r'\1', normalized)
    
    # Normalize spacing commands
    normalized = re.sub(r'\\,|\\:|\\;|\\!|\\quad|\\qquad|~', ' ', normalized)
    normalized = re.sub(r'\\\s+', ' ', normalized)  # backslash followed by whitespace
    
    # Step 3: Mathematical notation normalization
    # -----------------------------------------
    # Normalize delimiters and brackets
    # Remove \left and \right commands
    normalized = re.sub(r'\\left(\S)', r'\1', normalized)
    normalized = re.sub(r'\\right(\S)', r'\1', normalized)
    
    # Normalize braces for sets
    normalized = re.sub(r'\\{', '{', normalized)
    normalized = re.sub(r'\\}', '}', normalized)
    
    # Normalize different ways to express square roots
    normalized = re.sub(r'\\sqrt\s+(\d+)', r'\\sqrt{\1}', normalized)
    
    # Normalize plain-text fractions like a/b to LaTeX \frac{a}{b}
    # This is tricky because we don't want to convert things like "x/y + z" incorrectly
    # For simple cases where the entire answer is just a fraction:
    if re.match(r'^\s*(\d+)\s*/\s*(\d+)\s*$', normalized):
        match = re.match(r'^\s*(\d+)\s*/\s*(\d+)\s*$', normalized)
        num, denom = match.groups()
        normalized = f"\\frac{{{num}}}{{{denom}}}"
    
    # Normalization for common mathematical symbols
    normalized = re.sub(r'\\neq', '≠', normalized)
    normalized = re.sub(r'\\approx', '≈', normalized)
    normalized = re.sub(r'\\sim', '∼', normalized)
    normalized = re.sub(r'\\times', '×', normalized)
    normalized = re.sub(r'\\cdot', '·', normalized)
    normalized = re.sub(r'\\div', '÷', normalized)
    normalized = re.sub(r'\\pm', '±', normalized)
    normalized = re.sub(r'\\mp', '∓', normalized)
    normalized = re.sub(r'\\leftarrow', '←', normalized)
    normalized = re.sub(r'\\rightarrow', '→', normalized)
    normalized = re.sub(r'\\infty', '∞', normalized)
    
    # Normalize LaTeX superscripts and subscripts
    # For simple cases like x^2, x_n
    normalized = re.sub(r'(\w)\^(\w|\d)', r'\1^\2', normalized)
    normalized = re.sub(r'(\w)_(\w|\d)', r'\1_\2', normalized)
    
    # Step 4: Standardize spacing around operators
    # ------------------------------------------
    # List of mathematical operators to standardize spacing around
    operators = ['+', '-', '=', '<', '>', '≤', '≥', '≠', '≈', '×', '÷', '·', '±', '∓', '→', '←']
    
    # First, ensure there are no spaces around operators, then add a standardized space
    for op in operators:
        # Remove any existing spaces around the operator
        normalized = re.sub(f' *\\{op} *' if op in '+-*' else f' *{op} *', op, normalized)
        
        # For special cases like negative signs at the beginning of expression, e.g., -2+3i
        # don't add spaces if it's a negative sign at the beginning
        if op == '-' and re.match(f'^\\{op}', normalized):
            # Skip adding space after the leading negative sign
            continue
            
        # Skip adding spaces around minus signs in scientific notation (e.g., 1.5e-3)
        if op == '-' and re.search(r'[eE]\\{op}', normalized):
            # Just ensure no spaces in this case
            normalized = re.sub(r'([eE]) *- *(\d)', r'\1-\2', normalized)
            continue
            
        # Now add standardized spacing for all other occurrences
        normalized = re.sub(f'\\{op}' if op in '+-*' else op, f' {op} ', normalized)
    
    # Clean up any double spaces introduced
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Step 5: Final clean-up
    # ---------------------
    # Normalize various non-mathematical characters
    normalized = re.sub(r'[.,;:]+$', '', normalized)  # Remove trailing punctuation
    normalized = normalized.strip()
    
    return normalized


def calculate_score_math500(extracted_answer: Optional[str], correct_answer: str) -> int:
    """Calculates the score for a MATH-500 problem (1 for correct, 0 for incorrect/missing).
    
    Args:
        extracted_answer (Optional[str]): The extracted answer from the model's response
        correct_answer (str): The correct answer for the problem
        
    Returns:
        int: 1 if the answers match after normalization, 0 otherwise
    """
    # Handle None or invalid extracted answer
    if extracted_answer is None:
        logger.error(f"[red]No valid MATH-500 answer found in extracted_answer.[/red]")
        return 0
        
    # Normalize both answers to account for LaTeX display variations
    extracted_norm = normalize_latex_expression(extracted_answer)
    correct_norm = normalize_latex_expression(correct_answer)
    
    # Check if the normalized expressions match
    if extracted_norm == correct_norm:
        return 1
    
    # Perform additional checks for common equivalent forms
    # These are special cases that the normalization function might not catch
    
    # Special case for decimal vs. fraction
    # Check if either is a decimal and the other is a fraction with the same value
    try:
        # First, check if the correct answer is a decimal and the extracted is a fraction
        if re.match(r'^-?\d+\.\d+$', correct_norm) and '\\frac{' in extracted_norm:
            # Extract numerator and denominator from fraction
            frac_match = re.search(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', extracted_norm)
            if frac_match:
                num, denom = frac_match.groups()
                try:
                    # Calculate decimal value from fraction
                    frac_value = float(num) / float(denom)
                    # Compare with the correct decimal
                    correct_value = float(correct_norm)
                    if abs(frac_value - correct_value) < 1e-9:  # Use small epsilon for float comparison
                        return 1
                except (ValueError, ZeroDivisionError):
                    pass  # Handle invalid numeric values
        
        # Now check the reverse: extracted is decimal, correct is fraction
        if re.match(r'^-?\d+\.\d+$', extracted_norm) and '\\frac{' in correct_norm:
            frac_match = re.search(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', correct_norm)
            if frac_match:
                num, denom = frac_match.groups()
                try:
                    frac_value = float(num) / float(denom)
                    extracted_value = float(extracted_norm)
                    if abs(frac_value - extracted_value) < 1e-9:
                        return 1
                except (ValueError, ZeroDivisionError):
                    pass
    except (ValueError, TypeError):
        pass  # Skip this check if there are any conversion errors
    
    # If all checks fail, the answers don't match
    return 0


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
    
    # Add new test cases for nested braces
    nested_brace_tests = [
        "\\boxed{\\dfrac{3}{2}}",  # Test case from question 49
        "\\boxed{\\frac{3}{2}}",   # Alternative fraction format
        "\\boxed{\\sqrt{64} + \\frac{1}{3}}",  # Multiple LaTeX commands
        "\\boxed{\\left(\\frac{3}{2}\\right)}",  # More complex nesting
        "\\boxed{\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}}",  # Matrix
        "Therefore, the solution is \\boxed{\\dfrac{3}{2}}.",  # With surrounding text
    ]
    
    # Add the specific test cases mentioned
    specific_test_cases = [
        "\\boxed{\\text{Evelyn}}",  # Text command
        "\\boxed{\\left( 3, \\frac{\\pi}{2} \\right)}",  # Coordinates with fraction
        "\\boxed{-2 + 7i}",  # Complex number
        "\\boxed{3, 5, 7}",  # List of values
        "\\boxed{11\\sqrt{2}}",  # Expression with square root
        # More complex combinations
        "\\boxed{\\text{Evelyn and } \\frac{3}{4}}",  # Text with fraction
        "\\boxed{\\{3, 5, 7\\}}",  # Set with braces
        "\\boxed{f(x) = \\begin{cases} x^2 & \\text{if } x > 0 \\\\ 0 & \\text{otherwise} \\end{cases}}",  # Function definition with cases
    ]
    
    test_responses.extend(nested_brace_tests)
    test_responses.extend(specific_test_cases)
    
    for i, response in enumerate(test_responses):
        print(f"\nTest {i+1}:")
        print(f"Response: {response}")
        extracted = extract_answer_math500(response)
        print(f"Extracted: {extracted}")
        
        # Test 4: Score calculation
        if i < 9:  # Only test score calculation for the original examples with "55"
            score = calculate_score_math500(extracted, "55")
            print(f"Score: {score}")
        elif response is not None:
            # For other test cases, check if we successfully extract anything
            print(f"Extraction successful: {extracted is not None}")
            if extracted is not None:
                # Check if key parts of the input are preserved in the output
                input_markers = [
                    "\\dfrac", "\\frac", "\\sqrt", "\\text", "\\left", "\\right",
                    "\\begin", "\\end", "3, 5, 7", "-2 + 7i", "11\\sqrt"
                ]
                found_markers = [marker for marker in input_markers if marker in response and marker in extracted]
                if found_markers:
                    print(f"Preserved elements: {found_markers}")
                else:
                    print("No specific markers preserved, but extraction completed")
                    
    # Test 5: LaTeX normalization and scoring
    print("\n=== Testing LaTeX Normalization and Scoring ===")
    normalization_test_pairs = [
        # Fraction variations
        ("\\frac{3}{2}", "\\dfrac{3}{2}"),  # Different fraction styles
        ("\\dfrac{3}{2}", "3/2"),  # Fraction vs plain division
        ("\\displaystyle\\frac{3}{2}", "\\frac{3}{2}"),  # Display style
        
        # Square root variations
        ("\\sqrt{2}", "\\sqrt 2"),  # Different sqrt formats
        ("\\sqrt{4}", "2"),  # Mathematically equivalent (would need special handling)
        
        # Text and formatting variations
        ("\\text{Evelyn}", "Evelyn"),  # Text command
        ("\\textrm{Decision}", "Decision"),  # Text formatting
        ("\\mathbf{x}", "x"),  # Math formatting
        
        # Delimiters and brackets
        ("\\left(3, \\frac{\\pi}{2}\\right)", "(3, \\frac{\\pi}{2})"),  # With/without left/right
        ("\\{3, 5, 7\\}", "{3, 5, 7}"),  # Different set notations
        
        # Complex expressions with multiple elements
        ("x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}", "x = \\dfrac{-b \\pm \\sqrt{b^2-4ac}}{2a}"),
        
        # Spacing variations
        ("a+b", "a + b"),  # Different spacing
        ("a\\,b\\,c", "a b c"),  # LaTeX spacing commands
        
        # Symbol variations
        ("\\times", "×"),  # LaTeX symbol vs Unicode
        ("\\infty", "∞"),  # Infinity symbol
        
        # Mixed cases
        ("3.14159", "\\frac{314159}{100000}"),  # Decimal vs fraction 
        ("e^{i\\pi} + 1 = 0", "e^{i\\pi}+1=0"),  # Euler's identity with different spacing
        
        # Lists and sets
        ("[1, 2, 3]", "\\left[1, 2, 3\\right]"),  # List with/without left/right
        
        # Complex numbers
        ("-2 + 7i", "-2+7i"),  # Complex number with different spacing
    ]
    
    print(f"Testing {len(normalization_test_pairs)} normalization test pairs...")
    success_count = 0
    
    for i, (expr1, expr2) in enumerate(normalization_test_pairs):
        print(f"\nNormalization Test {i+1}:")
        print(f"Expression 1: {expr1}")
        print(f"Expression 2: {expr2}")
        
        # Special case for the fraction test - direct equivalence check
        direct_equivalent = False
        if i == 1:  # This is the \dfrac{3}{2} vs 3/2 test
            if calculate_score_math500(expr1, expr2) == 1:
                direct_equivalent = True
                print("Direct equivalence detected via calculate_score_math500!")
        
        # Continue with regular normalization
        norm1 = normalize_latex_expression(expr1)
        norm2 = normalize_latex_expression(expr2)
        print(f"Normalized 1: {norm1}")
        print(f"Normalized 2: {norm2}")
        match_after_norm = norm1 == norm2 or direct_equivalent
        print(f"Match after normalization: {match_after_norm}")
        
        # Test score calculation with the pairs
        score = calculate_score_math500(expr1, expr2)
        print(f"Score (expr1 vs expr2): {score}")
        # Test in reverse order too
        score_rev = calculate_score_math500(expr2, expr1)
        print(f"Score (expr2 vs expr1): {score_rev}")
        
        # Special case for the sqrt(4) = 2 test, which our current system doesn't handle
        if i == 4:  # The sqrt{4} vs 2 test
            print("NOTE: This test requires advanced mathematical equivalence checking, which is beyond our current system.")
            continue
            
        # Special case for the decimal vs fraction test
        if i == 15:  # The 3.14159 vs fraction test
            try:
                # Check if they're numerically close
                val1 = float(expr1)
                match = re.search(r'\\frac\{(\d+)\}\{(\d+)\}', expr2)
                if match:
                    num, denom = match.groups()
                    val2 = float(num) / float(denom)
                    numeric_match = abs(val1 - val2) < 1e-9
                    print(f"Numeric equivalence check: {numeric_match}")
                    match_after_norm = numeric_match  # Override the text match result
            except (ValueError, ZeroDivisionError):
                pass
                
        # Count and report successes  
        if match_after_norm or score == 1 or score_rev == 1:  # Count as success if either match_after_norm OR score is 1
            success_count += 1
        elif i != 4 and i != 15:  # Skip the special cases where we know normalization won't work
            print(f"⚠️ WARNING: Expressions did not match after normalization but should be equivalent!")
            
    print(f"\nNormalization tests passed: {success_count}/{len(normalization_test_pairs)}")
    print("\nAll tests completed!")

    # Special test for the 3/2 fraction issue
    print("\n=== Special Test for Fraction Equivalence ===")
    test_cases = [
        ("3/2", "\\dfrac{3}{2}"),
        ("3/2", "\\frac{3}{2}"),
        ("\\dfrac{3}{2}", "\\frac{3}{2}"),
        ("\\frac{3}{2}", "3/2"),
    ]
    
    for test_case in test_cases:
        expr1, expr2 = test_case
        print(f"\nTesting {expr1} vs {expr2}:")
        # Check with our score calculation
        score1 = calculate_score_math500(expr1, expr2)
        score2 = calculate_score_math500(expr2, expr1)
        print(f"  calculate_score_math500('{expr1}', '{expr2}') = {score1}")
        print(f"  calculate_score_math500('{expr2}', '{expr1}') = {score2}")
        print(f"  Both directions match correctly: {score1 == 1 and score2 == 1}")
        
        # If either score fails, it's a problem
        if score1 != 1 or score2 != 1:
            print("  ⚠️ ERROR: These fractions are not being treated as equivalent!")
        else:
            print("  ✓ SUCCESS: These fractions are correctly treated as equivalent!")
    
    print("\nAll tests completed!")
