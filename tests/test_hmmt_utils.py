"""Tests for prune.utils.hmmt_utils module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import sympy

from prune.utils.hmmt_utils import (
    load_data_hmmt,
    create_prompt_hmmt,
    extract_answer_hmmt,
    normalize_latex_expression,
    to_sympy,
    is_equivalent,
    calculate_score_hmmt
)


class TestLoadDataHmmt:
    """Test the load_data_hmmt function."""
    
    @patch('prune.utils.hmmt_utils.load_dataset')
    def test_load_data_hmmt_success(self, mock_load_dataset):
        """Test successful loading of HMMT dataset."""
        # Setup mock dataset with proper behavior after renaming
        mock_rows = [
            {"question": "Find all solutions to x^2 = 4", "answer": "\\pm 2", "id": "1"},
            {"question": "Calculate 2+2", "answer": "4", "id": "2"}
        ]
        
        # Create a proper mock that simulates rename behavior
        mock_dataset = Mock()
        mock_dataset.column_names = ["problem", "answer", "id"]
        
        def rename_column_side_effect(old_name, new_name):
            if old_name == "problem" and new_name == "question":
                # Update the mock rows to use 'question' instead of 'problem'
                for row in mock_rows:
                    if "problem" in row:
                        row["question"] = row.pop("problem")
                mock_dataset.column_names = ["question", "answer", "id"]
            return mock_dataset
        
        mock_dataset.rename_column.side_effect = rename_column_side_effect
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset

        # Test
        result = load_data_hmmt("hmmt", "train")

        # Assertions
        assert len(result) == 2
        assert result[0]["question"] == "Find all solutions to x^2 = 4"
        assert result[0]["answer"] == "\\pm 2"
        assert result[1]["question"] == "Calculate 2+2"
        assert result[1]["answer"] == "4"
        
        mock_load_dataset.assert_called_once_with("MathArena/hmmt_feb_2025", split="train")
        mock_dataset.rename_column.assert_called_once_with("problem", "question")
    
    @patch('prune.utils.hmmt_utils.load_dataset')
    def test_load_data_hmmt_no_rename_needed(self, mock_load_dataset):
        """Test when dataset already has 'question' column."""
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "answer"]
        mock_rows = [{"question": "What is 2+2?", "answer": "4"}]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        result = load_data_hmmt()
        
        assert len(result) == 1
        # Should not call rename_column since "problem" not in column_names
        assert not hasattr(mock_dataset, 'rename_column') or not mock_dataset.rename_column.called
    
    @patch('prune.utils.hmmt_utils.load_dataset')
    def test_load_data_hmmt_exception(self, mock_load_dataset):
        """Test handling of dataset loading exception."""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        with pytest.raises(Exception, match="Dataset not found"):
            load_data_hmmt("hmmt", "train")


class TestCreatePromptHmmt:
    """Test the create_prompt_hmmt function."""
    
    def test_create_prompt_hmmt_basic(self):
        """Test basic prompt creation."""
        example = {
            "question": "Find the derivative of x^2",
            "answer": "2x"
        }
        
        prompt, correct_answer = create_prompt_hmmt(example)
        
        # Check prompt structure
        assert "Answer the following math problem" in prompt
        assert "Find the derivative of x^2" in prompt
        assert "\\boxed{}" in prompt
        assert "Think step by step" in prompt
        assert "LaTeX format" in prompt
        
        # Check correct answer
        assert correct_answer == "2x"
    
    def test_create_prompt_hmmt_with_problem_key(self):
        """Test prompt creation with 'problem' key instead of 'question'."""
        example = {
            "problem": "Solve for x: x + 1 = 5",
            "answer": "4"
        }
        
        prompt, correct_answer = create_prompt_hmmt(example)
        
        assert "Solve for x: x + 1 = 5" in prompt
        assert correct_answer == "4"
    
    def test_create_prompt_hmmt_missing_question(self):
        """Test prompt creation with missing question/problem."""
        example = {"answer": "42"}
        
        prompt, correct_answer = create_prompt_hmmt(example)
        
        assert "N/A" in prompt
        assert correct_answer == "42"
    
    def test_create_prompt_hmmt_missing_answer(self):
        """Test prompt creation with missing answer."""
        example = {"question": "What is the meaning of life?"}
        
        prompt, correct_answer = create_prompt_hmmt(example)
        
        assert "What is the meaning of life?" in prompt
        assert correct_answer == "N/A"


class TestExtractAnswerHmmt:
    """Test the extract_answer_hmmt function."""
    
    def test_extract_answer_hmmt_none_content(self):
        """Test with None content."""
        result = extract_answer_hmmt(None)
        assert result is None
    
    def test_extract_answer_hmmt_single_boxed(self):
        """Test with single \\boxed{} answer."""
        content = "The solution is \\boxed{\\frac{1}{2}}"
        result = extract_answer_hmmt(content)
        assert result == "\\frac{1}{2}"
    
    def test_extract_answer_hmmt_multiple_boxed(self):
        """Test with multiple \\boxed{} answers."""
        content = "First answer: \\boxed{x} and second: \\boxed{y}"
        result = extract_answer_hmmt(content)
        assert result == "x, y"
    
    def test_extract_answer_hmmt_nested_braces(self):
        """Test with nested braces in LaTeX."""
        content = "The answer is \\boxed{\\frac{-1+\\sqrt{17}}{2}}"
        result = extract_answer_hmmt(content)
        assert result == "\\frac{-1+\\sqrt{17}}{2}"
    
    def test_extract_answer_hmmt_complex_latex(self):
        """Test with complex LaTeX expressions."""
        content = "Result: \\boxed{\\sum_{i=1}^{n} \\frac{1}{i^2}}"
        result = extract_answer_hmmt(content)
        assert result == "\\sum_{i=1}^{n} \\frac{1}{i^2}"
    
    def test_extract_answer_hmmt_no_answer(self):
        """Test with content containing no boxed answer."""
        content = "This is just some text without any boxed answers."
        result = extract_answer_hmmt(content)
        assert result is None
    
    def test_extract_answer_hmmt_unmatched_braces(self):
        """Test with unmatched braces."""
        content = "Incomplete: \\boxed{\\frac{1}{2}"
        result = extract_answer_hmmt(content)
        assert result is None
    
    def test_extract_answer_hmmt_empty_boxed(self):
        """Test with empty boxed expression."""
        content = "Empty answer: \\boxed{}"
        result = extract_answer_hmmt(content)
        assert result == ""
    
    def test_extract_answer_hmmt_whitespace_handling(self):
        """Test whitespace handling in answers."""
        content = "Answer: \\boxed{  \\frac{1}{2}  }"
        result = extract_answer_hmmt(content)
        assert result == "\\frac{1}{2}"


class TestNormalizeLatexExpression:
    """Test the normalize_latex_expression function."""
    
    def test_normalize_latex_empty(self):
        """Test with empty string."""
        result = normalize_latex_expression("")
        assert result == ""
    
    def test_normalize_latex_basic(self):
        """Test basic normalization."""
        result = normalize_latex_expression("  \\frac{1}{2}  ")
        assert result == "\\frac{1}{2}"
    
    def test_normalize_latex_commas_in_numbers(self):
        """Test removal of commas in numbers."""
        result = normalize_latex_expression("10,080")
        assert result == "10080"
    
    def test_normalize_latex_spacing_commands(self):
        """Test removal of LaTeX spacing commands."""
        test_cases = [
            ("a\\,b", "ab"),
            ("a\\:b", "ab"),
            ("a\\;b", "ab"),
            ("a\\!b", "ab"),
            ("a\\quad b", "ab"),
            ("a\\qquad b", "ab"),
            ("a~b", "ab"),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_latex_expression(input_str)
            assert result == expected
    
    def test_normalize_latex_fraction_commands(self):
        """Test normalization of fraction commands."""
        test_cases = [
            ("\\dfrac{1}{2}", "\\frac{1}{2}"),
            ("\\tfrac{1}{2}", "\\frac{1}{2}"),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_latex_expression(input_str)
            assert result == expected
    
    def test_normalize_latex_left_right_commands(self):
        """Test normalization of \\left and \\right commands."""
        test_cases = [
            ("\\left(x\\right)", "(x)"),
            ("\\left[x\\right]", "[x]"),
            ("\\left\\{x\\right\\}", "{x}"),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_latex_expression(input_str)
            assert result == expected
    
    def test_normalize_latex_text_commands(self):
        """Test removal of text formatting commands."""
        test_cases = [
            ("\\text{hello}", "hello"),
            ("\\textrm{world}", "world"),
            ("\\textbf{bold}", "bold"),
            ("\\mathbf{vector}", "vector"),
        ]
        
        for input_str, expected in test_cases:
            result = normalize_latex_expression(input_str)
            assert result == expected
    
    def test_normalize_latex_multiplication(self):
        """Test normalization of multiplication symbols."""
        result = normalize_latex_expression("a\\cdot b")
        assert result == "a\\timesb"
    
    def test_normalize_latex_whitespace_removal(self):
        """Test complete whitespace removal."""
        result = normalize_latex_expression("a + b - c")
        assert result == "a+b-c"


class TestToSympy:
    """Test the to_sympy function."""
    
    def test_to_sympy_simple_fraction(self):
        """Test conversion of simple fraction."""
        result = to_sympy("\\frac{1}{2}")
        expected = sympy.Rational(1, 2)
        assert result == expected
    
    def test_to_sympy_nested_fractions(self):
        """Test conversion of nested fractions."""
        result = to_sympy("\\frac{\\frac{1}{2}}{\\frac{3}{4}}")
        expected = sympy.Rational(2, 3)
        assert result == expected
    
    def test_to_sympy_square_root(self):
        """Test conversion of square root."""
        result = to_sympy("\\sqrt{4}")
        expected = sympy.sqrt(4)
        assert result == expected
    
    def test_to_sympy_exponents(self):
        """Test conversion of exponents."""
        result = to_sympy("x^{2}")
        expected = sympy.Symbol('x')**2
        assert result == expected
    
    def test_to_sympy_pi(self):
        """Test conversion of pi."""
        result = to_sympy("\\pi")
        expected = sympy.pi
        assert result == expected
    
    def test_to_sympy_multiplication(self):
        """Test implicit and explicit multiplication."""
        # Test implicit multiplication
        result1 = to_sympy("2x")
        expected1 = 2 * sympy.Symbol('x')
        assert result1 == expected1
        
        # Test explicit multiplication
        result2 = to_sympy("2\\cdot x")
        expected2 = 2 * sympy.Symbol('x')
        assert result2 == expected2
    
    def test_to_sympy_invalid_expression(self):
        """Test handling of invalid expressions."""
        result = to_sympy("\\invalid{command}{with}{too}{many}{braces}")
        assert result is None
    
    def test_to_sympy_complex_expression(self):
        """Test complex mathematical expression."""
        result = to_sympy("\\frac{-1+\\sqrt{17}}{2}")
        expected = (-1 + sympy.sqrt(17)) / 2
        assert result == expected


class TestIsEquivalent:
    """Test the is_equivalent function."""
    
    def test_is_equivalent_identical(self):
        """Test with identical expressions."""
        assert is_equivalent("\\frac{1}{2}", "\\frac{1}{2}") == True
    
    def test_is_equivalent_normalized_same(self):
        """Test expressions that are same after normalization."""
        assert is_equivalent("\\dfrac{1}{2}", "\\frac{1}{2}") == True
        assert is_equivalent("a \\cdot b", "a \\times b") == True
    
    def test_is_equivalent_mathematically_same(self):
        """Test mathematically equivalent expressions."""
        assert is_equivalent("\\frac{2}{4}", "\\frac{1}{2}") == True
        assert is_equivalent("x*y", "y*x") == True
    
    def test_is_equivalent_different(self):
        """Test with different expressions."""
        assert is_equivalent("\\frac{1}{2}", "\\frac{1}{3}") == False
        assert is_equivalent("x+y", "x-y") == False
    
    def test_is_equivalent_multiple_answers_same_order(self):
        """Test multiple answers in same order."""
        expr1 = "\\frac{1}{2}, \\frac{3}{4}"
        expr2 = "\\frac{1}{2}, \\frac{3}{4}"
        assert is_equivalent(expr1, expr2) == True
    
    def test_is_equivalent_multiple_answers_different_order(self):
        """Test multiple answers in different order."""
        expr1 = "\\frac{1}{2}, \\frac{3}{4}"
        expr2 = "\\frac{3}{4}, \\frac{1}{2}"
        assert is_equivalent(expr1, expr2) == True
    
    def test_is_equivalent_multiple_answers_different_count(self):
        """Test multiple answers with different counts."""
        expr1 = "\\frac{1}{2}, \\frac{3}{4}"
        expr2 = "\\frac{1}{2}"
        assert is_equivalent(expr1, expr2) == False
    
    def test_is_equivalent_sympy_conversion_failure(self):
        """Test when sympy conversion fails."""
        # These should fall back to string comparison
        expr1 = "invalid_expression_1"
        expr2 = "invalid_expression_1"
        assert is_equivalent(expr1, expr2) == True
        
        expr3 = "invalid_expression_1"
        expr4 = "invalid_expression_2"
        assert is_equivalent(expr3, expr4) == False


class TestCalculateScoreHmmt:
    """Test the calculate_score_hmmt function."""
    
    def test_calculate_score_hmmt_correct(self):
        """Test with correct answer."""
        result = calculate_score_hmmt("\\frac{1}{2}", "\\frac{1}{2}")
        assert result == 1
    
    def test_calculate_score_hmmt_incorrect(self):
        """Test with incorrect answer."""
        result = calculate_score_hmmt("\\frac{1}{2}", "\\frac{1}{3}")
        assert result == 0
    
    def test_calculate_score_hmmt_none_extracted(self):
        """Test with None extracted answer."""
        result = calculate_score_hmmt(None, "\\frac{1}{2}")
        assert result == 0
    
    def test_calculate_score_hmmt_equivalent_forms(self):
        """Test with mathematically equivalent forms."""
        result = calculate_score_hmmt("\\frac{2}{4}", "\\frac{1}{2}")
        assert result == 1
    
    def test_calculate_score_hmmt_different_latex_formatting(self):
        """Test with different LaTeX formatting of same expression."""
        result = calculate_score_hmmt("\\dfrac{1}{2}", "\\frac{1}{2}")
        assert result == 1
    
    def test_calculate_score_hmmt_multiple_answers(self):
        """Test with multiple answers."""
        extracted = "\\frac{1}{2}, \\frac{3}{4}"
        correct = "\\frac{3}{4}, \\frac{1}{2}"  # Different order
        result = calculate_score_hmmt(extracted, correct)
        assert result == 1


class TestHmmtUtilsIntegration:
    """Integration tests for HMMT utilities."""
    
    @patch('prune.utils.hmmt_utils.load_dataset')
    def test_full_pipeline(self, mock_load_dataset):
        """Test complete pipeline from dataset to scoring."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["problem", "answer"]
        mock_dataset.rename_column.return_value = mock_dataset
        mock_rows = [
            {"problem": "Find all real solutions to $x^2+x-4=0$.", 
             "answer": "\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}"}
        ]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Load data
        examples = load_data_hmmt()
        assert len(examples) == 1
        
        example = examples[0]
        
        # Create prompt
        prompt, correct_answer = create_prompt_hmmt(example)
        assert "x^2+x-4=0" in prompt
        assert "\\frac{-1+\\sqrt{17}}{2}" in correct_answer
        
        # Test various model responses
        test_responses = [
            # Correct answer in same order
            ("The solutions are \\boxed{\\frac{-1+\\sqrt{17}}{2}, \\frac{-1-\\sqrt{17}}{2}}", 1),
            # Correct answer in different order
            ("The roots are \\boxed{\\frac{-1-\\sqrt{17}}{2}, \\frac{-1+\\sqrt{17}}{2}}", 1),
            # Different LaTeX formatting but same math
            ("Answer: \\boxed{\\dfrac{-1+\\sqrt{17}}{2}, \\dfrac{-1-\\sqrt{17}}{2}}", 1),
            # Incorrect answer
            ("The answer is \\boxed{1, -4}", 0),
            # No answer
            ("I cannot solve this problem.", 0),
        ]
        
        for response, expected_score in test_responses:
            extracted = extract_answer_hmmt(response)
            score = calculate_score_hmmt(extracted, correct_answer)
            assert score == expected_score, f"Failed for response: {response}"
    
    def test_complex_latex_expressions(self):
        """Test handling of complex LaTeX expressions."""
        complex_expressions = [
            "\\sum_{i=1}^{n} \\frac{1}{i^2}",
            "\\int_{0}^{1} x^2 dx",
            "\\lim_{x \\to 0} \\frac{\\sin x}{x}",
            "\\begin{bmatrix} 1 & 2 \\\\ 3 & 4 \\end{bmatrix}",
        ]
        
        for expr in complex_expressions:
            content = f"The answer is \\boxed{{{expr}}}"
            extracted = extract_answer_hmmt(content)
            assert extracted == expr
    
    def test_mathematical_equivalence_edge_cases(self):
        """Test edge cases in mathematical equivalence."""
        equivalence_tests = [
            # Basic equivalences
            ("1/2", "0.5", True),
            ("\\sqrt{4}", "2", True),
            ("2^3", "8", True),
            
            # Non-equivalences
            ("1/2", "1/3", False),
            ("\\sqrt{2}", "2", False),
            
            # Complex equivalences
            ("x*y", "y*x", True),
            ("(x+y)^2", "(y+x)^2", True),
        ]
        
        for expr1, expr2, expected in equivalence_tests:
            result = is_equivalent(expr1, expr2)
            assert result == expected, f"Failed for {expr1} vs {expr2}"