"""Tests for prune.utils.math500_utils module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
import json
import tempfile
from pathlib import Path
import sys

from prune.utils.math500_utils import (
    load_data_math500,
    create_prompt_math500,
    extract_answer_math500,
    normalize_latex_expression,
    calculate_score_math500
)


class TestLoadDataMath500:
    """Test the load_data_math500 function."""
    
    @patch('prune.utils.math500_utils.load_dataset')
    def test_load_data_math500_success(self, mock_load_dataset):
        """Test successful loading of MATH-500 dataset."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.column_names = ["problem", "answer"]
        mock_dataset.rename_column = Mock(return_value=mock_dataset)
        
        # Mock the iteration properly
        mock_rows = [
            {"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 3+3?", "answer": "6"}
        ]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = load_data_math500()
        
        # Assertions
        assert len(result) == 2
        assert result[0]["problem"] == "What is 2+2?"  # Should remain as "problem" after mock
        assert result[0]["answer"] == "4"
        mock_load_dataset.assert_called_once_with("HuggingFaceH4/MATH-500", split="test")

    @patch('prune.utils.math500_utils.load_dataset')
    def test_load_data_math500_no_rename_needed(self, mock_load_dataset):
        """Test loading when dataset already has 'question' column."""
        # Setup mock dataset with 'question' column
        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "answer"]
        mock_rows = [{"question": "What is 2+2?", "answer": "4"}]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = load_data_math500()
        
        # Assertions
        assert len(result) == 1
        assert result[0]["question"] == "What is 2+2?"
        assert result[0]["answer"] == "4"

    @patch('prune.utils.math500_utils.load_dataset')
    def test_load_data_math500_exception(self, mock_load_dataset):
        """Test exception handling during dataset loading."""
        mock_load_dataset.side_effect = Exception("Network error")
        
        with pytest.raises(Exception) as exc_info:
            load_data_math500()
        
        assert "Network error" in str(exc_info.value)


class TestCreatePromptMath500:
    """Test the create_prompt_math500 function."""
    
    def test_create_prompt_math500_basic(self):
        """Test basic prompt creation."""
        example = {
            "question": "What is 2 + 2?",
            "answer": "4"
        }
        
        prompt, correct_answer = create_prompt_math500(example)
        
        assert "What is 2 + 2?" in prompt
        assert "\\boxed{}" in prompt
        assert "Think step by step" in prompt
        assert correct_answer == "4"

    def test_create_prompt_math500_with_problem_key(self):
        """Test prompt creation when example uses 'problem' key instead of 'question'."""
        example = {
            "problem": "What is the derivative of x^2?",
            "answer": "2x"
        }
        
        prompt, correct_answer = create_prompt_math500(example)
        
        assert "What is the derivative of x^2?" in prompt
        assert correct_answer == "2x"

    def test_create_prompt_math500_missing_keys(self):
        """Test prompt creation with missing keys."""
        example = {}
        
        prompt, correct_answer = create_prompt_math500(example)
        
        assert "N/A" in prompt
        assert correct_answer == "N/A"


class TestExtractAnswerMath500:
    """Test the extract_answer_math500 function."""
    
    def test_extract_answer_none_content(self):
        """Test extraction with None content."""
        result = extract_answer_math500(None)
        assert result is None

    def test_extract_answer_simple_boxed(self):
        """Test extraction with simple \\boxed{} pattern."""
        content = "The answer is \\boxed{42}."
        result = extract_answer_math500(content)
        assert result == "42"

    def test_extract_answer_nested_braces(self):
        """Test extraction with nested braces."""
        content = "The answer is \\boxed{\\frac{1}{2}}."
        result = extract_answer_math500(content)
        assert result == "\\frac{1}{2}"

    def test_extract_answer_multiple_boxed(self):
        """Test extraction with multiple \\boxed{} patterns (should return last)."""
        content = "First we have \\boxed{1} and then \\boxed{2}."
        result = extract_answer_math500(content)
        assert result == "2"

    def test_extract_answer_no_backslash(self):
        """Test extraction with boxed{} pattern without backslash."""
        content = "The answer is boxed{42}."
        result = extract_answer_math500(content)
        assert result == "42"

    def test_extract_answer_with_spaces(self):
        """Test extraction with spaces inside boxed{}."""
        content = "The answer is \\boxed{ 42 }."
        result = extract_answer_math500(content)
        assert result == "42"

    def test_extract_answer_unmatched_braces(self):
        """Test extraction with unmatched braces (should try regex fallback)."""
        content = "The answer is \\boxed{42 and incomplete"
        result = extract_answer_math500(content)
        # This should fail and return None since braces don't match
        assert result is None

    def test_extract_answer_no_boxed_pattern(self):
        """Test extraction with no \\boxed{} pattern."""
        content = "The answer is 42."
        result = extract_answer_math500(content)
        assert result is None

    def test_extract_answer_complex_nested(self):
        """Test extraction with complex nested expressions."""
        content = "The final answer is \\boxed{\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}}."
        result = extract_answer_math500(content)
        assert result == "\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}"


class TestNormalizeLatexExpression:
    """Test the normalize_latex_expression function."""
    
    def test_normalize_simple_expression(self):
        """Test normalization of simple expression."""
        expr = "42"
        result = normalize_latex_expression(expr)
        assert result == "42"

    def test_normalize_fraction(self):
        """Test normalization of fractions."""
        expr = "\\dfrac{1}{2}"
        result = normalize_latex_expression(expr)
        assert "\\frac{1}{2}" in result

    def test_normalize_degree_notation(self):
        """Test normalization of degree notation."""
        expr = "90^\\circ"
        result = normalize_latex_expression(expr)
        assert result == "90"

    def test_normalize_text_commands(self):
        """Test normalization of text commands."""
        expr = "\\text{meters}"
        result = normalize_latex_expression(expr)
        assert "meters" not in result or "text" not in result

    def test_normalize_set_spacing(self):
        """Test normalization of set spacing."""
        expr = "{1,2,3}"
        result = normalize_latex_expression(expr)
        assert "{1, 2, 3}" in result

    def test_normalize_sqrt(self):
        """Test normalization of square root."""
        expr = "\\sqrt 4"
        result = normalize_latex_expression(expr)
        assert "\\sqrt{4}" in result

    def test_normalize_plain_fraction(self):
        """Test normalization of plain fraction."""
        expr = "1/2"
        result = normalize_latex_expression(expr)
        assert "\\frac{1}{2}" in result

    def test_normalize_currency(self):
        """Test normalization of currency symbols."""
        expr = "\\$18.90"
        result = normalize_latex_expression(expr)
        assert "$" not in result
        assert "18.90" in result

    def test_normalize_percentage(self):
        """Test normalization of percentage."""
        expr = "10\\%"
        result = normalize_latex_expression(expr)
        assert "%" not in result
        assert "10" in result


class TestCalculateScoreMath500:
    """Test the calculate_score_math500 function."""
    
    def test_calculate_score_exact_match(self):
        """Test score calculation with exact match."""
        extracted = "42"
        correct = "42"
        score = calculate_score_math500(extracted, correct)
        assert score == 1

    def test_calculate_score_no_match(self):
        """Test score calculation with no match."""
        extracted = "42"
        correct = "24"
        score = calculate_score_math500(extracted, correct)
        assert score == 0

    def test_calculate_score_none_extracted(self):
        """Test score calculation with None extracted answer."""
        extracted = None
        correct = "42"
        score = calculate_score_math500(extracted, correct)
        assert score == 0

    def test_calculate_score_normalized_match(self):
        """Test score calculation where answers match after normalization."""
        extracted = "\\dfrac{1}{2}"
        correct = "\\frac{1}{2}"
        score = calculate_score_math500(extracted, correct)
        assert score == 1

    def test_calculate_score_degree_normalization(self):
        """Test score calculation with degree normalization."""
        extracted = "90^\\circ"
        correct = "90"
        score = calculate_score_math500(extracted, correct)
        assert score == 1

    def test_calculate_score_case_insensitive_true(self):
        """Test that scoring is case insensitive due to normalization."""
        extracted = "A"
        correct = "a" 
        score = calculate_score_math500(extracted, correct)
        # The normalize function converts to lowercase, so they should match
        assert score == 1


class TestMath500UtilsIntegration:
    """Integration tests for math500_utils functions."""
    
    def test_full_pipeline(self):
        """Test the full pipeline from example to score."""
        example = {
            "question": "What is 1 + 1?",
            "answer": "2"
        }
        
        # Create prompt
        prompt, correct_answer = create_prompt_math500(example)
        assert "What is 1 + 1?" in prompt
        assert correct_answer == "2"
        
        # Simulate model response
        model_response = "First, I need to add 1 + 1. The answer is \\boxed{2}."
        
        # Extract answer
        extracted = extract_answer_math500(model_response)
        assert extracted == "2"
        
        # Calculate score
        score = calculate_score_math500(extracted, correct_answer)
        assert score == 1

    def test_pipeline_with_latex(self):
        """Test pipeline with LaTeX expressions."""
        example = {
            "question": "What is half of 1?",
            "answer": "\\frac{1}{2}"
        }
        
        prompt, correct_answer = create_prompt_math500(example)
        model_response = "Half of 1 is \\boxed{\\dfrac{1}{2}}."
        
        extracted = extract_answer_math500(model_response)
        score = calculate_score_math500(extracted, correct_answer)
        
        # Should match after normalization
        assert score == 1