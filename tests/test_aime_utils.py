"""Tests for prune.utils.aime_utils module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging

from prune.utils.aime_utils import (
    load_data_aime,
    create_prompt_aime,
    extract_answer_aime,
    calculate_score_aime
)


class TestLoadDataAime:
    """Test the load_data_aime function."""
    
    @patch('prune.utils.aime_utils.load_dataset')
    def test_load_data_aime_success(self, mock_load_dataset):
        """Test successful loading of AIME dataset."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_rows = [
            {"question": "What is 2+2?", "Answer": 4},
            {"question": "What is 3*3?", "Answer": 9}
        ]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = load_data_aime("aime2024", "train")
        
        # Assertions
        assert len(result) == 2
        assert result[0]["question"] == "What is 2+2?"
        assert result[0]["Answer"] == 4
        assert result[1]["question"] == "What is 3*3?"
        assert result[1]["Answer"] == 9
        
        mock_load_dataset.assert_called_once_with("Maxwell-Jia/AIME_2024", split="train")
        mock_dataset.rename_column.assert_called_once_with("Problem", "question")
    
    @patch('prune.utils.aime_utils.load_dataset')
    def test_load_data_aime_exception(self, mock_load_dataset):
        """Test handling of dataset loading exception."""
        mock_load_dataset.side_effect = Exception("Dataset not found")
        
        with pytest.raises(Exception, match="Dataset not found"):
            load_data_aime("aime2024", "train")
    
    @patch('prune.utils.aime_utils.load_dataset')
    def test_load_data_aime_default_params(self, mock_load_dataset):
        """Test with default parameters."""
        mock_dataset = Mock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_dataset.__iter__ = Mock(return_value=iter([]))
        mock_load_dataset.return_value = mock_dataset
        
        result = load_data_aime()  # Use defaults
        
        mock_load_dataset.assert_called_once_with("Maxwell-Jia/AIME_2024", split="train")
        assert isinstance(result, list)


class TestCreatePromptAime:
    """Test the create_prompt_aime function."""
    
    def test_create_prompt_aime_basic(self):
        """Test basic prompt creation."""
        example = {
            "question": "Find the value of x if x^2 = 16",
            "Answer": 4
        }
        
        prompt, correct_answer = create_prompt_aime(example)
        
        # Check prompt structure
        assert "Answer the following math problem" in prompt
        assert "Find the value of x if x^2 = 16" in prompt
        assert "\\boxed{}" in prompt
        assert "Think step by step" in prompt
        
        # Check correct answer
        assert correct_answer == "4"
    
    def test_create_prompt_aime_missing_question(self):
        """Test prompt creation with missing question."""
        example = {"Answer": 42}
        
        prompt, correct_answer = create_prompt_aime(example)
        
        assert "N/A" in prompt
        assert correct_answer == "42"
    
    def test_create_prompt_aime_missing_answer(self):
        """Test prompt creation with missing answer."""
        example = {"question": "What is the meaning of life?"}
        
        prompt, correct_answer = create_prompt_aime(example)
        
        assert "What is the meaning of life?" in prompt
        assert correct_answer == "0"  # Default
    
    def test_create_prompt_aime_integer_answer(self):
        """Test with integer answer."""
        example = {
            "question": "Calculate 5 + 7",
            "Answer": 12
        }
        
        prompt, correct_answer = create_prompt_aime(example)
        
        assert correct_answer == "12"
    
    def test_create_prompt_aime_string_answer(self):
        """Test with string answer."""
        example = {
            "question": "What is 2^3?",
            "Answer": "8"
        }
        
        prompt, correct_answer = create_prompt_aime(example)
        
        assert correct_answer == "8"


class TestExtractAnswerAime:
    """Test the extract_answer_aime function."""
    
    def test_extract_answer_aime_none_content(self):
        """Test with None content."""
        result = extract_answer_aime(None)
        assert result is None
    
    def test_extract_answer_aime_valid_boxed(self):
        """Test with valid \\boxed{} answer."""
        content = "After solving step by step, \\boxed{042}"
        result = extract_answer_aime(content)
        assert result == "42"  # Should convert to integer and back to string
    
    def test_extract_answer_aime_boxed_no_backslash(self):
        """Test with boxed{} (missing backslash)."""
        content = "The answer is boxed{123}"
        result = extract_answer_aime(content)
        assert result == "123"
    
    def test_extract_answer_aime_boxed_with_spaces(self):
        """Test with spaces inside braces."""
        content = "Therefore, \\boxed{ 456 }"
        result = extract_answer_aime(content)
        assert result == "456"
    
    def test_extract_answer_aime_multiple_boxed(self):
        """Test with multiple boxed answers (should take last)."""
        content = "First try: \\boxed{100} but actually \\boxed{200}"
        result = extract_answer_aime(content)
        assert result == "200"
    
    def test_extract_answer_aime_no_answer(self):
        """Test with content containing no boxed answer."""
        content = "This is just some text without any answer format."
        result = extract_answer_aime(content)
        assert result is None
    
    def test_extract_answer_aime_invalid_number(self):
        """Test with invalid number in boxed answer."""
        content = "The answer is \\boxed{not_a_number}"
        result = extract_answer_aime(content)
        assert result is None
    
    def test_extract_answer_aime_zero_answer(self):
        """Test with zero as answer."""
        content = "The result is \\boxed{0}"
        result = extract_answer_aime(content)
        assert result == "0"
    
    def test_extract_answer_aime_large_number(self):
        """Test with large number."""
        content = "The solution is \\boxed{999}"
        result = extract_answer_aime(content)
        assert result == "999"
    
    def test_extract_answer_aime_leading_zeros(self):
        """Test with leading zeros (should be removed)."""
        content = "Answer: \\boxed{0042}"
        result = extract_answer_aime(content)
        assert result == "42"
    
    def test_extract_answer_aime_alternative_patterns(self):
        """Test various alternative patterns."""
        test_cases = [
            ("\\boxed{42}", "42"),
            ("boxed{42}", "42"),
            ("\\boxed{ 42 }", "42"),
            ("boxed{ 42 }", "42"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_aime(content)
            assert result == expected, f"Failed for content: {content}"


class TestCalculateScoreAime:
    """Test the calculate_score_aime function."""
    
    def test_calculate_score_aime_correct_match(self):
        """Test with correct answer match."""
        result = calculate_score_aime("42", "42")
        assert result == 1
    
    def test_calculate_score_aime_incorrect_match(self):
        """Test with incorrect answer."""
        result = calculate_score_aime("42", "43")
        assert result == 0
    
    def test_calculate_score_aime_none_extracted(self):
        """Test with None extracted answer."""
        result = calculate_score_aime(None, "42")
        assert result == 0
    
    def test_calculate_score_aime_leading_zeros(self):
        """Test with leading zeros in answers."""
        result = calculate_score_aime("042", "42")
        assert result == 1  # Should match after int conversion
    
    def test_calculate_score_aime_different_format(self):
        """Test with different string formats of same number."""
        result = calculate_score_aime("0042", "042")
        assert result == 1  # Both convert to same integer
    
    def test_calculate_score_aime_zero_answers(self):
        """Test with zero answers."""
        result = calculate_score_aime("0", "000")
        assert result == 1  # Both are zero
    
    def test_calculate_score_aime_invalid_extracted(self):
        """Test with invalid extracted answer."""
        result = calculate_score_aime("invalid", "42")
        assert result == 0
    
    def test_calculate_score_aime_invalid_correct(self):
        """Test with invalid correct answer."""
        result = calculate_score_aime("42", "invalid")
        assert result == 0
    
    def test_calculate_score_aime_both_invalid(self):
        """Test with both answers invalid."""
        result = calculate_score_aime("invalid1", "invalid2")
        assert result == 0
    
    def test_calculate_score_aime_string_integers(self):
        """Test with string representations of integers."""
        test_cases = [
            ("123", "123", 1),
            ("123", "124", 0),
            ("0", "0", 1),
            ("1", "01", 1),
            ("100", "0100", 1),
        ]
        
        for extracted, correct, expected in test_cases:
            result = calculate_score_aime(extracted, correct)
            assert result == expected, f"Failed for {extracted} vs {correct}"


class TestAimeUtilsIntegration:
    """Integration tests for AIME utilities."""
    
    @patch('prune.utils.aime_utils.load_dataset')
    def test_full_pipeline(self, mock_load_dataset):
        """Test complete pipeline from dataset to scoring."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.rename_column.return_value = mock_dataset
        
        # Create rows with the renamed column "question" instead of "Problem"
        mock_rows = [
            {"question": "If x^2 + 3x + 2 = 0, find the sum of the roots.", "Answer": "3"}
        ]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Load data
        examples = load_data_aime()
        assert len(examples) == 1
        
        example = examples[0]
        
        # Create prompt
        prompt, correct_answer = create_prompt_aime(example)
        assert "x^2 + 3x + 2 = 0" in prompt
        assert correct_answer == "3"
        
        # Test various model responses
        test_responses = [
            ("After solving: x^2 + 3x + 2 = (x+1)(x+2) = 0, so roots are -1 and -2. Sum = \\boxed{3}", 1),
            ("The sum of roots is \\boxed{-3}", 0),  # Incorrect
            ("I don't know the answer", 0),  # No answer
            ("The roots sum to \\boxed{003}", 1),  # Leading zeros, should still match
        ]
        
        for response, expected_score in test_responses:
            extracted = extract_answer_aime(response)
            score = calculate_score_aime(extracted, correct_answer)
            assert score == expected_score, f"Failed for response: {response}"
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Empty example
        empty_example = {}
        prompt, correct = create_prompt_aime(empty_example)
        assert "N/A" in prompt
        assert correct == "0"
        
        # Extract from empty/weird content
        weird_contents = [
            "",
            "\\boxed{}",  # Empty braces
            "\\boxed{not_a_number}",
            "Multiple \\boxed{1} and \\boxed{2} answers",  # Should take last
        ]
        
        for content in weird_contents:
            result = extract_answer_aime(content)
            # Most should return None, except the multiple answers case
            if "Multiple" in content:
                assert result == "2"
            else:
                assert result is None or result == ""
    
    def test_numerical_edge_cases(self):
        """Test numerical edge cases in scoring."""
        edge_cases = [
            # (extracted, correct, expected_score)
            ("0", "0", 1),
            ("000", "0", 1),
            ("1", "01", 1),
            ("999", "999", 1),
            ("100", "1000", 0),
            ("-5", "5", 0),  # Should not match negative with positive
        ]
        
        for extracted, correct, expected in edge_cases:
            score = calculate_score_aime(extracted, correct)
            assert score == expected, f"Failed for {extracted} vs {correct}"