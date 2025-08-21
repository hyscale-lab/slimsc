"""Tests for prune.utils.gpqa_utils module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
import random

from prune.utils.gpqa_utils import (
    load_data_gpqa,
    create_prompt_gpqa,
    extract_answer_gpqa,
    calculate_score_gpqa
)


class TestLoadDataGpqa:
    """Test the load_data_gpqa function."""
    
    @patch('prune.utils.gpqa_utils.load_dataset')
    @patch('prune.utils.gpqa_utils.random.Random')
    def test_load_data_gpqa_success(self, mock_random, mock_load_dataset):
        """Test successful loading of GPQA dataset."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_rows = [
            {"Question": "Test question 1", "id": "1", "Correct Answer": "A"},
            {"Question": "Test question 2", "id": "2", "Correct Answer": "B"}
        ]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Setup mock random for permutations
        mock_rng = Mock()
        mock_rng.sample.side_effect = [[0, 1, 2, 3], [1, 0, 3, 2]]  # Different permutations
        mock_random.return_value = mock_rng
        
        # Test
        result = load_data_gpqa("gpqa_diamond", "train")
        
        # Assertions
        assert len(result) == 2
        assert result[0]["question"] == "Test question 1"  # Renamed from Question
        assert result[0]["permutation"] == [0, 1, 2, 3]
        assert result[1]["permutation"] == [1, 0, 3, 2]
        
        mock_load_dataset.assert_called_once_with("Idavidrein/gpqa", name="gpqa_diamond", split="train")
        mock_dataset.rename_column.assert_called_once_with("Question", "question")
        mock_random.assert_called_once_with(0)  # Seeded for reproducibility
    
    @patch('prune.utils.gpqa_utils.load_dataset')
    def test_load_data_gpqa_exception(self, mock_load_dataset):
        """Test exception handling during dataset loading."""
        mock_load_dataset.side_effect = Exception("Network error")
        
        with pytest.raises(Exception) as exc_info:
            load_data_gpqa("gpqa_diamond", "test")
        
        assert "Network error" in str(exc_info.value)


class TestCreatePromptGpqa:
    """Test the create_prompt_gpqa function."""
    
    def test_create_prompt_gpqa_basic(self):
        """Test basic prompt creation with valid permutation."""
        example = {
            "question": "What is the atomic number of hydrogen?",
            "Correct Answer": "1",
            "Incorrect Answer 1": "2",
            "Incorrect Answer 2": "3", 
            "Incorrect Answer 3": "4",
            "permutation": [0, 1, 2, 3],  # Original order
            "id": "test_1"
        }
        
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # Check prompt structure
        assert "What is the atomic number of hydrogen?" in prompt
        assert "Answer: $LETTER" in prompt
        assert "A)" in prompt and "B)" in prompt and "C)" in prompt and "D)" in prompt
        
        # Check choices are in correct order
        assert choices == ["1", "2", "3", "4"]
        
        # Check correct answer
        assert correct_letter == "A"  # First choice is correct
    
    def test_create_prompt_gpqa_with_permutation(self):
        """Test prompt creation with different permutation."""
        example = {
            "question": "Test question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong1",
            "Incorrect Answer 2": "Wrong2",
            "Incorrect Answer 3": "Wrong3",
            "permutation": [3, 0, 1, 2],  # Correct answer moves to position B
            "id": "test_2"
        }
        
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # Check choices are permuted correctly
        assert choices == ["Wrong3", "Correct", "Wrong1", "Wrong2"]
        
        # Check correct answer letter is updated
        assert correct_letter == "B"  # Correct answer is now in position B
    
    def test_create_prompt_gpqa_missing_permutation(self):
        """Test prompt creation when permutation is missing."""
        example = {
            "question": "Test question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong1", 
            "Incorrect Answer 2": "Wrong2",
            "Incorrect Answer 3": "Wrong3",
            # No permutation key
            "id": "test_3"
        }
        
        with patch('prune.utils.gpqa_utils.random.sample') as mock_sample:
            mock_sample.return_value = [1, 0, 2, 3]
            
            prompt, choices, correct_letter = create_prompt_gpqa(example)
            
            # Should use fallback permutation
            assert choices == ["Wrong1", "Correct", "Wrong2", "Wrong3"]
            assert correct_letter == "B"
    
    def test_create_prompt_gpqa_invalid_permutation(self):
        """Test prompt creation with invalid permutation."""
        example = {
            "question": "Test question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong1",
            "Incorrect Answer 2": "Wrong2", 
            "Incorrect Answer 3": "Wrong3",
            "permutation": [5, 6, 7, 8],  # Invalid indices
            "id": "test_4"
        }
        
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # Should fallback to original order when permutation fails
        assert choices == ["Correct", "Wrong1", "Wrong2", "Wrong3"]
        assert correct_letter == "A"
    
    def test_create_prompt_gpqa_missing_fields(self):
        """Test prompt creation with missing fields."""
        example = {
            "question": "Test question?",
            # Missing some answer fields
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong1",
            # Missing Incorrect Answer 2 and 3
            "permutation": [0, 1, 2, 3],
            "id": "test_5"
        }
        
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # Should handle missing fields with "N/A"
        assert choices == ["Correct", "Wrong1", "N/A", "N/A"]
        assert correct_letter == "A"
        assert "N/A" in prompt
    
    def test_create_prompt_gpqa_with_n_a_correct_answer(self):
        """Test when correct answer field is missing (defaults to N/A)."""
        example = {
            "question": "Test question?",
            # "Correct Answer" field missing, will default to "N/A"
            "Incorrect Answer 1": "Wrong1",
            "Incorrect Answer 2": "Wrong2",
            "Incorrect Answer 3": "Wrong3",
            "permutation": [0, 1, 2, 3],
            "id": "test_6"
        }
        
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # choices_pool = ["N/A", "Wrong1", "Wrong2", "Wrong3"]
        # With permutation [0,1,2,3], choices = ["N/A", "Wrong1", "Wrong2", "Wrong3"]
        # "N/A" is found at index 0, so correct_letter should be "A"
        assert choices == ["N/A", "Wrong1", "Wrong2", "Wrong3"]
        assert correct_letter == "A"
        assert "N/A" in prompt


class TestExtractAnswerGpqa:
    """Test the extract_answer_gpqa function."""
    
    def test_extract_answer_none_content(self):
        """Test extraction with None content."""
        result = extract_answer_gpqa(None)
        assert result is None
    
    def test_extract_answer_standard_format(self):
        """Test extraction with standard Answer: format."""
        test_cases = [
            ("The answer is clearly A. Answer: A", "A"),
            ("After analysis, I conclude Answer: B", "B"), 
            ("My reasoning leads to Answer: C", "C"),
            ("Therefore Answer: D", "D"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_with_dollar_signs(self):
        """Test extraction with dollar signs."""
        test_cases = [
            ("Answer: $A$", "A"),
            ("Answer: $B", "B"),
            ("Answer: C$", "C"),
            ("The answer is Answer: $D", "D"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_case_insensitive(self):
        """Test case insensitive extraction."""
        test_cases = [
            ("answer: a", "A"),
            ("ANSWER: b", "B"),
            ("Answer: c", "C"),
            ("aNsWeR: d", "D"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_with_whitespace(self):
        """Test extraction with various whitespace."""
        test_cases = [
            ("Answer:   A", "A"),
            ("Answer :\tB", "B"),
            ("Answer  :  C", "C"),
            ("Answer\t:\t D", "D"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_multiple_answers(self):
        """Test extraction with multiple Answer: patterns (should use last)."""
        content = "First I thought Answer: A, but then Answer: B, finally Answer: C"
        result = extract_answer_gpqa(content)
        assert result == "C"  # Should use the last one
    
    def test_extract_answer_fallback_last_character(self):
        """Test fallback extraction using last character."""
        test_cases = [
            ("The answer is A", "A"),    # "answer is A" contains "answer"
            ("My conclusion: B", "B"),   # "nclusion: B" contains ":"
            ("Therefore, the answer C", "C"),  # "he answer C" contains "answer"
            ("So the result is D", "D"), # This might not work - let's be more explicit
            ("Final answer: D", "D"),    # "l answer: D" contains both "answer" and ":"
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_fallback_no_context(self):
        """Test fallback fails when no context."""
        test_cases = [
            ("Some random text A", None),  # No answer context
            ("Just the letter B here", None),  # No colon or answer word  
            ("Regular sentence ending in C", None),  # No indication it's an answer
            ("My choice: A", "A"),  # Has colon context, should extract
            ("The answer is A", "A"),  # Has "answer" context, should extract
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_invalid_letters(self):
        """Test extraction ignores invalid letters."""
        test_cases = [
            ("Answer: E", None),  # E not valid for GPQA
            ("Answer: F", None),  # F not valid 
            ("Answer: Z", None),  # Z not valid
            ("Answer: 1", None),  # Number not valid
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected
    
    def test_extract_answer_no_answer_found(self):
        """Test when no answer can be extracted."""
        test_cases = [
            ("This text has no answer pattern", None),
            ("Random content without format", None),
            ("", None),  # Empty string
            ("   ", None),  # Just whitespace
        ]
        
        for content, expected in test_cases:
            result = extract_answer_gpqa(content)
            assert result == expected


class TestCalculateScoreGpqa:
    """Test the calculate_score_gpqa function."""
    
    def test_calculate_score_correct_matches(self):
        """Test score calculation with correct matches."""
        test_cases = [
            ("A", "A", 1),
            ("B", "B", 1), 
            ("C", "C", 1),
            ("D", "D", 1),
        ]
        
        for extracted, correct, expected_score in test_cases:
            score = calculate_score_gpqa(extracted, correct)
            assert score == expected_score
    
    def test_calculate_score_case_insensitive(self):
        """Test score calculation is case insensitive."""
        test_cases = [
            ("a", "A", 1),
            ("B", "b", 1),
            ("c", "C", 1),
            ("D", "d", 1),
        ]
        
        for extracted, correct, expected_score in test_cases:
            score = calculate_score_gpqa(extracted, correct)
            assert score == expected_score
    
    def test_calculate_score_incorrect_matches(self):
        """Test score calculation with incorrect matches."""
        test_cases = [
            ("A", "B", 0),
            ("B", "C", 0),
            ("C", "D", 0),
            ("D", "A", 0),
        ]
        
        for extracted, correct, expected_score in test_cases:
            score = calculate_score_gpqa(extracted, correct)
            assert score == expected_score
    
    def test_calculate_score_none_extracted(self):
        """Test score calculation with None extracted answer."""
        test_cases = [
            (None, "A", 0),
            (None, "B", 0),
            (None, "C", 0),
            (None, "D", 0),
        ]
        
        for extracted, correct, expected_score in test_cases:
            score = calculate_score_gpqa(extracted, correct)
            assert score == expected_score
    
    def test_calculate_score_invalid_correct_answer(self):
        """Test score calculation with invalid correct answer."""
        test_cases = [
            ("A", "E", 0),  # Invalid letter
            ("A", "F", 0),  # Invalid letter
            ("A", "1", 0),  # Number
            ("A", "", 0),   # Empty string
            ("A", None, 0), # None
            ("A", "AB", 0), # Multiple letters
        ]
        
        for extracted, correct, expected_score in test_cases:
            score = calculate_score_gpqa(extracted, correct)
            assert score == expected_score
    
    def test_calculate_score_edge_cases(self):
        """Test score calculation edge cases."""
        test_cases = [
            ("", "A", 0),   # Empty extracted
            ("  ", "A", 0), # Whitespace extracted  
            ("A", "  ", 0), # Whitespace correct
        ]
        
        for extracted, correct, expected_score in test_cases:
            score = calculate_score_gpqa(extracted, correct)
            assert score == expected_score


class TestGpqaUtilsIntegration:
    """Integration tests for GPQA utils."""
    
    @patch('prune.utils.gpqa_utils.load_dataset')
    @patch('prune.utils.gpqa_utils.random.Random')
    def test_full_pipeline(self, mock_random, mock_load_dataset):
        """Test the full pipeline from loading to scoring."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_rows = [{
            "Question": "What is the symbol for gold?",
            "Correct Answer": "Au",
            "Incorrect Answer 1": "Ag", 
            "Incorrect Answer 2": "Fe",
            "Incorrect Answer 3": "Cu",
            "id": "gold_question"
        }]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Setup mock random (correct answer goes to position C)
        mock_rng = Mock()
        mock_rng.sample.return_value = [1, 2, 0, 3]  # Au moves to position C
        mock_random.return_value = mock_rng
        
        # Load dataset
        examples = load_data_gpqa("gpqa_diamond", "test")
        assert len(examples) == 1
        
        # Create prompt
        example = examples[0]
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        assert "What is the symbol for gold?" in prompt
        assert choices == ["Ag", "Fe", "Au", "Cu"]  # Permuted order
        assert correct_letter == "C"  # Au is now in position C
        
        # Simulate model response and extract answer
        model_response = "Gold's chemical symbol is Au. Answer: C"
        extracted = extract_answer_gpqa(model_response)
        assert extracted == "C"
        
        # Calculate score
        score = calculate_score_gpqa(extracted, correct_letter)
        assert score == 1  # Correct!
    
    @patch('prune.utils.gpqa_utils.load_dataset')
    @patch('prune.utils.gpqa_utils.random.Random')
    def test_pipeline_with_incorrect_answer(self, mock_random, mock_load_dataset):
        """Test pipeline with incorrect model response."""
        # Setup mock dataset
        mock_dataset = Mock()
        mock_dataset.rename_column.return_value = mock_dataset
        mock_rows = [{
            "Question": "Test question?",
            "Correct Answer": "Right",
            "Incorrect Answer 1": "Wrong1",
            "Incorrect Answer 2": "Wrong2", 
            "Incorrect Answer 3": "Wrong3",
            "id": "test"
        }]
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Setup mock random (no permutation)
        mock_rng = Mock()
        mock_rng.sample.return_value = [0, 1, 2, 3]
        mock_random.return_value = mock_rng
        
        # Full pipeline
        examples = load_data_gpqa()
        example = examples[0]
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # Model gives wrong answer
        model_response = "I think the answer is B"
        extracted = extract_answer_gpqa(model_response)
        score = calculate_score_gpqa(extracted, correct_letter)
        
        assert extracted == "B"
        assert correct_letter == "A"  # Correct answer
        assert score == 0  # Incorrect
    
    def test_pipeline_with_extraction_failure(self):
        """Test pipeline when answer extraction fails."""
        example = {
            "question": "Test question?",
            "Correct Answer": "Right",
            "Incorrect Answer 1": "Wrong1",
            "Incorrect Answer 2": "Wrong2",
            "Incorrect Answer 3": "Wrong3", 
            "permutation": [0, 1, 2, 3],
            "id": "test"
        }
        
        prompt, choices, correct_letter = create_prompt_gpqa(example)
        
        # Model response without clear answer
        model_response = "This is a difficult question with no clear answer format."
        extracted = extract_answer_gpqa(model_response)
        score = calculate_score_gpqa(extracted, correct_letter)
        
        assert extracted is None
        assert score == 0  # No answer extracted