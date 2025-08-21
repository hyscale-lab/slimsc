"""Tests for prune.utils.aqua_rat_utils module."""

import pytest
from unittest.mock import Mock, MagicMock, patch

from prune.utils.aqua_rat_utils import (
    load_data_aqua_rat,
    create_prompt_aqua_rat,
    extract_answer_aqua_rat,
    calculate_score_aqua_rat,
    AQUA_RAT_DATASET_PATH,
    AQUA_RAT_SUBSET_NAME,
    AQUA_RAT_SPLIT,
    N_OPTIONS_AQUA_RAT,
    OPTION_LABELS
)


class TestLoadDataAquaRat:
    """Test the load_data_aqua_rat function."""
    
    @patch('prune.utils.aqua_rat_utils.load_dataset')
    def test_load_data_aqua_rat_success(self, mock_load_dataset):
        """Test successful loading of AQUA-RAT dataset."""
        # Setup mock dataset
        mock_rows = [
            {
                "question": "If a train travels 60 km in 1 hour, how far will it travel in 2 hours?",
                "options": ["A) 60 km", "B) 120 km", "C) 180 km", "D) 240 km", "E) 300 km"],
                "correct": "B"
            },
            {
                "question": "What is 5 + 7?",
                "options": ["A) 10", "B) 11", "C) 12", "D) 13", "E) 14"],
                "correct": "C"
            }
        ]
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Test
        result = load_data_aqua_rat()
        
        # Assertions
        assert len(result) == 2
        assert result[0]["question"] == "If a train travels 60 km in 1 hour, how far will it travel in 2 hours?"
        assert result[0]["options"] == ["A) 60 km", "B) 120 km", "C) 180 km", "D) 240 km", "E) 300 km"]
        assert result[0]["correct"] == "B"
        assert result[0]["id"] == "aqua_rat_test_0"  # Auto-generated ID
        
        assert result[1]["question"] == "What is 5 + 7?"
        assert result[1]["correct"] == "C"
        assert result[1]["id"] == "aqua_rat_test_1"  # Auto-generated ID
        
        mock_load_dataset.assert_called_once_with(
            AQUA_RAT_DATASET_PATH, 
            name=AQUA_RAT_SUBSET_NAME, 
            split=AQUA_RAT_SPLIT
        )
    
    @patch('prune.utils.aqua_rat_utils.load_dataset')
    def test_load_data_aqua_rat_with_existing_ids(self, mock_load_dataset):
        """Test loading when items already have IDs."""
        mock_rows = [
            {
                "id": "existing_id_1",
                "question": "Test question",
                "options": ["A) Option A", "B) Option B"],
                "correct": "A"
            }
        ]
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        result = load_data_aqua_rat()
        
        assert len(result) == 1
        assert result[0]["id"] == "existing_id_1"  # Should preserve existing ID
    
    @patch('prune.utils.aqua_rat_utils.load_dataset')
    def test_load_data_aqua_rat_exception(self, mock_load_dataset):
        """Test handling of dataset loading exception."""
        mock_load_dataset.side_effect = Exception("Dataset access failed")
        
        with pytest.raises(Exception, match="Dataset access failed"):
            load_data_aqua_rat()


class TestCreatePromptAquaRat:
    """Test the create_prompt_aqua_rat function."""
    
    def test_create_prompt_aqua_rat_basic(self):
        """Test basic prompt creation."""
        example = {
            "question": "What is 2 + 2?",
            "options": ["A) 3", "B) 4", "C) 5", "D) 6", "E) 7"],
            "correct": "B"
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        # Check prompt structure
        assert "Answer the following multiple-choice question" in prompt
        assert "What is 2 + 2?" in prompt
        assert "A) 3" in prompt
        assert "B) 4" in prompt
        assert "C) 5" in prompt
        assert "D) 6" in prompt
        assert "E) 7" in prompt
        assert "Answer: $LETTER" in prompt
        
        # Check correct answer
        assert correct_answer == "B"
    
    def test_create_prompt_aqua_rat_missing_question(self):
        """Test prompt creation with missing question."""
        example = {
            "options": ["A) Option A", "B) Option B"],
            "correct": "A"
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        assert "N/A" in prompt
        assert correct_answer == "A"
    
    def test_create_prompt_aqua_rat_missing_options(self):
        """Test prompt creation with missing options."""
        example = {
            "question": "Test question?",
            "correct": "A"
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        assert "Test question?" in prompt
        assert correct_answer == "A"
    
    def test_create_prompt_aqua_rat_invalid_correct_answer(self):
        """Test with invalid correct answer."""
        example = {
            "question": "Test question?",
            "options": ["A) Option A", "B) Option B"],
            "correct": "Z"  # Invalid
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        assert correct_answer == "Z"  # Should preserve but log error
    
    def test_create_prompt_aqua_rat_lowercase_correct(self):
        """Test with lowercase correct answer."""
        example = {
            "question": "Test question?",
            "options": ["A) Option A", "B) Option B"],
            "correct": "b"  # Lowercase
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        assert correct_answer == "B"  # Should be uppercased
    
    def test_create_prompt_aqua_rat_fewer_options(self):
        """Test with fewer than 5 options."""
        example = {
            "question": "True or False: 2+2=4",
            "options": ["A) True", "B) False"],
            "correct": "A"
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        assert "A) True" in prompt
        assert "B) False" in prompt
        assert correct_answer == "A"
    
    def test_create_prompt_aqua_rat_missing_id(self):
        """Test with missing ID field."""
        example = {
            "question": "Test question?",
            "options": ["A) Option A"],
            "correct": "A"
            # No 'id' field
        }
        
        prompt, correct_answer = create_prompt_aqua_rat(example)
        
        assert prompt is not None
        assert correct_answer == "A"


class TestExtractAnswerAquaRat:
    """Test the extract_answer_aqua_rat function."""
    
    def test_extract_answer_aqua_rat_none_content(self):
        """Test with None content."""
        result = extract_answer_aqua_rat(None)
        assert result is None
    
    def test_extract_answer_aqua_rat_valid_answer_pattern(self):
        """Test with valid Answer: pattern."""
        test_cases = [
            ("Answer: A", "A"),
            ("Answer: B", "B"),
            ("Answer: C", "C"),
            ("Answer: D", "D"),
            ("Answer: E", "E"),
            ("Answer: $A", "A"),
            ("Answer: $B$", "B"),
            ("answer: c", "C"),  # Case insensitive
            ("ANSWER: d", "D"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_aqua_rat(content)
            assert result == expected, f"Failed for content: {content}"
    
    def test_extract_answer_aqua_rat_multiple_answers(self):
        """Test with multiple answer patterns (should take last)."""
        content = "First I thought Answer: A, but actually Answer: B"
        result = extract_answer_aqua_rat(content)
        assert result == "B"
    
    def test_extract_answer_aqua_rat_no_answer_pattern(self):
        """Test with content containing no Answer: pattern."""
        content = "This text doesn't contain the expected answer format."
        result = extract_answer_aqua_rat(content)
        assert result is None
    
    def test_extract_answer_aqua_rat_fallback_last_character(self):
        """Test fallback to last character."""
        # Test valid fallback cases
        test_cases = [
            ("The answer is A", "A"),
            ("I choose B", "B"),
            ("My final answer: C", "C"),
            ("Therefore D", "D"),
            ("The solution is E", "E"),
        ]
        
        for content, expected in test_cases:
            result = extract_answer_aqua_rat(content)
            assert result == expected, f"Failed for content: {content}"
    
    def test_extract_answer_aqua_rat_fallback_invalid_cases(self):
        """Test fallback cases that should not match."""
        invalid_cases = [
            "The answer is A)",  # Ends with ), should not match
            "Option B) is correct",  # Ends with ), should not match
            "Text ending with Z",  # Z is not a valid option
            "Number 5",  # Not a letter
            "",  # Empty string
        ]
        
        for content in invalid_cases:
            result = extract_answer_aqua_rat(content)
            assert result is None, f"Should not match for content: {content}"
    
    def test_extract_answer_aqua_rat_invalid_letters(self):
        """Test with invalid option letters."""
        invalid_cases = [
            "Answer: F",  # F not in ABCDE
            "Answer: Z",  # Z not valid
            "Answer: X",  # X not valid
        ]
        
        for content in invalid_cases:
            result = extract_answer_aqua_rat(content)
            assert result is None, f"Should not match invalid letter in: {content}"
    
    def test_extract_answer_aqua_rat_edge_cases(self):
        """Test various edge cases."""
        edge_cases = [
            ("Answer:A", "A"),  # No space after colon
            ("Answer : B", "B"),  # Space before colon
            ("Answer  :  C", "C"),  # Multiple spaces
            ("The final Answer: D is correct", "D"),  # Answer in middle
            ("I think the Answer: E", "E"),  # Answer at end
        ]
        
        for content, expected in edge_cases:
            result = extract_answer_aqua_rat(content)
            assert result == expected, f"Failed for edge case: {content}"


class TestCalculateScoreAquaRat:
    """Test the calculate_score_aqua_rat function."""
    
    def test_calculate_score_aqua_rat_correct_match(self):
        """Test with correct answer match."""
        result = calculate_score_aqua_rat("A", "A")
        assert result == 1
    
    def test_calculate_score_aqua_rat_incorrect_match(self):
        """Test with incorrect answer."""
        result = calculate_score_aqua_rat("A", "B")
        assert result == 0
    
    def test_calculate_score_aqua_rat_none_extracted(self):
        """Test with None extracted answer."""
        result = calculate_score_aqua_rat(None, "A")
        assert result == 0
    
    def test_calculate_score_aqua_rat_case_insensitive(self):
        """Test case insensitive comparison."""
        test_cases = [
            ("a", "A", 1),
            ("B", "b", 1),
            ("c", "C", 1),
            ("D", "d", 1),
            ("e", "E", 1),
            ("a", "B", 0),  # Different letters
        ]
        
        for extracted, correct, expected in test_cases:
            result = calculate_score_aqua_rat(extracted, correct)
            assert result == expected, f"Failed for {extracted} vs {correct}"
    
    def test_calculate_score_aqua_rat_invalid_correct_answer(self):
        """Test with invalid correct answer."""
        result = calculate_score_aqua_rat("A", "Z")  # Z is invalid
        assert result == 0
    
    def test_calculate_score_aqua_rat_all_valid_options(self):
        """Test scoring with all valid option letters."""
        for letter in OPTION_LABELS:
            # Correct match
            result = calculate_score_aqua_rat(letter, letter)
            assert result == 1
            
            # Incorrect match with next letter (wrap around at end)
            next_letter = OPTION_LABELS[(OPTION_LABELS.index(letter) + 1) % len(OPTION_LABELS)]
            result = calculate_score_aqua_rat(letter, next_letter)
            assert result == 0


class TestAquaRatUtilsIntegration:
    """Integration tests for AQUA-RAT utilities."""
    
    @patch('prune.utils.aqua_rat_utils.load_dataset')
    def test_full_pipeline(self, mock_load_dataset):
        """Test complete pipeline from dataset to scoring."""
        # Setup mock dataset
        mock_rows = [
            {
                "question": "A train travels 100 km in 2 hours. What is its speed?",
                "options": [
                    "A) 25 km/h",
                    "B) 50 km/h", 
                    "C) 75 km/h",
                    "D) 100 km/h",
                    "E) 200 km/h"
                ],
                "correct": "B"
            }
        ]
        mock_dataset = Mock()
        mock_dataset.__iter__ = Mock(return_value=iter(mock_rows))
        mock_load_dataset.return_value = mock_dataset
        
        # Load data
        examples = load_data_aqua_rat()
        assert len(examples) == 1
        
        example = examples[0]
        
        # Create prompt
        prompt, correct_answer = create_prompt_aqua_rat(example)
        assert "train travels 100 km" in prompt
        assert "A) 25 km/h" in prompt
        assert "B) 50 km/h" in prompt
        assert correct_answer == "B"
        
        # Test various model responses
        test_responses = [
            ("Speed = distance/time = 100/2 = 50 km/h. Answer: B", 1),
            ("The speed is 50 km per hour. Answer: B", 1),
            ("I think the answer is Answer: A", 0),  # Incorrect
            ("The calculation gives us Answer: C", 0),  # Incorrect
            ("I'm not sure about this question.", 0),  # No answer
            ("Distance is 100 km, time is 2 hours, so speed is 50. B", 1),  # Fallback to B
        ]
        
        for response, expected_score in test_responses:
            extracted = extract_answer_aqua_rat(response)
            score = calculate_score_aqua_rat(extracted, correct_answer)
            assert score == expected_score, f"Failed for response: {response}"
    
    def test_constants_validity(self):
        """Test that module constants are valid."""
        assert AQUA_RAT_DATASET_PATH == "deepmind/aqua_rat"
        assert AQUA_RAT_SUBSET_NAME == "raw"
        assert AQUA_RAT_SPLIT == "test"
        assert N_OPTIONS_AQUA_RAT == 5
        assert OPTION_LABELS == "ABCDE"
        assert len(OPTION_LABELS) == N_OPTIONS_AQUA_RAT
    
    def test_edge_cases_and_robustness(self):
        """Test various edge cases for robustness."""
        # Empty example
        empty_example = {}
        prompt, correct = create_prompt_aqua_rat(empty_example)
        assert "N/A" in prompt
        assert correct == "Z"  # Error default
        
        # Malformed options
        malformed_example = {
            "question": "Test?",
            "options": ["Not proper format", "Also wrong"],
            "correct": "A"
        }
        prompt, correct = create_prompt_aqua_rat(malformed_example)
        assert "Test?" in prompt
        assert "Not proper format" in prompt
        assert correct == "A"
        
        # Test extraction robustness
        edge_extractions = [
            ("", None),
            ("No answer here", None),
            ("Answer: answer: B", "B"),  # Multiple colons
            ("The Answer: A is correct, but Answer: B is wrong", "B"),  # Last match
            ("Final answer A", "A"),  # Fallback works
        ]
        
        for content, expected in edge_extractions:
            result = extract_answer_aqua_rat(content)
            assert result == expected, f"Failed for edge extraction: {content}"
    
    def test_scoring_edge_cases(self):
        """Test edge cases in scoring."""
        scoring_tests = [
            # (extracted, correct, expected_score)
            ("A", "A", 1),
            ("a", "A", 1),  # Case insensitive
            ("A", "a", 1),  # Case insensitive
            ("A", "B", 0),  # Wrong answer
            (None, "A", 0),  # No extraction
            ("F", "A", 0),  # Invalid letter
            ("A", "F", 0),  # Invalid correct
            ("", "A", 0),   # Empty extraction
        ]
        
        for extracted, correct, expected in scoring_tests:
            score = calculate_score_aqua_rat(extracted, correct)
            assert score == expected, f"Failed for {extracted} vs {correct}"