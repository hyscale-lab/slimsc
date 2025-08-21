"""Tests for prune.utils.dataset_handler module."""

import pytest
from unittest.mock import patch, Mock, MagicMock

from prune.utils.dataset_handler import DatasetHandler


class TestDatasetHandlerInit:
    """Test DatasetHandler initialization."""
    
    def test_init_valid_dataset(self):
        """Test initialization with valid dataset name."""
        handler = DatasetHandler("gpqa_diamond")
        assert handler.dataset_name == "gpqa_diamond"
        
        handler = DatasetHandler("math500")
        assert handler.dataset_name == "math500"
        
        handler = DatasetHandler("aime")
        assert handler.dataset_name == "aime"
        
        handler = DatasetHandler("aqua_rat")
        assert handler.dataset_name == "aqua_rat"
        
        handler = DatasetHandler("hmmt")
        assert handler.dataset_name == "hmmt"
    
    def test_init_invalid_dataset(self):
        """Test initialization with invalid dataset name."""
        with pytest.raises(ValueError) as exc_info:
            DatasetHandler("invalid_dataset")
        
        assert "Unknown dataset type: invalid_dataset" in str(exc_info.value)


class TestDatasetHandlerLoadDataset:
    """Test DatasetHandler load_dataset method."""
    
    @patch('prune.utils.dataset_handler.load_data_gpqa')
    def test_load_dataset_gpqa(self, mock_load_gpqa):
        """Test loading GPQA dataset."""
        mock_data = [{"question": "test", "answer": "A"}]
        mock_load_gpqa.return_value = mock_data
        
        handler = DatasetHandler("gpqa_diamond")
        result = handler.load_dataset("test")
        
        mock_load_gpqa.assert_called_once_with(dataset_name="gpqa_diamond", split="test")
        assert result == mock_data
    
    @patch('prune.utils.dataset_handler.load_data_math500')
    def test_load_dataset_math500(self, mock_load_math500):
        """Test loading MATH-500 dataset."""
        mock_data = [{"question": "test", "answer": "42"}]
        mock_load_math500.return_value = mock_data
        
        handler = DatasetHandler("math500")
        result = handler.load_dataset("test")
        
        mock_load_math500.assert_called_once_with(dataset_name="math500", split="test")
        assert result == mock_data
    
    @patch('prune.utils.dataset_handler.load_data_aime')
    def test_load_dataset_aime(self, mock_load_aime):
        """Test loading AIME dataset."""
        mock_data = [{"question": "test", "answer": "123"}]
        mock_load_aime.return_value = mock_data
        
        handler = DatasetHandler("aime")
        result = handler.load_dataset("test")
        
        mock_load_aime.assert_called_once_with(dataset_name="aime", split="test")
        assert result == mock_data
    
    @patch('prune.utils.dataset_handler.load_data_aqua_rat')
    def test_load_dataset_aqua_rat(self, mock_load_aqua_rat):
        """Test loading AQuA-RAT dataset."""
        mock_data = [{"question": "test", "answer": "A"}]
        mock_load_aqua_rat.return_value = mock_data
        
        handler = DatasetHandler("aqua_rat")
        result = handler.load_dataset("test")
        
        # Note: aqua_rat doesn't take split parameter
        mock_load_aqua_rat.assert_called_once_with()
        assert result == mock_data
    
    @patch('prune.utils.dataset_handler.load_data_hmmt')
    def test_load_dataset_hmmt(self, mock_load_hmmt):
        """Test loading HMMT dataset."""
        mock_data = [{"question": "test", "answer": "42"}]
        mock_load_hmmt.return_value = mock_data
        
        handler = DatasetHandler("hmmt")
        result = handler.load_dataset("test")
        
        mock_load_hmmt.assert_called_once_with(dataset_name="hmmt", split="test")
        assert result == mock_data


class TestDatasetHandlerCreatePrompt:
    """Test DatasetHandler create_prompt method."""
    
    @patch('prune.utils.dataset_handler.create_prompt_gpqa')
    def test_create_prompt_gpqa(self, mock_create_prompt):
        """Test creating prompt for GPQA."""
        example = {"question": "test"}
        mock_create_prompt.return_value = ("prompt", ["A", "B", "C", "D"], "A")
        
        handler = DatasetHandler("gpqa_diamond")
        result = handler.create_prompt(example)
        
        mock_create_prompt.assert_called_once_with(example)
        assert result == ("prompt", (["A", "B", "C", "D"], "A"))
    
    @patch('prune.utils.dataset_handler.create_prompt_math500')
    def test_create_prompt_math500(self, mock_create_prompt):
        """Test creating prompt for MATH-500."""
        example = {"question": "test"}
        mock_create_prompt.return_value = ("prompt", "42")
        
        handler = DatasetHandler("math500")
        result = handler.create_prompt(example)
        
        mock_create_prompt.assert_called_once_with(example)
        assert result == ("prompt", "42")
    
    @patch('prune.utils.dataset_handler.create_prompt_aime')
    def test_create_prompt_aime(self, mock_create_prompt):
        """Test creating prompt for AIME."""
        example = {"question": "test"}
        mock_create_prompt.return_value = ("prompt", "123")
        
        handler = DatasetHandler("aime")
        result = handler.create_prompt(example)
        
        mock_create_prompt.assert_called_once_with(example)
        assert result == ("prompt", "123")
    
    @patch('prune.utils.dataset_handler.create_prompt_aqua_rat')
    def test_create_prompt_aqua_rat(self, mock_create_prompt):
        """Test creating prompt for AQuA-RAT."""
        example = {"question": "test"}
        mock_create_prompt.return_value = ("prompt", "A")
        
        handler = DatasetHandler("aqua_rat")
        result = handler.create_prompt(example)
        
        mock_create_prompt.assert_called_once_with(example)
        assert result == ("prompt", "A")
    
    @patch('prune.utils.dataset_handler.create_prompt_hmmt')
    def test_create_prompt_hmmt(self, mock_create_prompt):
        """Test creating prompt for HMMT."""
        example = {"question": "test"}
        mock_create_prompt.return_value = ("prompt", "42")
        
        handler = DatasetHandler("hmmt")
        result = handler.create_prompt(example)
        
        mock_create_prompt.assert_called_once_with(example)
        assert result == ("prompt", "42")


class TestDatasetHandlerExtractAnswer:
    """Test DatasetHandler extract_answer method."""
    
    @patch('prune.utils.dataset_handler.extract_answer_gpqa')
    def test_extract_answer_gpqa(self, mock_extract):
        """Test extracting answer for GPQA."""
        content = "The answer is A"
        mock_extract.return_value = "A"
        
        handler = DatasetHandler("gpqa_diamond")
        result = handler.extract_answer(content)
        
        mock_extract.assert_called_once_with(content)
        assert result == "A"
    
    @patch('prune.utils.dataset_handler.extract_answer_math500')
    def test_extract_answer_math500(self, mock_extract):
        """Test extracting answer for MATH-500."""
        content = "The answer is \\boxed{42}"
        mock_extract.return_value = "42"
        
        handler = DatasetHandler("math500")
        result = handler.extract_answer(content)
        
        mock_extract.assert_called_once_with(content)
        assert result == "42"
    
    @patch('prune.utils.dataset_handler.extract_answer_aime')
    def test_extract_answer_aime(self, mock_extract):
        """Test extracting answer for AIME."""
        content = "The answer is 123"
        mock_extract.return_value = "123"
        
        handler = DatasetHandler("aime")
        result = handler.extract_answer(content)
        
        mock_extract.assert_called_once_with(content)
        assert result == "123"
    
    @patch('prune.utils.dataset_handler.extract_answer_aqua_rat')
    def test_extract_answer_aqua_rat(self, mock_extract):
        """Test extracting answer for AQuA-RAT."""
        content = "The answer is A"
        mock_extract.return_value = "A"
        
        handler = DatasetHandler("aqua_rat")
        result = handler.extract_answer(content)
        
        mock_extract.assert_called_once_with(content)
        assert result == "A"
    
    @patch('prune.utils.dataset_handler.extract_answer_hmmt')
    def test_extract_answer_hmmt(self, mock_extract):
        """Test extracting answer for HMMT."""
        content = "The answer is 42"
        mock_extract.return_value = "42"
        
        handler = DatasetHandler("hmmt")
        result = handler.extract_answer(content)
        
        mock_extract.assert_called_once_with(content)
        assert result == "42"
    
    def test_extract_answer_none_content(self):
        """Test extracting answer with None content."""
        handler = DatasetHandler("math500")
        # This should delegate to the specific function
        with patch('prune.utils.dataset_handler.extract_answer_math500') as mock_extract:
            mock_extract.return_value = None
            result = handler.extract_answer(None)
            mock_extract.assert_called_once_with(None)
            assert result is None


class TestDatasetHandlerCalculateScore:
    """Test DatasetHandler calculate_score method."""
    
    @patch('prune.utils.dataset_handler.calculate_score_gpqa')
    def test_calculate_score_gpqa(self, mock_calculate):
        """Test calculating score for GPQA."""
        mock_calculate.return_value = 1
        
        handler = DatasetHandler("gpqa_diamond")
        result = handler.calculate_score("A", "A")
        
        mock_calculate.assert_called_once_with("A", "A")
        assert result == 1
    
    @patch('prune.utils.dataset_handler.calculate_score_math500')
    def test_calculate_score_math500(self, mock_calculate):
        """Test calculating score for MATH-500."""
        mock_calculate.return_value = 1
        
        handler = DatasetHandler("math500")
        result = handler.calculate_score("42", "42")
        
        mock_calculate.assert_called_once_with("42", "42")
        assert result == 1
    
    @patch('prune.utils.dataset_handler.calculate_score_aime')
    def test_calculate_score_aime(self, mock_calculate):
        """Test calculating score for AIME."""
        mock_calculate.return_value = 1
        
        handler = DatasetHandler("aime")
        result = handler.calculate_score("123", "123")
        
        mock_calculate.assert_called_once_with("123", "123")
        assert result == 1
    
    @patch('prune.utils.dataset_handler.calculate_score_aqua_rat')
    def test_calculate_score_aqua_rat(self, mock_calculate):
        """Test calculating score for AQuA-RAT."""
        mock_calculate.return_value = 1
        
        handler = DatasetHandler("aqua_rat")
        result = handler.calculate_score("A", "A")
        
        mock_calculate.assert_called_once_with("A", "A")
        assert result == 1
    
    @patch('prune.utils.dataset_handler.calculate_score_hmmt')
    def test_calculate_score_hmmt(self, mock_calculate):
        """Test calculating score for HMMT."""
        mock_calculate.return_value = 1
        
        handler = DatasetHandler("hmmt")
        result = handler.calculate_score("42", "42")
        
        mock_calculate.assert_called_once_with("42", "42")
        assert result == 1
    
    def test_calculate_score_mismatch(self):
        """Test calculating score with mismatched answers."""
        handler = DatasetHandler("math500")
        with patch('prune.utils.dataset_handler.calculate_score_math500') as mock_calculate:
            mock_calculate.return_value = 0
            result = handler.calculate_score("41", "42")
            mock_calculate.assert_called_once_with("41", "42")
            assert result == 0


class TestDatasetHandlerIntegration:
    """Integration tests for DatasetHandler."""
    
    def test_full_pipeline_math500(self):
        """Test full pipeline for MATH-500."""
        # Mock all the individual functions
        with patch('prune.utils.dataset_handler.load_data_math500') as mock_load, \
             patch('prune.utils.dataset_handler.create_prompt_math500') as mock_prompt, \
             patch('prune.utils.dataset_handler.extract_answer_math500') as mock_extract, \
             patch('prune.utils.dataset_handler.calculate_score_math500') as mock_score:
            
            # Setup mocks
            mock_load.return_value = [{"question": "What is 2+2?", "answer": "4"}]
            mock_prompt.return_value = ("Solve: What is 2+2?", "4")
            mock_extract.return_value = "4"
            mock_score.return_value = 1
            
            # Test the pipeline
            handler = DatasetHandler("math500")
            
            # Load dataset
            dataset = handler.load_dataset("test")
            assert len(dataset) == 1
            
            # Create prompt
            example = dataset[0]
            prompt, correct_answer = handler.create_prompt(example)
            assert "Solve: What is 2+2?" == prompt
            assert correct_answer == "4"
            
            # Extract answer (simulating model response)
            model_response = "The answer is \\boxed{4}"
            extracted = handler.extract_answer(model_response)
            assert extracted == "4"
            
            # Calculate score
            score = handler.calculate_score(extracted, correct_answer)
            assert score == 1
    
    def test_full_pipeline_gpqa(self):
        """Test full pipeline for GPQA."""
        with patch('prune.utils.dataset_handler.load_data_gpqa') as mock_load, \
             patch('prune.utils.dataset_handler.create_prompt_gpqa') as mock_prompt, \
             patch('prune.utils.dataset_handler.extract_answer_gpqa') as mock_extract, \
             patch('prune.utils.dataset_handler.calculate_score_gpqa') as mock_score:
            
            # Setup mocks
            mock_load.return_value = [{"question": "Which element has atomic number 1?"}]
            mock_prompt.return_value = ("Question: Which element has atomic number 1?", ["A) Hydrogen", "B) Helium"], "A")
            mock_extract.return_value = "A"
            mock_score.return_value = 1
            
            # Test the pipeline
            handler = DatasetHandler("gpqa_diamond")
            
            # Load dataset
            dataset = handler.load_dataset("test")
            assert len(dataset) == 1
            
            # Create prompt (GPQA returns different structure)
            example = dataset[0]
            prompt, correct_answer_details = handler.create_prompt(example)
            choices, correct_letter = correct_answer_details
            assert "Question: Which element has atomic number 1?" == prompt
            assert choices == ["A) Hydrogen", "B) Helium"]
            assert correct_letter == "A"
            
            # Extract answer
            model_response = "The answer is A"
            extracted = handler.extract_answer(model_response)
            assert extracted == "A"
            
            # Calculate score
            score = handler.calculate_score(extracted, correct_letter)
            assert score == 1


class TestDatasetHandlerErrorHandling:
    """Test error handling in DatasetHandler."""
    
    def test_load_dataset_invalid_name(self):
        """Test that invalid dataset names raise appropriate errors."""
        # This should be caught at initialization, but let's test the method too
        handler = DatasetHandler.__new__(DatasetHandler)  # bypass __init__
        handler.dataset_name = "invalid"
        
        with pytest.raises(ValueError) as exc_info:
            handler.load_dataset("test")
        assert "Unhandled dataset name 'invalid' in load_dataset method" in str(exc_info.value)
    
    def test_create_prompt_invalid_name(self):
        """Test that invalid dataset names raise appropriate errors in create_prompt."""
        handler = DatasetHandler.__new__(DatasetHandler)
        handler.dataset_name = "invalid"
        
        with pytest.raises(ValueError) as exc_info:
            handler.create_prompt({})
        assert "Unhandled dataset name 'invalid' in create_prompt method" in str(exc_info.value)
    
    def test_extract_answer_invalid_name(self):
        """Test that invalid dataset names raise appropriate errors in extract_answer."""
        handler = DatasetHandler.__new__(DatasetHandler)
        handler.dataset_name = "invalid"
        
        with pytest.raises(ValueError) as exc_info:
            handler.extract_answer("test")
        assert "Unhandled dataset name 'invalid' in extract_answer method" in str(exc_info.value)
    
    def test_calculate_score_invalid_name(self):
        """Test that invalid dataset names raise appropriate errors in calculate_score."""
        handler = DatasetHandler.__new__(DatasetHandler)
        handler.dataset_name = "invalid"
        
        with pytest.raises(ValueError) as exc_info:
            handler.calculate_score("test", "test")
        assert "Unhandled dataset name 'invalid' in calculate_score method" in str(exc_info.value)