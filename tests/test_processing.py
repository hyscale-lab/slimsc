"""Tests for prune.evaluation.processing module."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, Optional, List, Any

from prune.evaluation.processing import (
    process_single_stream,
    process_question_sc_stream,
    process_question_esc_stream
)


class TestProcessSingleStream:
    """Test the process_single_stream function."""
    
    @pytest.mark.asyncio
    async def test_process_single_stream_success(self):
        """Test successful stream processing."""
        # Mock stream generator
        async def mock_stream_generator():
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            yield {"choices": [{"delta": {"content": " world"}}]}
            yield {"usage": {"completion_tokens": 2, "prompt_tokens": 5}}
        
        # Mock process_stream_chunks
        with patch('prune.evaluation.processing.process_stream_chunks') as mock_process:
            mock_process.return_value = {
                "full_content": "Hello world",
                "completion_tokens": 2,
                "prompt_tokens": 5
            }
            
            result = await process_single_stream(mock_stream_generator(), chain_index=1)
            
            assert result["full_content"] == "Hello world"
            assert result["completion_tokens"] == 2
            assert result["prompt_tokens"] == 5
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_single_stream_exception(self):
        """Test stream processing with exception."""
        # Mock stream generator that raises exception
        async def mock_stream_generator():
            yield {"choices": [{"delta": {"content": "Hello"}}]}
            raise Exception("Stream error")
        
        with patch('prune.evaluation.processing.process_stream_chunks') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            result = await process_single_stream(mock_stream_generator(), chain_index=1)
            
            assert "error" in result
            assert result["error"]["status"] == "stream_processing_failed"
            assert "Processing error" in result["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_process_single_stream_empty_chunks(self):
        """Test stream processing with empty chunks."""
        # Mock empty stream generator
        async def mock_stream_generator():
            return
            yield  # Never reached
        
        with patch('prune.evaluation.processing.process_stream_chunks') as mock_process:
            mock_process.return_value = {"full_content": "", "completion_tokens": 0}
            
            result = await process_single_stream(mock_stream_generator(), chain_index=1)
            
            assert result["full_content"] == ""
            assert result["completion_tokens"] == 0


class TestProcessQuestionScStream:
    """Test the process_question_sc_stream function."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Set up common mocks for SC stream processing tests."""
        with patch.multiple(
            'prune.evaluation.processing',
            DatasetHandler=Mock(),
            stream_vllm_request=AsyncMock(),
            process_single_stream=AsyncMock(),
            majority_vote=Mock(),
            extract_kv_cache_usage_for_question=Mock()
        ) as mocks:
            yield mocks
    
    @pytest.mark.asyncio
    async def test_process_question_sc_stream_basic(self, mock_dependencies):
        """Test basic SC stream processing."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", ("choices", "A"))
        mock_handler.extract_answer.return_value = "A"
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing
        mock_dependencies['process_single_stream'].return_value = {
            "full_content": "The answer is A",
            "completion_tokens": 10,
            "prompt_tokens": 5
        }
        
        # Setup mock voting
        mock_dependencies['majority_vote'].return_value = ("A", 1, ["A", "A", "B"])
        
        # Setup mock KV cache extraction
        mock_dependencies['extract_kv_cache_usage_for_question'].return_value = {
            "mean_gpu_cache_perc": 0.65
        }
        
        example = {"id": "test_1", "question": "Test question?", "correct_answer": "A"}
        paths = {
            "source_usage_file": "/tmp/usage.json",
            "summaries_dir": "/tmp/summaries"
        }
        
        result = await process_question_sc_stream(
            example=example,
            iteration=1,
            n_chains=3,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path="/path/to/tokenizer",
            dataset_name="gpqa_diamond"
        )
        
        # Verify result structure
        assert result is not None
        assert result["iteration"] == 1
        assert result["question_id"] == "test_1"
        assert result["n_chains"] == 3
        assert result["voted_answer"] == "A"
        assert result["score"] == 1
        assert "kv_cache_usage" in result
        
        # Verify mocks were called
        mock_handler.create_prompt.assert_called_once_with(example)
        assert mock_dependencies['stream_vllm_request'].call_count == 3  # n_chains
        assert mock_dependencies['process_single_stream'].call_count == 3
        mock_dependencies['majority_vote'].assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_question_sc_stream_prompt_creation_failure(self, mock_dependencies):
        """Test handling of prompt creation failure."""
        # Setup mock dataset handler to fail
        mock_handler = Mock()
        mock_handler.create_prompt.side_effect = Exception("Prompt creation failed")
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        example = {"id": "test_1", "question": "Test question?"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_sc_stream(
            example=example,
            iteration=1,
            n_chains=3,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="gpqa_diamond"
        )
        
        assert result is None
        mock_handler.create_prompt.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_question_sc_stream_stream_failures(self, mock_dependencies):
        """Test handling of stream processing failures."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", ("choices", "A"))
        mock_handler.extract_answer.return_value = "A"
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing with some failures
        mock_dependencies['process_single_stream'].side_effect = [
            {"full_content": "Answer A", "completion_tokens": 10},
            {"error": {"status": "failed", "message": "Stream failed"}},
            {"full_content": "Answer A", "completion_tokens": 12}
        ]
        
        # Setup mock voting
        mock_dependencies['majority_vote'].return_value = ("A", 1, ["A", "A"])
        
        example = {"id": "test_1", "question": "Test question?", "correct_answer": "A"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_sc_stream(
            example=example,
            iteration=1,
            n_chains=3,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="gpqa_diamond"
        )
        
        assert result is not None
        assert result["n_chains"] == 3
        assert result["n_successful_chains"] == 2  # Only 2 successful
        assert result["voted_answer"] == "A"
    
    @pytest.mark.asyncio
    async def test_process_question_sc_stream_different_dataset(self, mock_dependencies):
        """Test SC processing with different dataset (non-GPQA)."""
        # Setup mock dataset handler for AIME
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", "42")  # AIME returns answer directly
        mock_handler.extract_answer.return_value = "42"
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing
        mock_dependencies['process_single_stream'].return_value = {
            "full_content": "The answer is 42",
            "completion_tokens": 8
        }
        
        # Setup mock voting
        mock_dependencies['majority_vote'].return_value = ("42", 1, ["42", "42"])
        
        example = {"id": "aime_1", "question": "AIME question?", "answer": "42"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_sc_stream(
            example=example,
            iteration=1,
            n_chains=2,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="aime"
        )
        
        assert result is not None
        assert result["voted_answer"] == "42"
        assert result["score"] == 1


class TestProcessQuestionEscStream:
    """Test the process_question_esc_stream function."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Set up common mocks for ESC stream processing tests."""
        with patch.multiple(
            'prune.evaluation.processing',
            DatasetHandler=Mock(),
            stream_vllm_request=AsyncMock(),
            process_single_stream=AsyncMock(),
            majority_vote_for_sim_prune=Mock(),
            fallback_tie_break_logic=Mock(),
            extract_kv_cache_usage_for_question=Mock()
        ) as mocks:
            yield mocks
    
    @pytest.mark.asyncio
    async def test_process_question_esc_stream_early_consensus(self, mock_dependencies):
        """Test ESC processing with early consensus."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", ("choices", "A"))
        mock_handler.extract_answer.return_value = "A"
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing - all return same answer
        mock_dependencies['process_single_stream'].return_value = {
            "full_content": "The answer is A",
            "completion_tokens": 10
        }
        
        # Setup mock voting to return early consensus
        mock_dependencies['majority_vote_for_sim_prune'].return_value = (
            "winner", "A", 1, ["A", "A"], [], None
        )
        
        example = {"id": "test_1", "question": "Test question?", "correct_answer": "A"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_esc_stream(
            example=example,
            iteration=1,
            n_chains_max=6,
            window_size=2,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="gpqa_diamond"
        )
        
        assert result is not None
        assert result["voted_answer"] == "A"
        assert result["score"] == 1
        assert result["consensus_found"] == True
        assert result["consensus_window"] == 1  # Found in first window
        assert result["n_chains_generated"] == 2  # Only first window processed
    
    @pytest.mark.asyncio
    async def test_process_question_esc_stream_no_consensus(self, mock_dependencies):
        """Test ESC processing without reaching consensus."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", ("choices", "A"))
        mock_handler.extract_answer.side_effect = ["A", "B", "A", "C"]  # Mixed answers
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing
        mock_dependencies['process_single_stream'].side_effect = [
            {"full_content": "Answer A", "completion_tokens": 10},
            {"full_content": "Answer B", "completion_tokens": 12},
            {"full_content": "Answer A", "completion_tokens": 11},
            {"full_content": "Answer C", "completion_tokens": 9}
        ]
        
        # Setup mock voting - never reaches consensus, requires tie-break
        mock_dependencies['majority_vote_for_sim_prune'].side_effect = [
            ("REQUIRES_LLM_TIEBREAK", None, 0, ["A", "B"], 
             [{"extracted_answer": "A"}, {"extracted_answer": "B"}], ["A", "B"]),
            ("REQUIRES_LLM_TIEBREAK", None, 0, ["A", "B", "A", "C"], 
             [{"extracted_answer": "A"}, {"extracted_answer": "C"}], ["A", "C"])
        ]
        
        # Setup fallback tie-breaking
        mock_dependencies['fallback_tie_break_logic'].return_value = ("A", 1)
        
        example = {"id": "test_1", "question": "Test question?", "correct_answer": "A"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_esc_stream(
            example=example,
            iteration=1,
            n_chains_max=4,
            window_size=2,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path="/path/to/tokenizer",
            dataset_name="gpqa_diamond"
        )
        
        assert result is not None
        assert result["voted_answer"] == "A"
        assert result["score"] == 1
        assert result["consensus_found"] == False
        assert result["n_chains_generated"] == 4  # All chains processed
        assert result["n_windows_processed"] == 2
        
        # Verify fallback was called
        mock_dependencies['fallback_tie_break_logic'].assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_question_esc_stream_window_processing(self, mock_dependencies):
        """Test ESC window-based processing logic."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", ("choices", "A"))
        mock_handler.extract_answer.side_effect = ["A", "B", "A", "A", "A"]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing
        mock_dependencies['process_single_stream'].return_value = {
            "full_content": "Answer", "completion_tokens": 10
        }
        
        # Setup mock voting - consensus in second window
        mock_dependencies['majority_vote_for_sim_prune'].side_effect = [
            ("REQUIRES_LLM_TIEBREAK", None, 0, ["A", "B"], [], ["A", "B"]),  # Window 1: tie
            ("winner", "A", 1, ["A", "B", "A"], [], None)  # Window 2: consensus
        ]
        
        example = {"id": "test_1", "question": "Test question?", "correct_answer": "A"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_esc_stream(
            example=example,
            iteration=1,
            n_chains_max=6,
            window_size=2,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="gpqa_diamond"
        )
        
        assert result is not None
        assert result["consensus_found"] == True
        assert result["consensus_window"] == 2
        assert result["n_chains_generated"] == 3  # Stopped after consensus
        assert result["n_windows_processed"] == 2
    
    @pytest.mark.asyncio
    async def test_process_question_esc_stream_stream_failures(self, mock_dependencies):
        """Test ESC processing with stream failures."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("Test prompt", ("choices", "A"))
        mock_handler.extract_answer.side_effect = ["A", "A"]  # Only successful streams
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock stream processing with failures
        mock_dependencies['process_single_stream'].side_effect = [
            {"full_content": "Answer A", "completion_tokens": 10},
            {"error": {"status": "failed"}},  # Failure
            {"full_content": "Answer A", "completion_tokens": 10},
            {"error": {"status": "timeout"}}  # Another failure
        ]
        
        # Setup mock voting - consensus with successful streams
        mock_dependencies['majority_vote_for_sim_prune'].return_value = (
            "winner", "A", 1, ["A", "A"], [], None
        )
        
        example = {"id": "test_1", "question": "Test question?", "correct_answer": "A"}
        paths = {"source_usage_file": "/tmp/usage.json"}
        
        result = await process_question_esc_stream(
            example=example,
            iteration=1,
            n_chains_max=4,
            window_size=2,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="gpqa_diamond"
        )
        
        assert result is not None
        assert result["n_chains_generated"] == 4  # All attempted
        assert result["n_successful_chains"] == 2  # Only 2 successful
        assert result["voted_answer"] == "A"


class TestProcessingIntegration:
    """Integration tests for processing functionality."""
    
    @pytest.mark.asyncio
    @patch('prune.evaluation.processing.stream_vllm_request')
    @patch('prune.evaluation.processing.process_stream_chunks')
    @patch('prune.evaluation.processing.DatasetHandler')
    @patch('prune.evaluation.processing.majority_vote')
    async def test_sc_processing_complete_workflow(
        self, mock_majority_vote, mock_dataset_handler, 
        mock_process_chunks, mock_stream_request
    ):
        """Test complete SC processing workflow."""
        # Setup dataset handler
        mock_handler = Mock()
        mock_handler.create_prompt.return_value = ("What is 2+2?", ("A) 3", "B) 4", "C) 5"), "B")
        mock_handler.extract_answer.side_effect = ["B", "B", "A"]  # Majority B
        mock_dataset_handler.return_value = mock_handler
        
        # Setup stream processing
        async def mock_stream():
            yield {"choices": [{"delta": {"content": "The answer is B"}}]}
            yield {"usage": {"completion_tokens": 5}}
        
        mock_stream_request.return_value = mock_stream()
        mock_process_chunks.return_value = {
            "full_content": "The answer is B",
            "completion_tokens": 5,
            "prompt_tokens": 10
        }
        
        # Setup voting
        mock_majority_vote.return_value = ("B", 1, ["B", "B", "A"])
        
        example = {
            "id": "q1",
            "question": "What is 2+2?",
            "Correct Answer": "B"
        }
        paths = {
            "source_usage_file": "/tmp/usage.json",
            "summaries_dir": "/tmp/summaries"
        }
        
        result = await process_question_sc_stream(
            example=example,
            iteration=1,
            n_chains=3,
            paths=paths,
            vllm_url="http://localhost:8000",
            model_name="test_model",
            tokenizer_path=None,
            dataset_name="gpqa_diamond"
        )
        
        # Verify complete workflow
        assert result is not None
        assert result["question_id"] == "q1"
        assert result["voted_answer"] == "B"
        assert result["score"] == 1
        assert result["n_chains"] == 3
        assert "total_time" in result
        
        # Verify all components called
        mock_handler.create_prompt.assert_called_once()
        assert mock_stream_request.call_count == 3
        assert mock_process_chunks.call_count == 3
        assert mock_handler.extract_answer.call_count == 3
        mock_majority_vote.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery in processing."""
        with patch.multiple(
            'prune.evaluation.processing',
            DatasetHandler=Mock(),
            stream_vllm_request=AsyncMock(),
            process_single_stream=AsyncMock(),
            majority_vote=Mock()
        ) as mocks:
            
            # Setup handler that works
            mock_handler = Mock()
            mock_handler.create_prompt.return_value = ("prompt", "A")
            mock_handler.extract_answer.return_value = "A"
            mocks['DatasetHandler'].return_value = mock_handler
            
            # Setup stream processing with mixed success/failure
            mocks['process_single_stream'].side_effect = [
                Exception("First stream failed"),
                {"full_content": "Answer A", "completion_tokens": 5},
                {"error": {"status": "timeout"}}
            ]
            
            # Setup voting to work with available data
            mocks['majority_vote'].return_value = ("A", 1, ["A"])
            
            example = {"id": "test", "question": "Test?", "answer": "A"}
            paths = {"source_usage_file": "/tmp/usage.json"}
            
            # Should not raise exception despite stream failures
            result = await process_question_sc_stream(
                example=example,
                iteration=1,
                n_chains=3,
                paths=paths,
                vllm_url="http://localhost:8000",
                model_name="test_model",
                tokenizer_path=None,
                dataset_name="test_dataset"
            )
            
            # Should still produce result with partial data
            assert result is not None
            assert result["voted_answer"] == "A"
            assert result["n_chains"] == 3
            assert result["n_successful_chains"] == 1  # Only one succeeded
    
    def test_chain_result_data_structure(self):
        """Test that chain result data structures are consistent."""
        # This test verifies the expected structure of chain results
        # that would be used by voting functions
        
        expected_chain_result_keys = {
            "chain_index",
            "extracted_answer", 
            "full_content",
            "completion_tokens",
            "prompt_tokens",
            "reasoning_text",
            "final_answer_text"
        }
        
        # Mock a typical chain result
        chain_result = {
            "chain_index": 0,
            "extracted_answer": "A",
            "full_content": "Let me think... The answer is A",
            "completion_tokens": 15,
            "prompt_tokens": 50,
            "reasoning_text": "Let me think...",
            "final_answer_text": "The answer is A"
        }
        
        # Verify all expected keys are present
        assert all(key in chain_result for key in expected_chain_result_keys)
        
        # Verify data types
        assert isinstance(chain_result["chain_index"], int)
        assert isinstance(chain_result["extracted_answer"], str)
        assert isinstance(chain_result["completion_tokens"], int)
        assert isinstance(chain_result["prompt_tokens"], int)