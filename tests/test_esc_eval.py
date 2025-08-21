"""Tests for prune.evaluation.esc_eval module."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import pandas as pd
import os

from prune.evaluation.esc_eval import (
    run_esc_evaluation_async,
    flatten_dict,
    setup_output_directories
)


class TestFlattenDict:
    """Test the flatten_dict helper function."""
    
    def test_flatten_dict_simple(self):
        """Test flattening simple dictionary."""
        input_dict = {"a": 1, "b": 2}
        result = flatten_dict(input_dict)
        assert result == {"a": 1, "b": 2}
    
    def test_flatten_dict_nested(self):
        """Test flattening nested dictionary."""
        input_dict = {
            "level1": {
                "level2": {
                    "value": 42
                },
                "simple": "test"
            },
            "direct": "value"
        }
        
        result = flatten_dict(input_dict)
        expected = {
            "level1_level2_value": 42,
            "level1_simple": "test",
            "direct": "value"
        }
        
        assert result == expected
    
    def test_flatten_dict_custom_separator(self):
        """Test flattening with custom separator."""
        input_dict = {"a": {"b": {"c": 1}}}
        result = flatten_dict(input_dict, sep=".")
        assert result == {"a.b.c": 1}
    
    def test_flatten_dict_empty(self):
        """Test flattening empty dictionary."""
        result = flatten_dict({})
        assert result == {}
    
    def test_flatten_dict_with_parent_key(self):
        """Test flattening with parent key."""
        input_dict = {"nested": {"value": 10}}
        result = flatten_dict(input_dict, parent_key="prefix")
        assert result == {"prefix_nested_value": 10}
    
    def test_flatten_dict_mixed_types(self):
        """Test flattening with mixed value types."""
        input_dict = {
            "string": "text",
            "number": 42,
            "nested": {
                "list": [1, 2, 3],
                "bool": True,
                "null": None
            }
        }
        
        result = flatten_dict(input_dict)
        expected = {
            "string": "text",
            "number": 42,
            "nested_list": [1, 2, 3],
            "nested_bool": True,
            "nested_null": None
        }
        
        assert result == expected


class TestSetupOutputDirectories:
    """Test the setup_output_directories function."""
    
    @patch('prune.evaluation.esc_eval.os.makedirs')
    @patch('prune.evaluation.esc_eval.Path')
    def test_setup_output_directories_basic(self, mock_path, mock_makedirs):
        """Test basic directory setup."""
        # Mock Path behavior
        mock_base_dir = Mock()
        mock_model_dir = Mock()
        mock_dataset_dir = Mock()
        mock_run_dir = Mock()
        mock_summaries_dir = Mock()
        
        mock_path.return_value = mock_base_dir
        mock_base_dir.__truediv__ = Mock(side_effect=[mock_model_dir, mock_dataset_dir, mock_run_dir, mock_summaries_dir])
        mock_model_dir.__truediv__ = Mock(return_value=mock_dataset_dir)
        mock_dataset_dir.__truediv__ = Mock(return_value=mock_run_dir)
        mock_run_dir.__truediv__ = Mock(return_value=mock_summaries_dir)
        
        # Convert to string representations for paths
        mock_base_dir.__str__ = Mock(return_value="/base")
        mock_run_dir.__str__ = Mock(return_value="/base/model/dataset/run")
        mock_summaries_dir.__str__ = Mock(return_value="/base/model/dataset/run/summaries")
        
        result = setup_output_directories(
            "/base", "test_model", "test_dataset", 8, 2, 1
        )
        
        # Verify the structure of returned paths
        assert "base_output_dir" in result
        assert "model_output_dir" in result
        assert "dataset_output_dir" in result
        assert "run_output_dir" in result
        assert "summaries_dir" in result
        assert "results_csv_file" in result
        assert "source_usage_file" in result


class TestRunEscEvaluationAsync:
    """Test the main run_esc_evaluation_async function."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Set up common mocks for ESC evaluation tests."""
        with patch.multiple(
            'prune.evaluation.esc_eval',
            setup_output_directories=Mock(return_value={
                "results_csv_file": "/path/to/results.csv",
                "source_usage_file": "/path/to/usage.json",
                "summaries_dir": "/path/to/summaries"
            }),
            clear_source_kv_cache=Mock(),
            DatasetHandler=Mock(),
            process_question_esc_stream=AsyncMock(),
            pd=Mock()
        ) as mocks:
            yield mocks
    
    @pytest.mark.asyncio
    async def test_run_esc_evaluation_empty_dataset(self, mock_dependencies):
        """Test ESC evaluation with empty dataset."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = []
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Run evaluation
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1
        )
        
        # Verify dataset handler was called
        mock_handler.load_data.assert_called_once()
        mock_dependencies['setup_output_directories'].assert_called_once()
        mock_dependencies['clear_source_kv_cache'].assert_called_once()
    
    @pytest.mark.asyncio
    @patch('prune.evaluation.esc_eval.pd.DataFrame')
    async def test_run_esc_evaluation_single_question(self, mock_dataframe, mock_dependencies):
        """Test ESC evaluation with single question."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": "test_1", "question": "What is 2+2?", "answer": "4"}
        ]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock process result
        mock_result = {
            "iteration": 1,
            "question_id": "test_1",
            "n_chains_max": 4,
            "window_size": 2,
            "n_chains_generated": 3,
            "voted_answer": "4",
            "score": 1,
            "consensus_found": True,
            "consensus_window": 2
        }
        mock_dependencies['process_question_esc_stream'].return_value = mock_result
        
        # Setup mock DataFrame
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        mock_df.columns = list(mock_result.keys())
        mock_df.dropna.return_value = mock_df
        mock_df.sort_values.return_value = mock_df
        mock_df.drop_duplicates.return_value = mock_df
        
        # Run evaluation
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model", 
            model_identifier="test_identifier",
            tokenizer_path="/path/to/tokenizer",
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1
        )
        
        # Verify processing was called
        mock_dependencies['process_question_esc_stream'].assert_called_once()
        
        # Verify DataFrame operations
        mock_dataframe.assert_called()
        mock_df.to_csv.assert_called()
    
    @pytest.mark.asyncio
    async def test_run_esc_evaluation_with_iteration_range(self, mock_dependencies):
        """Test ESC evaluation with specific iteration range."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": f"test_{i}", "question": f"Question {i}?", "answer": f"{i}"}
            for i in range(1, 6)  # 5 questions
        ]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock process result
        mock_dependencies['process_question_esc_stream'].return_value = {
            "iteration": 1,
            "score": 1
        }
        
        # Run evaluation with specific range
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier", 
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1,
            start_iteration=2,
            end_iteration=4
        )
        
        # Should process iterations 2, 3, 4 (3 total)
        assert mock_dependencies['process_question_esc_stream'].call_count == 3
    
    @pytest.mark.asyncio
    async def test_run_esc_evaluation_with_specific_iterations(self, mock_dependencies):
        """Test ESC evaluation with specific iteration list."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": f"test_{i}", "question": f"Question {i}?", "answer": f"{i}"}
            for i in range(1, 11)  # 10 questions
        ]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock process result
        mock_dependencies['process_question_esc_stream'].return_value = {
            "iteration": 1,
            "score": 1
        }
        
        # Run evaluation with specific iterations
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1,
            specific_iterations=[1, 3, 5, 7]
        )
        
        # Should process only the specified iterations
        assert mock_dependencies['process_question_esc_stream'].call_count == 4
    
    @pytest.mark.asyncio
    async def test_run_esc_evaluation_processing_failure(self, mock_dependencies):
        """Test handling of processing failures."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": "test_1", "question": "What is 2+2?", "answer": "4"},
            {"id": "test_2", "question": "What is 3+3?", "answer": "6"}
        ]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock process results - first succeeds, second fails
        mock_dependencies['process_question_esc_stream'].side_effect = [
            {
                "iteration": 1,
                "question_id": "test_1",
                "score": 1
            },
            None  # Processing failure
        ]
        
        # Run evaluation
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1
        )
        
        # Both questions should be attempted
        assert mock_dependencies['process_question_esc_stream'].call_count == 2
    
    @pytest.mark.asyncio
    async def test_run_esc_evaluation_exception_handling(self, mock_dependencies):
        """Test exception handling during evaluation."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": "test_1", "question": "What is 2+2?", "answer": "4"}
        ]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock to raise exception
        mock_dependencies['process_question_esc_stream'].side_effect = Exception("Processing error")
        
        # Run evaluation - should not raise exception
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier", 
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1
        )
        
        # Should attempt processing despite exception
        mock_dependencies['process_question_esc_stream'].assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_esc_evaluation_csv_output(self, mock_dependencies):
        """Test CSV output functionality."""
        # Setup mock dataset handler
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": "test_1", "question": "What is 2+2?", "answer": "4"}
        ]
        mock_dependencies['DatasetHandler'].return_value = mock_handler
        
        # Setup mock process result with comprehensive data
        mock_result = {
            "iteration": 1,
            "question_id": "test_1",
            "n_chains_max": 4,
            "window_size": 2,
            "n_chains_generated": 3,
            "voted_answer": "4",
            "score": 1,
            "consensus_found": True,
            "consensus_window": 2,
            "total_time": 15.5,
            "kv_cache_usage": {"mean_gpu_cache_perc": 0.65}
        }
        mock_dependencies['process_question_esc_stream'].return_value = mock_result
        
        # Setup DataFrame mock
        mock_df = Mock()
        mock_dependencies['pd'].DataFrame.return_value = mock_df
        mock_df.columns = []
        mock_df.dropna.return_value = mock_df
        mock_df.sort_values.return_value = mock_df
        mock_df.drop_duplicates.return_value = mock_df
        
        # Run evaluation
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1
        )
        
        # Verify DataFrame was created and saved
        mock_dependencies['pd'].DataFrame.assert_called()
        mock_df.to_csv.assert_called()


class TestEscEvalIntegration:
    """Integration tests for ESC evaluation functionality."""
    
    @pytest.mark.asyncio
    @patch('prune.evaluation.esc_eval.process_question_esc_stream')
    @patch('prune.evaluation.esc_eval.DatasetHandler')
    @patch('prune.evaluation.esc_eval.setup_output_directories')
    @patch('prune.evaluation.esc_eval.clear_source_kv_cache')
    @patch('prune.evaluation.esc_eval.pd.DataFrame')
    async def test_complete_evaluation_workflow(
        self, mock_dataframe, mock_clear_cache, mock_setup_dirs, 
        mock_dataset_handler, mock_process_question
    ):
        """Test complete evaluation workflow."""
        # Setup directory paths
        mock_paths = {
            "results_csv_file": "/tmp/results.csv",
            "source_usage_file": "/tmp/usage.json",
            "summaries_dir": "/tmp/summaries"
        }
        mock_setup_dirs.return_value = mock_paths
        
        # Setup dataset
        mock_handler = Mock()
        mock_handler.load_data.return_value = [
            {"id": "q1", "question": "2+2=?", "answer": "4"},
            {"id": "q2", "question": "3+3=?", "answer": "6"}
        ]
        mock_dataset_handler.return_value = mock_handler
        
        # Setup processing results
        async def mock_process_side_effect(example, iteration, *args, **kwargs):
            return {
                "iteration": iteration,
                "question_id": example["id"],
                "n_chains_max": 4,
                "window_size": 2,
                "n_chains_generated": 3,
                "voted_answer": example["answer"],
                "score": 1,
                "consensus_found": True,
                "consensus_window": 1,
                "total_time": 10.0 + iteration,
                "kv_cache_usage": {"mean_gpu_cache_perc": 0.5 + 0.1 * iteration}
            }
        
        mock_process_question.side_effect = mock_process_side_effect
        
        # Setup DataFrame mock
        mock_df = Mock()
        mock_dataframe.return_value = mock_df
        mock_df.columns = ["iteration", "question_id", "score"]
        mock_df.dropna.return_value = mock_df
        mock_df.sort_values.return_value = mock_df
        mock_df.drop_duplicates.return_value = mock_df
        
        # Run evaluation
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path="/path/to/tokenizer",
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1
        )
        
        # Verify all components were called
        mock_setup_dirs.assert_called_once()
        mock_clear_cache.assert_called_once()
        mock_handler.load_data.assert_called_once()
        assert mock_process_question.call_count == 2
        mock_dataframe.assert_called()
        mock_df.to_csv.assert_called()
    
    def test_flatten_dict_real_usage(self):
        """Test flatten_dict with realistic ESC evaluation data."""
        # Realistic data structure from ESC evaluation
        eval_data = {
            "basic_info": {
                "iteration": 1,
                "question_id": "q1",
                "model_name": "test_model"
            },
            "results": {
                "voted_answer": "A",
                "score": 1,
                "consensus_found": True
            },
            "performance": {
                "timing": {
                    "total_time": 15.5,
                    "avg_time_per_chain": 5.2
                },
                "resource_usage": {
                    "kv_cache": {
                        "mean_gpu_cache_perc": 0.65,
                        "max_gpu_cache_perc": 0.78
                    }
                }
            }
        }
        
        flattened = flatten_dict(eval_data)
        
        # Verify key flattening
        assert "basic_info_iteration" in flattened
        assert "basic_info_question_id" in flattened
        assert "results_voted_answer" in flattened
        assert "performance_timing_total_time" in flattened
        assert "performance_resource_usage_kv_cache_mean_gpu_cache_perc" in flattened
        
        # Verify values preserved
        assert flattened["basic_info_iteration"] == 1
        assert flattened["results_score"] == 1
        assert flattened["performance_timing_total_time"] == 15.5
        assert flattened["performance_resource_usage_kv_cache_mean_gpu_cache_perc"] == 0.65


class TestEscEvalEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    @patch('prune.evaluation.esc_eval.DatasetHandler')
    @patch('prune.evaluation.esc_eval.setup_output_directories')
    @patch('prune.evaluation.esc_eval.clear_source_kv_cache')
    async def test_invalid_iteration_range(self, mock_clear_cache, mock_setup_dirs, mock_dataset_handler):
        """Test with invalid iteration range."""
        mock_setup_dirs.return_value = {"source_usage_file": "/tmp/usage.json"}
        
        # Setup dataset with 5 questions
        mock_handler = Mock()
        mock_handler.load_data.return_value = [{"id": f"q{i}"} for i in range(1, 6)]
        mock_dataset_handler.return_value = mock_handler
        
        # Test with start > end
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1,
            start_iteration=4,
            end_iteration=2  # Invalid: end < start
        )
        
        # Should handle gracefully without processing questions
        mock_handler.load_data.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('prune.evaluation.esc_eval.DatasetHandler')
    @patch('prune.evaluation.esc_eval.setup_output_directories')
    @patch('prune.evaluation.esc_eval.clear_source_kv_cache')
    async def test_out_of_bounds_iterations(self, mock_clear_cache, mock_setup_dirs, mock_dataset_handler):
        """Test with out-of-bounds iteration specifications."""
        mock_setup_dirs.return_value = {"source_usage_file": "/tmp/usage.json"}
        
        # Setup dataset with 3 questions
        mock_handler = Mock()
        mock_handler.load_data.return_value = [{"id": f"q{i}"} for i in range(1, 4)]
        mock_dataset_handler.return_value = mock_handler
        
        # Test with specific iterations beyond dataset size
        await run_esc_evaluation_async(
            dataset_name="test_dataset",
            model_name="test_model",
            model_identifier="test_identifier",
            tokenizer_path=None,
            n_chains=4,
            window_size=2,
            vllm_url="http://localhost:8000",
            base_output_dir="/tmp",
            run_index=1,
            specific_iterations=[1, 5, 10]  # 5 and 10 are out of bounds
        )
        
        # Should handle gracefully and only process valid iterations
        mock_handler.load_data.assert_called_once()
    
    def test_flatten_dict_edge_cases(self):
        """Test flatten_dict with edge cases."""
        # Test with None values
        input_dict = {"a": None, "b": {"c": None}}
        result = flatten_dict(input_dict)
        assert result == {"a": None, "b_c": None}
        
        # Test with empty nested dict
        input_dict = {"a": {}, "b": {"c": {}}}
        result = flatten_dict(input_dict)
        assert result == {}
        
        # Test with numeric keys (should convert to string)
        input_dict = {1: {"2": "value"}}
        result = flatten_dict(input_dict)
        assert result == {"1_2": "value"}
        
        # Test deep nesting
        input_dict = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        result = flatten_dict(input_dict)
        assert result == {"a_b_c_d_e": "deep"}