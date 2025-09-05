# tests/prune/evaluation/test_sc_control_eval.py

import pytest
import os
import json
import pandas as pd
import sys
from unittest.mock import patch, AsyncMock, MagicMock, call
import unittest

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# Import the module to be tested
from slimsc.prune.evaluation import sc_control_eval

# --- Unit Tests for Helper Functions ---

@pytest.mark.unit
class TestHelperFunctions:
    """Tests for standalone helper functions."""

    def test_flatten_dict(self):
        """Test the dictionary flattening logic."""
        nested_dict = {
            'a': 1,
            'b': {'c': 2, 'd': {'e': 3}},
            'f': 4
        }
        expected = {'a': 1, 'b_c': 2, 'b_d_e': 3, 'f': 4}
        assert sc_control_eval.flatten_dict(nested_dict) == expected

    def test_flatten_dict_empty(self):
        """Test flattening an empty dictionary."""
        assert sc_control_eval.flatten_dict({}) == {}

    def test_flatten_dict_already_flat(self):
        """Test flattening an already flat dictionary."""
        flat_dict = {'a': 1, 'b': 2}
        assert sc_control_eval.flatten_dict(flat_dict) == flat_dict

    def test_setup_output_directories(self, tmp_path):
        """Test directory creation and path generation."""
        base_dir = str(tmp_path)
        paths = sc_control_eval.setup_output_directories(
            base_output_dir=base_dir,
            model_name="test-model",
            dataset_name="test-ds",
            sc_value=5,
            run_index=1
        )

        expected_base = tmp_path / "test-model" / "test-ds" / "sc_5_control" / "run1"
        assert os.path.isdir(expected_base)
        assert os.path.isdir(paths["chains"])
        assert os.path.isdir(paths["summaries"])
        assert os.path.isdir(paths["kvcache_usages_dir"])
        assert paths["base"] == str(expected_base)
        assert paths["csv"] == str(expected_base / "evaluation_summary.csv")


@pytest.mark.unit
class TestCalculateMeanStats:
    """Tests for the aggregation of stats across multiple runs."""

    def test_calculate_and_save_mean_stats_happy_path(self, tmp_path):
        """Test successful calculation and saving of mean stats."""
        base_run_dir = tmp_path / "test-model" / "test-ds" / "sc_5_control"
        
        # Create Run 1
        run1_dir = base_run_dir / "run1"
        run1_dir.mkdir(parents=True)
        with open(run1_dir / "aggregated_metrics.json", "w") as f:
            json.dump({
                "config": {"n_chains": 5},
                "metrics": {
                    "overall_accuracy": 0.8,
                    "tokens": {"total": 100},
                    "mean_max_kv_cache_usage_per_question_perc": 50.0
                }
            }, f)

        # Create Run 2
        run2_dir = base_run_dir / "run2"
        run2_dir.mkdir()
        with open(run2_dir / "aggregated_metrics.json", "w") as f:
            json.dump({
                "config": {"n_chains": 5},
                "metrics": {
                    "overall_accuracy": 0.9,
                    "tokens": {"total": 120},
                    "mean_max_kv_cache_usage_per_question_perc": 60.0
                }
            }, f)

        sc_control_eval.calculate_and_save_mean_stats(str(base_run_dir))

        output_file = base_run_dir / "mean_aggregated_metrics.json"
        assert output_file.exists()
        with open(output_file, 'r') as f:
            result = json.load(f)

        assert result["num_runs_aggregated"] == 2
        assert result["config"] == {"n_chains": 5}
        assert result["mean"]["overall_accuracy"] == pytest.approx(0.85)
        assert result["mean"]["tokens_total"] == pytest.approx(110.0)
        assert "mean_mean_max_kv_cache_usage_per_question_perc" in result["mean"]
        assert result["mean"]["mean_mean_max_kv_cache_usage_per_question_perc"] == pytest.approx(55.0)
        assert result["std_dev"]["overall_accuracy_std"] == pytest.approx(pd.Series([0.8, 0.9]).std())
    
    def test_calculate_and_save_mean_stats_no_files(self, tmp_path, caplog):
        """Test behavior when no valid metric files are found."""
        base_run_dir = tmp_path / "test-model" / "test-ds" / "sc_5_control"
        (base_run_dir / "run1").mkdir(parents=True)
        
        sc_control_eval.calculate_and_save_mean_stats(str(base_run_dir))

        assert "No valid aggregated_metrics.json files found" in caplog.text
        assert not (base_run_dir / "mean_aggregated_metrics.json").exists()


# --- Mocks and Fixtures for Integration-style Tests ---

@pytest.fixture
def mock_paths(tmp_path):
    """Provides a mocked dictionary of output paths."""
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    return {
        "base": str(run_dir),
        "chains": str(run_dir / "chains"),
        "summaries": str(run_dir / "summaries"),
        "csv": str(run_dir / "evaluation_summary.csv"),
        "kvcache_usages_dir": str(run_dir / "kvcache_usages"),
        "source_usage_file": str(run_dir / "kvcache_usages.csv"),
        "aggregated_metrics_json": str(run_dir / "aggregated_metrics.json")
    }

@pytest.fixture
def mock_all_dependencies(mocker, mock_paths):
    """Mocks all external dependencies for run_sc_evaluation_async."""
    mocker.patch('slimsc.prune.evaluation.sc_control_eval.setup_output_directories', return_value=mock_paths)
    mocker.patch('slimsc.prune.evaluation.sc_control_eval.clear_source_kv_cache')
    
    mock_dataset_handler = MagicMock()
    mock_dataset_handler.load_dataset.return_value = [
        {"id": "q1", "question": "...", "answer": "A"},
        {"id": "q2", "question": "...", "answer": "B"},
        {"id": "q3", "question": "...", "answer": "C"}
    ]
    mocker.patch('slimsc.prune.evaluation.sc_control_eval.DatasetHandler', return_value=mock_dataset_handler)
    
    mock_process_question = mocker.patch(
        'slimsc.prune.evaluation.sc_control_eval.process_question_sc_stream',
        new_callable=AsyncMock
    )
    # Configure a default return value for the mock
    mock_process_question.return_value = {
        "iteration": 1, "question_id": "q1", "final_score": 1.0, 
        "total_completion_tokens": 100, "max_kv_cache_usage": 50.0
    }

    mocker.patch('slimsc.prune.evaluation.sc_control_eval.calculate_and_save_mean_stats')
    mocker.patch('slimsc.prune.evaluation.sc_control_eval.close_aiohttp_session', new_callable=AsyncMock)

    return {
        "setup_output": sc_control_eval.setup_output_directories,
        "clear_cache": sc_control_eval.clear_source_kv_cache,
        "dataset_handler": mock_dataset_handler,
        "process_question": mock_process_question,
        "calc_mean": sc_control_eval.calculate_and_save_mean_stats,
        "close_session": sc_control_eval.close_aiohttp_session,
    }


# --- Tests for the Main Async Evaluation Function ---

@pytest.mark.integration
class TestRunScEvaluationAsync:
    """Tests the main orchestration logic of the evaluation script."""

    async def test_fresh_run_with_range(self, mock_all_dependencies, mock_paths):
        """Test a new run processing a range of questions."""
        await sc_control_eval.run_sc_evaluation_async(
            dataset_name="test-ds",
            model_name="test-model",
            model_identifier="model-id",
            tokenizer_path="/fake/path",
            n_chains=5,
            vllm_url="http://localhost",
            base_output_dir=os.path.dirname(mock_paths["base"]),
            run_index=1,
            start_iteration=1,
            end_iteration=2
        )

        # Assert process_question was called for iterations 1 and 2
        assert mock_all_dependencies["process_question"].call_count == 2
        mock_all_dependencies["process_question"].assert_any_call(
            unittest.mock.ANY, 1, 5, mock_paths, "http://localhost", "model-id", "/fake/path", "test-ds"
        )
        mock_all_dependencies["process_question"].assert_any_call(
            unittest.mock.ANY, 2, 5, mock_paths, "http://localhost", "model-id", "/fake/path", "test-ds"
        )
        
        # Check that final CSV and JSON were created
        assert os.path.exists(mock_paths["csv"])
        assert os.path.exists(mock_paths["aggregated_metrics_json"])
        mock_all_dependencies["calc_mean"].assert_called_once()
        mock_all_dependencies["close_session"].assert_called_once()

    async def test_resuming_from_existing_csv(self, mock_all_dependencies, mock_paths):
        """Test that the script correctly resumes from a partial results file."""
        # Create a pre-existing CSV file for iteration 1
        existing_df = pd.DataFrame([{"iteration": 1, "question_id": "q1", "final_score": 1.0}])
        existing_df.to_csv(mock_paths["csv"], index=False)
        
        await sc_control_eval.run_sc_evaluation_async(
            dataset_name="test-ds",
            model_name="test-model",
            model_identifier="model-id",
            tokenizer_path=None,
            n_chains=5,
            vllm_url="http://localhost",
            base_output_dir=os.path.dirname(mock_paths["base"]),
            run_index=1,
            start_iteration=1,
            end_iteration=3 # Try to run up to 3
        )
        
        # Should only process iterations 2 and 3
        assert mock_all_dependencies["process_question"].call_count == 2
        # Check it didn't call for iteration 1 again
        for mock_call in mock_all_dependencies["process_question"].call_args_list:
            assert mock_call.args[1] != 1

    async def test_specific_iterations_argument(self, mock_all_dependencies):
        """Test that --iterations logic correctly selects questions."""
        await sc_control_eval.run_sc_evaluation_async(
            dataset_name="test-ds",
            model_name="test-model",
            model_identifier="model-id",
            tokenizer_path=None,
            n_chains=5,
            vllm_url="http://localhost",
            base_output_dir="/fake/dir",
            run_index=1,
            specific_iterations=[1, 3] # Only run questions 1 and 3
        )

        assert mock_all_dependencies["process_question"].call_count == 2
        assert mock_all_dependencies["process_question"].call_args_list[0].args[1] == 1
        assert mock_all_dependencies["process_question"].call_args_list[1].args[1] == 3

    async def test_aggregation_logic(self, mock_all_dependencies, mock_paths):
        """Verify the correctness of the final metrics aggregation."""
        # Setup mock to return different values for each question
        mock_all_dependencies["process_question"].side_effect = [
            {"iteration": 1, "question_id": "q1", "final_score": 1.0, "total_completion_tokens": 100, "max_kv_cache_usage": 50.0},
            {"iteration": 2, "question_id": "q2", "final_score": 0.0, "total_completion_tokens": 150, "max_kv_cache_usage": 70.0}
        ]
        
        await sc_control_eval.run_sc_evaluation_async(
            dataset_name="test-ds",
            model_name="test-model",
            model_identifier="model-id",
            tokenizer_path="/fake/path",
            n_chains=5,
            vllm_url="http://localhost",
            base_output_dir=os.path.dirname(mock_paths["base"]),
            run_index=1,
            end_iteration=2
        )
        
        with open(mock_paths["aggregated_metrics_json"], 'r') as f:
            metrics = json.load(f)

        assert metrics["metrics"]["num_questions_processed"] == 2
        assert metrics["metrics"]["overall_accuracy"] == pytest.approx(0.5)
        assert metrics["metrics"]["mean_total_completion_tokens_per_question"] == pytest.approx(125.0)
        assert metrics["metrics"]["mean_max_kv_cache_usage_per_question_perc"] == pytest.approx(60.0)


@pytest.mark.unit
class TestMainFunction:
    """Tests the argument parsing and main function invocation logic."""

    @patch('slimsc.prune.evaluation.sc_control_eval.run_sc_evaluation_async', new_callable=AsyncMock)
    @patch('slimsc.prune.evaluation.sc_control_eval.DatasetHandler')
    def test_main_with_num_qns(self, mock_dataset_handler, mock_run_sc_eval_async, mocker):
        """Test --num_qns argument parsing and random selection."""
        # Mock DatasetHandler to control the number of available questions
        mock_dataset_handler.return_value.load_dataset.return_value = [i for i in range(20)]
        
        # Mock sys.argv to simulate command-line arguments
        mocker.patch.object(sys, 'argv', [
            'sc_control_eval.py',
            '--n_start', '5',
            '--model_name', 'test-model',
            '--model_identifier', 'model-id',
            '--num_qns', '10'
        ])

        sc_control_eval.main()

        # Check that the async function was called once
        mock_run_sc_eval_async.assert_awaited_once()
        # Get the arguments passed to the async function
        call_args = mock_run_sc_eval_async.call_args.kwargs
        
        # Verify that specific_iterations is a list of 10 items
        assert isinstance(call_args['specific_iterations'], list)
        assert len(call_args['specific_iterations']) == 10
        # Verify the numbers are within the valid range (1-indexed)
        assert min(call_args['specific_iterations']) >= 1
        assert max(call_args['specific_iterations']) <= 20
    
    @patch('slimsc.prune.evaluation.sc_control_eval.run_sc_evaluation_async', new_callable=AsyncMock)
    def test_main_with_iterations_range(self, mock_run_sc_eval_async, mocker):
        """Test --iterations argument parsing with ranges."""
        mocker.patch.object(sys, 'argv', [
            'sc_control_eval.py',
            '--n_start', '5',
            '--model_name', 'test-model',
            '--model_identifier', 'model-id',
            '--iterations', '1,5,10-12'
        ])
        
        sc_control_eval.main()
        
        mock_run_sc_eval_async.assert_awaited_once()
        call_args = mock_run_sc_eval_async.call_args.kwargs
        assert call_args['specific_iterations'] == [1, 5, 10, 11, 12]

    @patch('slimsc.prune.evaluation.sc_control_eval.run_sc_evaluation_async', new_callable=AsyncMock)
    def test_main_with_start_end(self, mock_run_sc_eval_async, mocker):
        """Test that start/end are used when other options are absent."""
        mocker.patch.object(sys, 'argv', [
            'sc_control_eval.py',
            '--n_start', '5',
            '--model_name', 'test-model',
            '--model_identifier', 'model-id',
            '--start', '10',
            '--end', '20'
        ])
        
        sc_control_eval.main()
        
        mock_run_sc_eval_async.assert_awaited_once()
        call_args = mock_run_sc_eval_async.call_args.kwargs
        assert call_args['specific_iterations'] is None
        assert call_args['start_iteration'] == 10
        assert call_args['end_iteration'] == 20