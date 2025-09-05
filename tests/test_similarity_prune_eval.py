# tests/prune/evaluation/test_similarity_prune_eval.py

import pytest
import os
import json
import pandas as pd
import sys
from unittest.mock import patch, AsyncMock, MagicMock

# Mark all tests in this file as asyncio
pytestmark = pytest.mark.asyncio

# Import the module to be tested
# We assume the file is named similarity_prune_eval.py
from slimsc.prune.evaluation import similarity_prune_eval

# --- Unit Tests for Helper Functions ---

@pytest.mark.unit
class TestHelperFunctions:
    """Tests for standalone helper functions."""

    # This function is identical to the one in the other script,
    # but it's good practice to include its test here for completeness.
    def test_flatten_dict(self):
        """Test the dictionary flattening logic."""
        nested_dict = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}, 'f': 4}
        expected = {'a': 1, 'b_c': 2, 'b_d_e': 3, 'f': 4}
        assert similarity_prune_eval.flatten_dict(nested_dict) == expected

    def test_setup_output_directories_prune(self, tmp_path):
        """Test directory creation for the pruning script."""
        base_dir = str(tmp_path)
        paths = similarity_prune_eval.setup_output_directories_prune(
            base_output_dir=base_dir,
            model_name="test-model",
            dataset_name="test-ds",
            n_start=8,
            threshold=0.85,
            pruning_strategy="diversity",
            num_steps_to_delay_pruning=10,
            run_index=2
        )

        expected_run_name = "diversity_n8_thresh0.85_delay10"
        expected_base = tmp_path / "test-model" / "test-ds" / expected_run_name / "run2"
        
        assert os.path.isdir(expected_base)
        assert os.path.isdir(paths["chains"])
        assert os.path.isdir(paths["summaries"])
        assert paths["base"] == str(expected_base)
        assert paths["csv"] == str(expected_base / "evaluation_summary.csv")
        assert paths["aggregated_metrics_json"] == str(expected_base / "aggregated_metrics.json")


# The calculate_and_save_mean_stats function is identical, so its test is also reused.
@pytest.mark.unit
class TestCalculateMeanStats:
    """Tests for the aggregation of stats across multiple runs."""

    def test_calculate_and_save_mean_stats_happy_path(self, tmp_path):
        """Test successful calculation and saving of mean stats."""
        base_run_dir = tmp_path / "test-model" / "test-ds" / "diversity_n8_thresh0.85_delay10"
        
        # Create Run 1
        run1_dir = base_run_dir / "run1"
        run1_dir.mkdir(parents=True)
        with open(run1_dir / "aggregated_metrics.json", "w") as f:
            json.dump({
                "config": {"n_chains_start": 8},
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
                "config": {"n_chains_start": 8},
                "metrics": {
                    "overall_accuracy": 0.9,
                    "tokens": {"total": 120},
                    "mean_max_kv_cache_usage_per_question_perc": 60.0
                }
            }, f)

        similarity_prune_eval.calculate_and_save_mean_stats(str(base_run_dir))

        output_file = base_run_dir / "mean_aggregated_metrics.json"
        assert output_file.exists()
        with open(output_file, 'r') as f:
            result = json.load(f)

        assert result["num_runs_aggregated"] == 2
        assert result["config"] == {"n_chains_start": 8}
        assert result["mean"]["overall_accuracy"] == pytest.approx(0.85)
        assert "mean_mean_max_kv_cache_usage_per_question_perc" in result["mean"]
        assert result["mean"]["mean_mean_max_kv_cache_usage_per_question_perc"] == pytest.approx(55.0)

# --- Mocks and Fixtures for Integration-style Tests ---

@pytest.fixture
def mock_paths_prune(tmp_path):
    """Provides a mocked dictionary of output paths for the prune script."""
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
def mock_all_dependencies_prune(mocker, mock_paths_prune):
    """Mocks all external dependencies for the main async function."""
    mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.setup_output_directories_prune', return_value=mock_paths_prune)
    mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.clear_source_kv_cache')
    mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.get_embedding_model')
    
    mock_dataset_handler = MagicMock()
    mock_dataset_handler.load_dataset.return_value = [
        {"id": "q1", "question": "...", "answer": "A"},
        {"id": "q2", "question": "...", "answer": "B"},
        {"id": "q3", "question": "...", "answer": "C"}
    ]
    mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.DatasetHandler', return_value=mock_dataset_handler)
    
    mock_process_question = mocker.patch(
        'slimsc.prune.evaluation.similarity_prune_eval.process_question_similarity_prune',
        new_callable=AsyncMock
    )
    mock_process_question.return_value = {
        "iteration": 1, "question_id": "q1", "final_score": 1.0, 
        "total_completion_tokens": 100, "max_kv_cache_usage": 50.0
    }

    mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.calculate_and_save_mean_stats')
    mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.close_aiohttp_session', new_callable=AsyncMock)

    return {
        "process_question": mock_process_question,
        "calc_mean": similarity_prune_eval.calculate_and_save_mean_stats,
        "close_session": similarity_prune_eval.close_aiohttp_session,
    }


# --- Tests for the Main Async Evaluation Function ---

@pytest.mark.integration
class TestRunSimilarityPruningAsync:
    """Tests the main orchestration logic of the pruning evaluation script."""

    async def test_fresh_run_with_range(self, mock_all_dependencies_prune, mock_paths_prune):
        """Test a new run processing a range of questions."""
        await similarity_prune_eval.run_similarity_pruning_evaluation_async(
            dataset_name="test-ds",
            model_name="test-model",
            model_identifier="model-id",
            tokenizer_path="/fake/tokenizer",
            n_chains_start=8,
            similarity_threshold=0.85,
            pruning_strategy="diversity",
            vllm_url="http://localhost",
            base_output_dir=os.path.dirname(mock_paths_prune["base"]),
            run_index=1,
            seed_for_run=42,
            num_steps_to_delay_pruning=10,
            start_iteration=1,
            end_iteration=2
        )

        # Assert process_question was called for iterations 1 and 2
        assert mock_all_dependencies_prune["process_question"].call_count == 2
        
        # Check that final CSV and JSON were created
        assert os.path.exists(mock_paths_prune["csv"])
        assert os.path.exists(mock_paths_prune["aggregated_metrics_json"])
        mock_all_dependencies_prune["calc_mean"].assert_called_once()
        mock_all_dependencies_prune["close_session"].assert_called_once()

    async def test_embedding_model_load_failure(self, mocker, caplog):
        """Test that the script exits if the embedding model fails to load."""
        mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.setup_output_directories_prune', return_value={})
        mocker.patch('slimsc.prune.evaluation.similarity_prune_eval.get_embedding_model', side_effect=Exception("Model not found"))
        
        await similarity_prune_eval.run_similarity_pruning_evaluation_async(
            dataset_name="test-ds", model_name="test-model", model_identifier="model-id",
            tokenizer_path="/fake", n_chains_start=8, similarity_threshold=0.8,
            pruning_strategy="random", vllm_url="http://localhost",
            base_output_dir="/fake", run_index=1, seed_for_run=0, num_steps_to_delay_pruning=10
        )
        assert "Failed to load embedding model. Cannot proceed." in caplog.text

    async def test_aggregation_logic(self, mock_all_dependencies_prune, mock_paths_prune):
        """Verify the correctness of the final metrics aggregation."""
        mock_all_dependencies_prune["process_question"].side_effect = [
            {"iteration": 1, "question_id": "q1", "final_score": 1.0, "total_completion_tokens": 100, "max_kv_cache_usage": 50.0},
            {"iteration": 2, "question_id": "q2", "final_score": 0.0, "total_completion_tokens": 150, "max_kv_cache_usage": 70.0}
        ]
        
        await similarity_prune_eval.run_similarity_pruning_evaluation_async(
            dataset_name="test-ds", model_name="test-model", model_identifier="model-id",
            tokenizer_path="/fake", n_chains_start=8, similarity_threshold=0.8,
            pruning_strategy="random", vllm_url="http://localhost",
            base_output_dir=os.path.dirname(mock_paths_prune["base"]), run_index=1,
            seed_for_run=0, num_steps_to_delay_pruning=10, end_iteration=2
        )
        
        with open(mock_paths_prune["aggregated_metrics_json"], 'r') as f:
            metrics = json.load(f)

        assert metrics["metrics"]["num_qns_processed"] == 2
        assert metrics["metrics"]["overall_accuracy"] == pytest.approx(0.5)
        assert metrics["metrics"]["mean_total_completion_tokens_per_question"] == pytest.approx(125.0)
        assert metrics["metrics"]["mean_max_kv_cache_usage_per_question_perc"] == pytest.approx(60.0)


@pytest.mark.unit
class TestMainFunction:
    """Tests the argument parsing and main function invocation logic."""

    @patch('slimsc.prune.evaluation.similarity_prune_eval.run_similarity_pruning_evaluation_async', new_callable=AsyncMock)
    @patch('slimsc.prune.evaluation.similarity_prune_eval.DatasetHandler')
    def test_main_with_num_qns_and_batching(self, mock_dataset_handler, mock_run_eval_async, mocker):
        """Test --num_qns with batching arguments."""
        # Mock DatasetHandler to have 100 questions
        mock_dataset_handler.return_value.load_dataset.return_value = list(range(100))
        
        mocker.patch.object(sys, 'argv', [
            'similarity_prune_eval.py',
            '--n_start', '8',
            '--threshold', '0.8',
            '--pruning_strategy', 'diversity',
            '--model_name', 'test-model',
            '--model_identifier', 'model-id',
            '--tokenizer_path', '/fake/path',
            '--num_qns', '50', # Select 50 random questions
            '--seed', '42',
            '--batch_size', '10', # Batch size of 10
            '--batch_num', '1'   # Get the second batch (indices 10-19)
        ])

        similarity_prune_eval.main()

        mock_run_eval_async.assert_awaited_once()
        call_args = mock_run_eval_async.call_args.kwargs
        
        assert call_args['seed_for_run'] == 42
        assert len(call_args['specific_iterations']) == 10 # Batch size
        
        # Verify the batching logic by checking against a known random sequence
        import random
        random.seed(42)
        all_iters = list(range(1, 101))
        random.shuffle(all_iters)
        expected_selection = sorted(all_iters[:50])
        expected_batch = expected_selection[10:20]
        assert call_args['specific_iterations'] == expected_batch

    @patch('slimsc.prune.evaluation.similarity_prune_eval.run_similarity_pruning_evaluation_async', new_callable=AsyncMock)
    def test_main_with_invalid_threshold(self, mock_run_eval_async, mocker, caplog):
        """Test that main exits if the threshold is invalid."""
        mocker.patch.object(sys, 'argv', [
            'similarity_prune_eval.py',
            '--n_start', '8',
            '--threshold', '1.1', # Invalid threshold
            '--pruning_strategy', 'diversity',
            '--model_name', 'test-model',
            '--model_identifier', 'model-id',
            '--tokenizer_path', '/fake/path',
        ])

        similarity_prune_eval.main()

        # The main async function should NOT have been called
        mock_run_eval_async.assert_not_awaited()
        assert "Similarity threshold must be between 0.0" in caplog.text

    @patch('slimsc.prune.evaluation.similarity_prune_eval.run_similarity_pruning_evaluation_async', new_callable=AsyncMock)
    def test_main_passes_correct_args(self, mock_run_eval_async, mocker):
        """Test that regular arguments are passed correctly."""
        mocker.patch.object(sys, 'argv', [
            'similarity_prune_eval.py',
            '--n_start', '12',
            '--threshold', '0.9',
            '--pruning_strategy', 'most_thoughts',
            '--model_name', 'llama-test',
            '--model_identifier', 'meta-llama/Llama-2-7b-hf',
            '--tokenizer_path', '/models/llama',
            '--num_steps_to_delay_pruning', '5',
            '--run_index', '3',
            '--start', '10',
            '--end', '20'
        ])
        
        similarity_prune_eval.main()
        
        mock_run_eval_async.assert_awaited_once()
        call_args = mock_run_eval_async.call_args.kwargs
        
        assert call_args['n_chains_start'] == 12
        assert call_args['similarity_threshold'] == 0.9
        assert call_args['pruning_strategy'] == 'most_thoughts'
        assert call_args['model_name'] == 'llama-test'
        assert call_args['num_steps_to_delay_pruning'] == 5
        assert call_args['run_index'] == 3
        assert call_args['start_iteration'] == 10
        assert call_args['end_iteration'] == 20
        assert call_args['specific_iterations'] is None