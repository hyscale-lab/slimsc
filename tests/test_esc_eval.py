import pytest
import os
import json
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, call

# Import the module and functions to be tested
# Adjust the path if your structure is different
from slimsc.prune.evaluation.esc_eval import (
    flatten_dict,
    setup_output_directories,
    calculate_and_save_mean_stats,
    run_esc_evaluation_async,
)

# --- Fixtures ---

@pytest.fixture
def mock_dependencies(mocker):
    """A fixture to mock all external dependencies for the main async function."""
    mocks = {
        "setup_dirs": mocker.patch("slimsc.prune.evaluation.esc_eval.setup_output_directories"),
        "clear_cache": mocker.patch("slimsc.prune.evaluation.esc_eval.clear_source_kv_cache"),
        "dataset_handler": mocker.patch("slimsc.prune.evaluation.esc_eval.DatasetHandler"),
        "process_question": mocker.patch("slimsc.prune.evaluation.esc_eval.process_question_esc_stream", new_callable=AsyncMock),
        "close_session": mocker.patch("slimsc.prune.evaluation.esc_eval.close_aiohttp_session", new_callable=AsyncMock),
        "calculate_mean": mocker.patch("slimsc.prune.evaluation.esc_eval.calculate_and_save_mean_stats"),
    }
    return mocks

@pytest.fixture
def full_mock_process_result():
    """Provides a complete dictionary, matching all expected csv_cols."""
    return {
        "iteration": 1,
        "question_id": "q_1",
        "n_chains_max": 8,
        "window_size": 2,
        "n_chains_generated": 3,
        "stopped_early": True,
        "correct_answer": "A",
        "final_answer": "A",
        "final_score": 1.0,
        "prompt_tokens": 50,
        "total_completion_tokens": 100,
        "total_tokens": 150,
        "total_reasoning_tokens": 80,
        "total_non_reasoning_tokens": 20,
        "avg_kv_cache_usage": 0.8,
        "max_kv_cache_usage": 0.9,
        "processing_duration_sec": 10.5,
        "individual_answers_str": "['A', 'A', 'B']",
    }


# --- Tests for Utility Functions (No asyncio mark) ---

def test_flatten_dict():
    """Tests the dictionary flattening utility."""
    nested_dict = {
        'a': 1,
        'b': {'c': 2, 'd': {'e': 3}},
        'f': [4, 5],
        'g': None,
        1: {'h': 6}
    }
    expected = {'a': 1, 'b_c': 2, 'b_d_e': 3, 'f': [4, 5], 'g': None, '1_h': 6}
    assert flatten_dict(nested_dict) == expected

def test_flatten_dict_empty():
    """Tests flattening an empty dictionary."""
    assert flatten_dict({}) == {}


# --- Tests for File System and Setup Functions (No asyncio mark) ---

def test_setup_output_directories(tmp_path):
    """Tests that all necessary directories are created and paths are correct."""
    base_dir = tmp_path
    model = "test-model"
    dataset = "test-dataset"
    n_chains = 8
    window_size = 2
    run_index = 1
    paths = setup_output_directories(base_dir, model, dataset, n_chains, window_size, run_index)
    assert os.path.isdir(paths["base"])
    assert os.path.isdir(paths["chains"])
    expected_base_run_dir = os.path.join(base_dir, model, dataset, f"esc_n{n_chains}_w{window_size}")
    assert paths["base_run_dir"] == expected_base_run_dir


def test_calculate_and_save_mean_stats(tmp_path, caplog):
    """Tests the aggregation of metrics from multiple run directories."""
    base_run_dir = tmp_path / "test-model/test-dataset/esc_n8_w2"
    run1_dir = base_run_dir / "run1"
    run1_dir.mkdir(parents=True)
    run2_dir = base_run_dir / "run2"
    run2_dir.mkdir()
    run1_metrics = {"config": {"n_chains_max": 8}, "metrics": {"overall_accuracy": 0.8}}
    with open(run1_dir / "aggregated_metrics.json", "w") as f: json.dump(run1_metrics, f)
    run2_metrics = {"config": {"n_chains_max": 8}, "metrics": {"overall_accuracy": 0.9}}
    with open(run2_dir / "aggregated_metrics.json", "w") as f: json.dump(run2_metrics, f)
    calculate_and_save_mean_stats(str(base_run_dir))
    output_file = base_run_dir / "mean_aggregated_metrics.json"
    assert output_file.exists()
    with open(output_file, 'r') as f: mean_data = json.load(f)
    assert mean_data["num_runs_aggregated"] == 2
    assert pytest.approx(mean_data["mean"]["overall_accuracy"]) == 0.85

def test_calculate_and_save_mean_stats_no_files(tmp_path, caplog):
    """Tests the case where no valid metric files are found."""
    base_run_dir = tmp_path / "empty_run"
    base_run_dir.mkdir()
    calculate_and_save_mean_stats(str(base_run_dir))
    assert "No valid aggregated_metrics.json files found" in caplog.text


# --- Tests for the Main Asynchronous Evaluation Function (NEEDS asyncio mark) ---

@pytest.mark.asyncio
async def test_run_esc_evaluation_happy_path(tmp_path, mock_dependencies, full_mock_process_result):
    """Tests a successful run from start to finish without resuming."""
    # --- Setup Mocks ---
    run_specific_dir = tmp_path / "run1"
    run_specific_dir.mkdir(parents=True)
    mock_paths = {
        "base": str(run_specific_dir), "csv": str(run_specific_dir / "eval.csv"),
        "aggregated_metrics_json": str(run_specific_dir / "agg.json"),
        "base_run_dir": str(tmp_path)
    }
    mock_dependencies["setup_dirs"].return_value = mock_paths
    
    mock_dataset = [{"id": 1, "prompt": "Q1"}, {"id": 2, "prompt": "Q2"}]
    mock_dependencies["dataset_handler"].return_value.load_dataset.return_value = mock_dataset

    # Define results for the two calls using the complete fixture
    result1 = full_mock_process_result
    result2 = {**full_mock_process_result, "iteration": 2, "question_id": "q_2", "final_score": 0.0, "stopped_early": False}
    mock_dependencies["process_question"].side_effect = [result1, result2]

    # --- Run the Function ---
    await run_esc_evaluation_async(
        dataset_name="test_dataset", model_name="test_model", model_identifier="test/model",
        tokenizer_path=None, n_chains=8, window_size=2, vllm_url="fake_url",
        base_output_dir=str(tmp_path.parent), run_index=1, start_iteration=1, end_iteration=2
    )

    # --- Assertions ---
    mock_dependencies["setup_dirs"].assert_called_once()
    assert mock_dependencies["process_question"].call_count == 2
    
    final_csv_path = mock_paths["csv"]
    assert os.path.exists(final_csv_path)
    df = pd.read_csv(final_csv_path)
    assert len(df) == 2
    assert df.iloc[0]["final_score"] == 1.0
    assert 'n_chains_max' in df.columns


@pytest.mark.asyncio
async def test_run_esc_evaluation_resume_logic(tmp_path, mock_dependencies, full_mock_process_result):
    """Tests that the evaluation correctly resumes from a previous run."""
    # --- Setup Mocks ---
    run_specific_dir = tmp_path / "run1"
    run_specific_dir.mkdir(parents=True)
    csv_path = run_specific_dir / "eval.csv"
    
    # Create a pre-existing CSV file with a complete row
    existing_data = pd.DataFrame([full_mock_process_result])
    existing_data.to_csv(csv_path, index=False)

    mock_paths = {"csv": str(csv_path), "aggregated_metrics_json": str(run_specific_dir / "agg.json")}
    mock_dependencies["setup_dirs"].return_value = mock_paths

    mock_dataset = [{"id": 1, "prompt": "Q1"}, {"id": 2, "prompt": "Q2"}]
    mock_dependencies["dataset_handler"].return_value.load_dataset.return_value = mock_dataset
    
    # Mock only the result for the *new* question to be processed
    mock_dependencies["process_question"].return_value = {
        **full_mock_process_result, "iteration": 2, "question_id": "q_2", "final_score": 0.0
    }
    
    # --- Run the Function ---
    await run_esc_evaluation_async(
        dataset_name="test_dataset", model_name="test_model", model_identifier="test/model",
        tokenizer_path=None, n_chains=8, window_size=2, vllm_url="fake_url",
        base_output_dir=str(tmp_path.parent), run_index=1, start_iteration=1, end_iteration=2
    )
    
    # --- Assertions ---
    mock_dependencies["process_question"].assert_called_once()
    
    df = pd.read_csv(csv_path)
    assert len(df) == 2
    assert set(df['iteration']) == {1, 2}