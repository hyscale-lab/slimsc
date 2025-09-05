import pytest
import os
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, call

# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio

# Import the functions to be tested
from slimsc.prune.evaluation.processing import (
    process_question_sc_stream,
    process_question_esc_stream,
    process_single_stream,
)

# --- Fixtures ---

@pytest.fixture
def mock_paths(tmp_path):
    """Provides a temporary directory structure for test outputs."""
    paths = {
        "chains": tmp_path / "chains",
        "summaries": tmp_path / "summaries",
    }
    os.makedirs(paths["chains"])
    os.makedirs(paths["summaries"])
    return paths

@pytest.fixture
def mock_example():
    """Provides a sample dataset example."""
    return {"id": "test_q1", "question": "What is 2+2?"}

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for the processing functions."""
    
    # Mock the async generator returned by the client
    async def mock_stream_generator(*args, **kwargs):
        yield {"chunk": 1}
        yield {"chunk": 2}

    mocks = {
        "DatasetHandler": mocker.patch("slimsc.prune.evaluation.processing.DatasetHandler"),
        "stream_vllm_request": mocker.patch(
            "slimsc.prune.evaluation.processing.stream_vllm_request", 
            side_effect=mock_stream_generator
        ),
        "process_stream_chunks": mocker.patch("slimsc.prune.evaluation.processing.process_stream_chunks"),
        "extract_kv_cache": mocker.patch("slimsc.prune.evaluation.processing.extract_kv_cache_usage_for_question"),
        "majority_vote": mocker.patch("slimsc.prune.evaluation.processing.majority_vote"),
        "count_tokens": mocker.patch("slimsc.prune.evaluation.processing.count_tokens"),
    }
    
    # Default successful return values for mocks
    mocks["DatasetHandler"].return_value.create_prompt.return_value = ("Test Prompt", "Correct Answer")
    mocks["DatasetHandler"].return_value.extract_answer.return_value = "A"
    mocks["DatasetHandler"].return_value.calculate_score.return_value = 1.0
    mocks["process_stream_chunks"].return_value = {
        "chain_index": 1, "full_content": "The answer is A.", "final_answer_text": "A",
        "prompt_tokens": 10, "completion_tokens": 5,
    }
    mocks["extract_kv_cache"].return_value = {"avg_kv_cache_usage": 0.5, "max_kv_cache_usage": 0.6}
    mocks["majority_vote"].return_value = ("A", 1.0, ["A", "A", "B"])
    mocks["count_tokens"].return_value = 10 # Simulate successful token counting

    return mocks

# --- Tests for process_single_stream ---

async def test_process_single_stream(mocker):
    """Tests the helper coroutine that consumes a stream."""
    mock_process_chunks = mocker.patch("slimsc.prune.evaluation.processing.process_stream_chunks")
    
    async def mock_generator():
        yield {"data": "first"}
        yield {"data": "second"}

    stream_gen = mock_generator()
    await process_single_stream(stream_gen, chain_index=1)

    # Verify that it collected all chunks and passed them to the processing function
    expected_chunks = [{"data": "first"}, {"data": "second"}]
    mock_process_chunks.assert_called_once_with(expected_chunks, 1)


# --- Tests for process_question_sc_stream ---

async def test_sc_stream_happy_path(mock_dependencies, mock_paths, mock_example):
    """Tests the successful, standard execution of a self-consistency question."""
    n_chains = 3
    
    # Make process_stream_chunks return slightly different results for each chain
    mock_dependencies["process_stream_chunks"].side_effect = [
        {"chain_index": 1, "full_content": "Ans A", "final_answer_text": "A", "prompt_tokens": 10, "completion_tokens": 5},
        {"chain_index": 2, "full_content": "Ans A", "final_answer_text": "A", "prompt_tokens": 10, "completion_tokens": 6},
        {"chain_index": 3, "full_content": "Ans B", "final_answer_text": "B", "prompt_tokens": 10, "completion_tokens": 7},
    ]

    result = await process_question_sc_stream(
        example=mock_example, iteration=1, n_chains=n_chains, paths=mock_paths,
        vllm_url="fake_url", model_name="test_model", tokenizer_path="/fake/tokenizer", dataset_name="gpqa_diamond"
    )

    # Assertions
    assert result is not None
    assert result["final_score"] == 1.0
    assert result["voted_answer"] == "A"
    assert result["total_completion_tokens"] == 18 # 5 + 6 + 7
    assert mock_dependencies["stream_vllm_request"].call_count == n_chains
    assert mock_dependencies["majority_vote"].call_count == 1
    assert mock_dependencies["count_tokens"].call_count > 0 # Was called because tokenizer_path was provided

    # Check that summary and chain files were created
    summary_file = mock_paths["summaries"] / "question_1_summary.json"
    assert summary_file.exists()
    chain_file = mock_paths["chains"] / "question_1_chain_1_used_for_voting.txt"
    assert chain_file.exists()


async def test_sc_stream_prompt_creation_failure(mock_dependencies, mock_paths, mock_example, caplog):
    """Tests that the function handles errors during prompt creation."""
    mock_dependencies["DatasetHandler"].return_value.create_prompt.side_effect = ValueError("Bad prompt data")

    result = await process_question_sc_stream(
        example=mock_example, iteration=1, n_chains=3, paths=mock_paths,
        vllm_url="fake_url", model_name="test_model", tokenizer_path=None, dataset_name="gpqa_diamond"
    )
    
    assert result is None
    assert "Error creating prompt for question 1" in caplog.text
    
    # Check that a specific failure summary was written
    summary_file = mock_paths["summaries"] / "question_1_summary.json"
    assert summary_file.exists()
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    assert summary_data["status"] == "PROMPT_ERROR"


async def test_sc_stream_partial_failure(mock_dependencies, mock_paths, mock_example):
    """Tests handling when some streams fail and some succeed."""
    # Simulate one stream raising an exception and one returning an error dict
    mock_dependencies["process_stream_chunks"].side_effect = [
        {"chain_index": 1, "full_content": "Ans A", "final_answer_text": "A", "prompt_tokens": 10, "completion_tokens": 5},
        Exception("Task failed"),
        {"error": {"status": "client_error", "message": "Connection lost"}},
    ]

    result = await process_question_sc_stream(
        example=mock_example, iteration=1, n_chains=3, paths=mock_paths,
        vllm_url="fake_url", model_name="test_model", tokenizer_path=None, dataset_name="gpqa_diamond"
    )

    assert result is not None
    # majority_vote is mocked, so it will still return its default "A" based on the one successful chain
    assert result["voted_answer"] == "A"
    
    # Check summary for partial success status
    summary_file = mock_paths["summaries"] / "question_1_summary.json"
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    assert summary_data["status"] == "PARTIAL_SUCCESS (2_failed)"
    assert summary_data["n_chains_completed_stream_for_voting"] == 1
    assert len(summary_data["error_chain_details"]) == 2


async def test_sc_stream_no_extractable_answers(mock_dependencies, mock_paths, mock_example, caplog):
    """Tests the case where content is generated but no answers can be extracted."""
    mock_dependencies["DatasetHandler"].return_value.extract_answer.return_value = None

    result = await process_question_sc_stream(
        example=mock_example, iteration=1, n_chains=2, paths=mock_paths,
        vllm_url="fake_url", model_name="test_model", tokenizer_path=None, dataset_name="gpqa_diamond"
    )
    
    assert result is not None
    assert result["final_score"] == 0
    assert result["voted_answer"] is None
    assert "No chains with extracted answers" in caplog.text
    mock_dependencies["majority_vote"].assert_not_called()

# --- Tests for process_question_esc_stream ---

async def test_esc_stream_early_stop(mock_dependencies, mock_paths, mock_example):
    """Tests that ESC stops early when consensus is found in the first window."""
    n_chains_max, window_size = 8, 2
    
    # Mock stream chunks to return two identical answers in the first window
    mock_dependencies["process_stream_chunks"].side_effect = [
        {"chain_index": 1, "full_content": "The answer is A.", "completion_tokens": 5},
        {"chain_index": 2, "full_content": "I think it's A.", "completion_tokens": 6},
    ]
    # Mock extract_answer to return the same answer for both
    mock_dependencies["DatasetHandler"].return_value.extract_answer.return_value = "A"

    result = await process_question_esc_stream(
        example=mock_example, iteration=1, n_chains_max=n_chains_max, window_size=window_size,
        paths=mock_paths, vllm_url="fake_url", model_name="test_model", tokenizer_path=None, dataset_name="gpqa_diamond"
    )

    assert result["stopped_early"] is True
    assert result["n_chains_generated"] == window_size
    assert result["final_answer"] == "A"
    assert mock_dependencies["stream_vllm_request"].call_count == window_size
    mock_dependencies["majority_vote"].assert_not_called() # No vote needed on early stop


async def test_esc_stream_no_early_stop(mock_dependencies, mock_paths, mock_example):
    """Tests that ESC runs to max chains when no consensus is found."""
    n_chains_max, window_size = 4, 2

    # Mock different answers for each chain to prevent early stopping
    mock_dependencies["process_stream_chunks"].side_effect = [
        {"chain_index": 1, "full_content": "A", "completion_tokens": 5},
        {"chain_index": 2, "full_content": "B", "completion_tokens": 6},
        {"chain_index": 3, "full_content": "A", "completion_tokens": 7},
        {"chain_index": 4, "full_content": "C", "completion_tokens": 8},
    ]
    mock_dependencies["DatasetHandler"].return_value.extract_answer.side_effect = ["A", "B", "A", "C"]
    
    # Mock majority vote to return 'A' as it's the most common
    mock_dependencies["majority_vote"].return_value = ("A", 1.0, ["A", "B", "A", "C"])

    result = await process_question_esc_stream(
        example=mock_example, iteration=1, n_chains_max=n_chains_max, window_size=window_size,
        paths=mock_paths, vllm_url="fake_url", model_name="test_model", tokenizer_path=None, dataset_name="gpqa_diamond"
    )

    assert result["stopped_early"] is False
    assert result["n_chains_generated"] == n_chains_max
    assert result["final_answer"] == "A" # From the final majority vote
    assert mock_dependencies["stream_vllm_request"].call_count == n_chains_max
    mock_dependencies["majority_vote"].assert_called_once()