# tests/test_processing_similarity.py

import asyncio
import json
import logging
import pytest
import numpy as np

# Use a try-except block for imports to guide the user if dependencies are missing
try:
    from unittest.mock import AsyncMock, MagicMock
    from slimsc.prune.evaluation.processing_similarity import (
        calculate_mean_pairwise_similarity,
        stream_processing_worker,
        process_question_similarity_prune
    )
    # Adjust ANALYTICS_INTERVAL for faster tests
    import slimsc.prune.evaluation.processing_similarity as module_under_test
    module_under_test.ANALYSIS_INTERVAL_SECONDS = 0.01

except ImportError as e:
    pytest.fail(f"Failed to import necessary modules. Ensure slimsc is installed correctly. Error: {e}", pytrace=False)


# --- Helper for creating mock streams ---
async def mock_stream_generator(chunks: list):
    """An async generator to simulate a vLLM stream."""
    for chunk in chunks:
        yield chunk
        await asyncio.sleep(0)  # Yield control to the event loop

# --- Tests for calculate_mean_pairwise_similarity ---

@pytest.mark.unit
class TestCalculateMeanPairwiseSimilarity:
    def test_empty_list(self):
        assert calculate_mean_pairwise_similarity([]) == 0.0

    def test_single_embedding(self):
        assert calculate_mean_pairwise_similarity([np.array([1, 0])]) == 0.0

    def test_identical_embeddings(self):
        e = np.array([0.6, 0.8])
        assert calculate_mean_pairwise_similarity([e, e, e]) == pytest.approx(1.0)

    def test_orthogonal_embeddings(self):
        e1 = np.array([1, 0, 0])
        e2 = np.array([0, 1, 0])
        e3 = np.array([0, 0, 1])
        assert calculate_mean_pairwise_similarity([e1, e2, e3]) == pytest.approx(0.0)

    def test_mixed_embeddings(self):
        e1 = np.array([1, 0])
        e2 = np.array([0, 1])
        e3 = np.array([np.sqrt(0.5), np.sqrt(0.5)])
        expected_mean = (0 + np.sqrt(0.5) + np.sqrt(0.5)) / 3
        assert calculate_mean_pairwise_similarity([e1, e2, e3]) == pytest.approx(expected_mean)


# --- Tests for stream_processing_worker ---

@pytest.mark.unit
@pytest.mark.asyncio
class TestStreamProcessingWorker:
    @pytest.fixture
    def initial_chain_state(self):
        return {
            "id": "c1", "full_text": "", "is_active": True,
            "reasoning_complete": False, "finished": False,
            "prompt_tokens": 10, "completion_tokens": 0
        }

    async def test_normal_stream_consumption(self, initial_chain_state):
        stream = [
            {"choices": [{"delta": {"reasoning_content": "Step 1: "}}]},
            {"choices": [{"delta": {"reasoning_content": "Think..."}}]},
            {"choices": [{"delta": {"content": "The answer is A"}}]},
            {"choices": [{"finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 25}}
        ]
        event = asyncio.Event()
        await stream_processing_worker("c1", mock_stream_generator(stream), initial_chain_state, event)
        assert initial_chain_state["full_text"] == "Step 1: Think...The answer is A"
        assert initial_chain_state["reasoning_complete"] is True
        assert initial_chain_state["finished"] is True
        assert initial_chain_state["finish_reason"] == "stop"
        assert initial_chain_state["completion_tokens"] == 25

    async def test_stops_when_made_inactive(self, initial_chain_state):
        """Test that the worker stops if the chain is marked inactive."""
        async def long_running_stream():
            count = 0
            while count < 100: # Finite but long enough for the test
                yield {"choices": [{"delta": {"reasoning_content": "."}}]}
                await asyncio.sleep(0.01)
                count += 1
        
        async def pruner():
            # Give the worker a moment to enter its loop
            await asyncio.sleep(0.02)
            initial_chain_state["is_active"] = False
        
        event = asyncio.Event()
        worker_task = stream_processing_worker("c1", long_running_stream(), initial_chain_state, event)
        
        # Run worker and pruner concurrently, with a timeout to prevent hangs
        await asyncio.wait_for(asyncio.gather(worker_task, pruner()), timeout=1.0)
        
        assert initial_chain_state["finish_reason"] == "cancelled_inactive"
        assert initial_chain_state["finished"] is True

    async def test_handles_error_chunk(self, initial_chain_state):
        stream = [{"error": {"message": "Internal server error"}}]
        event = asyncio.Event()
        await stream_processing_worker("c1", mock_stream_generator(stream), initial_chain_state, event)
        assert "Internal server error" in initial_chain_state["error"]
        assert "error: Internal server error" in initial_chain_state["finish_reason"]

    async def test_handles_task_cancellation(self, initial_chain_state):
        async def infinite_stream():
            while True:
                yield {"choices": [{"delta": {"reasoning_content": "."}}]}
                await asyncio.sleep(0.01)
        event = asyncio.Event()
        task = asyncio.create_task(
            stream_processing_worker("c1", infinite_stream(), initial_chain_state, event)
        )
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert initial_chain_state["finished"] is True
        assert initial_chain_state["finish_reason"] == "cancelled_pruned"


# --- Tests for process_question_similarity_prune ---

@pytest.mark.unit
@pytest.mark.asyncio
class TestProcessQuestionSimilarityPrune:
    @pytest.fixture
    def mock_dependencies(self, mocker, tmp_path):
        mock_ds_handler = mocker.MagicMock()
        mock_ds_handler.create_prompt.return_value = (
            "prompt text",
            (["A", "B", "C"], "A")
        )
        mock_ds_handler.extract_answer.side_effect = lambda text: text[-1] if text else None
        mocker.patch('slimsc.prune.evaluation.processing_similarity.DatasetHandler', return_value=mock_ds_handler)

        mocker.patch('slimsc.prune.evaluation.processing_similarity.count_tokens', return_value=10)
        mocker.patch('slimsc.prune.evaluation.processing_similarity.get_embedding_model')

        mock_embed = mocker.patch('slimsc.prune.evaluation.processing_similarity.embed_segments')
        mock_embed.side_effect = lambda texts: [np.random.rand(4) for _ in texts]

        mock_faiss = mocker.MagicMock()
        mock_faiss.get_num_active_chains.return_value = 2
        mock_faiss.search_nearest_neighbor.return_value = (0.5, "c2", 0, "text")
        mocker.patch('slimsc.prune.evaluation.processing_similarity.FaissIndexManager', return_value=mock_faiss)

        mocker.patch('slimsc.prune.evaluation.processing_similarity.extract_kv_cache_usage_for_question', return_value={'avg_kv_cache_usage': 50.0, 'max_kv_cache_usage': 75.0})
        mocker.patch('slimsc.prune.evaluation.processing_similarity.majority_vote_for_sim_prune', return_value=("winner", "A", 1, ["A", "B"], None, None))
        
        mock_stream_req = mocker.patch('slimsc.prune.evaluation.processing_similarity.stream_vllm_request')

        paths = {"summaries": tmp_path / "summaries", "chains": tmp_path / "chains"}
        paths["summaries"].mkdir()
        paths["chains"].mkdir()

        return {
            "paths": paths, "mock_stream_req": mock_stream_req,
            "mock_faiss": mock_faiss, "mock_ds_handler": mock_ds_handler
        }

    # No changes needed in the test functions themselves. The fixture fix covers them.
    async def test_happy_path_no_pruning(self, mock_dependencies, caplog, mocker):
        stream1 = [
            {"choices": [{"delta": {"reasoning_content": "Chain 1 thought 1"}}]},
            {"choices": [{"delta": {"content": "A"}}]},
            {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 15}}
        ]
        stream2 = [
            {"choices": [{"delta": {"reasoning_content": "Chain 2 thought 1"}}]},
            {"choices": [{"delta": {"content": "B"}}]},
            {"choices": [{"finish_reason": "stop"}], "usage": {"completion_tokens": 18}}
        ]
        mock_dependencies["mock_stream_req"].side_effect = [
            mock_stream_generator(stream1), mock_stream_generator(stream2)
        ]
        mocker.patch('slimsc.prune.evaluation.processing_similarity.find_newly_completed_thoughts', return_value=([], []))
        result = await process_question_similarity_prune(
            example={"id": "q1"}, iteration=1, n_chains_start=2, paths=mock_dependencies["paths"],
            vllm_url="", model_name="", tokenizer_path="", similarity_threshold=0.95,
            pruning_strategy="fewest_thoughts", dataset_name="gpqa_diamond", num_steps_to_delay_pruning=0,
            max_analysis_steps=5
        )
        assert result is not None
        assert result["n_chains_pruned"] == 0

    async def test_prompt_creation_error(self, mock_dependencies):
        mock_dependencies["mock_ds_handler"].create_prompt.side_effect = ValueError("Bad prompt data")
        result = await process_question_similarity_prune(
            example={"id": "q1"}, iteration=1, n_chains_start=2, paths=mock_dependencies["paths"],
            vllm_url="", model_name="", tokenizer_path="", similarity_threshold=0.9,
            pruning_strategy="random", dataset_name="gpqa_diamond", num_steps_to_delay_pruning=0
        )
        assert result is None

    async def test_prompt_creation_error(self, mock_dependencies):
        mock_dependencies["mock_ds_handler"].create_prompt.side_effect = ValueError("Bad prompt data")
        result = await process_question_similarity_prune(
            example={"id": "q1"}, iteration=1, n_chains_start=2, paths=mock_dependencies["paths"],
            vllm_url="", model_name="", tokenizer_path="", similarity_threshold=0.9,
            pruning_strategy="random", dataset_name="gpqa_diamond", num_steps_to_delay_pruning=0
        )
        assert result is None
        summary_file = mock_dependencies["paths"]["summaries"] / "question_1_summary.json"
        assert summary_file.exists()
        with open(summary_file) as f:
            summary_data = json.load(f)
        assert summary_data["status"] == "PROMPT_ERROR"