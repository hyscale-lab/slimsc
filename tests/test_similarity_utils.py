import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Import the module and its functions/classes to be tested
from slimsc.prune.utils.similarity_utils import (
    get_embedding_model,
    embed_segments,
    find_thought_boundaries,
    find_newly_completed_thoughts,
    extract_final_thought,
    FaissIndexManager,
    TARGET_PHRASES,
)
# Import the module itself to reset its global variables
import slimsc.prune.utils.similarity_utils as sim_utils_module


# --- Fixtures and Setup ---

@pytest.fixture(autouse=True)
def reset_globals():
    """Fixture to reset global model caches before each test."""
    sim_utils_module._embedding_model = None
    yield
    sim_utils_module._embedding_model = None

@pytest.fixture
def mock_embedding_model(mocker):
    """Fixture to mock the SentenceTransformer model."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    mock_model.get_sentence_embedding_dimension.return_value = 3
    mocker.patch("slimsc.prune.utils.similarity_utils.SentenceTransformer", return_value=mock_model)
    return mock_model


# --- Tests for Model Loading and Embedding ---

def test_get_embedding_model_caching(mock_embedding_model, mocker):
    st_constructor = sim_utils_module.SentenceTransformer
    get_embedding_model()
    st_constructor.assert_called_once()
    get_embedding_model()
    st_constructor.assert_called_once()

def test_embed_segments_happy_path(mock_embedding_model):
    segments = ["This is a test segment."]
    embeddings = embed_segments(segments)
    mock_embedding_model.encode.assert_called_once()
    assert isinstance(embeddings, np.ndarray)

def test_embed_segments_empty_list(mock_embedding_model):
    embeddings = embed_segments([])
    assert embeddings.shape == (0, 3)
    mock_embedding_model.encode.assert_not_called()


# --- Tests for Thought Boundary Detection ---

@pytest.mark.parametrize("text, expected_boundaries", [
    ("First thought. Alternative second thought.", [0, 15]),
    ("Hello. Another idea. Oh wait, a third idea.", [0, 7, 21]),
    ("Wait, a thought at the beginning.", [0]),
    ("No boundaries here.", [0]),
    ("This is an alternative, but not a start.", [0, 11]),
    ("Some text... But wait, a new thought.", [0, 13]),
    ("", [0]),
])
def test_find_thought_boundaries(text, expected_boundaries):
    """Tests the pure logic of finding thought boundaries."""
    boundaries = find_thought_boundaries(text, TARGET_PHRASES)
    assert boundaries == expected_boundaries


# --- Tests for Segment Extraction ---

@patch("slimsc.prune.utils.similarity_utils.count_tokens")
def test_find_newly_completed_thoughts(mock_count_tokens):
    """Tests identifying new, valid thought segments."""
    mock_count_tokens.return_value = 30
    
    text = "Initial thought. Another thought is here. Oh wait, a final thought."
    processed_boundaries = [0]
    
    new_segments, updated_boundaries = find_newly_completed_thoughts(
        text, processed_boundaries, "fake/tokenizer", TARGET_PHRASES
    )

    assert len(new_segments) == 1
    start, end, segment_text = new_segments[0]
    assert start == 17
    assert end == 42 # "Oh wait" starts at 42
    assert segment_text == "Another thought is here."
    assert updated_boundaries == [0, 17]


@patch("slimsc.prune.utils.similarity_utils.count_tokens")
def test_find_newly_completed_thoughts_too_short(mock_count_tokens):
    mock_count_tokens.return_value = 10
    text = "First part. Alternative short part."
    new_segments, _ = find_newly_completed_thoughts(text, [], "fake/tokenizer", TARGET_PHRASES)
    assert len(new_segments) == 0


@patch("slimsc.prune.utils.similarity_utils.count_tokens")
def test_extract_final_thought(mock_count_tokens):
    """Tests extracting the final segment of text."""
    mock_count_tokens.return_value = 50
    
    text = "Thought one. Another thought two. But wait, this is the final thought."
    
    final_segment = extract_final_thought(text, [0, 13, 34], "fake/tokenizer")
    
    assert final_segment is not None
    start, end, text_content = final_segment
    
    # "But wait" starts at index 34
    assert start == 34
    assert end == len(text)
    
    assert text_content == "But wait, this is the final thought."


# --- Tests for FaissIndexManager ---

@pytest.fixture
def mock_faiss(mocker):
    """Mocks the faiss library to test FaissIndexManager in isolation."""
    mock_index_instance = MagicMock()
    # Configure the mock's ntotal attribute to be an integer that we can control
    mock_index_instance.ntotal = 0

    # Mock the search method to return a standard shape
    mock_index_instance.search.return_value = (np.array([[]]), np.array([[]]))
    
    # Patch the faiss.IndexFlatIP constructor to return our mock instance
    mocker.patch('faiss.IndexFlatIP', return_value=mock_index_instance)
    return mock_index_instance


class TestFaissIndexManager:
    def test_add_embedding(self, mock_faiss):
        """Test adding a valid embedding."""
        manager = FaissIndexManager(dimension=4, search_mode='similarity')
        embedding = np.random.rand(4).astype(np.float32)

        # Simulate the behavior of faiss.add
        def add_side_effect(*args):
            mock_faiss.ntotal += 1
        mock_faiss.add.side_effect = add_side_effect

        manager.add_embedding(embedding, "chain1", 0, "text segment")
        
        mock_faiss.add.assert_called_once()
        assert manager.get_num_embeddings() == 1

    def test_search_nearest_neighbor(self, mock_faiss):
        """Test searching for the nearest neighbor, excluding the query's own chain."""
        manager = FaissIndexManager(dimension=4, search_mode='similarity')
        
        # Manually populate metadata since we're not calling the real add
        manager.metadata_map = {
            0: ("chain1", 0, "text4"),
            1: ("chain2", 0, "text2"), # This should be the winner
            2: ("chain3", 0, "text3"),
        }
        mock_faiss.ntotal = 3

        # Mock the search result: distances and indices
        # Say query is closest to index 0, then 1, then 2
        mock_faiss.search.return_value = (
            np.array([[0.98, 0.95, 0.5]], dtype=np.float32), # Distances
            np.array([[0, 1, 2]], dtype=np.int64)          # Indices
        )
        
        query_emb = np.random.rand(4)
        result = manager.search_nearest_neighbor(query_emb, "chain1")
        
        assert result is not None
        score, chain_id, _, _ = result
        # The first result (index 0) is from 'chain1', so it should be skipped.
        # The second result (index 1) is from 'chain2', which is valid.
        assert chain_id == "chain2"
        assert score == pytest.approx(0.95)

    def test_remove_chain_embeddings(self, mock_faiss, mocker):
        """Test removing all embeddings associated with a chain."""
        mocker.patch('faiss.IDSelectorBatch') # Mock the selector helper class
        manager = FaissIndexManager(dimension=4, search_mode='similarity')
        manager.metadata_map = {
            0: ("chain1", 0, "text1"),
            1: ("chain2", 0, "text2"),
            2: ("chain1", 1, "text3"),
        }
        mock_faiss.ntotal = 3

        # Simulate the behavior of faiss.remove_ids
        def remove_side_effect(*args):
            mock_faiss.ntotal -= 2
            return 2
        mock_faiss.remove_ids.side_effect = remove_side_effect

        manager.remove_chain_embeddings("chain1")
        
        mock_faiss.remove_ids.assert_called_once()
        assert manager.get_num_embeddings() == 1
        assert list(manager.metadata_map.keys()) == [1]