"""Tests for prune.utils.similarity_utils module."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Dict, Optional, Tuple

from prune.utils.similarity_utils import (
    get_embedding_model,
    embed_segments,
    find_thought_boundaries,
    find_newly_completed_thoughts,
    extract_final_thought,
    FaissIndexManager,
    TARGET_PHRASES,
    EMBEDDING_MODEL_NAME,
    MIN_SEGMENT_TOKENS
)


class TestGetEmbeddingModel:
    """Test the get_embedding_model function."""
    
    @patch('prune.utils.similarity_utils.SentenceTransformer')
    def test_get_embedding_model_loads_successfully(self, mock_sentence_transformer):
        """Test successful loading of embedding model."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Reset global cache
        import prune.utils.similarity_utils
        prune.utils.similarity_utils._embedding_model = None
        
        result = get_embedding_model()
        
        assert result == mock_model
        mock_sentence_transformer.assert_called_once_with(EMBEDDING_MODEL_NAME, device='cpu')
    
    @patch('prune.utils.similarity_utils.SentenceTransformer')
    def test_get_embedding_model_returns_cached(self, mock_sentence_transformer):
        """Test that subsequent calls return cached model."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        # Reset and set up cache
        import prune.utils.similarity_utils
        prune.utils.similarity_utils._embedding_model = mock_model
        
        result = get_embedding_model()
        
        assert result == mock_model
        mock_sentence_transformer.assert_not_called()  # Should use cache
    
    @patch('prune.utils.similarity_utils.SentenceTransformer')
    def test_get_embedding_model_raises_on_failure(self, mock_sentence_transformer):
        """Test handling of model loading failure."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        # Reset global cache
        import prune.utils.similarity_utils
        prune.utils.similarity_utils._embedding_model = None
        
        with pytest.raises(Exception, match="Model loading failed"):
            get_embedding_model()


class TestEmbedSegments:
    """Test the embed_segments function."""
    
    @patch('prune.utils.similarity_utils.get_embedding_model')
    def test_embed_segments_empty_list(self, mock_get_model):
        """Test embedding empty segment list."""
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model
        
        result = embed_segments([])
        
        assert result is not None
        assert result.shape == (0, 384)
        assert result.dtype == np.float32
    
    @patch('prune.utils.similarity_utils.get_embedding_model')
    def test_embed_segments_successful(self, mock_get_model):
        """Test successful segment embedding."""
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
        mock_model.encode.return_value = mock_embeddings
        mock_get_model.return_value = mock_model
        
        segments = ["Hello world", "Another segment"]
        result = embed_segments(segments)
        
        assert result is not None
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, mock_embeddings.astype(np.float32))
        mock_model.encode.assert_called_once_with(
            segments, convert_to_numpy=True, normalize_embeddings=True
        )
    
    @patch('prune.utils.similarity_utils.get_embedding_model')
    def test_embed_segments_handles_exception(self, mock_get_model):
        """Test handling of embedding failure."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_get_model.return_value = mock_model
        
        result = embed_segments(["test segment"])
        
        assert result is None
    
    @patch('prune.utils.similarity_utils.get_embedding_model')
    def test_embed_segments_model_loading_failure(self, mock_get_model):
        """Test handling when model loading fails."""
        mock_get_model.side_effect = Exception("Model loading failed")
        
        result = embed_segments(["test segment"])
        
        assert result is None


class TestFindThoughtBoundaries:
    """Test the find_thought_boundaries function."""
    
    def test_find_thought_boundaries_empty_text(self):
        """Test with empty text."""
        result = find_thought_boundaries("", TARGET_PHRASES)
        assert result == [0]
    
    def test_find_thought_boundaries_no_phrases(self):
        """Test with text containing no target phrases."""
        text = "This is a simple text without any markers."
        result = find_thought_boundaries(text, TARGET_PHRASES)
        assert result == [0]
    
    def test_find_thought_boundaries_single_phrase(self):
        """Test with text containing a single target phrase."""
        text = "First thought. Wait, let me reconsider this approach."
        result = find_thought_boundaries(text, TARGET_PHRASES)
        assert result == [0, 15]  # "Wait" starts at position 15
    
    def test_find_thought_boundaries_multiple_phrases(self):
        """Test with text containing multiple target phrases."""
        text = "First thought. Alternative approach here. But another way to think about it."
        result = find_thought_boundaries(text, ["Alternative", "But another"])
        assert result == [0, 15, 42]
    
    def test_find_thought_boundaries_phrase_at_start(self):
        """Test with target phrase at the very start."""
        text = "Wait, this is wrong from the beginning."
        result = find_thought_boundaries(text, ["Wait"])
        assert result == [0]  # Should not add duplicate 0
    
    def test_find_thought_boundaries_context_check(self):
        """Test context checking for phrase boundaries."""
        # Should find "Wait" when preceded by whitespace
        text = "Some text. Wait, reconsider."
        result = find_thought_boundaries(text, ["Wait"])
        assert 11 in result  # "Wait" after space
        
        # Should not find "Wait" when it's part of another word
        text = "Don't await the results."
        result = find_thought_boundaries(text, ["Wait"])
        assert result == [0]  # Should only have initial boundary


class TestFindNewlyCompletedThoughts:
    """Test the find_newly_completed_thoughts function."""
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_find_newly_completed_thoughts_empty_text(self, mock_boundaries, mock_count_tokens):
        """Test with empty text."""
        mock_boundaries.return_value = [0]
        
        result, updated_boundaries = find_newly_completed_thoughts(
            "", [], "/path/to/tokenizer", TARGET_PHRASES
        )
        
        assert result == []
        assert updated_boundaries == []
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_find_newly_completed_thoughts_new_segment(self, mock_boundaries, mock_count_tokens):
        """Test finding a new completed thought segment."""
        text = "First thought with sufficient length. Wait, here's another complete thought."
        mock_boundaries.return_value = [0, 38]  # Two boundaries
        mock_count_tokens.return_value = 30  # Above minimum
        
        result, updated_boundaries = find_newly_completed_thoughts(
            text, [], "/path/to/tokenizer", TARGET_PHRASES, min_segment_tokens=25
        )
        
        assert len(result) == 1
        start_idx, end_idx, segment_text = result[0]
        assert start_idx == 0
        assert end_idx == 38
        assert segment_text == "First thought with sufficient length."
        assert updated_boundaries == [0]
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_find_newly_completed_thoughts_insufficient_tokens(self, mock_boundaries, mock_count_tokens):
        """Test skipping segments with insufficient tokens."""
        text = "Short. Wait, another short."
        mock_boundaries.return_value = [0, 7]
        mock_count_tokens.return_value = 15  # Below minimum
        
        result, updated_boundaries = find_newly_completed_thoughts(
            text, [], "/path/to/tokenizer", TARGET_PHRASES, min_segment_tokens=25
        )
        
        assert result == []
        assert updated_boundaries == []
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_find_newly_completed_thoughts_tokenization_failure(self, mock_boundaries, mock_count_tokens):
        """Test handling tokenization failure."""
        text = "Some text. Wait, more text."
        mock_boundaries.return_value = [0, 11]
        mock_count_tokens.return_value = None  # Tokenization failed
        
        result, updated_boundaries = find_newly_completed_thoughts(
            text, [], "/path/to/tokenizer", TARGET_PHRASES
        )
        
        assert result == []
        assert updated_boundaries == []


class TestExtractFinalThought:
    """Test the extract_final_thought function."""
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_extract_final_thought_empty_text(self, mock_boundaries, mock_count_tokens):
        """Test with empty text."""
        mock_boundaries.return_value = [0]
        
        result = extract_final_thought("", [], "/path/to/tokenizer")
        
        assert result is None
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_extract_final_thought_successful(self, mock_boundaries, mock_count_tokens):
        """Test successful final thought extraction."""
        text = "First thought. Wait, final thought with enough content."
        mock_boundaries.return_value = [0, 15]
        mock_count_tokens.return_value = 30  # Above minimum
        
        result = extract_final_thought(text, [0], "/path/to/tokenizer")
        
        assert result is not None
        start_idx, end_idx, segment_text = result
        assert start_idx == 15
        assert end_idx == len(text)
        assert segment_text == "final thought with enough content."
    
    @patch('prune.utils.similarity_utils.count_tokens')
    @patch('prune.utils.similarity_utils.find_thought_boundaries')
    def test_extract_final_thought_insufficient_tokens(self, mock_boundaries, mock_count_tokens):
        """Test final thought with insufficient tokens."""
        text = "First thought. Wait, short."
        mock_boundaries.return_value = [0, 15]
        mock_count_tokens.return_value = 15  # Below minimum
        
        result = extract_final_thought(text, [0], "/path/to/tokenizer", min_segment_tokens=25)
        
        assert result is None


class TestFaissIndexManager:
    """Test the FaissIndexManager class."""
    
    def test_faiss_index_manager_initialization_similarity(self):
        """Test initialization in similarity mode."""
        manager = FaissIndexManager(dimension=384, search_mode='similarity')
        
        assert manager.original_dimension == 384
        assert manager.index_dimension == 384
        assert manager.search_mode == 'similarity'
        assert manager.get_num_embeddings() == 0
    
    def test_faiss_index_manager_initialization_dissimilarity(self):
        """Test initialization in dissimilarity mode."""
        manager = FaissIndexManager(dimension=384, search_mode='dissimilarity')
        
        assert manager.original_dimension == 384
        assert manager.index_dimension == 385  # dimension + 1
        assert manager.search_mode == 'dissimilarity'
        assert manager.get_num_embeddings() == 0
    
    def test_add_embedding_similarity_mode(self):
        """Test adding embedding in similarity mode."""
        manager = FaissIndexManager(dimension=3, search_mode='similarity')
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        manager.add_embedding(embedding, "chain_1", 0, "test segment")
        
        assert manager.get_num_embeddings() == 1
        assert manager.get_num_active_chains() == 1
        assert 0 in manager.metadata_map
        assert manager.metadata_map[0] == ("chain_1", 0, "test segment")
    
    def test_add_embedding_dissimilarity_mode(self):
        """Test adding embedding in dissimilarity mode."""
        manager = FaissIndexManager(dimension=3, search_mode='dissimilarity')
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        manager.add_embedding(embedding, "chain_1", 0, "test segment")
        
        assert manager.get_num_embeddings() == 1
        assert manager.get_num_active_chains() == 1
    
    def test_add_embedding_dimension_mismatch(self):
        """Test handling dimension mismatch."""
        manager = FaissIndexManager(dimension=3, search_mode='similarity')
        embedding = np.array([0.1, 0.2], dtype=np.float32)  # Wrong dimension
        
        manager.add_embedding(embedding, "chain_1", 0, "test segment")
        
        assert manager.get_num_embeddings() == 0  # Should not add
    
    def test_search_nearest_neighbor_empty_index(self):
        """Test nearest neighbor search on empty index."""
        manager = FaissIndexManager(dimension=3, search_mode='similarity')
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        result = manager.search_nearest_neighbor(embedding, "query_chain")
        
        assert result is None
    
    def test_search_nearest_neighbor_wrong_mode(self):
        """Test nearest neighbor search in wrong mode."""
        manager = FaissIndexManager(dimension=3, search_mode='dissimilarity')
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        result = manager.search_nearest_neighbor(embedding, "query_chain")
        
        assert result is None
    
    def test_search_farthest_neighbor_wrong_mode(self):
        """Test farthest neighbor search in wrong mode."""
        manager = FaissIndexManager(dimension=3, search_mode='similarity')
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        result = manager.search_farthest_neighbor(embedding, "query_chain")
        
        assert result is None
    
    def test_remove_chain_embeddings(self):
        """Test removing embeddings for a specific chain."""
        manager = FaissIndexManager(dimension=3, search_mode='similarity')
        
        # Add embeddings for different chains
        embedding1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        embedding2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
        
        manager.add_embedding(embedding1, "chain_1", 0, "segment 1")
        manager.add_embedding(embedding2, "chain_2", 0, "segment 2")
        
        assert manager.get_num_embeddings() == 2
        assert manager.get_num_active_chains() == 2
        
        # Remove chain_1
        manager.remove_chain_embeddings("chain_1")
        
        assert manager.get_num_active_chains() == 1
        # Check that only chain_2 metadata remains
        remaining_chains = set(meta[0] for meta in manager.metadata_map.values())
        assert remaining_chains == {"chain_2"}
    
    def test_augment_methods(self):
        """Test augmentation methods for dissimilarity mode."""
        manager = FaissIndexManager(dimension=2, search_mode='dissimilarity')
        
        # Test database augmentation
        xb = np.array([[1.0, 2.0]], dtype=np.float32)
        augmented_db = manager._augment_for_database(xb)
        expected_db = np.array([[-2.0, -4.0, 5.0]], dtype=np.float32)  # [-2x, ||x||^2]
        np.testing.assert_array_almost_equal(augmented_db, expected_db)
        
        # Test query augmentation
        xq = np.array([[1.0, 2.0]], dtype=np.float32)
        augmented_q = manager._augment_for_query(xq)
        expected_q = np.array([[1.0, 2.0, 1.0]], dtype=np.float32)  # [q, 1]
        np.testing.assert_array_almost_equal(augmented_q, expected_q)


class TestIntegration:
    """Integration tests for similarity utilities."""
    
    @patch('prune.utils.similarity_utils.get_embedding_model')
    @patch('prune.utils.similarity_utils.count_tokens')
    def test_complete_workflow(self, mock_count_tokens, mock_get_model):
        """Test a complete workflow from text to embeddings."""
        # Setup mocks
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        mock_get_model.return_value = mock_model
        mock_count_tokens.return_value = 30
        
        # Simulate finding and embedding thought segments
        text = "First thought segment. Wait, here's another approach to consider."
        boundaries = find_thought_boundaries(text, ["Wait"])
        
        assert boundaries == [0, 22]
        
        segments, _ = find_newly_completed_thoughts(
            text, [], "/path/to/tokenizer", ["Wait"], min_segment_tokens=25
        )
        
        assert len(segments) == 1
        segment_text = segments[0][2]
        
        # Test embedding the segment
        embeddings = embed_segments([segment_text])
        assert embeddings is not None
        assert embeddings.shape[0] == 1