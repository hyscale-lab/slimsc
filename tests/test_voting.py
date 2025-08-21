"""Tests for prune.evaluation.voting module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import Counter

from prune.evaluation.voting import (
    majority_vote,
    majority_vote_for_sim_prune,
    fallback_tie_break_logic,
    _process_initial_vote,
    _get_best_chain_by_key,
    _tie_break_by_tokens,
    _tie_break_by_pruned_count
)


class TestProcessInitialVote:
    """Test the _process_initial_vote helper function."""
    
    def test_process_initial_vote_empty_results(self):
        """Test with empty chain results."""
        status, voted_answer, all_answers, valid_chains, tied_answers = _process_initial_vote([])
        
        assert status == "empty"
        assert voted_answer is None
        assert all_answers == []
        assert valid_chains == []
        assert tied_answers is None
    
    def test_process_initial_vote_no_valid_answers(self):
        """Test with chain results but no valid extracted answers."""
        chain_results = [
            {"chain_index": 0, "extracted_answer": None},
            {"chain_index": 1, "extracted_answer": None}
        ]
        
        status, voted_answer, all_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)
        
        assert status == "empty"
        assert voted_answer is None
        assert all_answers == []
        assert valid_chains == []
        assert tied_answers is None
    
    def test_process_initial_vote_clear_winner(self):
        """Test with clear majority winner."""
        chain_results = [
            {"chain_index": 0, "extracted_answer": "A"},
            {"chain_index": 1, "extracted_answer": "A"},
            {"chain_index": 2, "extracted_answer": "B"}
        ]
        
        status, voted_answer, all_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)
        
        assert status == "winner"
        assert voted_answer == "A"
        assert all_answers == ["A", "A", "B"]
        assert len(valid_chains) == 3
        assert tied_answers is None
    
    def test_process_initial_vote_tie(self):
        """Test with tied answers."""
        chain_results = [
            {"chain_index": 0, "extracted_answer": "A"},
            {"chain_index": 1, "extracted_answer": "B"},
            {"chain_index": 2, "extracted_answer": "A"},
            {"chain_index": 3, "extracted_answer": "B"}
        ]
        
        status, voted_answer, all_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)
        
        assert status == "tie"
        assert voted_answer is None
        assert all_answers == ["A", "B", "A", "B"]
        assert len(valid_chains) == 4
        assert set(tied_answers) == {"A", "B"}
    
    def test_process_initial_vote_single_answer(self):
        """Test with single answer (automatic winner)."""
        chain_results = [
            {"chain_index": 0, "extracted_answer": "A"}
        ]
        
        status, voted_answer, all_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)
        
        assert status == "winner"
        assert voted_answer == "A"
        assert all_answers == ["A"]
        assert len(valid_chains) == 1
        assert tied_answers is None


class TestGetBestChainByKey:
    """Test the _get_best_chain_by_key helper function."""
    
    def test_get_best_chain_by_key_minimize(self):
        """Test finding minimum value."""
        chains = [
            {"chain_index": 0, "tokens": 100},
            {"chain_index": 1, "tokens": 50},
            {"chain_index": 2, "tokens": 75}
        ]
        
        result = _get_best_chain_by_key(chains, "tokens", minimize=True)
        
        assert result["chain_index"] == 1
        assert result["tokens"] == 50
    
    def test_get_best_chain_by_key_maximize(self):
        """Test finding maximum value."""
        chains = [
            {"chain_index": 0, "score": 0.7},
            {"chain_index": 1, "score": 0.9},
            {"chain_index": 2, "score": 0.8}
        ]
        
        result = _get_best_chain_by_key(chains, "score", minimize=False)
        
        assert result["chain_index"] == 1
        assert result["score"] == 0.9
    
    def test_get_best_chain_by_key_tie_break_by_index(self):
        """Test tie breaking by chain index."""
        chains = [
            {"chain_index": 2, "tokens": 50},
            {"chain_index": 0, "tokens": 50},
            {"chain_index": 1, "tokens": 50}
        ]
        
        result = _get_best_chain_by_key(chains, "tokens", minimize=True, lower_index_better=True)
        
        assert result["chain_index"] == 0  # Lowest index wins
    
    def test_get_best_chain_by_key_missing_key(self):
        """Test with chains missing the key."""
        chains = [
            {"chain_index": 0},  # Missing "tokens"
            {"chain_index": 1, "tokens": 50},
            {"chain_index": 2}   # Missing "tokens"
        ]
        
        result = _get_best_chain_by_key(chains, "tokens", minimize=True)
        
        assert result["chain_index"] == 1  # Only one with the key
    
    def test_get_best_chain_by_key_all_missing_key(self):
        """Test when all chains are missing the key."""
        chains = [
            {"chain_index": 0},
            {"chain_index": 1}
        ]
        
        result = _get_best_chain_by_key(chains, "tokens", minimize=True)
        
        assert result is None


class TestTieBreakByTokens:
    """Test the _tie_break_by_tokens function."""
    
    def test_tie_break_by_tokens_usage_stats(self):
        """Test tie breaking using completion_tokens."""
        valid_chains = [
            {"chain_index": 0, "extracted_answer": "A", "completion_tokens": 100},
            {"chain_index": 1, "extracted_answer": "A", "completion_tokens": 50},
            {"chain_index": 2, "extracted_answer": "B", "completion_tokens": 75}
        ]
        tied_answers = ["A", "B"]
        
        result = _tie_break_by_tokens(valid_chains, tied_answers, None)
        
        assert result == "A"  # Chain 1 has fewest completion_tokens (50) among tied answers
    
    @patch('prune.evaluation.voting.count_tokens')
    def test_tie_break_by_tokens_tokenizer_fallback(self, mock_count_tokens):
        """Test tie breaking using tokenizer fallback."""
        valid_chains = [
            {"chain_index": 0, "extracted_answer": "A", "full_content": "Short answer"},
            {"chain_index": 1, "extracted_answer": "A", "full_content": "Much longer answer with more content"},
            {"chain_index": 2, "extracted_answer": "B", "full_content": "Medium length answer"}
        ]
        tied_answers = ["A", "B"]
        
        # Mock token counts
        mock_count_tokens.side_effect = [10, 30, 20]  # Corresponding to the chains
        
        result = _tie_break_by_tokens(valid_chains, tied_answers, "/path/to/tokenizer")
        
        assert result == "A"  # Chain 0 has fewest tokens (10) among A answers
    
    @patch('prune.evaluation.voting.count_tokens')
    def test_tie_break_by_tokens_index_fallback(self, mock_count_tokens):
        """Test falling back to chain index."""
        valid_chains = [
            {"chain_index": 2, "extracted_answer": "A"},
            {"chain_index": 0, "extracted_answer": "A"},
            {"chain_index": 1, "extracted_answer": "B"}
        ]
        tied_answers = ["A", "B"]
        
        # No usage stats, no tokenizer path
        result = _tie_break_by_tokens(valid_chains, tied_answers, None)
        
        assert result == "A"  # Chain 0 has lowest index among A answers
    
    @patch('prune.evaluation.voting.count_tokens')
    def test_tie_break_by_tokens_tokenization_failure(self, mock_count_tokens):
        """Test handling tokenization failure."""
        valid_chains = [
            {"chain_index": 0, "extracted_answer": "A", "full_content": "Content"},
            {"chain_index": 1, "extracted_answer": "B", "full_content": "Content"}
        ]
        tied_answers = ["A", "B"]
        
        # Tokenization fails
        mock_count_tokens.return_value = None
        
        result = _tie_break_by_tokens(valid_chains, tied_answers, "/path/to/tokenizer")
        
        assert result == "A"  # Falls back to index, chain 0 wins


class TestTieBreakByPrunedCount:
    """Test the _tie_break_by_pruned_count function."""
    
    def test_tie_break_by_pruned_count_highest_wins(self):
        """Test tie breaking by highest pruned count."""
        valid_chains = [
            {"chain_index": 0, "extracted_answer": "A", "pruned_count": 5},
            {"chain_index": 1, "extracted_answer": "A", "pruned_count": 10},
            {"chain_index": 2, "extracted_answer": "B", "pruned_count": 3}
        ]
        tied_answers = ["A", "B"]
        
        result = _tie_break_by_pruned_count(valid_chains, tied_answers)
        
        assert result == "A"  # Chain 1 has highest pruned_count (10)
    
    def test_tie_break_by_pruned_count_similarity_fallback(self):
        """Test fallback to internal similarity."""
        valid_chains = [
            {"chain_index": 0, "extracted_answer": "A", "pruned_count": 5, "final_internal_similarity": 0.8},
            {"chain_index": 1, "extracted_answer": "A", "pruned_count": 5, "final_internal_similarity": 0.6},
            {"chain_index": 2, "extracted_answer": "B", "pruned_count": 5, "final_internal_similarity": 0.7}
        ]
        tied_answers = ["A", "B"]
        
        result = _tie_break_by_pruned_count(valid_chains, tied_answers)
        
        assert result == "A"  # Chain 1 has lowest similarity (0.6) among tied pruned_counts
    
    def test_tie_break_by_pruned_count_missing_keys(self):
        """Test handling missing keys."""
        valid_chains = [
            {"chain_index": 0, "extracted_answer": "A"},  # Missing both keys
            {"chain_index": 1, "extracted_answer": "B"}   # Missing both keys
        ]
        tied_answers = ["A", "B"]
        
        result = _tie_break_by_pruned_count(valid_chains, tied_answers)
        
        assert result == "A"  # Falls back to first tied answer


class TestMajorityVote:
    """Test the majority_vote function."""
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_majority_vote_clear_winner(self, mock_dataset_handler):
        """Test majority vote with clear winner."""
        mock_handler = Mock()
        mock_handler.calculate_score.return_value = 1
        mock_dataset_handler.return_value = mock_handler
        
        chain_results = [
            {"extracted_answer": "A"},
            {"extracted_answer": "A"},
            {"extracted_answer": "B"}
        ]
        
        voted_answer, score, all_answers = majority_vote(
            chain_results, "A", "gpqa_diamond"
        )
        
        assert voted_answer == "A"
        assert score == 1
        assert all_answers == ["A", "A", "B"]
        mock_handler.calculate_score.assert_called_once_with("A", "A")
    
    @patch('prune.evaluation.voting.DatasetHandler')
    @patch('prune.evaluation.voting.random.choice')
    def test_majority_vote_tie_random_choice(self, mock_random_choice, mock_dataset_handler):
        """Test majority vote with tie resolved randomly."""
        mock_handler = Mock()
        mock_handler.calculate_score.return_value = 0
        mock_dataset_handler.return_value = mock_handler
        mock_random_choice.return_value = "B"
        
        chain_results = [
            {"extracted_answer": "A"},
            {"extracted_answer": "B"}
        ]
        
        voted_answer, score, all_answers = majority_vote(
            chain_results, "A", "gpqa_diamond"
        )
        
        assert voted_answer == "B"
        assert score == 0
        assert all_answers == ["A", "B"]
        mock_random_choice.assert_called_once_with(["A", "B"])
        mock_handler.calculate_score.assert_called_once_with("B", "A")
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_majority_vote_empty_results(self, mock_dataset_handler):
        """Test majority vote with empty results."""
        voted_answer, score, all_answers = majority_vote([], "A", "gpqa_diamond")
        
        assert voted_answer is None
        assert score == 0
        assert all_answers == []


class TestMajorityVoteForSimPrune:
    """Test the majority_vote_for_sim_prune function."""
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_majority_vote_for_sim_prune_winner(self, mock_dataset_handler):
        """Test with clear winner."""
        mock_handler = Mock()
        mock_handler.calculate_score.return_value = 1
        mock_dataset_handler.return_value = mock_handler
        
        chain_results = [
            {"extracted_answer": "A", "pruned_count": 5},
            {"extracted_answer": "A", "pruned_count": 3},
            {"extracted_answer": "B", "pruned_count": 1}
        ]
        
        status, voted_answer, score, all_answers, chains_for_tie, tied_answers = majority_vote_for_sim_prune(
            chain_results, "A", "gpqa_diamond"
        )
        
        assert status == "winner"
        assert voted_answer == "A"
        assert score == 1
        assert all_answers == ["A", "A", "B"]
        assert chains_for_tie == []
        assert tied_answers is None
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_majority_vote_for_sim_prune_tie(self, mock_dataset_handler):
        """Test with tie requiring LLM tie-breaking."""
        mock_handler = Mock()
        mock_dataset_handler.return_value = mock_handler
        
        chain_results = [
            {"extracted_answer": "A", "chain_index": 0},
            {"extracted_answer": "B", "chain_index": 1}
        ]
        
        status, voted_answer, score, all_answers, chains_for_tie, tied_answers = majority_vote_for_sim_prune(
            chain_results, "A", "gpqa_diamond"
        )
        
        assert status == "REQUIRES_LLM_TIEBREAK"
        assert voted_answer is None
        assert score == 0
        assert all_answers == ["A", "B"]
        assert len(chains_for_tie) == 2
        assert set(tied_answers) == {"A", "B"}
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_majority_vote_for_sim_prune_empty(self, mock_dataset_handler):
        """Test with empty results."""
        status, voted_answer, score, all_answers, chains_for_tie, tied_answers = majority_vote_for_sim_prune(
            [], "A", "gpqa_diamond"
        )
        
        assert status == "empty"
        assert voted_answer is None
        assert score == 0
        assert all_answers == []
        assert chains_for_tie == []
        assert tied_answers is None


class TestFallbackTieBreakLogic:
    """Test the fallback_tie_break_logic function."""
    
    @patch('prune.evaluation.voting.DatasetHandler')
    @patch('prune.evaluation.voting._tie_break_by_tokens')
    def test_fallback_tie_break_logic_success(self, mock_tie_break, mock_dataset_handler):
        """Test successful fallback tie breaking."""
        mock_handler = Mock()
        mock_handler.calculate_score.return_value = 1
        mock_dataset_handler.return_value = mock_handler
        mock_tie_break.return_value = "A"
        
        chains_in_tie = [
            {"extracted_answer": "A", "completion_tokens": 50},
            {"extracted_answer": "B", "completion_tokens": 100}
        ]
        tied_answers = ["A", "B"]
        
        voted_answer, score = fallback_tie_break_logic(
            chains_in_tie, tied_answers, "A", "gpqa_diamond", "/path/to/tokenizer"
        )
        
        assert voted_answer == "A"
        assert score == 1
        mock_tie_break.assert_called_once_with(chains_in_tie, tied_answers, "/path/to/tokenizer")
        mock_handler.calculate_score.assert_called_once_with("A", "A")
    
    @patch('prune.evaluation.voting.DatasetHandler')
    @patch('prune.evaluation.voting._tie_break_by_tokens')
    def test_fallback_tie_break_logic_none_result(self, mock_tie_break, mock_dataset_handler):
        """Test when tie breaking returns None."""
        mock_handler = Mock()
        mock_handler.calculate_score.return_value = 0
        mock_dataset_handler.return_value = mock_handler
        mock_tie_break.return_value = None
        
        chains_in_tie = [{"extracted_answer": "A"}]
        tied_answers = ["A", "B"]
        
        voted_answer, score = fallback_tie_break_logic(
            chains_in_tie, tied_answers, "C", "gpqa_diamond", None
        )
        
        assert voted_answer == "A"  # Falls back to first tied answer
        assert score == 0
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_fallback_tie_break_logic_empty_input(self, mock_dataset_handler):
        """Test with empty chains or tied answers."""
        voted_answer, score = fallback_tie_break_logic(
            [], [], "A", "gpqa_diamond", None
        )
        
        assert voted_answer is None
        assert score == 0


class TestVotingIntegration:
    """Integration tests for voting functionality."""
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_complete_voting_workflow(self, mock_dataset_handler):
        """Test complete voting workflow."""
        mock_handler = Mock()
        mock_handler.calculate_score.side_effect = lambda extracted, correct: 1 if extracted == correct else 0
        mock_dataset_handler.return_value = mock_handler
        
        # Scenario: Multiple chains with clear winner
        chain_results = [
            {"extracted_answer": "A", "completion_tokens": 100, "chain_index": 0},
            {"extracted_answer": "A", "completion_tokens": 150, "chain_index": 1},
            {"extracted_answer": "A", "completion_tokens": 200, "chain_index": 2},
            {"extracted_answer": "B", "completion_tokens": 120, "chain_index": 3},
            {"extracted_answer": "B", "completion_tokens": 180, "chain_index": 4},
        ]
        
        # Test majority vote
        voted_answer, score, all_answers = majority_vote(
            chain_results, "A", "gpqa_diamond"
        )
        
        assert voted_answer == "A"  # 3 vs 2, A wins
        assert score == 1  # Correct answer
        assert len(all_answers) == 5
        assert all_answers.count("A") == 3
        assert all_answers.count("B") == 2
    
    @patch('prune.evaluation.voting.DatasetHandler')
    def test_tie_breaking_scenarios(self, mock_dataset_handler):
        """Test various tie-breaking scenarios."""
        mock_handler = Mock()
        mock_handler.calculate_score.return_value = 1
        mock_dataset_handler.return_value = mock_handler
        
        # Perfect tie scenario
        chain_results = [
            {"extracted_answer": "A", "completion_tokens": 100, "chain_index": 0},
            {"extracted_answer": "B", "completion_tokens": 150, "chain_index": 1},
        ]
        
        # Test sim prune voting (should signal for LLM tie-break)
        status, voted_answer, score, all_answers, chains_for_tie, tied_answers = majority_vote_for_sim_prune(
            chain_results, "A", "gpqa_diamond"
        )
        
        assert status == "REQUIRES_LLM_TIEBREAK"
        assert len(chains_for_tie) == 2
        assert set(tied_answers) == {"A", "B"}
        
        # Test fallback logic
        voted_answer, score = fallback_tie_break_logic(
            chains_for_tie, tied_answers, "A", "gpqa_diamond", None
        )
        
        assert voted_answer == "A"  # Should win by lower completion_tokens or chain_index
        assert score == 1
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with chains missing extracted_answer
        chain_results = [
            {"chain_index": 0},  # Missing extracted_answer
            {"extracted_answer": "A", "chain_index": 1},
            {"extracted_answer": None, "chain_index": 2},  # Explicit None
        ]
        
        status, voted_answer, all_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)
        
        assert status == "winner"
        assert voted_answer == "A"
        assert len(valid_chains) == 1  # Only one valid chain
        assert all_answers == ["A"]
    
    def test_numerical_tie_breaking(self):
        """Test numerical comparisons in tie breaking."""
        chains = [
            {"chain_index": 0, "value": 10.5},
            {"chain_index": 1, "value": 10.1},
            {"chain_index": 2, "value": 10.9}
        ]
        
        # Test minimum
        result = _get_best_chain_by_key(chains, "value", minimize=True)
        assert result["chain_index"] == 1
        assert result["value"] == 10.1
        
        # Test maximum
        result = _get_best_chain_by_key(chains, "value", minimize=False)
        assert result["chain_index"] == 2
        assert result["value"] == 10.9