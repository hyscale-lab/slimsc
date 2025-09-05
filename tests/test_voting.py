# tests/prune/evaluation/test_voting.py

import pytest
from unittest.mock import patch, MagicMock

# Import the module to be tested
from slimsc.prune.evaluation import voting

# --- Fixtures for Reusable Test Data ---

@pytest.fixture
def chains_clear_winner():
    """Chain results with a clear majority winner ('B')."""
    return [
        {'chain_index': 0, 'extracted_answer': 'A'},
        {'chain_index': 1, 'extracted_answer': 'B'},
        {'chain_index': 2, 'extracted_answer': 'B'},
    ]

@pytest.fixture
def chains_simple_tie():
    """Chain results with a simple tie between 'A' and 'B'."""
    return [
        {'chain_index': 0, 'extracted_answer': 'A'},
        {'chain_index': 1, 'extracted_answer': 'B'},
        {'chain_index': 2, 'extracted_answer': 'A'},
        {'chain_index': 3, 'extracted_answer': 'B'},
        {'chain_index': 4, 'extracted_answer': 'C'}, # Loser
    ]

@pytest.fixture
def chains_no_valid_answer():
    """Chain results with no valid 'extracted_answer' key."""
    return [
        {'chain_index': 0, 'final_answer_text': 'A'},
        {'chain_index': 1, 'final_answer_text': 'B'},
    ]

@pytest.fixture
def chains_for_token_tiebreak():
    """Chains for testing token-based tie-breaking."""
    return [
        # Tie between A and B
        {
            'chain_index': 0, 'extracted_answer': 'A', 'full_content': 'Short A.',
            'completion_tokens': 100
        },
        {
            'chain_index': 1, 'extracted_answer': 'B', 'full_content': 'A very long B answer.',
            'completion_tokens': 150
        },
        {
            'chain_index': 2, 'extracted_answer': 'A', 'full_content': 'Medium length A.',
            'completion_tokens': 120
        },
        {
            'chain_index': 3, 'extracted_answer': 'B', 'full_content': 'Another long B text.',
            'completion_tokens': 150
        }
    ]

@pytest.fixture
def chains_for_pruned_count_tiebreak():
    """Chains for testing pruned_count and similarity tie-breaking."""
    return [
        # Tie between A and B
        {
            'chain_index': 0, 'extracted_answer': 'A', 'pruned_count': 5,
            'final_internal_similarity': 0.8
        },
        {
            'chain_index': 1, 'extracted_answer': 'B', 'pruned_count': 7, # Should win on pruned_count
            'final_internal_similarity': 0.9
        },
        {
            'chain_index': 2, 'extracted_answer': 'A', 'pruned_count': 5,
            'final_internal_similarity': 0.7 # Would win on similarity if pruned_counts were tied
        },
        {
            'chain_index': 3, 'extracted_answer': 'B', 'pruned_count': 6,
            'final_internal_similarity': 0.6
        }
    ]

# --- Unit Tests for Internal Helper Functions ---

@pytest.mark.unit
class TestInternalHelpers:
    """Tests for the internal _ helper functions."""

    def test_process_initial_vote_winner(self, chains_clear_winner):
        status, answer, all_ans, _, tied = voting._process_initial_vote(chains_clear_winner)
        assert status == "winner"
        assert answer == "B"
        assert set(all_ans) == {'A', 'B'}
        assert tied is None

    def test_process_initial_vote_tie(self, chains_simple_tie):
        status, answer, all_ans, _, tied = voting._process_initial_vote(chains_simple_tie)
        assert status == "tie"
        assert answer is None
        assert set(all_ans) == {'A', 'B', 'C'}
        assert set(tied) == {'A', 'B'}

    def test_process_initial_vote_empty(self, chains_no_valid_answer):
        status, answer, all_ans, _, tied = voting._process_initial_vote(chains_no_valid_answer)
        assert status == "empty"
        assert answer is None
        assert all_ans == []
        assert tied is None

    def test_get_best_chain_by_key(self):
        chains = [
            {'chain_index': 0, 'value': 10},
            {'chain_index': 1, 'value': 5}, # Min value
            {'chain_index': 2, 'value': 20}, # Max value
            {'chain_index': 3, 'value': 5}, # Tie with index 1
            {'chain_index': 4, 'other_key': 100}, # Missing key
        ]
        # Test minimize
        best_min = voting._get_best_chain_by_key(chains, 'value', minimize=True)
        assert best_min['chain_index'] == 1 # Lower index wins tie

        # Test maximize
        best_max = voting._get_best_chain_by_key(chains, 'value', minimize=False)
        assert best_max['chain_index'] == 2

        # Test no valid chains
        assert voting._get_best_chain_by_key(chains, 'non_existent_key', minimize=True) is None

    def test_tie_break_by_tokens_priority_usage(self, chains_for_token_tiebreak):
        """Should pick winner based on fewest 'completion_tokens'."""
        tied_answers = ['A', 'B']
        winner = voting._tie_break_by_tokens(chains_for_token_tiebreak, tied_answers, tokenizer_path=None)
        # Chain 0 has 'A' and the lowest completion_tokens (100)
        assert winner == 'A'

    @patch('slimsc.prune.evaluation.voting.count_tokens')
    def test_tie_break_by_tokens_priority_tokenizer(self, mock_count_tokens, chains_for_token_tiebreak):
        """Should fall back to tokenizer count when usage tokens are tied or absent."""
        # Remove usage tokens to force fallback
        for chain in chains_for_token_tiebreak:
            del chain['completion_tokens']
        
        # Make chain 2 ('A') have the fewest tokens when counted by tokenizer
        mock_count_tokens.side_effect = lambda text, path: {'Short A.': 20, 'Medium length A.': 10, 'A very long B answer.': 30, 'Another long B text.': 35}[text]

        tied_answers = ['A', 'B']
        winner = voting._tie_break_by_tokens(chains_for_token_tiebreak, tied_answers, tokenizer_path="/fake/path")
        assert winner == 'A'
        assert mock_count_tokens.call_count == 4

    def test_tie_break_by_tokens_priority_index(self, chains_for_token_tiebreak):
        """Should fall back to lowest chain_index as the last resort."""
        # Make all token counts equal or absent
        for chain in chains_for_token_tiebreak:
            if 'completion_tokens' in chain:
                del chain['completion_tokens']
        
        tied_answers = ['A', 'B']
        # Without tokenizer_path, it will fail token counts and fall back to index
        winner = voting._tie_break_by_tokens(chains_for_token_tiebreak, tied_answers, tokenizer_path=None)
        # Chain 0 has answer 'A' and is the lowest index in the tie
        assert winner == 'A'
        
    def test_tie_break_by_pruned_count_priority_count(self, chains_for_pruned_count_tiebreak):
        """Should pick winner based on highest 'pruned_count'."""
        tied_answers = ['A', 'B']
        winner = voting._tie_break_by_pruned_count(chains_for_pruned_count_tiebreak, tied_answers)
        # Chain 1 has answer 'B' and highest pruned_count (7)
        assert winner == 'B'

    def test_tie_break_by_pruned_count_priority_similarity(self, chains_for_pruned_count_tiebreak):
        """Should fall back to lowest 'final_internal_similarity' if pruned_counts are tied."""
        # Make the highest pruned_counts equal
        chains_for_pruned_count_tiebreak[1]['pruned_count'] = 6 # Now chain 1 and 3 are tied at 6
        
        tied_answers = ['A', 'B']
        winner = voting._tie_break_by_pruned_count(chains_for_pruned_count_tiebreak, tied_answers)
        # Between chain 1 (sim 0.9) and chain 3 (sim 0.6), chain 3 has lower similarity
        assert winner == 'B' # Answer from chain 3


# --- Tests for Public Voting Functions ---

@patch('slimsc.prune.evaluation.voting.DatasetHandler')
class TestPublicVotingFunctions:
    """Tests for the main public voting functions."""

    def test_majority_vote_winner(self, mock_dataset_handler, chains_clear_winner):
        """Test a clear win with random tie-breaking."""
        mock_handler_instance = MagicMock()
        mock_handler_instance.calculate_score.return_value = 1 # Assume 'B' is correct
        mock_dataset_handler.return_value = mock_handler_instance

        answer, score, all_ans = voting.majority_vote(chains_clear_winner, 'B', 'gpqa_diamond')
        
        assert answer == 'B'
        assert score == 1
        assert set(all_ans) == {'A', 'B'}
        mock_handler_instance.calculate_score.assert_called_once_with('B', 'B')

    @patch('slimsc.prune.evaluation.voting.random.choice')
    def test_majority_vote_tie(self, mock_random_choice, mock_dataset_handler, chains_simple_tie):
        """Test random tie-breaking is handled correctly."""
        mock_handler_instance = MagicMock()
        # Let's say the correct answer is 'A', so if 'A' is chosen score is 1, else 0.
        mock_handler_instance.calculate_score = lambda chosen, correct: 1 if chosen == correct else 0
        mock_dataset_handler.return_value = mock_handler_instance

        # Force random.choice to pick 'A'
        mock_random_choice.return_value = 'A'
        answer, score, _ = voting.majority_vote(chains_simple_tie, 'A', 'gpqa_diamond')

        mock_random_choice.assert_called_once()
        # Check that the choices passed to random.choice were the tied answers
        assert set(mock_random_choice.call_args[0][0]) == {'A', 'B'}
        assert answer == 'A'
        assert score == 1
    
    def test_majority_vote_for_sim_prune_winner(self, mock_dataset_handler, chains_clear_winner):
        """Test that a clear winner is returned directly."""
        mock_handler_instance = MagicMock()
        mock_handler_instance.calculate_score.return_value = 1
        mock_dataset_handler.return_value = mock_handler_instance

        status, answer, score, _, _, _ = voting.majority_vote_for_sim_prune(chains_clear_winner, 'B')

        assert status == "winner"
        assert answer == "B"
        assert score == 1

    def test_majority_vote_for_sim_prune_requires_tiebreak(self, mock_dataset_handler, chains_simple_tie):
        """Test that a tie correctly signals for an LLM tie-break."""
        mock_dataset_handler.return_value = MagicMock() # Not used in tie case

        status, answer, score, all_ans, chains_for_tb, tied_ans = voting.majority_vote_for_sim_prune(chains_simple_tie, 'C')

        assert status == "REQUIRES_LLM_TIEBREAK"
        assert answer is None
        assert score == 0
        assert set(all_ans) == {'A', 'B', 'C'}
        assert set(tied_ans) == {'A', 'B'}
        # Check that ONLY the tied chains are returned for tie-breaking
        assert len(chains_for_tb) == 4
        assert all(c['extracted_answer'] in ('A', 'B') for c in chains_for_tb)

    @patch('slimsc.prune.evaluation.voting._tie_break_by_tokens')
    def test_fallback_tie_break_logic(self, mock_tie_break, mock_dataset_handler, chains_simple_tie):
        """Test the fallback logic orchestrator."""
        mock_handler_instance = MagicMock()
        mock_handler_instance.calculate_score.return_value = 1 # Assume 'A' is correct
        mock_dataset_handler.return_value = mock_handler_instance
        
        # Chains involved in the tie
        chains_in_tie = [c for c in chains_simple_tie if c['extracted_answer'] in ('A', 'B')]
        tied_answers = ['A', 'B']

        # Mock the internal tie-breaker to return a specific winner
        mock_tie_break.return_value = 'A'
        
        answer, score = voting.fallback_tie_break_logic(
            chains_in_tie, tied_answers, 'A', 'gpqa_diamond', tokenizer_path="/fake/path"
        )
        
        mock_tie_break.assert_called_once_with(
            valid_chain_results=chains_in_tie,
            tied_answers=tied_answers,
            tokenizer_path="/fake/path"
        )
        assert answer == 'A'
        assert score == 1