# slimsc/prune/evaluation/voting.py
from collections import Counter
from typing import List, Dict, Tuple, Optional, Literal
import random

from ..utils import count_tokens
from ..utils import DatasetHandler

import logging

logger = logging.getLogger(__name__)
random.seed(42) # For reproducibility

# --- Internal Helper Functions ---

def _process_initial_vote(
    chain_results: List[Dict],
) -> Tuple[Literal["winner", "tie", "empty"], Optional[str], List[str], List[Dict], Optional[List[str]]]:
    """
    Handles the common initial steps of majority voting:
    1. Filters for valid chains (with extracted_answer).
    2. Counts answers.
    3. Checks for a clear winner.
    4. Identifies tied answers if no clear winner.

    Returns:
        A tuple containing:
        - status: "winner", "tie", or "empty".
        - voted_answer: The winning answer if status is "winner", otherwise None.
        - all_extracted_answers: List of answers from all valid chains.
        - valid_chain_results: The filtered list of chain dictionaries.
        - tied_answers: List of answers involved in the tie if status is "tie", otherwise None.
    """
    valid_chain_results = [cr for cr in chain_results if cr.get("extracted_answer") is not None]
    valid_answers = [chain["extracted_answer"] for chain in valid_chain_results]

    if not valid_answers:
        logger.warning("[yellow]No valid chains with extracted answers found for majority vote.[/yellow]")
        return "empty", None, [], [], None

    answer_counts = Counter(valid_answers)
    most_common = answer_counts.most_common()

    # Check for clear winner
    if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
        voted_answer = most_common[0][0]
        logger.info(f"Clear majority winner: {voted_answer} (count: {most_common[0][1]})")
        return "winner", voted_answer, valid_answers, valid_chain_results, None
    else:
        # Handle Tie
        max_count = most_common[0][1]
        tied_answers = [ans for ans, count in most_common if count == max_count]
        logger.info(f"Tie detected among answers {tied_answers}. Tie-breaking required.")
        return "tie", None, valid_answers, valid_chain_results, tied_answers


def _get_best_chain_by_key(
    chains: List[Dict],
    key: str,
    minimize: bool, # True to find minimum value, False for maximum
    lower_index_better: bool = True, # Secondary tie-breaker
) -> Optional[Dict]:
    """Finds the best chain based on a numeric key, with index as tie-breaker."""
    best_chain: Optional[Dict] = None
    best_value = float('inf') if minimize else float('-inf')

    candidate_chains = [] # Chains with the current best_value

    for chain in chains:
        value = chain.get(key)
        if value is None:
            continue # Skip chains missing the key

        comparison = (value < best_value) if minimize else (value > best_value)

        if best_chain is None or comparison:
            best_value = value
            candidate_chains = [chain] # Reset candidates
        elif value == best_value:
            candidate_chains.append(chain) # Add to candidates

    if not candidate_chains:
        return None # No chains had the key

    if len(candidate_chains) == 1:
        best_chain = candidate_chains[0]
    else:
        # Tie in primary key, use chain index
        index_key = 'chain_index'
        index_default = float('inf') if lower_index_better else float('-inf')
        candidate_chains.sort(key=lambda c: c.get(index_key, index_default), reverse=not lower_index_better)
        best_chain = candidate_chains[0] # Pick the one with the 'best' index

    return best_chain


def _tie_break_by_tokens(
    valid_chain_results: List[Dict],
    tied_answers: List[str],
    tokenizer_path: Optional[str]
) -> Optional[str]:
    """
    Tie-breaking logic based on token counts (fewest is best).
    Priority: completion_tokens (usage) -> tokenizer count -> chain_index.
    """
    logger.debug("Attempting tie-breaker (fewest tokens).")
    chains_in_tie = [
        chain for chain in valid_chain_results
        if chain.get("extracted_answer") in tied_answers
    ]

    if not chains_in_tie: return None # Should not happen

    # 1. Try 'completion_tokens' (usage)
    logger.debug("Trying 'completion_tokens' (usage) for tie-break.")
    best_chain_usage = _get_best_chain_by_key(chains_in_tie, 'completion_tokens', minimize=True)

    if best_chain_usage:
        winner_answer = best_chain_usage.get("extracted_answer")
        logger.info(f"Tie broken via usage stats: Chose chain {best_chain_usage.get('chain_index', 'N/A')} "
                    f"with {best_chain_usage.get('completion_tokens')} completion tokens (Answer: {winner_answer})")
        return winner_answer

    logger.debug("'completion_tokens' unavailable for all tied chains or tie remained. Trying tokenizer fallback.")

    # 2. Try tokenizer count fallback
    if not tokenizer_path:
        logger.warning("Tokenizer path not provided. Cannot use tokenizer count fallback for tie-breaking.")
    else:
        token_counting_successful = True
        temp_token_counts = {} # {chain_index: count}
        chains_with_counts = []

        for chain in chains_in_tie:
            content_to_count = chain.get("full_content", "") # Use full_content if available
            if not content_to_count:
                 # Fallback further to reasoning or final answer if full_content missing
                 content_to_count = chain.get("reasoning_text", "") or chain.get("final_answer_text", "")

            if not content_to_count:
                logger.warning(f"Chain {chain.get('chain_index', 'N/A')} has no content ('full_content', 'reasoning_text', 'final_answer_text') for token counting.")
                # Treat as if counting failed for this chain if it matters for tie-break
                continue

            tokens_fallback = count_tokens(content_to_count, tokenizer_path)
            chain_idx = chain.get('chain_index')

            if tokens_fallback is None:
                token_counting_successful = False
                logger.warning(f"Tokenizer counting failed for chain {chain_idx}. Cannot reliably use tokenizer tie-break.")
                break # Stop trying if any fails
            if chain_idx is not None:
                temp_token_counts[chain_idx] = tokens_fallback
                chain['tokenizer_tokens_fallback'] = tokens_fallback # Add count to chain dict for _get_best_chain_by_key
                chains_with_counts.append(chain)
            else:
                 logger.warning(f"Chain missing 'chain_index', cannot use reliably in tokenizer tie-break.")


        if token_counting_successful and chains_with_counts:
            best_chain_tokenizer = _get_best_chain_by_key(chains_with_counts, 'tokenizer_tokens_fallback', minimize=True)
            if best_chain_tokenizer:
                winner_answer = best_chain_tokenizer.get("extracted_answer")
                logger.info(f"Tie broken via tokenizer count fallback: Chose chain {best_chain_tokenizer.get('chain_index', 'N/A')} "
                            f"with {best_chain_tokenizer.get('tokenizer_tokens_fallback')} total tokens (Answer: {winner_answer})")
                return winner_answer
        elif not chains_with_counts:
             logger.warning("No tied chains could be processed for tokenizer fallback counting.")


    # 3. Final fallback: Lowest chain index
    logger.warning("[yellow]Tie-breaking failed using token counts. Arbitrarily choosing lowest chain index among tied chains.[/yellow]")
    best_chain_index = _get_best_chain_by_key(chains_in_tie, 'chain_index', minimize=True)

    if best_chain_index:
        winner_answer = best_chain_index.get("extracted_answer")
        logger.info(f"Tie broken via lowest index: Chose chain {best_chain_index.get('chain_index', 'N/A')} (Answer: {winner_answer})")
        return winner_answer
    else:
        # Should be extremely rare - means no tied chains even had an index?
        logger.error("[red]Cannot break tie even by index. Falling back to first tied answer.[/red]")
        return tied_answers[0]


def _tie_break_by_pruned_count(
    valid_chain_results: List[Dict],
    tied_answers: List[str],
) -> Optional[str]:
    """
    Tie-breaking logic based on the highest accumulated pruned_count.
    Fallback Priority:
    1. pruned_count (highest)
    2. final_internal_similarity (lowest)
    3. chain_index (lowest)
    Requires chain_results dicts to contain 'pruned_count' and 'final_internal_similarity'.
    """
    logger.debug("Attempting tie-breaker (highest pruned count).")
    chains_in_tie = [
        chain for chain in valid_chain_results
        if chain.get("extracted_answer") in tied_answers
    ]

    if not chains_in_tie:
        logger.error("[red]Cannot perform tie-break: No chains found matching tied answers.[/red]")
        return None # Should not happen

    # --- Primary Tie-breaker: Highest Pruned Count ---
    # Use _get_best_chain_by_key to find the chain with the maximum pruned_count
    # maximize=True means higher pruned_count is better
    # lower_index_better=True (default) ensures lowest index wins ONLY if counts are equal
    best_chain_pruned_count = _get_best_chain_by_key(
        chains_in_tie,
        key='pruned_count',
        minimize=False # We want the MAXIMUM count
    )

    # Check if the primary tie-breaker produced a unique winner *before* falling back to index
    # This requires checking if multiple chains shared the best pruned_count value.
    # We can do this by filtering chains_in_tie for the best count and seeing if > 1 remain.
    potential_winners = []
    if best_chain_pruned_count:
        best_count = best_chain_pruned_count.get('pruned_count')
        if best_count is not None:
            potential_winners = [
                c for c in chains_in_tie
                if c.get('pruned_count') == best_count
            ]

    # If _get_best_chain_by_key resolved the tie (implicitly using index if counts were equal)
    # or if there was only one chain with the best count, we have our winner.
    if best_chain_pruned_count and len(potential_winners) <= 1:
         winner_answer = best_chain_pruned_count.get("extracted_answer")
         winner_index = best_chain_pruned_count.get('chain_index', 'N/A')
         winner_count = best_chain_pruned_count.get('pruned_count', 'N/A')
         logger.info(f"Tie broken via highest pruned count (or index if counts tied): Chose chain {winner_index} "
                     f"with pruned_count={winner_count} (Answer: {winner_answer})")
         return winner_answer
    else:
        # --- Fallback Tie-breaker: Lowest Internal Similarity ---
        # This block is reached if:
        # a) 'pruned_count' key was missing from all tied chains.
        # b) Multiple chains tied for the highest 'pruned_count' (and _get_best_chain_by_key's index tie-break is bypassed by this logic structure, which is intended here).
        logger.warning("[yellow]Pruned count tie-breaking failed or resulted in a tie. "
                       "Falling back to lowest final_internal_similarity.[/yellow]")

        # We now only consider the 'potential_winners' if they exist (i.e., if the tie was in pruned_count)
        # Otherwise, we use the original 'chains_in_tie' (if pruned_count key was missing)
        chains_for_similarity_tiebreak = potential_winners if potential_winners else chains_in_tie

        best_chain_similarity = _get_best_chain_by_key(
            chains_for_similarity_tiebreak,
            key='final_internal_similarity',
            minimize=True # Lower similarity is better
            # lower_index_better=True (default) is the final fallback if similarities are also tied
        )

        if best_chain_similarity:
            winner_answer = best_chain_similarity.get("extracted_answer")
            winner_index = best_chain_similarity.get('chain_index', 'N/A')
            winner_sim = best_chain_similarity.get('final_internal_similarity', float('nan'))
            winner_count_orig = best_chain_similarity.get('pruned_count', 'N/A') # Log original count for context
            logger.info(f"Tie broken via lowest internal similarity fallback: Chose chain {winner_index} "
                        f"with internal_similarity={winner_sim:.4f} (Original pruned_count={winner_count_orig}) "
                        f"(Answer: {winner_answer})")
            return winner_answer
        else:
            # This should be extremely rare: means chains_in_tie was empty OR
            # none of the chains had 'pruned_count' AND none had 'final_internal_similarity'.
            logger.error("[red]Cannot break tie even by internal similarity fallback. "
                         "Arbitrarily choosing the first tied answer.[/red]")
            # Use the original tied_answers list as the ultimate fallback
            return tied_answers[0] if tied_answers else None


# --- Public Voting Functions ---

def majority_vote(
    chain_results: List[Dict],
    correct_answer_letter: str,
    dataset_name: str,
    tokenizer_path: Optional[str] = None
) -> Tuple[Optional[str], int, List[str]]:
    """
    Performs majority voting. Breaks ties by selecting the final answer randomly
    from the tied options.

    Requires 'extracted_answer'.

    Args:
        chain_results: List of dicts for completed chains.
        correct_answer_letter: The correct answer.
        dataset_name: Name of the dataset to use for scoring.
        tokenizer_path: Path to tokenizer for fallback token counting.

    Returns:
        Tuple[Optional[str], int, List[str]]: Voted answer, score, list of all valid extracted answers.
    """
    dataset_handler = DatasetHandler(dataset_name=dataset_name)

    status, voted_answer, all_extracted_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)

    if status == "empty":
        return None, 0, []
    if status == "winner":
        score = dataset_handler.calculate_score(voted_answer, correct_answer_letter)
        return voted_answer, score, all_extracted_answers

    # Status is "tie"
    # Break ties randomly
    if tied_answers: # Ensure there are answers to choose from (should be guaranteed by _process_initial_vote)
        final_voted_answer = random.choice(tied_answers)
        logger.info(f"Tie broken randomly: Chose answer '{final_voted_answer}' from {tied_answers}")
    else:
        # This case should theoretically not be reached if status is "tie", but as a safeguard:
        logger.error("[red]Tie status reported but no tied answers found. Returning None.[/red]")
        final_voted_answer = None

    # Calculate score based on the tie-broken answer
    score = dataset_handler.calculate_score(final_voted_answer, correct_answer_letter)
    return final_voted_answer, score, all_extracted_answers


def majority_vote_for_sim_prune(
    chain_results: List[Dict],
    correct_answer_letter: str,
    dataset_name: str = "gpqa_diamond"  # Default to GPQA for backward compatibility
) -> Tuple[Optional[str], int, List[str]]:
    """
    Specialized majority voting for similarity pruning evaluation.
    Uses internal similarity for tie-breaking (mean_sim / num_thoughts).
    Requires chain_results dicts to contain 'final_internal_similarity'.

    Args:
        chain_results: List of chain result dictionaries.
        correct_answer_letter: The correct answer for scoring.
        dataset_name: The dataset type ("gpqa_diamond", "aime", "math500").
    Performs majority voting using pruned count tie-breaking (highest count is best).
    Falls back to lowest internal similarity, then lowest chain index.
    Requires 'extracted_answer'. Optional keys for tie-breaking:
    'pruned_count', 'final_internal_similarity', 'chain_index'.

    Args:
        chain_results: List of dicts for completed chains. Must contain
                       'pruned_count' and 'final_internal_similarity'
                       if tie-breaking is needed.
        correct_answer_letter: The correct answer.

    Returns:
        Tuple of (voted_answer, score, all_extracted_answers).
    """
    dataset_handler = DatasetHandler(dataset_name=dataset_name)

    # Assumes chain_results dictionaries now contain 'pruned_count' and 'final_internal_similarity'
    status, voted_answer, all_extracted_answers, valid_chains, tied_answers = _process_initial_vote(chain_results)

    if status == "empty":
        return None, 0, []
    if status == "winner":
        # Log winner details
        winner_chain = next((c for c in valid_chains if c.get("extracted_answer") == voted_answer), None)
        winner_count = winner_chain.get('pruned_count', 'N/A') if winner_chain else 'N/A'
        winner_sim = winner_chain.get('final_internal_similarity', 'N/A') if winner_chain else 'N/A'
        sim_str = f"{winner_sim:.4f}" if isinstance(winner_sim, (int, float)) else 'N/A'
        logger.info(f"Clear majority winner {voted_answer} had pruned_count: {winner_count}, internal_sim: {sim_str}")
        score = dataset_handler.calculate_score(voted_answer, correct_answer_letter)
        return voted_answer, score, all_extracted_answers
    
    # Status is "tie"
    final_voted_answer = _tie_break_by_pruned_count(valid_chains, tied_answers)

    # Calculate score based on the tie-broken answer
    score = dataset_handler.calculate_score(final_voted_answer, correct_answer_letter)
    return final_voted_answer, score, all_extracted_answers
