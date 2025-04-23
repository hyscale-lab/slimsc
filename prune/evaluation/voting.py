# slimsc/prune/evaluation/voting.py
from collections import Counter
from typing import List, Dict, Tuple, Optional
from ..utils import calculate_score_gpqa, count_tokens

import logging

logger = logging.getLogger(__name__)

def majority_vote(
    chain_results: List[Dict], # List of processed chain result dicts that finished streams successfully and produced content
    correct_answer_letter: str,
    tokenizer_path: Optional[str] = None # Pass tokenizer path if needed for tie-break
) -> Tuple[Optional[str], int, List[str]]:
    """
    Performs majority voting on the extracted answers from SC chains.
    Uses completion_tokens from chain_results for N=2 tie-breaking if available.
    Falls back to total token count via tokenizer if completion_tokens (usage) is missing.
    Assumes input `chain_results` already filtered to chains that finished their streams
    and produced content, but we filter again for chains that had extractable answers.
    """
    # Filter out chains that did not produce an extracted answer
    valid_chain_results = [cr for cr in chain_results if cr.get("extracted_answer") is not None]

    # Get extracted answers from valid chains for counting and final list
    valid_answers = [chain["extracted_answer"] for chain in valid_chain_results]
    all_extracted_answers = valid_answers # This will be the list used in summary JSON/CSV

    if not valid_answers:
        logger.warning("[yellow]No valid chains with extracted answers found for majority vote.[/yellow]")
        # Return an empty list for all_extracted_answers as none were valid for voting
        return None, 0, []

    answer_counts = Counter(valid_answers)
    most_common = answer_counts.most_common()

    # Check if there is a clear winner (count > next highest count)
    if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
        voted_answer = most_common[0][0]
        score = calculate_score_gpqa(voted_answer, correct_answer_letter)
        logger.info(f"Clear majority winner: {voted_answer} (count: {most_common[0][1]})")
        return voted_answer, score, all_extracted_answers

    # Handle Ties - occurs when most_common[0][1] == most_common[1][1]
    max_count = most_common[0][1]
    tied_answers = [ans for ans, count in most_common if count == max_count]
    final_voted_answer = None

    # --- N=2 Tie-breaking Logic ---
    # Check if the tie involves exactly 2 valid chains with extracted answers
    # and there are no other chains with the same count.
    is_n2_tie = (len(valid_chain_results) == 2 and len(tied_answers) == 2)
    is_general_tie = (len(tied_answers) > 1) # Any tie situation

    if is_n2_tie:
        logger.info(f"N=2 tie detected among answers {tied_answers}. Attempting tie-breaker (fewest tokens).")
        min_tokens = float('inf')
        best_chain_data: Optional[Dict] = None # Use Optional type hint
        tie_break_successful = False

        # Attempt tie-breaking using 'completion_tokens' (usage) first
        chains_in_tie_with_usage = [
            chain for chain in valid_chain_results
            if chain.get("extracted_answer") in tied_answers and chain.get("completion_tokens") is not None
        ]

        if len(chains_in_tie_with_usage) > 0: # Need at least one chain with usage info
             logger.debug("Using 'completion_tokens' (usage) for N=2 tie-breaker.")
             for chain in chains_in_tie_with_usage:
                  tokens = chain["completion_tokens"]
                  # In case of equal token counts, prefer the chain with the lower index
                  # Need to handle initial best_chain_data being None
                  if best_chain_data is None or tokens < min_tokens or (tokens == min_tokens and chain.get('chain_index', float('inf')) < best_chain_data.get('chain_index', float('inf'))):
                       min_tokens = tokens
                       best_chain_data = chain
             if best_chain_data:
                  final_voted_answer = best_chain_data.get("extracted_answer")
                  logger.info(f"N=2 tie broken via usage stats: Chose chain {best_chain_data.get('chain_index', 'N/A')} with {min_tokens} completion tokens (Answer: {final_voted_answer})")
                  tie_break_successful = True
        else:
             logger.debug("No chains in tie had 'completion_tokens' usage stats available.")


        # If usage-based tie-breaking failed, attempt tie-breaking using tokenizer count as a fallback
        if not tie_break_successful and tokenizer_path:
             logger.debug("Usage stats tie-break failed. Attempting tokenizer counting fallback for N=2 tie.")
             min_tokens_fallback = float('inf')
             best_chain_data_fallback: Optional[Dict] = None # Use Optional type hint
             token_counting_available_for_all_tied = True

             chains_in_tie_with_tokenizer_count = []
             for chain in valid_chain_results:
                  if chain.get("extracted_answer") in tied_answers:
                      # Use full content for counting
                      content_to_count = chain.get("full_content", "")
                      tokens_fallback = count_tokens(content_to_count, tokenizer_path)
                      if tokens_fallback is not None: # Check if counting was successful
                          chain['total_tokens_counted_fallback'] = tokens_fallback # Store fallback count
                          chains_in_tie_with_tokenizer_count.append(chain)
                      else:
                          token_counting_available_for_all_tied = False
                          logger.warning(f"Tokenizer counting failed for chain {chain.get('chain_index', 'N/A')}. Cannot use tokenizer tie-break.")
                          break # Cannot use tokenizer tie-break if it fails for any tied chain

             if token_counting_available_for_all_tied and len(chains_in_tie_with_tokenizer_count) > 0:
                 for chain in chains_in_tie_with_tokenizer_count:
                     tokens = chain['total_tokens_counted_fallback']
                     # In case of equal token counts, prefer the chain with the lower index
                     if best_chain_data_fallback is None or tokens < min_tokens_fallback or (tokens == min_tokens_fallback and chain.get('chain_index', float('inf')) < best_chain_data_fallback.get('chain_index', float('inf'))):
                           min_tokens_fallback = tokens
                           best_chain_data_fallback = chain

                 if best_chain_data_fallback:
                      final_voted_answer = best_chain_data_fallback.get("extracted_answer")
                      logger.info(f"N=2 tie broken via tokenizer count fallback: Chose chain {best_chain_data_fallback.get('chain_index', 'N/A')} with {min_tokens_fallback} total tokens (Answer: {final_voted_answer})")
                      tie_break_successful = True
                 # else: # This case covered by len(chains_in_tie_with_tokenizer_count) check
             else:
                 logger.warning("[yellow]Tokenizer counting tie-break fallback failed or unavailable for all tied chains.[/yellow]")


        # If both usage and tokenizer tie-breaking failed for N=2
        if not tie_break_successful:
            logger.warning("[yellow]N=2 tie-breaking failed (token counts unavailable/equal or tokenizer issue). Arbitrarily choosing first tied answer.[/yellow]")
            final_voted_answer = tied_answers[0] # Arbitrarily choose the first tied answer

    elif is_general_tie:
         # Any other tie scenario (N > 2, or N=2 tie not between two answers)
         logger.warning(f"[yellow]Tie detected among: {tied_answers} (not a standard N=2 tie). Arbitrarily choosing first tied answer.[/yellow]")
         final_voted_answer = tied_answers[0]
    else:
        # This case should not be reached if most_common has > 1 elements and counts are equal
        # It's here as a safeguard.
        logger.error("[red]Unexpected voting scenario: Tie detected but fell through tie handling logic.[/red]")
        final_voted_answer = most_common[0][0] # Default to first most common

    score = calculate_score_gpqa(final_voted_answer, correct_answer_letter)
    return final_voted_answer, score, all_extracted_answers
