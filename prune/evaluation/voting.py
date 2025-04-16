from collections import Counter
from typing import List, Dict, Tuple, Optional
from ..utils import calculate_score_gpqa, count_tokens

import logging

logger = logging.getLogger(__name__)

def majority_vote(
    chain_results: List[Dict],
    correct_answer_letter: str,
    tokenizer_path: Optional[str] = None # Pass tokenizer path if needed for tie-break
) -> Tuple[Optional[str], int, List[str]]:
    """
    Performs majority voting on the extracted answers from SC chains.
    Uses completion_tokens from chain_results for N=2 tie-breaking if available.
    """
    # Filter out chains that resulted in an error
    valid_chain_results = [cr for cr in chain_results if "error" not in cr]

    if not valid_chain_results:
        logger.warning("[yellow]No valid chains found for majority vote.[/yellow]")
        return None, 0, []


    extracted_answers = [chain.get("extracted_answer") for chain in valid_chain_results]
    valid_answers = [ans for ans in extracted_answers if ans is not None]

    if not valid_answers:
        return None, 0, extracted_answers

    answer_counts = Counter(valid_answers)
    most_common = answer_counts.most_common()

    if len(most_common) == 1 or (len(most_common) > 1 and most_common[0][1] > most_common[1][1]):
        voted_answer = most_common[0][0]
        score = calculate_score_gpqa(voted_answer, correct_answer_letter)
        return voted_answer, score, extracted_answers

    # Handle Ties
    max_count = most_common[0][1]
    tied_answers = [ans for ans, count in most_common if count == max_count]
    final_voted_answer = None

    if len(valid_chain_results) == 2 and len(tied_answers) == 2: # Use count of valid chains
        logger.info(f"N=2 tie detected ({tied_answers}). Applying tie-breaker (fewest completion tokens).")
        min_tokens = float('inf')
        best_chain_idx_in_valid = -1 # Index within the valid_chain_results list
        token_counts_available = True

        for i, chain in enumerate(valid_chain_results):
            if chain.get("extracted_answer") in tied_answers:
                tokens = chain.get("completion_tokens")
                if tokens is not None:
                    if tokens < min_tokens:
                        min_tokens = tokens
                        best_chain_idx_in_valid = i
                else:
                    token_counts_available = False
                    break

        if token_counts_available and best_chain_idx_in_valid != -1:
            best_chain_data = valid_chain_results[best_chain_idx_in_valid]
            final_voted_answer = best_chain_data.get("extracted_answer")
            logger.info(f"Tie broken via usage stats: Chose chain {best_chain_data['chain_index']} with {min_tokens} tokens (Answer: {final_voted_answer})")
        else:
            logger.info("Usage stats missing for tie-break, attempting fallback to tokenizer counting...")
            if tokenizer_path:
                 min_tokens_fallback = float('inf')
                 best_chain_idx_fallback_in_valid = -1
                 for i, chain in enumerate(valid_chain_results):
                     if chain.get("extracted_answer") in tied_answers:
                         # Use full content for counting, assuming answer format is robust
                         content_to_count = chain.get("full_content", "")
                         tokens_fallback = count_tokens(content_to_count, tokenizer_path)
                         if tokens_fallback != -1 and tokens_fallback < min_tokens_fallback:
                              min_tokens_fallback = tokens_fallback
                              best_chain_idx_fallback_in_valid = i

                 if best_chain_idx_fallback_in_valid != -1:
                     best_chain_data_fallback = valid_chain_results[best_chain_idx_fallback_in_valid]
                     final_voted_answer = best_chain_data_fallback.get("extracted_answer")
                     logger.info(f"Tie broken via tokenizer count: Chose chain {best_chain_data_fallback['chain_index']} with {min_tokens_fallback} tokens (Answer: {final_voted_answer})")
                 else:
                     logger.warning("[yellow]Tokenizer counting failed. Arbitrarily choosing first tied answer.[/yellow]")
                     final_voted_answer = tied_answers[0]
            else:
                logger.warning("[yellow]Tokenizer path not provided for N=2 tie-breaking fallback. Arbitrarily choosing first tied answer.[/yellow]")
                final_voted_answer = tied_answers[0]

    elif len(tied_answers) > 1:
         logger.warning(f"[yellow]Tie detected among: {tied_answers}. Arbitrarily choosing first tied answer.[/yellow]")
         final_voted_answer = tied_answers[0]
    else:
        logger.warning(f"[yellow]Unexpected voting scenario. Tied answers: {tied_answers}. Choosing first.[/yellow]")
        final_voted_answer = tied_answers[0]

    score = calculate_score_gpqa(final_voted_answer, correct_answer_letter)
    return final_voted_answer, score, extracted_answers