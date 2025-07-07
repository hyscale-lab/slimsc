# slimsc/prune/utils/similarity_utils.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple, Callable
import logging
import torch
from transformers import AutoTokenizer
from .count_tokens import count_tokens

logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
MIN_SEGMENT_TOKENS = 25 # Define minimum thought length (tokens)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Caches ---
_embedding_model = None
# Tokenizer caching for utils is handled by utils.gpqa_utils.count_tokens
_tokenizer_for_utils = None
_tokenizer_path_loaded_for_utils = None

TARGET_PHRASES = ["alternative", "Alternative", "Another", "But another", "perhaps another", "Wait", "Oh wait", "But wait"]

def get_embedding_model() -> SentenceTransformer:
    """Loads or returns the cached SentenceTransformer model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} to device: {DEVICE}")
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.exception("[red]Failed to load Sentence Transformer model[/red]")
            raise e
    return _embedding_model


def embed_segments(segments: List[str]) -> Optional[np.ndarray]:
    """Computes embeddings for a list of text segments."""
    if not segments:
        # Ensure consistency: return empty array with correct dimensions
        try:
            model = get_embedding_model()
            dim = model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)
        except Exception:
             logger.error("[red]Could not get embedding dimension for empty segment list.[/red]")
             return None # Return None if model loading failed

    try:
        model = get_embedding_model()
        embeddings = model.encode(segments, convert_to_numpy=True, normalize_embeddings=True)
        # Ensure embeddings are float32 as required by FAISS
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        return embeddings
    except Exception as e:
        logger.exception(f"[red]Error during segment embedding: {e}[/red]")
        return None

# --- Thought Boundary and Segment Extraction Functions ---

def find_thought_boundaries(text: str, target_phrases: List[str]) -> List[int]:
    """
    Finds the starting character indices of all thought segments in the text.
    The first thought always starts at index 0. Subsequent thoughts are marked
    by target_phrases, ideally preceded by whitespace or start of line.
    """
    boundaries = [0] # First thought always starts at 0
    if not text:
        return boundaries

    found_positions = []
    for phrase in target_phrases:
        start_index = 0
        phrase_len = len(phrase)
        while True:
            pos = text.find(phrase, start_index)
            if pos == -1:
                break # Phrase not found (anymore) in the remaining text

            # Check context: must be preceded by start of string (pos == 0) or a non-word character/whitespace.
            # This helps avoid matching phrases mid-word or in unrelated contexts.
            is_likely_start = False
            if pos == 0:
                is_likely_start = True
            elif pos > 0:
                preceding_char = text[pos - 1]
                if preceding_char.isspace() or not preceding_char.isalnum():
                    is_likely_start = True
                # More specific check for common separators just before
                elif preceding_char in '.,;!?\'"': # Add common punctuation before phrase
                    # Check if the char before punctuation is also non-word/space or start of line
                    if pos > 1:
                         char_before_punct = text[pos - 2]
                         if char_before_punct.isspace() or not char_before_punct.isalnum():
                              is_likely_start = True
                    elif pos == 1: # If punctuation is the very first char
                         is_likely_start = True # Unlikely scenario for a thought boundary marker but safe


            if is_likely_start:
                found_positions.append(pos)

            # Move search start past the found position to find the next occurrence
            start_index = pos + 1 # Move search start just past the found position

    # Combine, sort, and remove duplicates
    # Also, ensure no boundary is equal to the index of the first thought start (0)
    if found_positions:
        # Remove duplicates and sort
        unique_sorted_positions = sorted(list(set(found_positions)))
        # Add boundaries from found_positions, excluding 0 if it was found
        # This might happen if a target phrase is at the very start, but thought 0 already covers that.
        boundaries.extend([pos for pos in unique_sorted_positions if pos > 0])

    # Ensure boundaries are unique and sorted at the end
    boundaries = sorted(list(set(boundaries)))

    # logger.debug(f"Found boundaries in text (len {len(text)}): {boundaries}")
    return boundaries


def find_newly_completed_thoughts(
    full_text: str,
    processed_boundaries: List[int], # Start indices of thoughts already processed
    tokenizer_path: str,
    target_phrases: List[str],
    min_segment_tokens: int = MIN_SEGMENT_TOKENS
) -> Tuple[List[Tuple[int, int, str]], List[int]]:
    """
    Identifies newly completed thought segments based on detected boundaries.
    
    A segment is considered 'completed' when its start boundary is detected,
    and the next boundary in the text has also appeared.

    Returns:
        Tuple: (new_segments, updated_processed_boundaries)
        - new_segments: List of (start_idx, end_idx, segment_text) for valid new thoughts.
        - updated_processed_boundaries: The list of start indices including newly processed ones.
    """
    if not full_text:
        return [], processed_boundaries

    # Find all potential boundaries in the current full text
    all_current_boundaries = find_thought_boundaries(full_text, target_phrases)

    newly_completed_segments = []
    # Track starts found and processed as a start of a newly completed segment in *this* function call
    new_segment_starts_processed_this_call = []

    # Ensure processed_boundaries is sorted and unique for efficient checking
    current_processed_boundaries = sorted(list(set(processed_boundaries)))
    # The index of the boundary that starts the LAST segment we finished processing previously.
    # We only look for new segment boundaries starting >= after this.
    last_processed_start = current_processed_boundaries[-1] if current_processed_boundaries else -1

    # Iterate through pairs of boundaries (start, end)
    # A segment is defined by `all_current_boundaries[i]` (start) and `all_current_boundaries[i+1]` (end)
    for i in range(len(all_current_boundaries) - 1):
        boundary_start = all_current_boundaries[i]
        boundary_end = all_current_boundaries[i+1]

        # Check if this segment's START boundary is NEW relative to what we've processed in PREVIOUS calls
        # AND has not been marked as a new segment start in *this* call yet.
        if boundary_start >= last_processed_start and boundary_start not in new_segment_starts_processed_this_call:

             # This boundary_start defines the start of a potential new segment.
             # Check if this specific boundary_start was already processed in any previous call.
             # We only process boundary_start if it's strictly greater than the last start boundary
             # we added to `processed_boundaries` in a previous call, or if it's a boundary
             # that appeared for the first time now.
             # The logic `boundary_start >= last_processed_start` covers most cases,
             # but `boundary_start not in current_processed_boundaries` is the most robust check
             # to see if this *specific* start index has ever defined a processed segment boundary before.

             if boundary_start not in current_processed_boundaries:
                  segment_text = full_text[boundary_start:boundary_end].strip()

                  if not segment_text: # Skip empty segments
                       continue

                  # Check token length using the shared utility function
                  num_tokens = count_tokens(segment_text, tokenizer_path)

                  if num_tokens is None:
                       logger.warning(f"Could not tokenize segment [{boundary_start}:{boundary_end}] for length check. Skipping segment.")
                       continue # Skip segment if tokenization fails

                  if num_tokens >= min_segment_tokens:
                      # logger.debug(f"Found NEW completed thought segment [{boundary_start}:{boundary_end}], tokens={num_tokens}.") # Text: '{segment_text[:80]}...'")
                      newly_completed_segments.append((boundary_start, boundary_end, segment_text))
                      new_segment_starts_processed_this_call.append(boundary_start) # Mark this start as processed NOW
                  # else: logger.debug(f"New segment [{boundary_start}:{boundary_end}] too short (tokens={num_tokens}).")

    # Combine old and newly processed start boundaries found in this call
    # Add the starts found *in this call* to the existing processed boundaries.
    # Sort and make unique to maintain a clean list.
    updated_processed_boundaries = sorted(list(set(current_processed_boundaries + new_segment_starts_processed_this_call)))

    return newly_completed_segments, updated_processed_boundaries


def extract_final_thought(
    full_text: str,
    processed_boundaries: List[int], # Already updated list from the last step
    tokenizer_path: str,
    min_segment_tokens: int = MIN_SEGMENT_TOKENS
) -> Optional[Tuple[int, int, str]]:
    """Extracts the final thought segment when a chain finishes, if valid."""
    if not full_text:
        return None

    # The final segment starts after the last boundary that was processed as a segment start
    # If processed_boundaries is empty (no boundaries found or processed ever), the first segment starts at 0.
    # The final segment starts after the *last* boundary in the list of all boundaries found in the text,
    # unless there are no boundaries other than 0, in which case the whole text is the first/only segment.

    all_current_boundaries = find_thought_boundaries(full_text, TARGET_PHRASES) # Find all boundaries one last time

    start_pos = 0 # Default if no boundaries were ever found other than 0
    if len(all_current_boundaries) > 1:
         # The last segment starts at the second-to-last boundary index found
         # E.g., boundaries [0, 100, 250]. Segments are [0, 100), [100, 250), [250, end).
         # The final segment starts at the last boundary index found (250).
         start_pos = all_current_boundaries[-1]
    # If only [0] or empty, start_pos remains 0.

    end_pos = len(full_text)
    segment_text = full_text[start_pos:end_pos].strip() # Use end_pos for clarity

    if not segment_text:
        return None

    # Check token length using the shared utility function
    num_tokens = count_tokens(segment_text, tokenizer_path)

    if num_tokens is None:
         logger.warning(f"Could not tokenize final segment [{start_pos}:{end_pos}] for length check. Skipping final segment extraction.")
         return None

    if num_tokens >= min_segment_tokens:
        logger.debug(f"Found FINAL thought segment [{start_pos}:{end_pos}], tokens={num_tokens}.") # Text: '{segment_text[:80]}...'")
        return (start_pos, end_pos, segment_text)
    else:
        logger.debug(f"Final segment [{start_pos}:{end_pos}] too short (tokens={num_tokens}).")
        return None

# --- FaissIndexManager Class ---
class FaissIndexManager:
    """Manages a FAISS index for similarity checking during pruning."""
    def __init__(self, dimension: int, search_mode: str = 'similarity'):
        self.original_dimension = dimension
        self.search_mode = search_mode

        if self.search_mode == 'dissimilarity':
            # For farthest neighbor search, augment dimension by 1
            self.index_dimension = self.original_dimension + 1
            logger.info(f"FaissIndexManager initialized in 'dissimilarity' mode (dim={self.index_dimension}).")
        else: # Default 'similarity' mode
            self.index_dimension = self.original_dimension
            logger.info(f"FaissIndexManager initialized in 'similarity' mode (dim={self.index_dimension}).")

        self.index = faiss.IndexFlatIP(self.index_dimension)
        self.metadata_map: Dict[int, Tuple[str, int, str]] = {}
        self.next_id = 0
        self.original_embeddings: Dict[int, np.ndarray] = {}  # Store original embeddings for cosine similarity

    def _augment_for_database(self, xb: np.ndarray) -> np.ndarray:
        """Augments database vectors for farthest neighbor search. x' = [-2x, ||x||^2]"""
        norms_sq = (xb ** 2).sum(axis=1)
        return np.hstack((-2 * xb, norms_sq[:, np.newaxis])).astype(np.float32)

    def _augment_for_query(self, xq: np.ndarray) -> np.ndarray:
        """Augments query vectors for farthest neighbor search. q' = [q, 1]"""
        extra_column = np.ones((len(xq), 1), dtype=xq.dtype)
        return np.hstack((xq, extra_column)).astype(np.float32)

    def add_embedding(self, embedding: np.ndarray, chain_id: str, thought_index: int, text_segment: str):
        """Adds a single embedding and its metadata to the index."""
        if embedding.ndim == 1: embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.original_dimension:
             logger.error(f"Embedding dim mismatch: expected {self.original_dimension}, got {embedding.shape[1]}. Skip add.")
             return

        if self.search_mode == 'dissimilarity':
            embedding_to_add = self._augment_for_database(embedding)
        else:
            embedding_to_add = embedding

        faiss_id = self.next_id
        try:
             embedding_to_add = np.ascontiguousarray(embedding_to_add, dtype=np.float32)
             self.index.add(embedding_to_add)
             self.metadata_map[faiss_id] = (chain_id, thought_index, text_segment)
             self.original_embeddings[faiss_id] = embedding[0].copy()  # Store the original embedding
             self.next_id += 1
        except Exception as e:
             logger.exception(f"[red]FAISS add failed for chain={chain_id}, thought={thought_index}: {e}[/red]")

    def search_farthest_neighbor(self, query_embedding: np.ndarray, query_chain_id: str) -> Optional[Tuple[float, str, int, str]]:
        """
        Searches for the farthest neighbor (maximum L2 distance)
        EXCLUDING embeddings from the query_chain_id.
        Returns (l2_distance, neighbor_chain_id, neighbor_thought_idx, neighbor_text).
        """
        if self.search_mode != 'dissimilarity':
            logger.error("`search_farthest_neighbor` called but manager not in 'dissimilarity' mode.")
            return None
        if self.index.ntotal == 0:
            return None
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(1, -1)

        # Augment query and calculate its norm for distance correction
        augmented_query = self._augment_for_query(query_embedding)
        query_norm_sq = (query_embedding ** 2).sum()

        k = min(max(1, self.index.ntotal), 10)
        try:
            # D_aug is the augmented inner product: -2<q,x> + ||x||^2
            D_aug, I = self.index.search(augmented_query, k)
        except Exception as e:
            logger.exception(f"[red]FAISS farthest neighbor search failed: {e}[/red]")
            return None

        for i in range(I.shape[1]):
            faiss_id = I[0, i]
            if faiss_id == -1: continue

            if faiss_id in self.metadata_map:
                neighbor_chain_id, neighbor_thought_idx, neighbor_text = self.metadata_map[faiss_id]
                if neighbor_chain_id != query_chain_id:
                    # Correct the distance: L2_sq = ||q-x||^2 = ||q||^2 - 2<q,x> + ||x||^2
                    # The augmented search returns D_aug = -2<q,x> + ||x||^2
                    # So, L2_sq = ||q||^2 + D_aug
                    l2_dist_sq = query_norm_sq + D_aug[0, i]
                    # Return the actual L2 distance, not squared
                    l2_distance = np.sqrt(max(0, l2_dist_sq)) # max(0,...) for float stability
                    return float(l2_distance), neighbor_chain_id, neighbor_thought_idx, neighbor_text

        return None # No valid neighbor from a different chain found
    
    def search_farthest_neighbor_return_sim_score(self, query_embedding: np.ndarray, query_chain_id: str) -> Optional[Tuple[float, str, int, str]]:
        """
        Searches for the farthest neighbor (maximum L2 distance)
        EXCLUDING embeddings from the query_chain_id.
        Returns (cosine_similarity, neighbor_chain_id, neighbor_thought_idx, neighbor_text).
        """
        if self.search_mode != 'dissimilarity':
            logger.error("`search_farthest_neighbor` called but manager not in 'dissimilarity' mode.")
            return None
        if self.index.ntotal == 0:
            return None
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(1, -1)

        # Augment query and calculate its norm for distance correction
        augmented_query = self._augment_for_query(query_embedding)
        query_norm_sq = (query_embedding ** 2).sum()

        k = min(max(1, self.index.ntotal), 10)
        try:
            D_aug, I = self.index.search(augmented_query, k)
        except Exception as e:
            logger.exception(f"[red]FAISS farthest neighbor search failed: {e}[/red]")
            return None

        for i in range(I.shape[1]):
            faiss_id = I[0, i]
            if faiss_id == -1: continue

            if faiss_id in self.metadata_map:
                neighbor_chain_id, neighbor_thought_idx, neighbor_text = self.metadata_map[faiss_id]
                if neighbor_chain_id != query_chain_id:
                    # Get the original embedding of the neighbor
                    neighbor_embedding = self.original_embeddings[faiss_id]
                    # Compute cosine similarity
                    q = query_embedding[0]
                    n = neighbor_embedding
                    cosine_sim = float(np.dot(q, n) / (np.linalg.norm(q) * np.linalg.norm(n) + 1e-8))
                    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
                    cosine_sim = (cosine_sim + 1) / 2  # Rescale to [0, 1]
                    return cosine_sim, neighbor_chain_id, neighbor_thought_idx, neighbor_text

        return None  # No valid neighbor from a different chain found

    def search_nearest_neighbor(self, query_embedding: np.ndarray, query_chain_id: str) -> Optional[Tuple[float, str, int, str]]:
        """
        Searches for the nearest neighbor (highest cosine similarity)
        EXCLUDING embeddings from the query_chain_id.
        Returns (similarity_score, neighbor_chain_id, neighbor_thought_idx, neighbor_text).
        """
        if self.search_mode != 'similarity':
            logger.error("`search_nearest_neighbor` called but manager not in 'similarity' mode.")
            return None
        if self.index.ntotal == 0:
            logger.debug("FAISS index is empty, cannot search.")
            return None
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.original_dimension:
             logger.error(f"Query dim mismatch: expected {self.original_dimension}, got {query_embedding.shape[1]}. Skip search.")
             return None

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        k = min(max(1, self.index.ntotal), 10)
        if k == 0:
            logger.debug("FAISS index size is 0, cannot search.")
            return None

        try:
            D, I = self.index.search(query_embedding, k)
        except Exception as e:
            logger.exception(f"[red]FAISS search failed for query_chain_id={query_chain_id}: {e}[/red]")
            return None

        for i in range(I.shape[1]):
            faiss_id = I[0, i]
            if faiss_id == -1: continue
            if faiss_id in self.metadata_map:
                neighbor_chain_id, neighbor_thought_idx, neighbor_text = self.metadata_map[faiss_id]
                if neighbor_chain_id != query_chain_id:
                    similarity_score = float(D[0, i])
                    similarity_score = np.clip(similarity_score, -1.0, 1.0)
                    return similarity_score, neighbor_chain_id, neighbor_thought_idx, neighbor_text
        return None

    def remove_chain_embeddings(self, chain_id_to_remove: str):
        """Removes all embeddings associated with a given chain ID."""
        ids_to_remove = [faiss_id for faiss_id, meta in list(self.metadata_map.items()) if meta[0] == chain_id_to_remove]

        if not ids_to_remove:
            return

        try:
            remove_selector = faiss.IDSelectorBatch(np.array(ids_to_remove, dtype='int64'))
            self.index.remove_ids(remove_selector)

            for faiss_id in ids_to_remove:
                 if faiss_id in self.metadata_map:
                      del self.metadata_map[faiss_id]

            logger.info(f"Removed embeddings from FAISS for pruned chain {chain_id_to_remove}. "
                        f"({len(ids_to_remove)} embeddings requested for removal)")
        except Exception as e:
             logger.exception(f"[red]Failed to remove embeddings for chain {chain_id_to_remove}: {e}[/red]")


    def get_num_embeddings(self) -> int:
        """Returns the number of vectors currently in the FAISS index."""
        return self.index.ntotal

    def get_num_active_chains(self) -> int:
        """
        Returns the number of unique chain IDs present in the metadata map.
        This reflects the number of chains whose embeddings we are actively considering
        for similarity comparisons.
        """
        return len(set(meta[0] for meta in self.metadata_map.values()))