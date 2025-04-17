# slimsc/prune/utils/similarity_utils.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging
import torch
from transformers import AutoTokenizer # Added import

logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
MIN_SEGMENT_TOKENS = 25 # Define minimum thought length (tokens)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Global Caches ---
_embedding_model = None
_tokenizer_for_utils = None
_tokenizer_path_loaded_for_utils = None

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

def _load_tokenizer_for_utils(tokenizer_path: str) -> AutoTokenizer:
    """Loads tokenizer or returns cached one for utility functions."""
    global _tokenizer_for_utils, _tokenizer_path_loaded_for_utils
    if not tokenizer_path:
        raise ValueError("Tokenizer path must be provided for thought segmentation.")
    if _tokenizer_for_utils is None or tokenizer_path != _tokenizer_path_loaded_for_utils:
        logger.info(f"[Utils] Loading tokenizer from: {tokenizer_path}")
        try:
            _tokenizer_for_utils = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            _tokenizer_path_loaded_for_utils = tokenizer_path
            logger.info("[Utils] Tokenizer loaded.")
        except Exception as e:
            logger.exception(f"[red]Failed to load tokenizer in utils from {tokenizer_path}[/red]")
            _tokenizer_for_utils = None
            _tokenizer_path_loaded_for_utils = None
            raise e # Critical failure
    return _tokenizer_for_utils


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
        return embeddings.astype(np.float32)
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
    # Search for target phrases starting from the second character
    for phrase in target_phrases:
        start_index = 1
        while True:
            pos = text.find(phrase, start_index)
            if pos == -1:
                break # Phrase not found (anymore) in the remaining text

            # Check context: preceded by whitespace, specific punctuation, or newline.
            # This helps avoid matching phrases mid-word or in unrelated contexts.
            is_likely_start = False
            if pos > 0:
                preceding_char = text[pos - 1]
                if preceding_char.isspace() or preceding_char in '."\')':
                    is_likely_start = True
                # Check for preceding newline specifically
                elif preceding_char == '\n':
                     is_likely_start = True
                     # Optional: Check for double newline for stronger signal?
                     # if pos > 1 and text[pos - 2] == '\n': is_likely_start = True

            if is_likely_start:
                found_positions.append(pos)

            # Move search start past the current find to avoid overlapping matches of same phrase
            start_index = pos + len(phrase)

    # Combine, sort, and remove duplicates
    if found_positions:
        unique_sorted_positions = sorted(list(set(found_positions)))
        boundaries.extend(unique_sorted_positions)

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

    Returns:
        Tuple: (new_segments, updated_processed_boundaries)
        - new_segments: List of (start_idx, end_idx, segment_text) for valid new thoughts.
        - updated_processed_boundaries: The list of start indices including newly processed ones.
    """
    if not full_text:
        return [], processed_boundaries

    try:
        tokenizer = _load_tokenizer_for_utils(tokenizer_path)
    except Exception:
        logger.error("[red]Tokenizer unavailable in find_newly_completed_thoughts.[/red]")
        return [], processed_boundaries # Cannot proceed without tokenizer

    # 1. Find all potential boundaries in the current full text
    all_current_boundaries = find_thought_boundaries(full_text, target_phrases)

    # 2. Identify which boundaries define the *end* of a newly completed segment
    newly_completed_segments = []
    newly_processed_starts = [] # Track starts processed in this call

    # Ensure processed_boundaries is sorted and unique for efficient checking
    current_processed_boundaries = sorted(list(set(processed_boundaries)))
    last_processed_start = current_processed_boundaries[-1] if current_processed_boundaries else -1

    for i, boundary_start in enumerate(all_current_boundaries):
        # A segment is defined between boundary_start (all_current_boundaries[i])
        # and the *next* boundary (all_current_boundaries[i+1])
        if i + 1 < len(all_current_boundaries):
            boundary_end = all_current_boundaries[i+1]

            # Check if this segment's START boundary is new relative to what we've processed
            # We process a segment once its start boundary appears and its *end* boundary is also found.
            # This means we look for segments where `boundary_start` is >= the last processed start,
            # AND `boundary_start` itself has not been added to `newly_processed_starts` in *this* function call yet.
            if boundary_start >= last_processed_start and boundary_start not in newly_processed_starts:
                 # Check if this segment start was already processed in previous calls
                 if boundary_start not in current_processed_boundaries:
                    segment_text = full_text[boundary_start:boundary_end].strip()

                    if not segment_text: # Skip empty segments
                         continue

                    # 3. Check token length
                    try:
                        tokens = tokenizer.encode(segment_text, add_special_tokens=False)
                        num_tokens = len(tokens)
                    except Exception as e:
                        logger.warning(f"Could not tokenize segment [{boundary_start}:{boundary_end}] for length check: {e}. Skipping.")
                        continue # Skip segment if tokenization fails

                    if num_tokens >= min_segment_tokens:
                        logger.debug(f"Found NEW completed thought segment [{boundary_start}:{boundary_end}], tokens={num_tokens}.") # Text: '{segment_text[:80]}...'")
                        newly_completed_segments.append((boundary_start, boundary_end, segment_text))
                        newly_processed_starts.append(boundary_start) # Mark this start as processed now
                    # else: logger.debug(f"New segment [{boundary_start}:{boundary_end}] too short (tokens={num_tokens}).")

    # Combine old and newly processed start boundaries
    updated_processed_boundaries = sorted(list(set(current_processed_boundaries + newly_processed_starts)))

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

    start_pos = 0 # Default if no boundaries were ever processed
    if processed_boundaries:
        start_pos = processed_boundaries[-1] # Start from the beginning of the last processed thought

    end_pos = len(full_text)
    segment_text = full_text[start_pos:].strip()

    if not segment_text:
        return None

    try:
        tokenizer = _load_tokenizer_for_utils(tokenizer_path)
        tokens = tokenizer.encode(segment_text, add_special_tokens=False)
        num_tokens = len(tokens)
    except Exception as e:
        logger.warning(f"Could not tokenize final segment [{start_pos}:{end_pos}] for length check: {e}. Skipping.")
        return None

    if num_tokens >= min_segment_tokens:
        logger.debug(f"Found FINAL thought segment [{start_pos}:{end_pos}], tokens={num_tokens}.") # Text: '{segment_text[:80]}...'")
        return (start_pos, end_pos, segment_text)
    else:
        logger.debug(f"Final segment [{start_pos}:{end_pos}] too short (tokens={num_tokens}).")
        return None

# --- FaissIndexManager Class (Keep as is, using 'thought_index') ---
class FaissIndexManager:
    """Manages a FAISS index for similarity checking during pruning."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # IP for normalized embeddings = Cosine
        # Map FAISS vector IDs to (chain_id, thought_index, text_segment)
        self.metadata_map: Dict[int, Tuple[str, int, str]] = {}
        self.next_id = 0

    def add_embedding(self, embedding: np.ndarray, chain_id: str, thought_index: int, text_segment: str):
        """Adds a single embedding and its metadata to the index."""
        if embedding.ndim == 1: embedding = embedding.reshape(1, -1)
        if embedding.shape[1] != self.dimension:
             logger.error(f"Embedding dim mismatch: expected {self.dimension}, got {embedding.shape[1]}. Skip add.")
             return

        faiss_id = self.next_id
        self.index.add(embedding.astype(np.float32))
        self.metadata_map[faiss_id] = (chain_id, thought_index, text_segment)
        self.next_id += 1
        # logger.debug(f"Added faiss_id={faiss_id} for chain={chain_id}, thought={thought_index}")

    def search_nearest_neighbor(self, query_embedding: np.ndarray, query_chain_id: str) -> Optional[Tuple[float, str, int, str]]:
        """Searches for nearest neighbor EXCLUDING the query_chain_id."""
        if self.index.ntotal == 0: return None
        if query_embedding.ndim == 1: query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape[1] != self.dimension:
             logger.error(f"Query dim mismatch: expected {self.dimension}, got {query_embedding.shape[1]}. Skip search.")
             return None

        k = min(10, self.index.ntotal) # Search more neighbors to increase chance of finding one from another chain
        try:
            D, I = self.index.search(query_embedding.astype(np.float32), k)
        except Exception as e:
            logger.exception(f"[red]FAISS search failed: {e}[/red]")
            return None

        for i in range(I.shape[1]): # Iterate through returned neighbors
            faiss_id = I[0, i]
            if faiss_id == -1 or faiss_id >= self.next_id: continue # Invalid index
            if faiss_id in self.metadata_map:
                neighbor_chain_id, neighbor_thought_idx, neighbor_text = self.metadata_map[faiss_id]
                if neighbor_chain_id != query_chain_id:
                    similarity_score = float(D[0, i])
                    # Clamp score just in case of float issues with normalized vectors
                    similarity_score = np.clip(similarity_score, -1.0, 1.0)
                    # logger.debug(f"NN for {query_chain_id}: score={similarity_score:.4f}, neighbor={neighbor_chain_id} (thought {neighbor_thought_idx})")
                    return similarity_score, neighbor_chain_id, neighbor_thought_idx, neighbor_text
            else:
                 logger.warning(f"FAISS ID {faiss_id} in search result but not in metadata_map.")

        # logger.debug(f"No valid neighbor found for {query_chain_id} from other chains.")
        return None # No neighbor found from a different chain

    def remove_chain_embeddings(self, chain_id_to_remove: str):
        """Removes all embeddings associated with a given chain ID."""
        ids_to_remove = [faiss_id for faiss_id, meta in self.metadata_map.items() if meta[0] == chain_id_to_remove]
        if not ids_to_remove: return

        try:
            # Using list of IDs directly for removal selector
            remove_selector = faiss.IDSelectorBatch(np.array(ids_to_remove, dtype='int64'))
            num_removed = self.index.remove_ids(remove_selector)
            # If num_removed != len(ids_to_remove): logger.warning("FAISS remove_ids count mismatch") # Optional check

            logger.info(f"Removed {num_removed} embeddings from FAISS for pruned chain {chain_id_to_remove}.")
            # Update metadata map safely
            current_keys = list(self.metadata_map.keys())
            for faiss_id in current_keys:
                 if self.metadata_map[faiss_id][0] == chain_id_to_remove:
                      del self.metadata_map[faiss_id]

        except Exception as e:
             logger.exception(f"[red]Failed to remove embeddings for chain {chain_id_to_remove}: {e}[/red]")
             # Consider implications: metadata might be out of sync with index


    def get_num_embeddings(self) -> int:
        return self.index.ntotal

    def get_num_active_chains(self) -> int:
        # Returns number of unique chains currently represented in the index metadata
        return len(set(meta[0] for meta in self.metadata_map.values()))