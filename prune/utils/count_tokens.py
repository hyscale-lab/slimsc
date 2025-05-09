import transformers
from typing import Optional

import logging

logger = logging.getLogger(__name__)

# --- Optional: Token counting utility (Ensure tokenizer_path is handled) ---
_tokenizer = None
_tokenizer_path_loaded = None

def count_tokens(text: str, tokenizer_path: Optional[str] = None) -> Optional[int]:
        """
        Counts tokens using Hugging Face tokenizer. Loads tokenizer on first call or if path changes.

        Args:
            text (str): The text to tokenize.
            tokenizer_path (Optional[str]): Path to the Hugging Face tokenizer directory.

        Returns:
            Optional[int]: The number of tokens, or None if tokenization fails or tokenizer is unavailable.
        """
        global _tokenizer, _tokenizer_path_loaded
        if not text:
            return 0

        # Load or reload tokenizer if path is provided and different from loaded one, or if not loaded yet
        if tokenizer_path and (tokenizer_path != _tokenizer_path_loaded or _tokenizer is None):
            try:
                # logger.info(f"Loading tokenizer from {tokenizer_path}...") # Verbose
                _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
                _tokenizer_path_loaded = tokenizer_path
                # logger.info("Tokenizer loaded.") # Verbose
            except Exception as e:
                logger.exception(f"[red]ERROR: Failed to load tokenizer from {tokenizer_path}. Token counting disabled.[/red]")
                _tokenizer = None
                _tokenizer_path_loaded = None
                return None # Indicate failure

        # If tokenizer_path was not provided at all, _tokenizer remains None.
        # If loading failed, _tokenizer is None.
        if _tokenizer:
            try:
                # Encode the text. Handle potential errors during encoding.
                # add_special_tokens=False is typical for counting content tokens
                tokens = _tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                logger.exception(f"[red]ERROR: Failed to encode text with tokenizer for counting[/red]")
                return None # Indicate failure
        else:
            # Only print a warning about tokenizer not being available if the path was expected (provided)
            # If tokenizer_path is None, the caller likely knew counting wasn't possible.
            # If loading failed previously (_tokenizer_path_loaded is not None but _tokenizer is None),
            # a warning was already printed on the first attempt.
            # Avoid repeated warnings here during every count call.
            # logger.debug("Tokenizer not available. Cannot count tokens.")
            return None # Indicate unavailability/failure


