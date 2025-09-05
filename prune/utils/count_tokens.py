# prune/utils/count_tokens.py

import transformers
from typing import Optional

import logging

logger = logging.getLogger(__name__)

_tokenizer = None
_tokenizer_path_loaded = None

def count_tokens(text: Optional[str], tokenizer_path: Optional[str] = None) -> Optional[int]:
    """
    Counts tokens using HuggingFace tokenizer. Loads tokenizer on first call or if path changes.

    Args:
        text (Optional[str]): The text to tokenize.
        tokenizer_path (Optional[str]): Path to the HuggingFace tokenizer directory. This must be provided to count tokens.

    Returns:
        Optional[int]: The number of tokens, or None if tokenization fails or tokenizer is unavailable.
    """
    global _tokenizer, _tokenizer_path_loaded
    if text is None or text == "":
        return 0

    # A tokenizer path must be provided to count tokens for non-empty text.
    if tokenizer_path is None:
        return None

    # Load or reload tokenizer if the requested path is different from the loaded one, or if not yet loaded.
    if tokenizer_path != _tokenizer_path_loaded or _tokenizer is None:
        try:
            _tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            _tokenizer_path_loaded = tokenizer_path
        except Exception as e:
            logger.exception(f"[red]ERROR: Failed to load tokenizer from {tokenizer_path}. Token counting disabled.[/red]")
            _tokenizer = None
            _tokenizer_path_loaded = None
            return None # Indicate failure

    # At this point, a tokenizer for the given path should be loaded.
    if _tokenizer:
        try:
            tokens = _tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            logger.exception(f"[red]ERROR: Failed to encode text with tokenizer for counting[/red]")
            return None # Indicate failure
    else:
        # Fallback in case loading silently failed to assign _tokenizer
        return None