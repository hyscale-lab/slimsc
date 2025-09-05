import pytest
import logging
from unittest.mock import MagicMock, call

# Assuming the file is located at prune/utils/count_tokens.py
# Adjust the import path if your project structure is different.
from prune.utils.count_tokens import count_tokens
import prune.utils.count_tokens as ct_module # Import module to modify globals

# Pytest fixture to ensure a clean state for each test by resetting the global tokenizer
@pytest.fixture(autouse=True)
def reset_tokenizer_globals():
    """
    This fixture runs automatically before each test. It resets the global
    tokenizer and its path to ensure that tests are isolated and do not
    interfere with each other's state.
    """
    # Setup: Reset globals before the test runs
    ct_module._tokenizer = None
    ct_module._tokenizer_path_loaded = None
    yield
    # Teardown: Optional, but good practice to clean up after.
    # The setup part already handles isolation for the next test.
    ct_module._tokenizer = None
    ct_module._tokenizer_path_loaded = None


def test_count_tokens_with_none_or_empty_text():
    """
    Test that count_tokens returns 0 for None or empty string inputs,
    without attempting to load a tokenizer.
    """
    assert count_tokens(None, tokenizer_path="any/path") == 0
    assert count_tokens("", tokenizer_path="any/path") == 0


def test_count_tokens_with_no_tokenizer_path():
    """
    Test that count_tokens returns None when text is provided but
    the tokenizer_path is not.
    """
    assert count_tokens("some sample text", tokenizer_path=None) is None


def test_count_tokens_happy_path(mocker):
    """
    Test the standard successful token counting scenario. Mocks the tokenizer.
    """
    # Mock the tokenizer and its encode method
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [101, 2535, 2017, 102]  # A list of 4 tokens
    
    # Mock the from_pretrained class method to return our mock tokenizer
    mock_from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )

    text = "hello world"
    tokenizer_path = "gpt2"
    
    # First call should load the tokenizer and return the count
    token_count = count_tokens(text, tokenizer_path)
    
    assert token_count == 4
    # Verify the tokenizer was loaded exactly once with the correct path
    mock_from_pretrained.assert_called_once_with(tokenizer_path, trust_remote_code=True)
    # Verify the text was encoded
    mock_tokenizer.encode.assert_called_once_with(text, add_special_tokens=False)


def test_count_tokens_tokenizer_caching(mocker):
    """
    Test that the tokenizer is cached and not reloaded on subsequent calls
    with the same path.
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )

    # First call
    count_tokens("first call", "bert-base-uncased")
    
    # Second call with the same tokenizer path
    count_tokens("second call", "bert-base-uncased")

    # The tokenizer should only be loaded once
    mock_from_pretrained.assert_called_once()


def test_count_tokens_reloads_on_path_change(mocker):
    """
    Test that the tokenizer is reloaded if the tokenizer_path changes.
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_from_pretrained = mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )

    path1 = "model/path/v1"
    path2 = "model/path/v2"

    # First call with the first path
    count_tokens("text for model 1", path1)
    
    # Second call with a different path
    count_tokens("text for model 2", path2)

    # The from_pretrained method should have been called twice
    assert mock_from_pretrained.call_count == 2
    # Verify it was called with the correct paths in order
    mock_from_pretrained.assert_has_calls([
        call(path1, trust_remote_code=True),
        call(path2, trust_remote_code=True)
    ])


def test_count_tokens_tokenizer_loading_failure(mocker, caplog):
    """
    Test that the function handles exceptions during tokenizer loading,
    returns None, and logs an error.
    """
    # Configure the mock to raise an exception
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        side_effect=OSError("Model not found")
    )
    
    invalid_path = "non/existent/path"
    
    with caplog.at_level(logging.ERROR):
        result = count_tokens("some text", invalid_path)

    # The function should return None to indicate failure
    assert result is None
    # Verify an error was logged
    assert "ERROR: Failed to load tokenizer" in caplog.text
    assert invalid_path in caplog.text


def test_count_tokens_encoding_failure(mocker, caplog):
    """
    Test that the function handles exceptions during the encoding process,
    returns None, and logs an error.
    """
    # Mock the tokenizer and make its encode method raise an error
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = ValueError("Encoding failed")
    
    mocker.patch(
        "transformers.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer
    )

    tokenizer_path = "gpt2"

    with caplog.at_level(logging.ERROR):
        result = count_tokens("some text that will fail to encode", tokenizer_path)

    # The function should return None to indicate failure
    assert result is None
    # Verify an error was logged
    assert "ERROR: Failed to encode text with tokenizer for counting" in caplog.text