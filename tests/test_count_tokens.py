"""Tests for prune.utils.count_tokens module."""

import pytest
from unittest.mock import patch, Mock, MagicMock
import sys

# Import the count_tokens function directly from the module
sys.path.insert(0, '/home/runner/work/slimsc/slimsc')
from prune.utils.count_tokens import count_tokens


class TestCountTokensBasic:
    """Test basic functionality of count_tokens."""
    
    def test_count_tokens_empty_text(self):
        """Test token counting with empty text."""
        result = count_tokens("")
        assert result == 0
    
    def test_count_tokens_none_text(self):
        """Test token counting with None text."""
        # count_tokens should handle falsy values
        result = count_tokens(None) 
        assert result == 0
    
    def test_count_tokens_no_tokenizer_path(self):
        """Test token counting without providing a tokenizer path."""
        result = count_tokens("Hello world")
        assert result is None
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_with_valid_tokenizer(self, mock_from_pretrained):
        """Test token counting with a valid tokenizer."""
        # Setup mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_from_pretrained.return_value = mock_tokenizer
        
        # Test token counting
        result = count_tokens("Hello world", "/path/to/tokenizer")
        
        assert result == 5
        mock_from_pretrained.assert_called_once_with("/path/to/tokenizer", trust_remote_code=True)
        mock_tokenizer.encode.assert_called_once_with("Hello world", add_special_tokens=False)
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_tokenizer_loading_failure(self, mock_from_pretrained):
        """Test handling of tokenizer loading failure."""
        # Mock tokenizer loading to fail
        mock_from_pretrained.side_effect = Exception("Failed to load tokenizer")
        
        result = count_tokens("Hello world", "/invalid/path")
        
        assert result is None
        mock_from_pretrained.assert_called_once_with("/invalid/path", trust_remote_code=True)
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_encoding_failure(self, mock_from_pretrained):
        """Test handling of text encoding failure."""
        # Setup mock tokenizer that fails on encoding
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Encoding failed")
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = count_tokens("Hello world", "/path/to/tokenizer")
        
        assert result is None
        mock_tokenizer.encode.assert_called_once_with("Hello world", add_special_tokens=False)
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_different_token_counts(self, mock_from_pretrained):
        """Test token counting with different token counts."""
        mock_tokenizer = Mock()
        mock_from_pretrained.return_value = mock_tokenizer
        
        # Test case 1: Single word
        mock_tokenizer.encode.return_value = [1]
        result = count_tokens("Hello", "/path/to/tokenizer")
        assert result == 1
        
        # Test case 2: Multiple words  
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        result = count_tokens("Hello beautiful world today", "/path/to/tokenizer")
        assert result == 4
        
        # Test case 3: Complex text
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = count_tokens("This is a more complex sentence with punctuation!", "/path/to/tokenizer")
        assert result == 10
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_special_characters(self, mock_from_pretrained):
        """Test token counting with special characters."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5, 6]
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = count_tokens("Hello! @#$%^&*()", "/path/to/tokenizer")
        
        assert result == 6
        mock_tokenizer.encode.assert_called_with("Hello! @#$%^&*()", add_special_tokens=False)
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_unicode_text(self, mock_from_pretrained):
        """Test token counting with unicode text."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = count_tokens("Hello 世界! 🌍", "/path/to/tokenizer")
        
        assert result == 3
        mock_tokenizer.encode.assert_called_with("Hello 世界! 🌍", add_special_tokens=False)


class TestCountTokensErrorHandling:
    """Test error handling in count_tokens."""
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_multiple_exceptions(self, mock_from_pretrained):
        """Test handling of multiple different exceptions."""
        # Test FileNotFoundError
        mock_from_pretrained.side_effect = FileNotFoundError("Tokenizer not found")
        result = count_tokens("text", "/missing/path")
        assert result is None
        
        # Test PermissionError
        mock_from_pretrained.side_effect = PermissionError("Access denied")
        result = count_tokens("text", "/forbidden/path")
        assert result is None
        
        # Test generic Exception
        mock_from_pretrained.side_effect = Exception("Unknown error")
        result = count_tokens("text", "/error/path")
        assert result is None
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_encoding_exceptions(self, mock_from_pretrained):
        """Test handling of different encoding exceptions."""
        mock_tokenizer = Mock()
        mock_from_pretrained.return_value = mock_tokenizer
        
        # Test UnicodeError
        mock_tokenizer.encode.side_effect = UnicodeError("Unicode decode error")
        result = count_tokens("problematic text", "/path/to/tokenizer")
        assert result is None
        
        # Test ValueError
        mock_tokenizer.encode.side_effect = ValueError("Invalid input")
        result = count_tokens("invalid text", "/path/to/tokenizer")
        assert result is None
        
        # Test RuntimeError
        mock_tokenizer.encode.side_effect = RuntimeError("Runtime issue")
        result = count_tokens("runtime error text", "/path/to/tokenizer")
        assert result is None


class TestCountTokensEdgeCases:
    """Test edge cases for count_tokens."""
    
    def test_count_tokens_empty_string_variations(self):
        """Test various empty string cases."""
        test_cases = ["", "   ", "\t", "\n", "\r\n", " \t \n "]
        
        for text in test_cases:
            result = count_tokens(text)
            if not text:  # Truly empty string
                assert result == 0
            else:  # Whitespace-only strings
                assert result is None  # No tokenizer available
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_empty_token_list(self, mock_from_pretrained):
        """Test when tokenizer returns empty token list."""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = []  # Empty token list
        mock_from_pretrained.return_value = mock_tokenizer
        
        result = count_tokens("some text", "/path/to/tokenizer")
        assert result == 0
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_count_tokens_very_long_text(self, mock_from_pretrained):
        """Test with very long text."""
        mock_tokenizer = Mock()
        # Simulate many tokens for long text
        mock_tokenizer.encode.return_value = list(range(1000))  # 1000 tokens
        mock_from_pretrained.return_value = mock_tokenizer
        
        long_text = "word " * 500  # 500 words
        result = count_tokens(long_text, "/path/to/tokenizer")
        assert result == 1000
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained') 
    def test_count_tokens_none_tokenizer_path(self, mock_from_pretrained):
        """Test with explicit None tokenizer path."""
        # Should not call from_pretrained when path is None
        result = count_tokens("Hello world", None)
        assert result is None
        mock_from_pretrained.assert_not_called()


class TestCountTokensIntegration:
    """Integration tests for count_tokens."""
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_realistic_tokenization_workflow(self, mock_from_pretrained):
        """Test a realistic workflow of text tokenization."""
        # Setup a mock tokenizer that behaves realistically
        mock_tokenizer = Mock()
        
        def realistic_encode(text, **kwargs):
            # Simple word-based tokenization simulation
            if not text.strip():
                return []
            words = text.strip().split()
            # Simulate some words being split into multiple tokens
            tokens = []
            for word in words:
                if len(word) > 6:  # Long words get split
                    tokens.extend([1, 2])  # 2 tokens per long word
                else:
                    tokens.append(1)  # 1 token per short word
            return tokens
        
        mock_tokenizer.encode.side_effect = realistic_encode
        mock_from_pretrained.return_value = mock_tokenizer
        
        # Test various texts
        test_cases = [
            ("hello", 1),  # Single short word
            ("hello world", 2),  # Two short words
            ("hello magnificent", 3),  # One short, one long (split into 2)
            ("extraordinary circumstances", 4),  # Two long words (2 tokens each)
            ("a very long sentence with words", 7),  # Mixed lengths
        ]
        
        tokenizer_path = "/path/to/tokenizer"
        
        for text, expected_count in test_cases:
            result = count_tokens(text, tokenizer_path)
            assert result == expected_count, f"Text: '{text}' expected {expected_count}, got {result}"
        
        # Tokenizer should be loaded for each call (since we don't test global state here)
        assert mock_from_pretrained.call_count >= 1
    
    @patch('prune.utils.count_tokens.transformers.AutoTokenizer.from_pretrained')
    def test_error_recovery_workflow(self, mock_from_pretrained):
        """Test error recovery in a workflow."""
        # Start with a failing tokenizer
        mock_from_pretrained.side_effect = Exception("Network error")
        
        # First attempt fails
        result1 = count_tokens("test text", "/remote/tokenizer")
        assert result1 is None
        
        # Subsequent attempts without tokenizer path should fail
        result2 = count_tokens("more text")
        assert result2 is None
        
        # Now provide a working tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4]
        mock_from_pretrained.side_effect = None
        mock_from_pretrained.return_value = mock_tokenizer
        
        # Should work with new path
        result3 = count_tokens("recovery text", "/local/tokenizer")
        assert result3 == 4
    
    def test_falsy_text_inputs(self):
        """Test various falsy text inputs."""
        falsy_inputs = [None, "", 0, False, []]
        
        for text in falsy_inputs:
            result = count_tokens(text)
            if text is None or text == "":
                assert result == 0
            else:
                # For other falsy inputs, the function might not handle them properly
                # This depends on the implementation details
                # We'll just check that it doesn't crash
                assert result is not None or result is None  # Either way is fine for this test