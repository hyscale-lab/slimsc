"""Tests for prune.clients.vllm_client module."""

import pytest
import asyncio
import aiohttp
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from aiohttp import ClientError, ClientTimeout

from prune.clients.vllm_client import (
    get_aiohttp_session,
    close_aiohttp_session,
    stream_vllm_request,
    process_stream_chunks
)


class TestGetAiohttpSession:
    """Test the get_aiohttp_session function."""
    
    @pytest.mark.asyncio
    async def test_get_aiohttp_session_creates_new(self):
        """Test creating new session when none exists."""
        # Reset global session
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = None
        
        with patch('prune.clients.vllm_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = Mock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            session = await get_aiohttp_session()
            
            assert session == mock_session
            mock_session_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_aiohttp_session_returns_cached(self):
        """Test returning cached session when available."""
        # Set up cached session
        mock_session = Mock()
        mock_session.closed = False
        
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = mock_session
        
        with patch('prune.clients.vllm_client.aiohttp.ClientSession') as mock_session_class:
            session = await get_aiohttp_session()
            
            assert session == mock_session
            mock_session_class.assert_not_called()  # Should use cached
    
    @pytest.mark.asyncio
    async def test_get_aiohttp_session_recreates_closed(self):
        """Test recreating session when existing one is closed."""
        # Set up closed session
        mock_old_session = Mock()
        mock_old_session.closed = True
        
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = mock_old_session
        
        with patch('prune.clients.vllm_client.aiohttp.ClientSession') as mock_session_class:
            mock_new_session = Mock()
            mock_new_session.closed = False
            mock_session_class.return_value = mock_new_session
            
            session = await get_aiohttp_session()
            
            assert session == mock_new_session
            mock_session_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_aiohttp_session_timeout_config(self):
        """Test that session is created with correct timeout configuration."""
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = None
        
        with patch('prune.clients.vllm_client.aiohttp.ClientSession') as mock_session_class:
            with patch('prune.clients.vllm_client.aiohttp.ClientTimeout') as mock_timeout:
                with patch('prune.clients.vllm_client.aiohttp.TCPConnector') as mock_connector:
                    
                    await get_aiohttp_session()
                    
                    # Verify timeout configuration
                    mock_timeout.assert_called_once_with(
                        total=None,
                        connect=60,
                        sock_connect=60,
                        sock_read=3600
                    )
                    
                    # Verify connector configuration
                    mock_connector.assert_called_once_with(limit_per_host=100)
                    
                    # Verify session creation
                    mock_session_class.assert_called_once()


class TestCloseAiohttpSession:
    """Test the close_aiohttp_session function."""
    
    @pytest.mark.asyncio
    async def test_close_aiohttp_session_existing(self):
        """Test closing existing session."""
        mock_session = AsyncMock()
        
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = mock_session
        
        await close_aiohttp_session()
        
        mock_session.close.assert_called_once()
        assert prune.clients.vllm_client._aiohttp_session is None
    
    @pytest.mark.asyncio
    async def test_close_aiohttp_session_none(self):
        """Test closing when no session exists."""
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = None
        
        # Should not raise exception
        await close_aiohttp_session()
        
        assert prune.clients.vllm_client._aiohttp_session is None


class TestStreamVllmRequest:
    """Test the stream_vllm_request function."""
    
    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = AsyncMock()
        with patch('prune.clients.vllm_client.get_aiohttp_session', return_value=session):
            yield session
    
    @pytest.mark.asyncio
    async def test_stream_vllm_request_success(self, mock_session):
        """Test successful streaming request."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        
        # Mock streaming content
        async def mock_content_iter():
            yield b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n'
            yield b'data: {"choices": [{"delta": {"content": " world"}}]}\n'
            yield b'data: [DONE]\n'
        
        mock_response.content = mock_content_iter()
        
        # Setup session post context manager
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Collect stream results
        chunks = []
        async for chunk in stream_vllm_request(
            prompt="Test prompt",
            vllm_url="http://localhost:8000",
            model_name="test_model",
            request_id="test_req_1"
        ):
            chunks.append(chunk)
        
        # Verify chunks
        assert len(chunks) == 2
        assert chunks[0]["choices"][0]["delta"]["content"] == "Hello"
        assert chunks[1]["choices"][0]["delta"]["content"] == " world"
        
        # Verify session was called correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "http://localhost:8000/v1/chat/completions"
        assert "json" in call_args[1]
        assert call_args[1]["json"]["model"] == "test_model"
        assert call_args[1]["json"]["stream"] == True
    
    @pytest.mark.asyncio
    async def test_stream_vllm_request_with_parameters(self, mock_session):
        """Test streaming request with various parameters."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        mock_response.content = iter([b'data: [DONE]\n'])
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Test with parameters
        stream_gen = stream_vllm_request(
            prompt="Test prompt",
            vllm_url="http://localhost:8000",
            model_name="test_model",
            request_id="test_req_1",
            temperature=0.8,
            max_tokens=1000,
            stop_sequences=["STOP", "END"],
            logprobs=5
        )
        
        # Consume generator
        chunks = [chunk async for chunk in stream_gen]
        
        # Verify request payload
        call_args = mock_session.post.call_args
        payload = call_args[1]["json"]
        
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 1000
        assert payload["stop"] == ["STOP", "END"]
        assert payload["logprobs"] == 1
        assert payload["top_logprobs"] == 5
    
    @pytest.mark.asyncio
    async def test_stream_vllm_request_http_error(self, mock_session):
        """Test handling of HTTP errors."""
        # Mock response that raises HTTP error
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=Mock(), history=(), status=500, message="Server Error"
        )
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Collect stream results
        chunks = []
        async for chunk in stream_vllm_request(
            prompt="Test prompt",
            vllm_url="http://localhost:8000",
            model_name="test_model",
            request_id="test_req_1",
            max_retries=1
        ):
            chunks.append(chunk)
        
        # Should get error chunk
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"]["status"] == "failed_after_retries"
    
    @pytest.mark.asyncio
    async def test_stream_vllm_request_retry_logic(self, mock_session):
        """Test retry logic on transient errors."""
        # Mock session to fail first time, succeed second time
        call_count = 0
        
        async def mock_post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            mock_ctx = AsyncMock()
            
            if call_count == 1:
                # First call fails
                mock_ctx.__aenter__.side_effect = aiohttp.ClientError("Connection failed")
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
            else:
                # Second call succeeds
                mock_response = AsyncMock()
                mock_response.raise_for_status = Mock()
                mock_response.content = iter([b'data: {"choices": [{"delta": {"content": "Success"}}]}\n', b'data: [DONE]\n'])
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
            
            return mock_ctx
        
        mock_session.post.side_effect = mock_post_side_effect
        
        # Collect stream results
        chunks = []
        async for chunk in stream_vllm_request(
            prompt="Test prompt",
            vllm_url="http://localhost:8000", 
            model_name="test_model",
            request_id="test_req_1",
            max_retries=2,
            initial_backoff=0.1  # Fast retry for testing
        ):
            chunks.append(chunk)
        
        # Should eventually succeed
        assert len(chunks) == 1
        assert chunks[0]["choices"][0]["delta"]["content"] == "Success"
        assert call_count == 2  # Failed once, succeeded second time
    
    @pytest.mark.asyncio
    async def test_stream_vllm_request_json_decode_error(self, mock_session):
        """Test handling of JSON decode errors in stream."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = Mock()
        
        # Mock content with invalid JSON
        async def mock_content_iter():
            yield b'data: {"valid": "json"}\n'
            yield b'data: invalid json here\n'
            yield b'data: {"more": "valid"}\n'
            yield b'data: [DONE]\n'
        
        mock_response.content = mock_content_iter()
        
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
        
        # Collect stream results
        chunks = []
        async for chunk in stream_vllm_request(
            prompt="Test prompt",
            vllm_url="http://localhost:8000",
            model_name="test_model",
            request_id="test_req_1"
        ):
            chunks.append(chunk)
        
        # Should get valid chunks, skip invalid JSON
        assert len(chunks) == 2
        assert chunks[0]["valid"] == "json"
        assert chunks[1]["more"] == "valid"
    
    @pytest.mark.asyncio
    async def test_stream_vllm_request_unexpected_error(self, mock_session):
        """Test handling of unexpected errors."""
        # Mock unexpected error
        mock_session.post.side_effect = RuntimeError("Unexpected error")
        
        # Collect stream results
        chunks = []
        async for chunk in stream_vllm_request(
            prompt="Test prompt",
            vllm_url="http://localhost:8000",
            model_name="test_model",
            request_id="test_req_1"
        ):
            chunks.append(chunk)
        
        # Should get error chunk
        assert len(chunks) == 1
        assert "error" in chunks[0]
        assert chunks[0]["error"]["status"] == "unexpected"
        assert "RuntimeError" in chunks[0]["error"]["message"]


class TestProcessStreamChunks:
    """Test the process_stream_chunks function."""
    
    def test_process_stream_chunks_basic(self):
        """Test basic chunk processing."""
        chunks = [
            {
                "choices": [{
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }]
            },
            {
                "choices": [{
                    "delta": {"content": " world"},
                    "finish_reason": None
                }]
            },
            {
                "choices": [{
                    "delta": {},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 5,
                    "total_tokens": 7
                }
            }
        ]
        
        result = process_stream_chunks(chunks, chain_index=1)
        
        assert result["full_content"] == "Hello world"
        assert result["completion_tokens"] == 2
        assert result["prompt_tokens"] == 5
        assert result["finish_reason"] == "stop"
        assert result["chain_index"] == 1
    
    def test_process_stream_chunks_with_reasoning(self):
        """Test chunk processing with reasoning/content separation."""
        chunks = [
            {
                "choices": [{
                    "delta": {"content": "Let me think about this..."},
                    "finish_reason": None
                }]
            },
            {
                "choices": [{
                    "delta": {"content": "\n\nThe answer is 42"},
                    "finish_reason": "stop"
                }],
                "usage": {"completion_tokens": 10}
            }
        ]
        
        result = process_stream_chunks(chunks, chain_index=0)
        
        assert result["full_content"] == "Let me think about this...\n\nThe answer is 42"
        assert result["reasoning_text"] == "Let me think about this..."
        assert result["final_answer_text"] == "\n\nThe answer is 42"
        assert result["completion_tokens"] == 10
    
    def test_process_stream_chunks_error_handling(self):
        """Test chunk processing with error chunks."""
        chunks = [
            {
                "choices": [{
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }]
            },
            {
                "error": {
                    "status": "timeout",
                    "message": "Request timed out"
                }
            }
        ]
        
        result = process_stream_chunks(chunks, chain_index=2)
        
        assert "error" in result
        assert result["error"]["status"] == "timeout"
        assert result["chain_index"] == 2
    
    def test_process_stream_chunks_empty_chunks(self):
        """Test processing empty chunk list."""
        result = process_stream_chunks([], chain_index=0)
        
        assert result["full_content"] == ""
        assert result["completion_tokens"] is None
        assert result["finish_reason"] is None
        assert result["chain_index"] == 0
    
    def test_process_stream_chunks_invalid_chunks(self):
        """Test processing with invalid/malformed chunks."""
        chunks = [
            None,  # Invalid chunk
            {},    # Empty chunk
            {"choices": []},  # No choices
            {
                "choices": [{
                    "delta": {"content": "Valid content"}
                }]
            },
            {"invalid": "structure"}  # Invalid structure
        ]
        
        result = process_stream_chunks(chunks, chain_index=1)
        
        # Should process only the valid chunk
        assert result["full_content"] == "Valid content"
        assert result["chain_index"] == 1
    
    def test_process_stream_chunks_multiple_choices(self):
        """Test processing chunks with multiple choices (should use first)."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": "First choice"},
                        "finish_reason": None
                    },
                    {
                        "delta": {"content": "Second choice"},
                        "finish_reason": None
                    }
                ]
            }
        ]
        
        result = process_stream_chunks(chunks, chain_index=0)
        
        # Should use first choice
        assert result["full_content"] == "First choice"
    
    def test_process_stream_chunks_usage_stats(self):
        """Test processing of usage statistics."""
        chunks = [
            {
                "choices": [{
                    "delta": {"content": "Content"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "completion_tokens": 15,
                    "prompt_tokens": 50,
                    "total_tokens": 65
                }
            }
        ]
        
        result = process_stream_chunks(chunks, chain_index=0)
        
        assert result["completion_tokens"] == 15
        assert result["prompt_tokens"] == 50
        assert result["total_tokens"] == 65
    
    def test_process_stream_chunks_finish_reasons(self):
        """Test handling of different finish reasons."""
        test_cases = [
            ("stop", "stop"),
            ("length", "length"),
            ("content_filter", "content_filter"),
            (None, None)
        ]
        
        for finish_reason, expected in test_cases:
            chunks = [
                {
                    "choices": [{
                        "delta": {"content": "test"},
                        "finish_reason": finish_reason
                    }]
                }
            ]
            
            result = process_stream_chunks(chunks, chain_index=0)
            assert result["finish_reason"] == expected


class TestVllmClientIntegration:
    """Integration tests for vLLM client functionality."""
    
    @pytest.mark.asyncio
    async def test_complete_streaming_workflow(self):
        """Test complete workflow from request to processed result."""
        with patch('prune.clients.vllm_client.get_aiohttp_session') as mock_get_session:
            # Setup mock session and response
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session
            
            mock_response = AsyncMock()
            mock_response.raise_for_status = Mock()
            
            # Mock realistic streaming response
            async def mock_content():
                yield b'data: {"choices": [{"delta": {"content": "The answer"}}]}\n'
                yield b'data: {"choices": [{"delta": {"content": " is 42"}}]}\n'
                yield b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"completion_tokens": 3, "prompt_tokens": 10}}\n'
                yield b'data: [DONE]\n'
            
            mock_response.content = mock_content()
            mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.post.return_value.__aexit__ = AsyncMock(return_value=False)
            
            # Stream the request
            chunks = []
            async for chunk in stream_vllm_request(
                prompt="What is the answer to life?",
                vllm_url="http://localhost:8000",
                model_name="test_model",
                request_id="integration_test"
            ):
                chunks.append(chunk)
            
            # Process the chunks
            result = process_stream_chunks(chunks, chain_index=0)
            
            # Verify end-to-end result
            assert result["full_content"] == "The answer is 42"
            assert result["completion_tokens"] == 3
            assert result["prompt_tokens"] == 10
            assert result["finish_reason"] == "stop"
            assert result["chain_index"] == 0
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test session creation and cleanup lifecycle."""
        # Ensure clean state
        import prune.clients.vllm_client
        prune.clients.vllm_client._aiohttp_session = None
        
        with patch('prune.clients.vllm_client.aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            # Get session (should create new)
            session1 = await get_aiohttp_session()
            assert session1 == mock_session
            mock_session_class.assert_called_once()
            
            # Get session again (should reuse)
            session2 = await get_aiohttp_session()
            assert session2 == mock_session
            assert mock_session_class.call_count == 1  # Still only called once
            
            # Close session
            await close_aiohttp_session()
            mock_session.close.assert_called_once()
            
            # Get session after close (should create new)
            mock_session_class.reset_mock()
            session3 = await get_aiohttp_session()
            mock_session_class.assert_called_once()  # Called again after close
    
    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self):
        """Test various error recovery scenarios."""
        with patch('prune.clients.vllm_client.get_aiohttp_session') as mock_get_session:
            mock_session = AsyncMock()
            mock_get_session.return_value = mock_session
            
            # Test scenario: Network error followed by success
            call_count = 0
            
            async def mock_post_behavior(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                
                if call_count == 1:
                    raise aiohttp.ClientError("Network error")
                else:
                    # Success on retry
                    mock_response = AsyncMock()
                    mock_response.raise_for_status = Mock()
                    mock_response.content = iter([
                        b'data: {"choices": [{"delta": {"content": "Recovered"}}]}\n',
                        b'data: [DONE]\n'
                    ])
                    
                    mock_ctx = AsyncMock()
                    mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
                    mock_ctx.__aexit__ = AsyncMock(return_value=False)
                    return mock_ctx
            
            mock_session.post.side_effect = mock_post_behavior
            
            # Should recover from error
            chunks = []
            async for chunk in stream_vllm_request(
                prompt="Test recovery",
                vllm_url="http://localhost:8000",
                model_name="test_model", 
                request_id="recovery_test",
                max_retries=2,
                initial_backoff=0.01
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 1
            assert chunks[0]["choices"][0]["delta"]["content"] == "Recovered"
            assert call_count == 2  # Failed once, succeeded on retry