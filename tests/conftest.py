"""Pytest configuration and common fixtures for prune module tests."""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Mock heavy dependencies that might not be available
mock_modules = [
    'numpy', 'pandas', 'datasets', 'transformers', 'torch', 
    'faiss', 'faiss-cpu', 'sentence_transformers', 'sympy',
    'sklearn', 'scikit-learn', 'scipy', 'matplotlib', 'seaborn',
    'tqdm', 'rich', 'aiohttp', 'aiosignal', 'yarl',
    'huggingface_hub'
]

for module in mock_modules:
    if module not in sys.modules:
        sys.modules[module] = MagicMock()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_json_file(temp_dir):
    """Create a mock JSON file for testing."""
    def _create_json_file(data, filename="test.json"):
        file_path = temp_dir / filename
        with open(file_path, 'w') as f:
            json.dump(data, f)
        return file_path
    return _create_json_file


@pytest.fixture
def mock_env():
    """Mock environment variables."""
    with patch.dict(os.environ, {"USER": "test_user"}, clear=False):
        yield


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    return [
        {
            "id": "test_1",
            "question": "What is 2 + 2?",
            "answer": "4",
            "target": "4"
        },
        {
            "id": "test_2", 
            "question": "What is the capital of France?",
            "answer": "Paris",
            "target": "Paris"
        }
    ]


@pytest.fixture
def mock_huggingface_dataset():
    """Mock HuggingFace dataset response."""
    mock_dataset = Mock()
    mock_dataset.column_names = ["question", "answer"]
    mock_dataset.__iter__ = Mock(return_value=iter([
        {"question": "Test question 1", "answer": "Test answer 1"},
        {"question": "Test question 2", "answer": "Test answer 2"}
    ]))
    return mock_dataset


@pytest.fixture
def sample_kv_cache_data():
    """Sample KV cache data for testing."""
    return {
        "avg_kv_cache_usage": 0.75,
        "total_tokens": 1000,
        "kv_cache_stats": {
            "mean": 0.75,
            "std": 0.1
        }
    }


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for async tests."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json = MagicMock(return_value={"choices": [{"text": "test response"}]})
    mock_response.status = 200
    mock_session.post = MagicMock(return_value=mock_response)
    return mock_session