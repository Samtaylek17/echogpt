from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Automatically mock environment variables for all tests"""
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test_openai_key", "HUGGINGFACE_TOKEN": "test_hf_token"},
    ):
        yield


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response"""
    return {"choices": [{"message": {"content": "Test response"}}]}


@pytest.fixture
def mock_huggingface_response():
    """Mock Hugging Face model response"""
    return "Test response"
