from unittest.mock import Mock, patch

import pytest

from llm_interface import LLMInterface


@pytest.fixture
def llm_interface():
    with patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "test_openai_key", "HUGGINGFACE_TOKEN": "test_hf_token"},
    ):
        return LLMInterface()


def test_init(llm_interface):
    """Test initialization of LLMInterface"""
    assert llm_interface.openai_client is not None
    assert llm_interface.hf_token == "test_hf_token"
    assert isinstance(llm_interface._tokenizer_cache, dict)
    assert isinstance(llm_interface._model_cache, dict)


@patch("openai.OpenAI")
def test_query_openai(mock_openai, llm_interface):
    """Test querying OpenAI models"""
    # Setup mock response
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test response"))]
    mock_openai.return_value.chat.completions.create.return_value = mock_response

    # Test GPT-3.5
    response = llm_interface.query_openai("test prompt", "gpt-3.5-turbo")
    assert response == "Test response"
    mock_openai.return_value.chat.completions.create.assert_called_once()

    # Test GPT-4
    response = llm_interface.query_openai("test prompt", "gpt-4")
    assert response == "Test response"

    # Test o3
    response = llm_interface.query_openai("test prompt", "o3")
    assert response == "Test response"


@patch("transformers.AutoTokenizer")
@patch("transformers.AutoModelForCausalLM")
def test_query_huggingface(mock_model, mock_tokenizer, llm_interface):
    """Test querying Hugging Face models"""
    # Setup mocks
    mock_tokenizer_instance = Mock()
    mock_model_instance = Mock()
    mock_tokenizer.return_value = mock_tokenizer_instance
    mock_model.return_value = mock_model_instance

    # Mock tokenizer and model behavior
    mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3]}
    mock_model_instance.generate.return_value = [1, 2, 3, 4, 5]
    mock_tokenizer_instance.decode.return_value = "Test response"

    # Test Gemma
    response = llm_interface.query_huggingface("test prompt", "google/gemma-2-9b-it")
    assert response == "Test response"
    mock_tokenizer.assert_called_once()
    mock_model.assert_called_once()

    # Test Llama
    response = llm_interface.query_huggingface(
        "test prompt", "meta-llama/Llama-2-7b-chat-hf"
    )
    assert response == "Test response"


def test_query_invalid_model(llm_interface):
    """Test querying with invalid model"""
    with pytest.raises(ValueError, match="Unsupported model"):
        llm_interface.query("test prompt", "invalid-model")


@patch("openai.OpenAI")
def test_query_openai_error(mock_openai, llm_interface):
    """Test OpenAI API error handling"""
    mock_openai.return_value.chat.completions.create.side_effect = Exception(
        "API Error"
    )

    with pytest.raises(Exception, match="Error querying OpenAI"):
        llm_interface.query_openai("test prompt", "gpt-3.5-turbo")


@patch("transformers.AutoTokenizer")
def test_query_huggingface_error(mock_tokenizer, llm_interface):
    """Test Hugging Face error handling"""
    mock_tokenizer.side_effect = Exception("Model Error")

    with pytest.raises(Exception, match="Error querying Hugging Face model"):
        llm_interface.query_huggingface("test prompt", "google/gemma-2-9b-it")
