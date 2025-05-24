from unittest.mock import Mock, patch

import pytest

from app import LLMApp


@pytest.fixture
def llm_app():
    return LLMApp()


def test_init(llm_app):
    """Test initialization of LLMApp"""
    assert len(llm_app.available_models) > 0
    assert "gpt-3.5-turbo" in llm_app.available_models
    assert "o3" in llm_app.available_models
    assert "google/gemma-2-9b-it" in llm_app.available_models


def test_display_models(llm_app, capsys):
    """Test model display functionality"""
    llm_app.display_models()
    captured = capsys.readouterr()
    assert "Available models:" in captured.out
    for model in llm_app.available_models:
        assert model in captured.out


@patch("builtins.input")
def test_get_model_choice_valid(llm_app, mock_input):
    """Test valid model selection"""
    mock_input.return_value = "1"
    model = llm_app.get_model_choice()
    assert model == llm_app.available_models[0]


@patch("builtins.input")
def test_get_model_choice_quit(llm_app, mock_input):
    """Test quitting model selection"""
    mock_input.return_value = "q"
    model = llm_app.get_model_choice()
    assert model is None


@patch("builtins.input")
def test_get_model_choice_invalid(llm_app, mock_input, capsys):
    """Test invalid model selection"""
    mock_input.side_effect = ["999", "q"]
    llm_app.get_model_choice()
    captured = capsys.readouterr()
    assert "Invalid choice" in captured.out


@patch("builtins.input")
def test_get_prompt_direct(llm_app, mock_input):
    """Test direct prompt input"""
    mock_input.return_value = "test prompt"
    prompt = llm_app.get_prompt()
    assert prompt == "test prompt"


@patch("builtins.input")
def test_get_prompt_quit(llm_app, mock_input):
    """Test quitting prompt input"""
    mock_input.return_value = "q"
    prompt = llm_app.get_prompt()
    assert prompt is None


@patch("builtins.input")
def test_get_prompt_file(llm_app, mock_input, tmp_path):
    """Test reading prompt from file"""
    # Create a temporary file with a prompt
    prompt_file = tmp_path / "test_prompt.txt"
    prompt_file.write_text("test prompt from file")

    # Mock the input sequence for file input
    mock_input.side_effect = ["2", str(prompt_file)]

    prompt = llm_app.get_prompt()
    assert prompt == "test prompt from file"


@patch("builtins.input")
def test_get_prompt_multiline(llm_app, mock_input):
    """Test multi-line prompt input"""
    mock_input.side_effect = ["3", "line 1", "line 2", "END"]
    prompt = llm_app.get_prompt()
    assert prompt == "line 1\nline 2"


@patch("app.LLMInterface")
def test_save_response(llm_app, mock_llm_interface, tmp_path):
    """Test saving response to file"""
    with patch("os.path.exists", return_value=False), patch("os.makedirs"), patch(
        "builtins.open", create=True
    ) as mock_open:
        llm_app.save_response("test prompt", "test response", "gpt-3.5-turbo")
        mock_open.assert_called_once()
