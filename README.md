# LLM Interface

A Python interface for querying multiple Large Language Models (LLMs) including OpenAI models, Google's Gemma, and Meta's Llama-2.

## Supported Models

- OpenAI models (e.g., GPT-3.5-Turbo, GPT-4)
- Google's Gemma (gemma-2-9b-it)
- Meta's Llama-2 (Llama-2-7b-chat-hf)

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

## Usage

### Interactive Application

The easiest way to use this interface is through the interactive command-line application:

```bash
python app.py
```

This will start an interactive session where you can:
1. Select from available models
2. Enter your prompts
3. Get responses from the selected model
4. Continue querying different models or exit

### Programmatic Usage

You can also use the interface programmatically in your Python code:

```python
from llm_interface import LLMInterface

# Initialize the interface
llm = LLMInterface()

# Query a model
response = llm.query(
    prompt="What is the capital of France?",
    model="gpt-3.5-turbo"  # or "google/gemma-2-9b-it" or "meta-llama/Llama-2-7b-chat-hf"
)
print(response)
```

You can also run the example script:
```bash
python example.py
```

## Features

- Interactive command-line interface for easy model querying
- Unified interface for multiple LLM providers
- Automatic model caching for Hugging Face models
- Error handling and proper exception management
- Support for both API-based (OpenAI) and local (Hugging Face) models

## Requirements

- Python 3.8+
- OpenAI API key
- Hugging Face token (for accessing Gemma and Llama-2 models)
- Sufficient GPU memory for running local models (recommended)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
