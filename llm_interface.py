import os
from typing import Any, Dict, Optional

import openai
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMInterface:
    def __init__(self):
        load_dotenv()
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")

        # Initialize model caches
        self._tokenizer_cache: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}

    def query_openai(self, prompt: str, model: str = "o3") -> str:
        """
        Query OpenAI models.

        Args:
            prompt (str): The input prompt
            model (str): The OpenAI model to use (default: o3)

        Returns:
            str: The model's response
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error querying OpenAI: {str(e)}")

    def query_huggingface(self, prompt: str, model_name: str) -> str:
        """
        Query Hugging Face models (Gemma or Llama).

        Args:
            prompt (str): The input prompt
            model_name (str): The Hugging Face model name

        Returns:
            str: The model's response
        """
        try:
            # Get or load tokenizer
            if model_name not in self._tokenizer_cache:
                self._tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                    model_name, token=self.hf_token
                )
            tokenizer = self._tokenizer_cache[model_name]

            # Get or load model
            if model_name not in self._model_cache:
                self._model_cache[model_name] = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    token=self.hf_token,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            model = self._model_cache[model_name]

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the input prompt from the response
            response = response[len(prompt) :].strip()
            return response

        except Exception as e:
            raise Exception(f"Error querying Hugging Face model: {str(e)}")

    def query(self, prompt: str, model: str) -> str:
        """
        Unified interface to query any supported model.

        Args:
            prompt (str): The input prompt
            model (str): The model to use. Can be:
                - OpenAI models (e.g., "gpt-3.5-turbo", "gpt-4", "o1-preview", "o3")
                - "google/gemma-2-9b-it"
                - "meta-llama/Llama-2-7b-chat-hf"

        Returns:
            str: The model's response
        """
        if model in ["gpt-3.5-turbo", "gpt-4", "o1-preview", "o3"]:
            return self.query_openai(prompt, model)
        elif model in ["google/gemma-2-9b-it", "meta-llama/Llama-2-7b-chat-hf"]:
            return self.query_huggingface(prompt, model)
        else:
            raise ValueError(f"Unsupported model: {model}")
