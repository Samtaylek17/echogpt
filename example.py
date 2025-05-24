from llm_interface import LLMInterface


def main():
    # Initialize the LLM interface
    llm = LLMInterface()

    # Example prompt
    prompt = "What is the capital of France?"

    # Query different models
    try:
        # Query OpenAI
        print("\nQuerying GPT-3.5-Turbo:")
        response = llm.query(prompt, "gpt-3.5-turbo")
        print(f"Response: {response}")

        # Query Gemma
        print("\nQuerying Gemma:")
        response = llm.query(prompt, "google/gemma-2-9b-it")
        print(f"Response: {response}")

        # Query Llama
        print("\nQuerying Llama:")
        response = llm.query(prompt, "meta-llama/Llama-2-7b-chat-hf")
        print(f"Response: {response}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
