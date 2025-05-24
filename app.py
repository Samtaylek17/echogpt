import datetime
import os
import sys
from typing import List, Optional

from llm_interface import LLMInterface


class LLMApp:
    def __init__(self):
        self.llm = LLMInterface()
        self.available_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "o1-preview",
            "o3",
            "google/gemma-2-9b-it",
            "meta-llama/Llama-2-7b-chat-hf",
        ]

    def display_models(self):
        print("\nAvailable models:")
        for i, model in enumerate(self.available_models, 1):
            print(f"{i}. {model}")

    def get_model_choice(self) -> Optional[str]:
        while True:
            try:
                choice = input("\nSelect a model (number) or 'q' to quit: ").strip()
                if choice.lower() == "q":
                    return None

                index = int(choice) - 1
                if 0 <= index < len(self.available_models):
                    return self.available_models[index]
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    def get_prompt(self) -> Optional[str]:
        print("\nChoose input method:")
        print("1. Enter prompt directly")
        print("2. Read from file")
        print("3. Multi-line input (type 'END' on a new line when done)")
        print("q. Quit")

        choice = input("> ").strip()

        if choice.lower() == "q":
            return None
        elif choice == "1":
            print("\nEnter your prompt:")
            return input("> ").strip()
        elif choice == "2":
            while True:
                file_path = input("\nEnter the path to your prompt file: ").strip()
                if file_path.lower() == "q":
                    return None
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        return f.read().strip()
                except FileNotFoundError:
                    print(
                        f"Error: File '{file_path}' not found. Try again or enter 'q' to quit."
                    )
                except Exception as e:
                    print(
                        f"Error reading file: {str(e)}. Try again or enter 'q' to quit."
                    )
        elif choice == "3":
            print("\nEnter your prompt (type 'END' on a new line when done):")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            return "\n".join(lines)
        else:
            print("Invalid choice. Please try again.")
            return self.get_prompt()

    def save_response(self, prompt: str, response: str, model: str):
        """Save the prompt and response to a file."""
        if not os.path.exists("responses"):
            os.makedirs("responses")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"responses/response_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Model: {model}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("\nPrompt:\n")
            f.write("-" * 50 + "\n")
            f.write(prompt)
            f.write("\n\nResponse:\n")
            f.write("-" * 50 + "\n")
            f.write(response)

        print(f"\nResponse saved to: {filename}")

    def run(self):
        print("Welcome to the LLM Query Interface!")
        print("This application allows you to query different LLM models.")

        while True:
            self.display_models()
            model = self.get_model_choice()

            if model is None:
                break

            prompt = self.get_prompt()
            if prompt is None:
                break

            try:
                print(f"\nQuerying {model}...")
                response = self.llm.query(prompt, model)
                print("\nResponse:")
                print("-" * 50)
                print(response)
                print("-" * 50)

                # Ask if user wants to save the response
                save_response = (
                    input("\nWould you like to save this response? (y/n): ")
                    .strip()
                    .lower()
                )
                if save_response == "y":
                    self.save_response(prompt, response, model)

            except Exception as e:
                print(f"\nError: {str(e)}")

            continue_query = (
                input("\nWould you like to query another model? (y/n): ")
                .strip()
                .lower()
            )
            if continue_query != "y":
                break

        print("\nThank you for using the LLM Query Interface!")


def main():
    try:
        app = LLMApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
