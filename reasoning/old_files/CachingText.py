import openai
import uuid
import json
import os

CACHE_FILE = "cache.json"


def load_cache():
    """Load the cached prompts from a JSON file."""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_cache(cache):
    """Save the current cache to a JSON file."""
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def cache_prompt(prompt):
    """Cache the prompt and return a unique key."""
    cache = load_cache()
    key = str(uuid.uuid4())
    cache[key] = prompt
    save_cache(cache)
    return key


def main():
    # Take API key from console input and set it for OpenAI.
    api_key = input("Enter your OpenAI API key: ")
    openai.api_key = api_key

    # Ask for a prompt to cache.
    #prompt = input("Enter your prompt: ")
    # read from prompt.txt
    with open('prompt_cached.txt', 'r') as file:
        prompt = file.read()

    # Cache the prompt and get a unique key.
    prompt_key = cache_prompt(prompt)
    print("Prompt has been cached with key:", prompt_key)

    # Optionally, you can retrieve the prompt later:
    cache = load_cache()
    retrieved_prompt = cache.get(prompt_key)
    print("Retrieved prompt from cache:", retrieved_prompt)


if __name__ == "__main__":
    main()
