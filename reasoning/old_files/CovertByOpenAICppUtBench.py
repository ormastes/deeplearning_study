import json
import openai
import time


def main():
    # Take API key from console input and set it for OpenAI.
    api_key = input("Enter your OpenAI API key: ")
    openai.api_key = api_key

    # Read the JSON file.
    json_file_path = "/workspace/data/model/reasoning/raw/cpp_ut_bench_json/train.json"
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # Print the number of items in the JSON file.
    num_items = len(json_data)
    print(f"JSON file contains {num_items} items.")

    # Read the prompt template from file.
    try:
        prompt_file = "prompt_cached.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    # Read the prompt template from file.
    try:
        prompt_file = "prompt_cached_after.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return

    # Set the model to "o3-mini".
    model_name = "o3-mini"

    # Define session_question_count to prompt the user every 10 iterations.
    session_question_count = 10

    # Initialize the item index.
    item_idx = 0

    # Open a file to save responses.
    output_file_path = f"responses_{item_idx}.txt"
    try:
        output_file = open(output_file_path, "w", encoding="utf-8")
    except Exception as e:
        print(f"Error opening output file: {e}")
        return

    # Loop through each JSON item.
    for idx in range(item_idx, num_items):
        item = json_data[idx]
        # Convert the JSON item to text.
        if isinstance(item, dict):
            item_text = json.dumps(item, indent=2)
        else:
            item_text = str(item)

        # Replace the placeholder in the prompt template with the JSON item text.
        prompt_body = prompt_template.replace('${json_to_covert}', item_text)
        # Apply the cached prompt before the prompt body.
        final_prompt = prompt_body

        # Call the OpenAI API using the "o3-mini" model.
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.7,
            )
            answer = response.choices[0].message.content
            answer_trimmed = answer.strip()
        except Exception as e:
            answer_trimmed = f"Error during API call: {e}"

        # Check if the trimmed answer ends with triple backticks.
        if not answer_trimmed.endswith("```"):
            print(f"Warning: Answer for item {item_idx} does not end with triple backticks!")

        # Create the output line.
        output_line = f"Response for item {item_idx}: {answer_trimmed}\n"
        print(output_line)
        output_file.write(output_line)

        # Increase the item index.
        item_idx += 1

        # Every session_question_count iterations, prompt the user to continue.
        if item_idx % session_question_count == 0:
            input(f"Processed {item_idx} items. Press Enter to continue...")

        # Optional: Sleep between API calls to avoid rate limiting.
        time.sleep(1)

    output_file.close()
    print("Processing complete. Responses saved to", output_file_path)


if __name__ == "__main__":
    main()
