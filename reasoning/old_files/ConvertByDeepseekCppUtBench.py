import json
import time
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name on Hugging Face
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_name_short = "DeepSeek-R1-Distill-Qwen-14B"

# Load the tokenizer and the 8-bit quantized model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",      # Automatically assign the model to available devices
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.config.pad_token_id = model.config.eos_token_id

current_src_file_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_src_file_dir, "output")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def chat(prompt, item_idx, max_new_tokens=2000, temperature=0.7):
    # Tokenize the input and send tensors to the same device as the model.
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate a response with sampling enabled for creativity.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature
    )
    # Check max_new_tokens is hit.
    token_len = len(outputs[0])
    if token_len >= max_new_tokens-3:
        log(f"Max tokens reached for item {item_idx}", item_idx)
    # Decode and return the generated text.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, token_len

def get_file(file_name):
    a_file_path = os.path.join(out_dir, file_name)
    try:
        a_file = open(a_file_path, "w", encoding="utf-8")
    except Exception as e:
        print(f"Error opening error file: {e}")
        return None
    return a_file

def read_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None
    return text

def read_json_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    return data

output_file = get_file(f"responses_{model_name_short}.txt")
short_response = get_file(f"short_responses_{model_name_short}.txt")
error_file = get_file(f"errors_{model_name_short}.txt")

def log(error_text, idx):
    error_file.write(error_text + "\n")
    error_file.flush()
    short_response.write(f"Error on: {idx}\n")
    short_response.flush()
    print(error_text)


def main():
    # Read the JSON file.
    input_json_data = read_json_file("/workspace/data/model/reasoning/raw/cpp_ut_bench_json/train.json")

    num_items = len(input_json_data)
    print(f"JSON file contains {num_items} items.")

    # Read the system prompt from file.
    system_prompt = read_file("prompt_cached.txt")
    prompt_template = read_file("prompt_cached_after.txt")

    # Define a batch count for user prompts.
    session_question_count = 10
    item_idx = 0
    last_token_len = 0
    results = []

    # Process each JSON item.
    for idx in range(item_idx, num_items):
        item = input_json_data[str(idx)]
        print(f"Processing item {idx}/{num_items}, Last token length: {last_token_len}")
        if isinstance(item, dict):
            item_text = json.dumps(item, indent=2)
        else:
            item_text = str(item)

        # Replace the placeholder with the JSON item text.
        prompt_body = prompt_template.replace('${json_to_covert}', item_text)
        # Combine the system prompt and the user prompt.
        final_prompt = system_prompt + "\n" + prompt_body

        try:
            # Use local DeepSeek chat function instead of OpenAI API.
            answer, last_token_len = chat(final_prompt, item_idx)
            answer_trimmed = answer.strip()
        except Exception as e:
            answer_trimmed = f"Error during local chat call: {e}"

        # Check if the answer ends with triple backticks.
        if answer_trimmed.endswith("\n```"):
            # remove the triple backticks
            json_format_text = answer_trimmed[:-3]
            # load json from the text
            try:
                output_json_data = json.loads(json_format_text)
                # check has keys Test Target, Test Object, Input Data, Expected Output, Clang-repl Test
                for key in ["Test Target", "Test Object", "Input Data", "Expected Output", "Clang-repl Test"]:
                    if key not in output_json_data:
                        # throw Exception
                        raise Exception(f"Key {key} not found in JSON data")
                results.append(output_json_data)
            except Exception as e:
                log(f"Error loading JSON from text: {e}", idx)
        else:
            log(f"Answer for item {item_idx} does not end with triple backticks!", idx)

        item_idx += 1

        # Every session_question_count iterations, prompt the user to continue.
        #if item_idx % session_question_count == 0:
            #input(f"Processed {item_idx} items. Press Enter to continue...")

        # Optional: Sleep between calls to avoid overloading resources.
        time.sleep(1)
    # write array of json to file
    output_file.write(json.dumps(results, indent=2))
    output_file.close()
    print("Processing complete. Responses saved to", output_file.name)

if __name__ == "__main__":
    main()
