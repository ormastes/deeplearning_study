import json
import time
import torch
import os
import transformer_engine.pytorch as te  # NVIDIA Transformer Engine for PyTorch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name on Hugging Face
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_name_short = "DeepSeek-R1-Distill-Qwen-14B"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically assign the model to available devices
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Create an output directory if it doesn't exist
current_src_file_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(current_src_file_dir, "output")
os.makedirs(out_dir, exist_ok=True)


def chat(prompt, item_idx, max_new_tokens=12000, temperature=0.1, stop_sequence="```\n"):
    # Tokenize with padding to get both input_ids and attention_mask.
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    # Start with the initial tokens.
    output_ids = input_ids
    # Tokenize the stop sequence.
    stop_ids = tokenizer(stop_sequence, add_special_tokens=False).input_ids[0]

    # Generate tokens one by one.
    for _ in range(max_new_tokens):
        # Generate the next token while passing the attention mask and pad_token_id.
        next_token = model.generate(
            input_ids=output_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            do_sample=True,  # set to False for greedy decoding
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        # Get the newly generated token (last token of the output).
        new_token = next_token[:, -1:]
        # Append the new token to the output_ids.
        output_ids = torch.cat([output_ids, new_token], dim=1)
        # Update attention_mask by appending a 1 for the new token.
        new_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, new_mask], dim=1)

        # Check if the generated tokens end with the stop sequence.
        if torch.equal(output_ids[0, stop_ids.shape[0]:],
                       torch.tensor(stop_ids, device=output_ids.device)):
            break

    token_len = output_ids.shape[1]
    if token_len >= max_new_tokens - 3:
        log(f"Max tokens reached for item {item_idx}", "Token output", item_idx)
    # Decode and return the generated text.
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response, token_len


def get_file(file_name):
    file_path = os.path.join(out_dir, file_name)
    try:
        return open(file_path, "w", encoding="utf-8")
    except Exception as e:
        print(f"Error opening file: {e}")
        return None


def read_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def read_json_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


output_file = get_file(f"responses_{model_name_short}.txt")
short_response = get_file(f"short_responses_{model_name_short}.txt")
error_file = get_file(f"errors_{model_name_short}.txt")

output_file.write("[\n")


def log(error_text, answer, idx):
    error_file.write(error_text + "\n")
    error_file.write(answer + "\n")
    error_file.flush()
    short_response.write(f"Error on: {idx}\n")
    short_response.flush()
    print(error_text)
    print(answer)


def main():
    # Read input JSON data.
    input_json_data = read_json_file("/workspace/data/model/reasoning/raw/cpp_ut_bench_json/train.json")
    num_items = len(input_json_data)
    print(f"JSON file contains {num_items} items.")

    system_prompt = read_file("prompt_cached.txt")
    prompt_template = read_file("prompt_cached_after.txt")
    session_question_count = 10
    item_idx = 0
    last_token_len = 0
    results = []

    for idx in range(item_idx, num_items):
        item = input_json_data[str(idx)]
        print(f"Processing item {idx}/{num_items}, Last token length: {last_token_len}")
        item_text = json.dumps(item, indent=2) if isinstance(item, dict) else str(item)
        prompt_body = prompt_template.replace('${json_to_covert}', item_text)
        final_prompt = system_prompt + "\n" + prompt_body

        try:
            answer, last_token_len = chat(final_prompt, item_idx)
            answer_trimmed = answer.strip()
        except Exception as e:
            answer_trimmed = f"Error during chat call: {e}"

        if answer_trimmed.endswith("\n```"):
            json_format_text = answer_trimmed[:-3]
            last_dotdotdot_idx = json_format_text.rfind("```json\n")
            if last_dotdotdot_idx != -1:
                json_format_text = json_format_text[last_dotdotdot_idx + 7:]
            json_format_text = json_format_text.strip()
            try:
                output_json_data = json.loads(json_format_text)
                for key in ["Test Target", "Test Object", "Input Data", "Expected Output", "Clang-repl Test"]:
                    if key not in output_json_data:
                        raise Exception(f"Key {key} not found in JSON data")
                results.append(output_json_data)
                output_file.write(output_json_data)
                if idx < num_items - 1:
                    output_file.write(",\n")
                else:
                    output_file.write("\n")
                output_file.flush()
            except Exception as e:
                log(f"Error loading JSON from text: {e}", answer_trimmed, idx)
        else:
            log(f"Answer for item {item_idx} does not end with triple backticks!", answer_trimmed, idx)

        item_idx += 1
        time.sleep(1)

    output_file.write("]\n")
    output_file.close()
    print("Processing complete. Responses saved to", output_file.name)


if __name__ == "__main__":
    main()
