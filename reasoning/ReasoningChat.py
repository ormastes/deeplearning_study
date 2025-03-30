
import torch

def chat(model, tokenizer, prompt, item_idx, config, stop_sequence="```\n"):
    # Tokenize with padding to get both input_ids and attention_mask.
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=config.max_new_tokens,
        do_sample=True,  # set to False for greedy decoding
        temperature=config.temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    try:
        # Decode and return the generated text.
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error decoding output: {e}")
        return ""
    if response.strip().endswith(stop_sequence):
        return response
    else:
        config.log_answer(f"Stop sequence not reached for item:", response, item_idx)
        return response
