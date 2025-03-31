import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name on Hugging Face
model_name = "yuanzu/DeepSeek-R1-INT8"
save_dir = "/workspace/data/model/reasoning/raw/WizardCoderModel"

# Load the tokenizer and the 8-bit quantized model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(
    save_dir,
    device_map="auto",      # Automatically assign the model to available devices
    load_in_8bit=True       # Load the model in 8-bit precision
)


# Move the model to the GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)


def chat(prompt, max_new_tokens=100, temperature=0.7):
    # Tokenize input and move tensors to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # Generate response (sampling enabled for creativity)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature
    )
    # Decode and return the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Chat with DeepSeek-R1-INT8 (type 'exit' to quit)")
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            break
        answer = chat(user_input)
        print("Bot:", answer)
