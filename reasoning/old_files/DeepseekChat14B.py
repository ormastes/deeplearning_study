import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Specify the model name on Hugging Face
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"#"bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF"
#save_dir = "/workspace/data/model/reasoning/raw/WizardCoderModel"

# Create a quantization configuration for 8-bit
#uant_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the tokenizer and the 8-bit quantized model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",      # Automatically assign the model to available devices
    torch_dtype=torch.float16,
    trust_remote_code=True
)


# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def chat(prompt, max_new_tokens=1000, temperature=0.7):
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
