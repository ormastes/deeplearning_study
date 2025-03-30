import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

save_dir = "/workspace/data/model/reasoning/raw/WizardCoderModel"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(save_dir)
model = AutoModelForCausalLM.from_pretrained(save_dir)

# Move the model to the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

prompt = "Write a C function to add two int."
inputs = tokenizer(prompt, return_tensors="pt")

# Move inputs to the GPU
inputs = {key: value.to(device) for key, value in inputs.items()}

# Generate output
outputs = model.generate(**inputs, max_new_tokens=400)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
