from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "vanillaOVO/WizardCoder-Python-7B-V1.0"
save_dir = "/workspace/data/model/reasoning/raw/WizardCoderModel"

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Download the model and tokenizer (this will cache them locally)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Save the model and tokenizer to the specified directory
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"Model and tokenizer saved to {save_dir}")

# Now you can use the model for inference, for example:
prompt = "Write a Python function to reverse a string."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
