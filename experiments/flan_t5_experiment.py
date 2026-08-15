import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_NAME = "google/flan-t5-small"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")

text = """
Rewrite the following sentence using different words while keeping exactly the same meaning.

Original: I am very happy because I got a new job.

Rewritten:
"""
# Convert text into tokens
inputs = tokenizer(text, return_tensors="pt")

# Generate output
with torch.no_grad():
  outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    num_beams=5,
    do_sample=False
)

# Convert tokens back into text
result = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nInput:")
print(text)

print("\nAI Output:")
print(result)